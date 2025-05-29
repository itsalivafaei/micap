"""
Entity Recognition System for Brand and Product Detection
Implements custom NER for market intelligence
"""

import json
import logging
from typing import Dict, List, Tuple, Set, Optional
from difflib import SequenceMatcher
from src.utils.path_utils import get_path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf, explode, array, when, lit, lower
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, FloatType
# import spacy
# from spacy.matcher import Matcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrandRecognizer:
    """
    Custom brand recognition system with fuzzy matching
    """

    def __init__(
            self,
            config_path: str = get_path('config/brands/brand_config.json'),
            use_spacy: bool = True
    ):
        """
        Initialize brand recognizer with configuration

        Args:
            config_path: Path to brand configuration JSON
        """
        self.config = self._load_config(config_path)
        self.brands = self._extract_all_brands()
        self.brand_patterns = self._create_brand_patterns()

        # Load spaCy model for advanced NER
        # Load spaCy model only when requested (heavy -> pulls in Metal/MPS)
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                # self.nlp = spacy.load("en_core_web_sm")
                self.nlp = spacy.load("not_now")
            except:
                logger.warning("spaCy model not found. Using pattern matching only.")
                # self.nlp = None

        logger.info(f"Initialized BrandRecognizer with {len(self.brands)} brands")

    def _load_config(self, config_path: str) -> Dict:
        """Load brand configuration from JSON"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def _extract_all_brands(self) -> Dict[str, Dict]:
        """Extract all brands from configuration"""
        brands = {}
        for industry, industry_data in self.config['industries'].items():
            for brand_data in industry_data['brands']:
                brand_name = brand_data['name'].lower()
                brands[brand_name] = {
                    'industry': industry,
                    'aliases': [alias.lower() for alias in brand_data.get('aliases', [])],
                    'products': [product.lower() for product in brand_data.get('products', [])],
                    'keywords': [keyword.lower() for keyword in brand_data.get('keywords', [])],
                    'competitors': [comp.lower() for comp in brand_data.get('competitors', [])]
                }
        return brands

    def _create_brand_patterns(self) -> Dict[str, List[str]]:
        """Create regex patterns for brand matching"""
        patterns = {}
        for brand, data in self.brands.items():
            # Create patterns for brand name and aliases
            brand_patterns = [brand] + data['aliases']
            # Add product patterns
            brand_patterns.extend(data['products'])
            # Create regex pattern
            patterns[brand] = brand_patterns
        return patterns

    @property
    def competitor_map(self) -> Dict[str, Set[str]]:
        """
        Bidirectional map {brand → set(competitors)} built from the
        config.  Constructed lazily so it costs zero time when unused.
        """
        if not hasattr(self, "_competitor_map"):
            comp_map: Dict[str, Set[str]] = {}
            for brand, data in self.brands.items():
                comp_map.setdefault(brand, set()).update(data['competitors'])
                for rival in data['competitors']:
                    comp_map.setdefault(rival, set()).add(brand)
            self._competitor_map = comp_map
        return self._competitor_map

    def recognize_brands(self, text: str) -> List[Tuple[str, float]]:
        """
        Recognize brands in text with confidence scores

        Args:
            text: Input text

        Returns:
            List of (brand, confidence) tuples
        """
        text_lower = text.lower()
        found_brands = []

        # Direct pattern matching
        for brand, patterns in self.brand_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Calculate confidence based on match quality
                    confidence = self._calculate_confidence(pattern, text_lower)
                    found_brands.append((brand, confidence))
                    break

        # Fuzzy matching for variations
        words = text_lower.split()
        for word in words:
            for brand in self.brands:
                if self._fuzzy_match(word, brand) > 0.85:
                    found_brands.append((brand, 0.85))

        # Remove duplicates and return highest confidence for each brand
        brand_dict = {}
        for brand, conf in found_brands:
            if brand not in brand_dict or conf > brand_dict[brand]:
                brand_dict[brand] = conf

        return [(brand, conf) for brand, conf in brand_dict.items()]

    def _calculate_confidence(self, pattern: str, text: str) -> float:
        """Calculate confidence score for brand match"""
        # Base confidence
        confidence = 0.9

        # Boost for exact match
        if f" {pattern} " in f" {text} ":
            confidence = 1.0
        # Reduce for partial match
        elif pattern in text:
            confidence = 0.8

        return confidence

    def _fuzzy_match(self, word: str, brand: str) -> float:
        """Calculate fuzzy match score between word and brand"""
        return SequenceMatcher(None, word, brand).ratio()


class ProductExtractor:
    """
    Extract product mentions and features from text
    """

    def __init__(self, brand_recognizer: BrandRecognizer):
        """
        Initialize product extractor

        Args:
            brand_recognizer: Initialized BrandRecognizer instance
        """
        self.brand_recognizer = brand_recognizer
        self.product_patterns = self._create_product_patterns()

    def _create_product_patterns(self) -> Dict[str, List[str]]:
        """Create patterns for product extraction"""
        patterns = {}
        for brand, data in self.brand_recognizer.brands.items():
            patterns[brand] = data['products']
        return patterns

    def extract_products(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Extract product mentions with associated brands

        Args:
            text: Input text

        Returns:
            List of (product, brand, confidence) tuples
        """
        text_lower = text.lower()
        found_products = []

        # First, identify brands in text
        brands = self.brand_recognizer.recognize_brands(text)
        brand_names = [b[0] for b in brands]

        # Extract products for identified brands
        for brand in brand_names:
            if brand in self.product_patterns:
                for product in self.product_patterns[brand]:
                    if product in text_lower:
                        confidence = 0.9 if f" {product} " in f" {text_lower} " else 0.7
                        found_products.append((product, brand, confidence))

        return found_products


class CompetitorMapper:
    """
    Map and analyze competitor relationships
    """

    def __init__(self, brand_recognizer: BrandRecognizer):
        """
        Initialize competitor mapper

        Args:
            brand_recognizer: Initialized BrandRecognizer instance
        """
        self.brand_recognizer = brand_recognizer
        self.competitor_map = self._build_competitor_map()

    def _build_competitor_map(self) -> Dict[str, Set[str]]:
        """Build bidirectional competitor relationships"""
        comp_map = {}

        for brand, data in self.brand_recognizer.brands.items():
            if brand not in comp_map:
                comp_map[brand] = set()

            # Add declared competitors
            for competitor in data['competitors']:
                comp_map[brand].add(competitor)

                # Add reverse relationship
                if competitor not in comp_map:
                    comp_map[competitor] = set()
                comp_map[competitor].add(brand)

        return comp_map

    def get_competitors(self, brand: str) -> List[str]:
        """Get competitors for a brand"""
        brand_lower = brand.lower()
        return list(self.competitor_map.get(brand_lower, set()))

    def identify_competitive_context(self, text: str) -> List[Dict]:
        """
        Identify competitive contexts in text

        Args:
            text: Input text

        Returns:
            List of competitive context dictionaries
        """
        # Find all brands mentioned
        mentioned_brands = self.brand_recognizer.recognize_brands(text)
        brand_names = [b[0] for b in mentioned_brands]

        competitive_contexts = []

        # Check for competitor pairs
        for i, brand1 in enumerate(brand_names):
            for brand2 in brand_names[i + 1:]:
                if brand2 in self.competitor_map.get(brand1, set()):
                    competitive_contexts.append({
                        'brand1': brand1,
                        'brand2': brand2,
                        'relationship': 'direct_competitor',
                        'context': self._extract_comparison_context(text, brand1, brand2)
                    })

        return competitive_contexts

    def _extract_comparison_context(self, text: str, brand1: str, brand2: str) -> str:
        """Extract comparison context between two brands"""
        # Simple implementation - can be enhanced with more sophisticated NLP
        comparison_words = ['better', 'worse', 'vs', 'versus', 'compared to',
                            'more than', 'less than', 'superior', 'inferior']

        text_lower = text.lower()
        for word in comparison_words:
            if word in text_lower:
                return 'comparison'

        return 'co-mention'


class EntityDisambiguator:
    """
    Disambiguate entity mentions using context
    """

    def __init__(self, brand_recognizer: BrandRecognizer):
        """
        Initialize entity disambiguator

        Args:
            brand_recognizer: Initialized BrandRecognizer instance
        """
        self.brand_recognizer = brand_recognizer
        self.context_clues = self._build_context_clues()

    def _build_context_clues(self) -> Dict[str, Dict[str, List[str]]]:
        """Build context clues for disambiguation"""
        clues = {}

        for brand, data in self.brand_recognizer.brands.items():
            clues[brand] = {
                'industry': data['industry'],
                'keywords': data['keywords'],
                'products': data['products']
            }

        return clues

    def disambiguate(self, text: str, entities: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Disambiguate entities using context

        Args:
            text: Input text
            entities: List of (entity, confidence) tuples

        Returns:
            Disambiguated entities with updated confidence scores
        """
        text_lower = text.lower()
        disambiguated = []

        for entity, confidence in entities:
            if entity in self.context_clues:
                # Check for context keywords
                context_score = 0
                for keyword in self.context_clues[entity]['keywords']:
                    if keyword in text_lower:
                        context_score += 0.1

                # Check for product mentions
                for product in self.context_clues[entity]['products']:
                    if product in text_lower:
                        context_score += 0.15

                # Update confidence based on context
                new_confidence = min(1.0, confidence + context_score)
                disambiguated.append((entity, new_confidence))
            else:
                disambiguated.append((entity, confidence))

        return disambiguated


# Spark UDF wrappers
def create_brand_recognition_udf(brand_recognizer: BrandRecognizer):
    """Create Spark UDF for brand recognition"""

    def recognize_brands_wrapper(text):
        if text is None:
            return []
        brands = brand_recognizer.recognize_brands(text)
        return [f"{brand}:{confidence:.2f}" for brand, confidence in brands]

    return udf(recognize_brands_wrapper, ArrayType(StringType()))


def create_product_extraction_udf(product_extractor: ProductExtractor):
    """Create Spark UDF for product extraction"""

    def extract_products_wrapper(text):
        if text is None:
            return []
        products = product_extractor.extract_products(text)
        return [f"{product}|{brand}:{confidence:.2f}" for product, brand, confidence in products]

    return udf(extract_products_wrapper, ArrayType(StringType()))


# Main function for testing
def main():
    """Test entity recognition system"""
    # Initialize components
    brand_recognizer = BrandRecognizer()
    product_extractor = ProductExtractor(brand_recognizer)
    competitor_mapper = CompetitorMapper(brand_recognizer)
    disambiguator = EntityDisambiguator(brand_recognizer)

    # Test texts
    test_texts = [
        "Just got the new iPhone 15 Pro! Much better camera than my old Samsung Galaxy.",
        "Tesla Model 3 is the best electric car, way ahead of Ford's EVs.",
        "Comparing MacBook Pro vs Dell XPS for programming.",
        "Apple Watch Series 9 has better health features than Galaxy Watch."
    ]

    for text in test_texts:
        print(f"\nText: {text}")

        # Recognize brands
        brands = brand_recognizer.recognize_brands(text)
        print(f"Brands: {brands}")

        # Extract products
        products = product_extractor.extract_products(text)
        print(f"Products: {products}")

        # Identify competitive context
        contexts = competitor_mapper.identify_competitive_context(text)
        print(f"Competitive contexts: {contexts}")

        # Disambiguate
        disambiguated = disambiguator.disambiguate(text, brands)
        print(f"Disambiguated: {disambiguated}")


if __name__ == "__main__":
    main()