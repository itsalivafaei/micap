"""Hybrid Entity Recognition Module for MICAP.

Uses fuzzywuzzy as the primary matching approach for robust entity detection
Dependencies: fuzzywuzzy (required), spacy (optional)
"""

import json
import logging
import re
from typing import Dict, List, Tuple, Set, Optional, Union
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf, explode, array, when, lit, lower
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, FloatType

from src.utils.path_utils import get_path

# fuzzywuzzy is now a required dependency
try:
    from fuzzywuzzy import fuzz, process
    HAS_FUZZYWUZZY = True
except ImportError:
    HAS_FUZZYWUZZY = False
    raise ImportError(
        "fuzzywuzzy is required for entity recognition. "
        "Install with: pip install fuzzywuzzy[speedup]"
    )

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/entity_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BrandRecognizer:
    """Advanced brand recognizer using fuzzywuzzy as the primary matching approach.
    
    Features:
    - fuzzywuzzy-first fuzzy matching with multiple scoring algorithms
    - Multi-level confidence scoring based on different match types
    - Comprehensive brand detection including misspellings and variations
    - Optional spaCy integration for enhanced NER
    - Detailed match metadata and debugging information
    """
    
    def __init__(
        self,
        config_path: str = None,
        fuzzy_threshold: float = 70,  # Changed from similarity to fuzzywuzzy score
        exact_threshold: float = 90,
        use_spacy: bool = False,
        scorer_weights: Dict[str, float] = None
    ):
        """Initialize brand recognizer with fuzzywuzzy-first approach.
        
        Args:
            config_path: Path to brand configuration JSON
            fuzzy_threshold: Minimum fuzzywuzzy score for matches (0-100)
            exact_threshold: Score threshold for considering matches "exact" (0-100)
            use_spacy: Whether to load spaCy model for additional NER
            scorer_weights: Weights for different fuzzywuzzy scorers
        """
        if not HAS_FUZZYWUZZY:
            raise RuntimeError("fuzzywuzzy is required but not available")
            
        if config_path is None:
            config_path = str(get_path('config/brands/brand_config.json'))
            
        self.fuzzy_threshold = fuzzy_threshold
        self.exact_threshold = exact_threshold
        
        # Configure multiple fuzzywuzzy scorers with weights
        self.scorer_weights = scorer_weights or {
            'ratio': 0.4,           # Standard ratio
            'partial_ratio': 0.2,   # Substring matching
            'token_sort_ratio': 0.25, # Order-independent
            'token_set_ratio': 0.15   # Set-based matching
        }
        
        # Load configuration and build lookup structures
        self.config = self._load_config(config_path)
        self.brands = self._extract_all_brands()
        
        # Create optimized lookup structures for fuzzywuzzy
        self.search_terms = self._build_search_terms()
        self.term_to_brand = self._build_term_mapping()
        
        # Initialize spaCy if requested
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load spaCy model: {e}")
                
        logger.info(f"Initialized BrandRecognizer with {len(self.brands)} brands")
        logger.info(f"Using fuzzywuzzy with {len(self.search_terms)} search terms")
        logger.info(f"Scorer weights: {self.scorer_weights}")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load brand configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded brand config from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
            
    def _extract_all_brands(self) -> Dict[str, Dict]:
        """Extract all brands from configuration."""
        brands = {}
        for industry, industry_data in self.config.get('industries', {}).items():
            for brand_data in industry_data.get('brands', []):
                brand_name = brand_data['name'].lower()
                brands[brand_name] = {
                    'original_name': brand_data['name'],
                    'industry': industry,
                    'aliases': [alias.lower() for alias in brand_data.get('aliases', [])],
                    'products': [product.lower() for product in brand_data.get('products', [])],
                    'keywords': [keyword.lower() for keyword in brand_data.get('keywords', [])],
                    'competitors': [comp.lower() for comp in brand_data.get('competitors', [])]
                }
        return brands
        
    def _build_search_terms(self) -> List[str]:
        """Build comprehensive list of search terms for fuzzywuzzy matching."""
        terms = []
        for brand_name, data in self.brands.items():
            # Add brand name
            terms.append(data['original_name'])
            # Add all aliases
            terms.extend([alias for alias in data['aliases']])
            # Add products
            terms.extend([product for product in data['products']])
            # Add important keywords (filter out common words)
            keywords = [kw for kw in data['keywords'] if len(kw) > 3]
            terms.extend(keywords)
        
        # Remove duplicates and sort by length (longer terms first for better matching)
        unique_terms = list(set(terms))
        unique_terms.sort(key=len, reverse=True)
        
        logger.info(f"Built {len(unique_terms)} unique search terms")
        return unique_terms
        
    def _build_term_mapping(self) -> Dict[str, str]:
        """Map search terms back to their canonical brand names."""
        mapping = {}
        for brand_name, data in self.brands.items():
            original = data['original_name']
            
            # Map all variations to original brand name
            mapping[original] = original
            for alias in data['aliases']:
                mapping[alias] = original
            for product in data['products']:
                mapping[product] = original
            for keyword in data['keywords']:
                if len(keyword) > 3:  # Only significant keywords
                    mapping[keyword] = original
                    
        return mapping
        
    def recognize_brands(self, text: str, return_details: bool = True) -> Union[List[Dict], List[Tuple]]:
        """Recognize brand mentions using fuzzywuzzy-first approach.
        
        Args:
            text: Input text to analyze
            return_details: If True, return detailed dicts; if False, return simple tuples
            
        Returns:
            List of brand detections with confidence and metadata
        """
        if not text or not text.strip():
            return []
            
        logger.debug(f"Processing text: {text[:100]}...")
        
        found_brands = []
        processed_positions = set()
        
        # 1. Primary approach: Advanced fuzzywuzzy matching on text chunks
        text_chunks = self._extract_text_chunks(text)
        
        for chunk in text_chunks:
            if chunk['position'] in processed_positions:
                continue
                
            matches = self._fuzzy_match_chunk(chunk['text'], chunk['position'])
            
            for match in matches:
                if match['position'] not in processed_positions:
                    found_brands.append(match)
                    # Mark range as processed
                    start, end = match['position'], match.get('end_position', match['position'] + len(match['matched_text']))
                    processed_positions.update(range(start, end + 1))
        
        # 2. Secondary approach: Word-level fuzzy matching for missed entities
        words = self._extract_words_with_positions(text)
        
        for word_info in words:
            if any(pos in processed_positions for pos in range(word_info['start'], word_info['end'])):
                continue
                
            if len(word_info['word']) >= 3:  # Skip very short words
                match = self._fuzzy_match_word(word_info)
                if match and match['confidence'] >= self.fuzzy_threshold:
                    found_brands.append(match)
                    processed_positions.update(range(word_info['start'], word_info['end']))
        
        # 3. Tertiary approach: spaCy NER integration
        if self.nlp:
            spacy_matches = self._spacy_entity_extraction(text, processed_positions)
            found_brands.extend(spacy_matches)
        
        # 4. Context-based detection using co-occurrence patterns
        context_matches = self._context_based_detection(text, found_brands)
        found_brands.extend(context_matches)
        
        # Post-processing: deduplicate and rank
        found_brands = self._deduplicate_and_rank(found_brands)
        
        logger.debug(f"Found {len(found_brands)} brand mentions")
        
        # Return format based on request
        if not return_details:
            return [(d['brand'], d['confidence'] / 100.0) for d in found_brands]
            
        # Convert scores to 0-1 range for consistency
        for brand in found_brands:
            brand['confidence'] = brand['confidence'] / 100.0
            
        return found_brands
    
    def _extract_text_chunks(self, text: str) -> List[Dict]:
        """Extract meaningful text chunks for fuzzy matching."""
        chunks = []
        
        # Extract phrases (noun phrases, multi-word terms)
        # Simple heuristic: sequences of 1-4 words
        words = re.findall(r'\b\w+\b', text)
        positions = [m.start() for m in re.finditer(r'\b\w+\b', text)]
        
        for i in range(len(words)):
            for chunk_size in range(1, min(5, len(words) - i + 1)):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                # Only consider chunks with reasonable length
                if len(chunk_text) >= 3:
                    chunks.append({
                        'text': chunk_text,
                        'position': positions[i],
                        'size': chunk_size
                    })
        
        # Sort by size (prefer longer matches) then position
        chunks.sort(key=lambda x: (-x['size'], x['position']))
        return chunks
    
    def _fuzzy_match_chunk(self, chunk_text: str, position: int) -> List[Dict]:
        """Perform fuzzy matching on a text chunk using multiple scorers."""
        matches = []
        
        # Use fuzzywuzzy's process.extract for efficient matching
        candidates = process.extract(
            chunk_text, 
            self.search_terms, 
            limit=5,  # Get top 5 candidates
            scorer=fuzz.ratio
        )
        
        for term, base_score in candidates:
            if base_score < self.fuzzy_threshold:
                continue
                
            # Calculate composite score using multiple scorers
            composite_score = self._calculate_composite_score(chunk_text, term)
            
            if composite_score >= self.fuzzy_threshold:
                # Determine match type based on score
                match_type = 'exact' if composite_score >= self.exact_threshold else 'fuzzy'
                
                brand_name = self.term_to_brand.get(term, term)
                
                match = {
                    'brand': brand_name,
                    'confidence': composite_score,
                    'match_type': match_type,
                    'position': position,
                    'matched_text': chunk_text,
                    'matched_term': term,
                    'end_position': position + len(chunk_text),
                    'scorer_details': {
                        'ratio': fuzz.ratio(chunk_text, term),
                        'partial_ratio': fuzz.partial_ratio(chunk_text, term),
                        'token_sort_ratio': fuzz.token_sort_ratio(chunk_text, term),
                        'token_set_ratio': fuzz.token_set_ratio(chunk_text, term)
                    }
                }
                matches.append(match)
        
        return matches
    
    def _calculate_composite_score(self, text1: str, text2: str) -> float:
        """Calculate weighted composite score using multiple fuzzywuzzy scorers."""
        scores = {
            'ratio': fuzz.ratio(text1, text2),
            'partial_ratio': fuzz.partial_ratio(text1, text2),
            'token_sort_ratio': fuzz.token_sort_ratio(text1, text2),
            'token_set_ratio': fuzz.token_set_ratio(text1, text2)
        }
        
        # Calculate weighted average
        composite = sum(scores[scorer] * weight 
                       for scorer, weight in self.scorer_weights.items())
        
        # Apply bonus for exact case match
        if text1.lower() == text2.lower():
            composite = min(composite * 1.1, 100)
        
        return composite
    
    def _extract_words_with_positions(self, text: str) -> List[Dict]:
        """Extract individual words with their positions."""
        words = []
        for match in re.finditer(r'\b\w+\b', text):
            words.append({
                'word': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        return words
    
    def _fuzzy_match_word(self, word_info: Dict) -> Optional[Dict]:
        """Perform fuzzy matching on a single word."""
        word = word_info['word']
        
        # Use process.extractOne for single best match
        result = process.extractOne(
            word, 
            self.search_terms, 
            scorer=fuzz.ratio
        )
        
        if not result:
            return None
            
        matched_term, score = result
        
        if score >= self.fuzzy_threshold:
            # Calculate composite score for final confidence
            composite_score = self._calculate_composite_score(word, matched_term)
            
            brand_name = self.term_to_brand.get(matched_term, matched_term)
            
            return {
                'brand': brand_name,
                'confidence': composite_score,
                'match_type': 'exact' if composite_score >= self.exact_threshold else 'fuzzy',
                'position': word_info['start'],
                'matched_text': word,
                'matched_term': matched_term,
                'end_position': word_info['end']
            }
        
        return None
    
    def _spacy_entity_extraction(self, text: str, processed_positions: Set[int]) -> List[Dict]:
        """Extract entities using spaCy NER and match with fuzzywuzzy."""
        if not self.nlp:
            return []
            
        matches = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'PERSON']:
                # Check if position is already processed
                if any(pos in processed_positions 
                       for pos in range(ent.start_char, ent.end_char)):
                    continue
                
                # Use fuzzywuzzy to match spaCy entity
                result = process.extractOne(
                    ent.text, 
                    self.search_terms, 
                    scorer=fuzz.token_sort_ratio
                )
                
                if result and result[1] >= self.fuzzy_threshold * 0.8:  # Slightly lower threshold for NER
                    matched_term, score = result
                    brand_name = self.term_to_brand.get(matched_term, matched_term)
                    
                    # Apply penalty for NER-based detection
                    adjusted_score = score * 0.9
                    
                    match = {
                        'brand': brand_name,
                        'confidence': adjusted_score,
                        'match_type': 'spacy_fuzzy',
                        'position': ent.start_char,
                        'matched_text': ent.text,
                        'matched_term': matched_term,
                        'end_position': ent.end_char,
                        'spacy_label': ent.label_
                    }
                    matches.append(match)
        
        return matches
    
    def _context_based_detection(self, text: str, existing_brands: List[Dict]) -> List[Dict]:
        """Detect brands based on context clues and keyword co-occurrence."""
        if len(existing_brands) >= 3:  # Skip if we already found many brands.
            return []
            
        matches = []
        text_lower = text.lower()
        found_brand_names = {b['brand'] for b in existing_brands}
        
        # Look for brands mentioned through keywords
        for brand_name, data in self.brands.items():
            original_name = data['original_name']
            if original_name in found_brand_names:
                continue
                
            keywords = [kw.lower() for kw in data.get('keywords', []) if len(kw) > 3]
            if not keywords:
                continue
                
            # Count keyword matches using fuzzy matching
            keyword_matches = 0
            matched_keywords = []
            
            for keyword in keywords:
                # Use fuzzy matching for keywords too
                text_words = re.findall(r'\b\w+\b', text_lower)
                for word in text_words:
                    if len(word) >= 3:
                        score = fuzz.ratio(word, keyword)
                        if score >= 80:  # Higher threshold for keyword matching
                            keyword_matches += 1
                            matched_keywords.append(f"{word}~{keyword}")
                            break
            
            # Require multiple keyword matches for context-based detection
            if keyword_matches >= 2:
                confidence = min(50 + (keyword_matches * 10), 85)  # Cap at 85 for context
                
                match = {
                    'brand': original_name,
                    'confidence': confidence,
                    'match_type': 'context_fuzzy',
                    'position': -1,  # No specific position
                    'matched_text': f"{keyword_matches} fuzzy keywords",
                    'keyword_matches': matched_keywords,
                    'keyword_count': keyword_matches
                }
                matches.append(match)
        
        return matches
    
    def _deduplicate_and_rank(self, detections: List[Dict]) -> List[Dict]:
        """Remove duplicates and rank by confidence and match quality."""
        if not detections:
            return []
            
        # Group by brand
        brand_groups = {}
        for detection in detections:
            brand = detection['brand']
            if brand not in brand_groups:
                brand_groups[brand] = []
            brand_groups[brand].append(detection)
        
        # Keep best detection for each brand
        final_detections = []
        for brand, group in brand_groups.items():
            # Sort by multiple criteria
            def sort_key(d):
                # Prioritize exact matches, then fuzzy, then context
                type_priority = {
                    'exact': 0, 
                    'fuzzy': 1, 
                    'spacy_fuzzy': 2, 
                    'context_fuzzy': 3
                }
                return (
                    type_priority.get(d['match_type'], 4),
                    -d['confidence'],  # Higher confidence first
                    d.get('position', float('inf'))  # Earlier position first
                )
            
            group.sort(key=sort_key)
            best_detection = group[0]
            
            # Enhance with additional info from other detections
            if len(group) > 1:
                best_detection['alternative_matches'] = len(group) - 1
                
            final_detections.append(best_detection)
        
        # Final sort by confidence and position
        final_detections.sort(key=lambda x: (-x['confidence'], x.get('position', float('inf'))))
        
        return final_detections

    def get_brand_info(self, brand_name: str) -> Optional[Dict]:
        """Get detailed information about a brand."""
        # Look up by original name first, then by any search term.
        for brand_key, data in self.brands.items():
            if (data['original_name'].lower() == brand_name.lower() or 
                brand_name.lower() in [alias.lower() for alias in data['aliases']]):
                return data
        return None
        
    @property
    def competitor_map(self) -> Dict[str, Set[str]]:
        """Lazy-loaded bidirectional competitor map."""
        if not hasattr(self, '_competitor_map'):
            comp_map = {}
            for brand, data in self.brands.items():
                original = data['original_name']
                comp_map[original] = set()
                for comp in data['competitors']:
                    # Find original name for competitor
                    comp_data = next((d for d in self.brands.values() 
                                    if comp.lower() in [d['original_name'].lower()] + 
                                    [alias.lower() for alias in d['aliases']]), None)
                    comp_original = comp_data['original_name'] if comp_data else comp.title()
                    comp_map[original].add(comp_original)
                    # Bidirectional
                    if comp_original not in comp_map:
                        comp_map[comp_original] = set()
                    comp_map[comp_original].add(original)
            self._competitor_map = comp_map
        return self._competitor_map


class ProductExtractor:
    """Extract product mentions with brand association using fuzzywuzzy.
    Enhanced with pattern variations and version detection.
    """
    def __init__(self, brand_recognizer: BrandRecognizer):
        """Initialize with brand recognizer instance."""
        self.brand_recognizer = brand_recognizer
        self.product_terms = self._build_product_terms()
        self.product_to_brand = self._build_product_mapping()
        
    def _build_product_terms(self) -> List[str]:
        """Build list of product terms for fuzzywuzzy matching."""
        terms = []
        for brand_name, data in self.brand_recognizer.brands.items():
            for product in data.get('products', []):
                terms.append(product)
                # Add variations with common descriptors
                for descriptor in ['pro', 'plus', 'max', 'mini', 'ultra', 'lite']:
                    terms.append(f"{product} {descriptor}")
                # Add numbered versions (1-20 should cover most cases)
                for num in range(1, 21):
                    terms.append(f"{product} {num}")
        
        return list(set(terms))  # Remove duplicates
        
    def _build_product_mapping(self) -> Dict[str, str]:
        """Map product terms to their brand names."""
        mapping = {}
        for brand_name, data in self.brand_recognizer.brands.items():
            original = data['original_name']
            for product in data.get('products', []):
                mapping[product] = original
                # Map variations too
                for descriptor in ['pro', 'plus', 'max', 'mini', 'ultra', 'lite']:
                    mapping[f"{product} {descriptor}"] = original
                for num in range(1, 21):
                    mapping[f"{product} {num}"] = original
        return mapping
        
    def extract_products(self, text: str, detected_brands: List[Dict] = None) -> List[Dict]:
        """Extract product mentions from text using fuzzywuzzy.
        
        Args:
            text: Input text
            detected_brands: Previously detected brands for context
            
        Returns:
            List of product detections
        """
        if not text or not text.strip():
            return []
            
        found_products = []
        
        # Get brand context for confidence boosting
        brand_context = set()
        if detected_brands:
            brand_context = {d['brand'] for d in detected_brands}
            
        # Extract words and phrases from text
        words = re.findall(r'\b\w+(?:\s+\w+){0,2}\b', text)  # 1-3 word phrases
        
        for word_phrase in words:
            if len(word_phrase.strip()) < 3:
                continue
                
            # Use fuzzywuzzy to find best product match
            result = process.extractOne(
                word_phrase, 
                self.product_terms, 
                scorer=fuzz.token_sort_ratio
            )
            
            if result and result[1] >= 70:  # 70% threshold for products
                matched_product, score = result
                brand_name = self.product_to_brand.get(matched_product)
                
                if brand_name:
                    # Boost confidence if brand was detected in context
                    confidence_boost = 1.1 if brand_name in brand_context else 1.0
                    final_confidence = min(score * confidence_boost, 100)
                    
                    # Find position in original text
                    position = text.lower().find(word_phrase.lower())
                    
                    product_detection = {
                        'product': matched_product,
                        'brand': brand_name,
                        'confidence': final_confidence,
                        'position': position,
                        'matched_text': word_phrase,
                        'fuzzy_score': score
                    }
                    found_products.append(product_detection)
                    
        # Deduplicate products
        unique_products = {}
        for product in found_products:
            key = (product['product'].lower(), product['brand'])
            if key not in unique_products or product['confidence'] > unique_products[key]['confidence']:
                unique_products[key] = product
                
        return list(unique_products.values())


class CompetitorMapper:
    """Enhanced competitor mapping with relationship analysis using fuzzywuzzy.
    """
    def __init__(self, brand_recognizer: BrandRecognizer):
        """Initialize with brand recognizer."""
        self.brand_recognizer = brand_recognizer
        self.competitor_map = brand_recognizer.competitor_map
        
    def identify_competitive_context(self, text: str, brands: List[Dict]) -> List[Dict]:
        """Identify competitive contexts in text.
        
        Args:
            text: Input text
            brands: Detected brand entities
            
        Returns:
            List of competitive contexts
        """
        if len(brands) < 2:
            return []
            
        text_lower = text.lower()
        contexts = []
        
        # Enhanced comparison indicators using fuzzy matching
        comparison_terms = [
            'better than', 'worse than', 'vs', 'versus', 'compared to', 
            'comparison', 'prefer', 'instead of', 'switched from', 
            'upgraded from', 'downgraded from', 'beats', 'outperforms'
        ]
        
        # Check each brand pair
        for i, brand1 in enumerate(brands):
            for brand2 in brands[i+1:]:
                b1_name = brand1['brand']
                b2_name = brand2['brand']
                
                # Check if they're competitors (with fuzzy matching)
                is_competitor = (b2_name in self.competitor_map.get(b1_name, set()) or
                               self._are_fuzzy_competitors(b1_name, b2_name))
                
                if is_competitor:
                    # Look for comparison context with fuzzy matching
                    context_type = 'co-mention'
                    for term in comparison_terms:
                        # Use fuzzy matching to find comparison indicators
                        text_words = text_lower.split()
                        for word_phrase in [' '.join(text_words[i:i+len(term.split())]) 
                                          for i in range(len(text_words))]:
                            if fuzz.ratio(word_phrase, term) >= 80:
                                context_type = 'comparison'
                                break
                        if context_type == 'comparison':
                            break
                            
                    contexts.append({
                        'brand1': b1_name,
                        'brand2': b2_name,
                        'relationship': 'direct_competitor' if is_competitor else 'potential_competitor',
                        'context_type': context_type,
                        'confidence': max(brand1['confidence'], brand2['confidence'])
                    })
                    
        return contexts
    
    def _are_fuzzy_competitors(self, brand1: str, brand2: str) -> bool:
        """Check if two brands might be competitors using fuzzy industry matching."""
        brand1_info = self.brand_recognizer.get_brand_info(brand1)
        brand2_info = self.brand_recognizer.get_brand_info(brand2)
        
        if not (brand1_info and brand2_info):
            return False
            
        # Same industry = potential competitors
        industry1 = brand1_info.get('industry', '').lower()
        industry2 = brand2_info.get('industry', '').lower()
        
        if industry1 and industry2:
            return fuzz.ratio(industry1, industry2) >= 90
        
        return False


class EntityDisambiguator:
    """Advanced disambiguation using multiple signals and fuzzywuzzy.
    """
    def __init__(self, brand_recognizer: BrandRecognizer):
        """Initialize disambiguator."""
        self.brand_recognizer = brand_recognizer
        
    def disambiguate(self, entities: List[Dict], text: str) -> List[Dict]:
        """Disambiguate entities using context and overlap detection.
        
        Args:
            entities: List of detected entities
            text: Original text
            
        Returns:
            Disambiguated entities
        """
        if not entities:
            return []
            
        # Remove overlapping entities
        entities = self._remove_overlaps(entities)
        
        # Apply context boosting using fuzzy matching
        entities = self._apply_fuzzy_context_boost(entities, text)
        
        # Re-sort by position and confidence
        entities.sort(key=lambda x: (x.get('position', float('inf')), -x['confidence']))
        
        return entities
        
    def _remove_overlaps(self, entities: List[Dict]) -> List[Dict]:
        """Remove overlapping entity detections, keeping higher confidence ones."""
        # Sort by position.
        positioned = [e for e in entities if e.get('position', -1) >= 0]
        positioned.sort(key=lambda x: x['position'])
        
        non_overlapping = []
        last_end = -1
        
        for entity in positioned:
            start = entity['position']
            end = entity.get('end_position', start + len(entity.get('matched_text', '')))
            
            if start >= last_end:
                non_overlapping.append(entity)
                last_end = end
            else:
                # Keep higher confidence
                if non_overlapping and entity['confidence'] > non_overlapping[-1]['confidence']:
                    non_overlapping[-1] = entity
                    last_end = end
                    
        # Add context-based detections (no position)
        context_entities = [e for e in entities if e.get('position', -1) < 0]
        non_overlapping.extend(context_entities)
        
        return non_overlapping
        
    def _apply_fuzzy_context_boost(self, entities: List[Dict], text: str) -> List[Dict]:
        """Boost confidence based on fuzzy context matching."""
        text_lower = text.lower()
        
        # Enhanced industry/category keywords
        industry_keywords = {
            'technology': ['tech', 'software', 'hardware', 'device', 'computer', 'digital', 'app', 'platform'],
            'automotive': ['car', 'vehicle', 'drive', 'auto', 'electric', 'engine', 'motor', 'transport'],
            'retail': ['shop', 'store', 'buy', 'purchase', 'retail', 'shopping', 'sale', 'price'],
            'food': ['food', 'restaurant', 'eat', 'meal', 'cuisine', 'dining', 'taste', 'flavor'],
            'fashion': ['fashion', 'clothing', 'wear', 'style', 'apparel', 'outfit', 'design']
        }
        
        text_words = re.findall(r'\b\w+\b', text_lower)
        
        for entity in entities:
            brand_info = self.brand_recognizer.get_brand_info(entity['brand'])
            if brand_info:
                industry = brand_info.get('industry', '').lower()
                if industry in industry_keywords:
                    # Use fuzzy matching for keyword detection
                    keyword_matches = 0
                    for keyword in industry_keywords[industry]:
                        for word in text_words:
                            if fuzz.ratio(word, keyword) >= 85:  # Fuzzy keyword match
                                keyword_matches += 1
                                break
                    
                    if keyword_matches > 0:
                        boost = min(0.05 * keyword_matches, 0.15)  # Max 15% boost
                        entity['confidence'] = min(entity['confidence'] + boost, 1.0)
                        entity['context_boost'] = boost
                        entity['fuzzy_keyword_matches'] = keyword_matches
                        
        return entities


# Spark UDF creators
def create_brand_recognition_udf(config_path: str = None):
    """Create Spark UDF for brand recognition using fuzzywuzzy."""
    recognizer = BrandRecognizer(config_path, use_spacy=False)
    
    def recognize_wrapper(text):
        if not text:
            return []
        try:
            brands = recognizer.recognize_brands(text, return_details=False)
            return [f"{brand}:{conf:.2f}" for brand, conf in brands]
        except Exception as e:
            logger.error(f"Error in brand UDF: {e}")
            return []
            
    return udf(recognize_wrapper, ArrayType(StringType()))


def create_product_extraction_udf(config_path: str = None):
    """Create Spark UDF for product extraction using fuzzywuzzy."""
    recognizer = BrandRecognizer(config_path, use_spacy=False)
    extractor = ProductExtractor(recognizer)
    
    def extract_wrapper(text):
        if not text:
            return []
        try:
            # First detect brands for context
            brands = recognizer.recognize_brands(text)
            products = extractor.extract_products(text, brands)
            return [f"{p['product']}|{p['brand']}:{p['confidence']:.2f}" 
                   for p in products]
        except Exception as e:
            logger.error(f"Error in product UDF: {e}")
            return []
            
    return udf(extract_wrapper, ArrayType(StringType()))


def main():
    """Test the fuzzywuzzy-first entity recognition system."""
    logger.info("Testing fuzzywuzzy-first entity recognition system")
    
    # Initialize components with fuzzywuzzy settings
    try:
        recognizer = BrandRecognizer(
            use_spacy=False, 
            fuzzy_threshold=65,  # Lower threshold to catch more variations
            exact_threshold=90,
            scorer_weights={
                'ratio': 0.3,
                'partial_ratio': 0.3,
                'token_sort_ratio': 0.25,
                'token_set_ratio': 0.15
            }
        )
        extractor = ProductExtractor(recognizer)
        mapper = CompetitorMapper(recognizer)
        disambiguator = EntityDisambiguator(recognizer)
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        print(f"Error: {e}")
        print("Make sure fuzzywuzzy is installed: pip install fuzzywuzzy[speedup]")
        return
    
    # Enhanced test cases with more variations and typos
    test_texts = [
        "Just got the new iPhone 15 Pro! Much better camera than my old Samsung Galaxy S23.",
        "Tesla Model 3 is the best electric car, way ahead of Ford's EVs.",
        "Comparing MacBook Pro M3 vs Dell XPS for programming work.",
        "Apple Watch Series 9 has better health features than Galaxy Watch 6.",
        "I love my new iPhne 15 (typo intended) - it's amazing!",  # Intentional typo
        "Tim Cook announced new products at the Apple event yesterday.",
        "Macdonalds has better burgers than Burger King",  # Typo in McDonald's
        "Nike Air Jordan beats Adidas any day",
        "My Samsng phone crashed again, switching to Apple",  # Typo in Samsung
        "Google Pixel camera quality is better than iPhone",
        "Microsoft Surface vs MacBook for productivity work",
        "Coca Cola taste better than Pepsi in my opinion"
    ]
    
    total_detections = 0
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {text}")
        print('='*80)
        
        # Detect brands with detailed output
        brands = recognizer.recognize_brands(text, return_details=True)
        total_detections += len(brands)
        
        print(f"\nBrand Detections ({len(brands)} found):")
        for b in brands:
            confidence_display = f"{b['confidence']:.3f}"
            match_details = f"type: {b['match_type']}"
            
            if 'scorer_details' in b:
                # Show fuzzywuzzy scorer breakdown
                scores = b['scorer_details']
                score_breakdown = f"scores: R:{scores['ratio']}, P:{scores['partial_ratio']}, " \
                                f"TS:{scores['token_sort_ratio']}, TSE:{scores['token_set_ratio']}"
                match_details += f", {score_breakdown}"
            
            if 'matched_term' in b:
                match_details += f", term: '{b['matched_term']}'"
                
            print(f"  ✓ {b['brand']} (conf: {confidence_display}, {match_details})")
            print(f"    matched: '{b['matched_text']}' at pos: {b.get('position', 'N/A')}")
            
            if 'alternative_matches' in b:
                print(f"    alternatives: {b['alternative_matches']} other matches found")
            
        # Extract products
        products = extractor.extract_products(text, brands)
        if products:
            print(f"\nProduct Detections ({len(products)} found):")
            for p in products:
                conf_display = f"{p['confidence']/100:.3f}" if p['confidence'] > 1 else f"{p['confidence']:.3f}"
                print(f"  ✓ {p['product']} by {p['brand']} (conf: {conf_display})")
                if 'fuzzy_score' in p:
                    print(f"    fuzzy score: {p['fuzzy_score']}, matched: '{p['matched_text']}'")
                
        # Find competitive contexts
        contexts = mapper.identify_competitive_context(text, brands)
        if contexts:
            print(f"\nCompetitive Contexts ({len(contexts)} found):")
            for c in contexts:
                print(f"  ⚔️  {c['brand1']} vs {c['brand2']} ({c['context_type']}, {c['relationship']})")
                
        # Disambiguate all entities
        all_entities = brands + [{'brand': p['brand'], 'confidence': p['confidence']/100 if p['confidence'] > 1 else p['confidence'], 
                                 'position': p['position'], 'match_type': 'product'} 
                                for p in products]
        disambiguated = disambiguator.disambiguate(all_entities, text)
        
        print(f"\nFinal entities after disambiguation: {len(disambiguated)}")
        for entity in disambiguated:
            boost_info = ""
            if 'context_boost' in entity:
                boost_info = f" (+{entity['context_boost']:.3f} context boost)"
            if 'fuzzy_keyword_matches' in entity:
                boost_info += f" [{entity['fuzzy_keyword_matches']} fuzzy keywords]"
            print(f"  → {entity['brand']} (final conf: {entity['confidence']:.3f}){boost_info}")
        
    # Test configuration summary
    print(f"\n{'='*80}")
    print("Fuzzywuzzy Configuration Summary:")
    print(f"  - Total brands in config: {len(recognizer.brands)}")
    print(f"  - Search terms built: {len(recognizer.search_terms)}")
    print(f"  - Fuzzy threshold: {recognizer.fuzzy_threshold}")
    print(f"  - Exact threshold: {recognizer.exact_threshold}")
    print(f"  - Scorer weights: {recognizer.scorer_weights}")
    print(f"  - Product terms: {len(extractor.product_terms)}")
    print(f"  - Competitor relationships: {sum(len(v) for v in recognizer.competitor_map.values()) // 2}")
    print(f"  - Total detections across all tests: {total_detections}")
    
    # Performance test with typos
    print(f"\n{'='*80}")
    print("Typo Resistance Test:")
    typo_tests = [
        ("Appl iPhone", "Apple iPhone"),
        ("Samsyng Galaxy", "Samsung Galaxy"),
        ("Microsft Surface", "Microsoft Surface"),
        ("Gogle Pixel", "Google Pixel"),
        ("Tesle Model", "Tesla Model")
    ]
    
    for typo_text, correct_text in typo_tests:
        typo_brands = recognizer.recognize_brands(typo_text, return_details=False)
        correct_brands = recognizer.recognize_brands(correct_text, return_details=False)
        
        typo_detected = len(typo_brands) > 0
        correct_detected = len(correct_brands) > 0
        
        status = "✓" if typo_detected else "✗"
        print(f"  {status} '{typo_text}' -> {typo_brands if typo_detected else 'No detection'}")
        print(f"    vs '{correct_text}' -> {correct_brands if correct_detected else 'No detection'}")


if __name__ == "__main__":
    main()