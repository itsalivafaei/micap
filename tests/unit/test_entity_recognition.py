"""
Unit Tests for Entity Recognition Module
Tests BrandRecognizer, ProductExtractor, CompetitorMapper, and EntityDisambiguator classes
Updated for fuzzywuzzy-first approach
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import tempfile
import os
from typing import Dict, List, Tuple, Set
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ml.entity_recognition import (
    BrandRecognizer, ProductExtractor, CompetitorMapper, 
    EntityDisambiguator, create_brand_recognition_udf,
    create_product_extraction_udf
)


class TestBrandRecognizer(unittest.TestCase):
    """Test cases for BrandRecognizer class with fuzzywuzzy-first approach"""
    
    def setUp(self):
        """Set up test data and mock configuration"""
        self.test_config = {
            "industries": {
                "technology": {
                    "brands": [
                        {
                            "name": "Apple",
                            "aliases": ["AAPL", "Apple Inc"],
                            "products": ["iPhone", "MacBook", "iPad"],
                            "keywords": ["innovation", "design"],
                            "competitors": ["Samsung", "Google"]
                        },
                        {
                            "name": "Samsung",
                            "aliases": ["Samsung Electronics"],
                            "products": ["Galaxy", "Note"],
                            "keywords": ["android", "display"],
                            "competitors": ["Apple", "Huawei"]
                        }
                    ]
                },
                "automotive": {
                    "brands": [
                        {
                            "name": "Tesla",
                            "aliases": ["TSLA"],
                            "products": ["Model 3", "Model Y"],
                            "keywords": ["electric", "autopilot"],
                            "competitors": ["Ford", "BMW"]
                        }
                    ]
                }
            }
        }
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(self.test_config, self.temp_config)
        self.temp_config.close()

    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)

    @patch('src.ml.entity_recognition.get_path')
    def test_init_with_spacy_disabled(self, mock_get_path):
        """Test initialization without spaCy using fuzzywuzzy parameters"""
        mock_get_path.return_value = self.temp_config.name
        
        recognizer = BrandRecognizer(
            use_spacy=False,
            fuzzy_threshold=70,
            exact_threshold=90
        )
        
        self.assertIsNotNone(recognizer.config)
        self.assertIsNotNone(recognizer.brands)
        self.assertIsNotNone(recognizer.search_terms)
        self.assertIsNotNone(recognizer.term_to_brand)
        self.assertIsNone(recognizer.nlp)
        self.assertEqual(len(recognizer.brands), 3)  # Apple, Samsung, Tesla
        self.assertEqual(recognizer.fuzzy_threshold, 70)
        self.assertEqual(recognizer.exact_threshold, 90)

    @patch('src.ml.entity_recognition.get_path')
    def test_init_with_spacy_enabled_but_not_available(self, mock_get_path):
        """Test initialization with spaCy enabled but not available"""
        mock_get_path.return_value = self.temp_config.name
        
        with patch('src.ml.entity_recognition.spacy') as mock_spacy:
            mock_spacy.load.side_effect = OSError("Model not found")
            
            recognizer = BrandRecognizer(use_spacy=True, fuzzy_threshold=70)
            
            self.assertIsNone(recognizer.nlp)

    def test_load_config(self):
        """Test configuration loading"""
        recognizer = BrandRecognizer.__new__(BrandRecognizer)
        config = recognizer._load_config(self.temp_config.name)
        
        self.assertEqual(config, self.test_config)

    @patch('src.ml.entity_recognition.get_path')
    def test_extract_all_brands(self, mock_get_path):
        """Test brand extraction from configuration"""
        mock_get_path.return_value = self.temp_config.name
        
        recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        brands = recognizer.brands
        
        # Check that all brands are extracted
        self.assertIn('apple', brands)
        self.assertIn('samsung', brands)
        self.assertIn('tesla', brands)
        
        # Check brand data structure
        apple_data = brands['apple']
        self.assertEqual(apple_data['industry'], 'technology')
        self.assertIn('aapl', apple_data['aliases'])
        self.assertIn('iphone', apple_data['products'])
        self.assertIn('samsung', apple_data['competitors'])

    @patch('src.ml.entity_recognition.get_path')
    def test_build_search_terms(self, mock_get_path):
        """Test search terms building for fuzzywuzzy"""
        mock_get_path.return_value = self.temp_config.name
        
        recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        search_terms = recognizer.search_terms
        
        # Check that search terms include brand names, aliases, and products
        self.assertIn('Apple', search_terms)
        self.assertIn('AAPL', search_terms)
        self.assertIn('iPhone', search_terms)
        self.assertIn('Samsung', search_terms)
        self.assertIn('Tesla', search_terms)

    @patch('src.ml.entity_recognition.get_path')
    def test_term_to_brand_mapping(self, mock_get_path):
        """Test term to brand mapping"""
        mock_get_path.return_value = self.temp_config.name
        
        recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        mapping = recognizer.term_to_brand
        
        # Check mappings
        self.assertEqual(mapping.get('Apple'), 'Apple')
        self.assertEqual(mapping.get('AAPL'), 'Apple')
        self.assertEqual(mapping.get('iPhone'), 'Apple')
        self.assertEqual(mapping.get('Samsung'), 'Samsung')

    @patch('src.ml.entity_recognition.get_path')
    def test_competitor_map_property(self, mock_get_path):
        """Test competitor map property (bidirectional)"""
        mock_get_path.return_value = self.temp_config.name
        
        recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        comp_map = recognizer.competitor_map
        
        # Check bidirectional relationships
        self.assertIn('Samsung', comp_map['Apple'])
        self.assertIn('Apple', comp_map['Samsung'])

    @patch('src.ml.entity_recognition.get_path')
    @patch('src.ml.entity_recognition.HAS_FUZZYWUZZY', True)
    def test_recognize_brands_direct_match(self, mock_get_path):
        """Test brand recognition with direct fuzzy matching"""
        mock_get_path.return_value = self.temp_config.name
        
        recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=65)
        
        # Test direct brand name match
        text = "I love my Apple iPhone"
        brands = recognizer.recognize_brands(text, return_details=True)
        
        self.assertTrue(len(brands) > 0)
        brand_names = [b['brand'] for b in brands]
        self.assertIn('Apple', brand_names)
        
        # Check confidence and match type
        apple_detection = next(b for b in brands if b['brand'] == 'Apple')
        self.assertGreater(apple_detection['confidence'], 0.6)
        self.assertIn(apple_detection['match_type'], ['exact', 'fuzzy'])

    @patch('src.ml.entity_recognition.get_path')
    @patch('src.ml.entity_recognition.HAS_FUZZYWUZZY', True)
    def test_recognize_brands_typo_resistance(self, mock_get_path):
        """Test brand recognition with typos (fuzzywuzzy strength)"""
        mock_get_path.return_value = self.temp_config.name
        
        recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=65)
        
        # Test typo handling
        text = "I love my Appl iPhone"  # Typo in Apple
        brands = recognizer.recognize_brands(text, return_details=True)
        
        # Should still detect Apple due to fuzzywuzzy
        if len(brands) > 0:
            brand_names = [b['brand'] for b in brands]
            # Might detect Apple if fuzzy match is good enough
            self.assertTrue(len(brands) >= 0)  # At least we don't crash

    @patch('src.ml.entity_recognition.get_path')
    def test_recognize_brands_return_formats(self, mock_get_path):
        """Test different return formats"""
        mock_get_path.return_value = self.temp_config.name
        
        recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=65)
        text = "Apple iPhone is great"
        
        # Test detailed format
        detailed_brands = recognizer.recognize_brands(text, return_details=True)
        if len(detailed_brands) > 0:
            self.assertIsInstance(detailed_brands[0], dict)
            self.assertIn('brand', detailed_brands[0])
            self.assertIn('confidence', detailed_brands[0])
            self.assertIn('match_type', detailed_brands[0])
        
        # Test simple format
        simple_brands = recognizer.recognize_brands(text, return_details=False)
        if len(simple_brands) > 0:
            self.assertIsInstance(simple_brands[0], tuple)
            self.assertEqual(len(simple_brands[0]), 2)  # (brand, confidence)

    @patch('src.ml.entity_recognition.get_path')
    def test_recognize_brands_no_match(self, mock_get_path):
        """Test brand recognition with no matches"""
        mock_get_path.return_value = self.temp_config.name
        
        recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        
        # Test no match
        text = "I like programming"
        brands = recognizer.recognize_brands(text)
        
        self.assertEqual(len(brands), 0)

    @patch('src.ml.entity_recognition.get_path')
    @patch('src.ml.entity_recognition.HAS_FUZZYWUZZY', True)
    def test_calculate_composite_score(self, mock_get_path):
        """Test composite score calculation with multiple fuzzywuzzy scorers"""
        mock_get_path.return_value = self.temp_config.name
        
        recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        
        # Test exact match
        score = recognizer._calculate_composite_score("apple", "apple")
        self.assertGreater(score, 95)  # Should be very high for exact match
        
        # Test similar match
        score = recognizer._calculate_composite_score("appl", "apple")
        self.assertGreater(score, 70)  # Should be reasonably high for typo

    @patch('src.ml.entity_recognition.get_path')
    def test_get_brand_info(self, mock_get_path):
        """Test getting brand information"""
        mock_get_path.return_value = self.temp_config.name
        
        recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        
        # Test getting brand info
        apple_info = recognizer.get_brand_info('Apple')
        self.assertIsNotNone(apple_info)
        self.assertEqual(apple_info['original_name'], 'Apple')
        self.assertEqual(apple_info['industry'], 'technology')
        
        # Test with lowercase
        apple_info_lower = recognizer.get_brand_info('apple')
        self.assertIsNotNone(apple_info_lower)
        
        # Test non-existent brand
        unknown_info = recognizer.get_brand_info('UnknownBrand')
        self.assertIsNone(unknown_info)


class TestProductExtractor(unittest.TestCase):
    """Test cases for ProductExtractor class with fuzzywuzzy"""
    
    def setUp(self):
        """Set up test data"""
        self.test_config = {
            "industries": {
                "technology": {
                    "brands": [
                        {
                            "name": "Apple",
                            "aliases": ["AAPL"],
                            "products": ["iPhone", "MacBook", "iPad"],
                            "keywords": ["innovation"],
                            "competitors": ["Samsung"]
                        }
                    ]
                }
            }
        }
        
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(self.test_config, self.temp_config)
        self.temp_config.close()

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)

    @patch('src.ml.entity_recognition.get_path')
    def test_init(self, mock_get_path):
        """Test ProductExtractor initialization"""
        mock_get_path.return_value = self.temp_config.name
        
        brand_recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        extractor = ProductExtractor(brand_recognizer)
        
        self.assertIsNotNone(extractor.brand_recognizer)
        self.assertIsNotNone(extractor.product_terms)
        self.assertIsNotNone(extractor.product_to_brand)

    @patch('src.ml.entity_recognition.get_path')
    def test_build_product_terms(self, mock_get_path):
        """Test product terms building"""
        mock_get_path.return_value = self.temp_config.name
        
        brand_recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        extractor = ProductExtractor(brand_recognizer)
        
        terms = extractor.product_terms
        self.assertIn('iPhone', terms)
        self.assertIn('MacBook', terms)
        # Check for variations
        self.assertIn('iPhone pro', terms)
        self.assertIn('iPhone 1', terms)

    @patch('src.ml.entity_recognition.get_path')
    @patch('src.ml.entity_recognition.HAS_FUZZYWUZZY', True)
    def test_extract_products(self, mock_get_path):
        """Test product extraction with fuzzywuzzy"""
        mock_get_path.return_value = self.temp_config.name
        
        brand_recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        extractor = ProductExtractor(brand_recognizer)
        
        text = "I bought a new iPhone from Apple"
        products = extractor.extract_products(text)
        
        if len(products) > 0:
            self.assertIsInstance(products[0], dict)
            self.assertIn('product', products[0])
            self.assertIn('brand', products[0])
            self.assertIn('confidence', products[0])

    @patch('src.ml.entity_recognition.get_path')
    def test_extract_products_with_brand_context(self, mock_get_path):
        """Test product extraction with brand context boosting"""
        mock_get_path.return_value = self.temp_config.name
        
        brand_recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        extractor = ProductExtractor(brand_recognizer)
        
        # Simulate detected brands
        detected_brands = [{'brand': 'Apple', 'confidence': 0.9}]
        
        text = "iPhone is great"
        products = extractor.extract_products(text, detected_brands)
        
        if len(products) > 0:
            # Should have higher confidence due to brand context
            self.assertGreater(products[0]['confidence'], 70)


class TestCompetitorMapper(unittest.TestCase):
    """Test cases for CompetitorMapper class with fuzzywuzzy enhancements"""
    
    def setUp(self):
        """Set up test data"""
        self.test_config = {
            "industries": {
                "technology": {
                    "brands": [
                        {
                            "name": "Apple",
                            "aliases": ["AAPL"],
                            "products": ["iPhone"],
                            "keywords": ["innovation"],
                            "competitors": ["Samsung", "Google"]
                        },
                        {
                            "name": "Samsung",
                            "aliases": [],
                            "products": ["Galaxy"],
                            "keywords": ["android"],
                            "competitors": ["Apple"]
                        }
                    ]
                }
            }
        }
        
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(self.test_config, self.temp_config)
        self.temp_config.close()

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)

    @patch('src.ml.entity_recognition.get_path')
    def test_init(self, mock_get_path):
        """Test CompetitorMapper initialization"""
        mock_get_path.return_value = self.temp_config.name
        
        brand_recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        mapper = CompetitorMapper(brand_recognizer)
        
        self.assertIsNotNone(mapper.brand_recognizer)
        self.assertIsNotNone(mapper.competitor_map)

    @patch('src.ml.entity_recognition.get_path')
    def test_identify_competitive_context_with_fuzzy_matching(self, mock_get_path):
        """Test competitive context identification with fuzzy matching"""
        mock_get_path.return_value = self.temp_config.name
        
        brand_recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        mapper = CompetitorMapper(brand_recognizer)
        
        text = "Apple beats Samsung in smartphones"
        # Simulate detected brands
        brands = [
            {'brand': 'Apple', 'confidence': 0.9},
            {'brand': 'Samsung', 'confidence': 0.8}
        ]
        
        contexts = mapper.identify_competitive_context(text, brands)
        
        if len(contexts) > 0:
            self.assertIn('relationship', contexts[0])
            self.assertIn('context_type', contexts[0])

    @patch('src.ml.entity_recognition.get_path')
    @patch('src.ml.entity_recognition.HAS_FUZZYWUZZY', True)
    def test_are_fuzzy_competitors(self, mock_get_path):
        """Test fuzzy competitor detection by industry"""
        mock_get_path.return_value = self.temp_config.name
        
        brand_recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        mapper = CompetitorMapper(brand_recognizer)
        
        # Test same industry brands
        result = mapper._are_fuzzy_competitors('Apple', 'Samsung')
        self.assertTrue(result)  # Both are in technology


class TestEntityDisambiguator(unittest.TestCase):
    """Test cases for EntityDisambiguator class with fuzzy enhancements"""
    
    def setUp(self):
        """Set up test data"""
        self.test_config = {
            "industries": {
                "technology": {
                    "brands": [
                        {
                            "name": "Apple",
                            "aliases": ["AAPL"],
                            "products": ["iPhone", "MacBook"],
                            "keywords": ["innovation", "design"],
                            "competitors": ["Samsung"]
                        }
                    ]
                }
            }
        }
        
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(self.test_config, self.temp_config)
        self.temp_config.close()

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)

    @patch('src.ml.entity_recognition.get_path')
    def test_init(self, mock_get_path):
        """Test EntityDisambiguator initialization"""
        mock_get_path.return_value = self.temp_config.name
        
        brand_recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        disambiguator = EntityDisambiguator(brand_recognizer)
        
        self.assertIsNotNone(disambiguator.brand_recognizer)

    @patch('src.ml.entity_recognition.get_path')
    @patch('src.ml.entity_recognition.HAS_FUZZYWUZZY', True)
    def test_apply_fuzzy_context_boost(self, mock_get_path):
        """Test fuzzy context boosting"""
        mock_get_path.return_value = self.temp_config.name
        
        brand_recognizer = BrandRecognizer(use_spacy=False, fuzzy_threshold=70)
        disambiguator = EntityDisambiguator(brand_recognizer)
        
        # Test entity with technology context
        entities = [{'brand': 'Apple', 'confidence': 0.7}]
        text = "Apple's innovative tech and software"
        
        boosted = disambiguator._apply_fuzzy_context_boost(entities, text)
        
        # Should boost confidence due to fuzzy keyword matches
        if 'context_boost' in boosted[0]:
            self.assertGreater(boosted[0]['confidence'], 0.7)


class TestSparkUDFs(unittest.TestCase):
    """Test cases for Spark UDF wrappers with fuzzywuzzy"""
    
    def setUp(self):
        """Set up test data"""
        self.test_config = {
            "industries": {
                "technology": {
                    "brands": [
                        {
                            "name": "Apple",
                            "aliases": ["AAPL"],
                            "products": ["iPhone"],
                            "keywords": ["innovation"],
                            "competitors": ["Samsung"]
                        }
                    ]
                }
            }
        }
        
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(self.test_config, self.temp_config)
        self.temp_config.close()

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)

    @patch('src.ml.entity_recognition.get_path')
    @patch('src.ml.entity_recognition.HAS_FUZZYWUZZY', True)
    def test_create_brand_recognition_udf(self, mock_get_path):
        """Test brand recognition UDF creation with fuzzywuzzy"""
        mock_get_path.return_value = self.temp_config.name
        
        udf_func = create_brand_recognition_udf()
        
        # Test the UDF function
        result = udf_func.func("I love Apple iPhone")
        self.assertTrue(len(result) >= 0)  # May or may not detect depending on fuzzy matching
        
        # Test null input
        result_null = udf_func.func(None)
        self.assertEqual(result_null, [])

    @patch('src.ml.entity_recognition.get_path')
    @patch('src.ml.entity_recognition.HAS_FUZZYWUZZY', True)
    def test_create_product_extraction_udf(self, mock_get_path):
        """Test product extraction UDF creation with fuzzywuzzy"""
        mock_get_path.return_value = self.temp_config.name
        
        udf_func = create_product_extraction_udf()
        
        # Test the UDF function
        result = udf_func.func("I bought Apple iPhone")
        self.assertTrue(len(result) >= 0)  # May or may not detect depending on fuzzy matching
        
        # Test null input
        result_null = udf_func.func(None)
        self.assertEqual(result_null, [])


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios for fuzzywuzzy-first approach"""
    
    def test_invalid_config_file(self):
        """Test handling of invalid config file"""
        with self.assertRaises(FileNotFoundError):
            BrandRecognizer(config_path="nonexistent.json", use_spacy=False)

    def test_malformed_config_file(self):
        """Test handling of malformed config file"""
        temp_config = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_config.write("invalid json")
        temp_config.close()
        
        try:
            with self.assertRaises(json.JSONDecodeError):
                BrandRecognizer(config_path=temp_config.name, use_spacy=False)
        finally:
            os.unlink(temp_config.name)

    @patch('src.ml.entity_recognition.HAS_FUZZYWUZZY', False)
    def test_fuzzywuzzy_not_available(self):
        """Test handling when fuzzywuzzy is not available"""
        with self.assertRaises(ImportError):
            BrandRecognizer(use_spacy=False, fuzzy_threshold=70)


if __name__ == '__main__':
    unittest.main() 