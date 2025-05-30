"""
Lightweight Unit Tests for Phase 2 Pipeline
Tests core functionality with minimal resources and proper mocking
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
import tempfile
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class TestEntityRecognitionUnit(unittest.TestCase):
    """Unit tests for entity recognition without heavy dependencies"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_data = {
            "industries": {
                "technology": {
                    "brands": [
                        {
                            "name": "Apple",
                            "aliases": ["AAPL"],
                            "products": ["iPhone", "iPad"],
                            "keywords": ["innovation"],
                            "competitors": ["Samsung"]
                        },
                        {
                            "name": "Samsung", 
                            "aliases": ["SMSN"],
                            "products": ["Galaxy"],
                            "keywords": ["android"],
                            "competitors": ["Apple"]
                        }
                    ]
                }
            }
        }
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        with open(self.config_file, 'w') as f:
            json.dump(self.config_data, f)
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('src.ml.entity_recognition.fuzz')
    @patch('src.ml.entity_recognition.process')
    def test_brand_recognizer_initialization(self, mock_process, mock_fuzz):
        """Test BrandRecognizer initializes correctly"""
        from src.ml.entity_recognition import BrandRecognizer
        
        recognizer = BrandRecognizer(self.config_file, use_spacy=False)
        
        # Verify initialization
        self.assertEqual(len(recognizer.brands), 2)
        self.assertIn('apple', recognizer.brands)
        self.assertIn('samsung', recognizer.brands)
        self.assertGreater(len(recognizer.search_terms), 0)
    
    @patch('src.ml.entity_recognition.fuzz')
    @patch('src.ml.entity_recognition.process')
    def test_brand_recognition_basic(self, mock_process, mock_fuzz):
        """Test basic brand recognition functionality"""
        from src.ml.entity_recognition import BrandRecognizer
        
        # Mock fuzzywuzzy responses
        mock_process.extract.return_value = [('Apple', 95), ('iPhone', 85)]
        mock_process.extractOne.return_value = ('Apple', 95)
        mock_fuzz.ratio.return_value = 95
        mock_fuzz.partial_ratio.return_value = 90
        mock_fuzz.token_sort_ratio.return_value = 92
        mock_fuzz.token_set_ratio.return_value = 88
        
        recognizer = BrandRecognizer(self.config_file, use_spacy=False)
        
        # Test recognition
        result = recognizer.recognize_brands("New Apple iPhone is great", return_details=False)
        
        # Should return tuples of (brand, confidence)
        self.assertIsInstance(result, list)
        if result:  # If mocking worked
            for item in result:
                self.assertIsInstance(item, tuple)
                self.assertEqual(len(item), 2)
    
    def test_product_extractor_initialization(self):
        """Test ProductExtractor initializes correctly"""
        from src.ml.entity_recognition import BrandRecognizer, ProductExtractor
        
        with patch('src.ml.entity_recognition.fuzz'), \
             patch('src.ml.entity_recognition.process'):
            
            recognizer = BrandRecognizer(self.config_file, use_spacy=False)
            extractor = ProductExtractor(recognizer)
            
            # Verify initialization
            self.assertIsNotNone(extractor.brand_recognizer)
            self.assertGreater(len(extractor.product_terms), 0)
            self.assertIn('iPhone', extractor.product_terms)
    
    def test_udf_creation(self):
        """Test UDF creation functions"""
        from src.ml.entity_recognition import create_brand_recognition_udf, create_product_extraction_udf
        
        with patch('src.ml.entity_recognition.BrandRecognizer'), \
             patch('src.ml.entity_recognition.ProductExtractor'):
            
            # Test UDF creation doesn't crash
            brand_udf = create_brand_recognition_udf(self.config_file)
            product_udf = create_product_extraction_udf(self.config_file)
            
            # Verify UDFs are callable objects
            self.assertTrue(callable(brand_udf))
            self.assertTrue(callable(product_udf))


class TestCompetitorAnalysisUnit(unittest.TestCase):
    """Unit tests for competitor analysis with mocked Spark operations"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_spark = Mock()
        self.mock_brand_recognizer = Mock()
        self.mock_brand_recognizer.competitor_map = {
            'Apple': {'Samsung', 'Google'},
            'Samsung': {'Apple', 'Google'},
            'Google': {'Apple', 'Samsung'}
        }
        
    def test_competitor_analyzer_initialization(self):
        """Test CompetitorAnalyzer initializes correctly"""
        from src.spark.competitor_analysis import CompetitorAnalyzer
        
        analyzer = CompetitorAnalyzer(self.mock_spark, self.mock_brand_recognizer)
        
        # Verify initialization
        self.assertEqual(analyzer.spark, self.mock_spark)
        self.assertEqual(analyzer.brand_recognizer, self.mock_brand_recognizer)
        self.assertIn('hourly', analyzer.time_windows)
    
    def test_parse_brand_confidence(self):
        """Test brand confidence parsing"""
        from src.spark.competitor_analysis import CompetitorAnalyzer
        
        # Mock DataFrame operations
        mock_df = Mock()
        mock_df.select.return_value = mock_df
        mock_df.filter.return_value = mock_df
        mock_df.withColumn.return_value = mock_df
        
        analyzer = CompetitorAnalyzer(self.mock_spark, self.mock_brand_recognizer)
        
        # Test parse_brand_confidence method
        result = analyzer.parse_brand_confidence(mock_df)
        
        # Verify DataFrame operations were called
        mock_df.select.assert_called()
        mock_df.withColumn.assert_called()
    
    def test_generate_competitive_insights_structure(self):
        """Test competitive insights generation returns correct structure"""
        from src.spark.competitor_analysis import CompetitorAnalyzer
        
        # Mock DataFrame with minimal required operations
        mock_df = Mock()
        mock_df.filter.return_value = mock_df
        mock_df.count.return_value = 1
        mock_df.agg.return_value = mock_df
        mock_df.collect.return_value = [Mock(
            avg_sentiment=75.0,
            avg_mentions=10.0,
            total_mentions=100,
            sentiment_volatility=5.0,
            avg_sov=25.0
        )]
        mock_df.orderBy.return_value = mock_df
        mock_df.first.return_value = Mock(
            sentiment_score=75.0,
            mention_count=10,
            positive_ratio=0.8,
            market_position='Leader',
            share_of_voice=25.0,
            sov_rank=1,
            sentiment_momentum=2.5,
            momentum_trend='uptrend',
            momentum_signal='accelerating_positive',
            projected_sentiment=80.0
        )
        mock_df.groupBy.return_value = mock_df
        mock_df.limit.return_value = mock_df
        
        analyzer = CompetitorAnalyzer(self.mock_spark, self.mock_brand_recognizer)
        
        # Test insights generation
        insights = analyzer.generate_competitive_insights(mock_df, "Apple", ["Samsung", "Google"])
        
        # Verify structure
        required_keys = ['brand', 'summary', 'metrics', 'competitors', 'trends', 'recommendations', 'insights']
        for key in required_keys:
            self.assertIn(key, insights)
        
        # Verify content
        self.assertEqual(insights['brand'], 'Apple')
        self.assertIsInstance(insights['summary'], dict)
        self.assertIsInstance(insights['recommendations'], list)
        
        # Verify revised pipeline compatibility
        self.assertIn('insights', insights)
        if 'market_position' in insights['insights']:
            market_pos = insights['insights']['market_position']
            self.assertIn('share_of_voice', market_pos)
            self.assertIn('sentiment_score', market_pos)


class TestPipelineIntegrationUnit(unittest.TestCase):
    """Unit tests for pipeline integration with proper mocking"""
    
    def setUp(self):
        """Set up mocks"""
        self.mock_spark = Mock()
        self.mock_df = Mock()
        
        # Setup DataFrame method chaining
        self.mock_df.sample.return_value = self.mock_df
        self.mock_df.cache.return_value = self.mock_df
        self.mock_df.count.return_value = 100
        self.mock_df.withColumn.return_value = self.mock_df
        self.mock_df.withColumnRenamed.return_value = self.mock_df
        self.mock_df.filter.return_value = self.mock_df
        self.mock_df.unpersist.return_value = None
        
        # Setup Spark session
        self.mock_spark.read.parquet.return_value = self.mock_df
        self.mock_spark.stop.return_value = None
    
    @patch('scripts.run_phase2_pipeline.os.makedirs')
    @patch('scripts.run_phase2_pipeline.json.dump')
    @patch('builtins.open')
    @patch('src.utils.path_utils.get_path', side_effect=lambda x: x)
    @patch('config.spark_config.create_spark_session')
    def test_pipeline_function_signature(self, mock_create_spark, mock_get_path, 
                                       mock_open, mock_json_dump, mock_makedirs):
        """Test pipeline function has correct signature and can be called"""
        from scripts.run_phase2_pipeline import run_phase2_pipeline
        
        mock_create_spark.return_value = self.mock_spark
        
        # Mock all the components
        with patch('src.ml.entity_recognition.BrandRecognizer'), \
             patch('src.ml.entity_recognition.ProductExtractor'), \
             patch('src.ml.entity_recognition.create_brand_recognition_udf') as mock_brand_udf, \
             patch('src.ml.entity_recognition.create_product_extraction_udf') as mock_product_udf, \
             patch('src.spark.competitor_analysis.CompetitorAnalyzer') as mock_analyzer, \
             patch('src.ml.trend_detection.TopicModeler'), \
             patch('src.ml.trend_detection.TrendForecaster'), \
             patch('src.ml.trend_detection.AnomalyDetector'), \
             patch('pyspark.sql.functions.col'), \
             patch('pyspark.sql.functions.size'), \
             patch('pyspark.sql.functions.expr'), \
             patch('pyspark.sql.functions.when'):
            
            # Setup UDF mocks
            mock_brand_udf.return_value = Mock()
            mock_product_udf.return_value = Mock()
            
            # Setup analyzer mock
            mock_analyzer_instance = Mock()
            mock_analyzer.return_value = mock_analyzer_instance
            mock_analyzer_instance.aggregate_brand_sentiment.return_value = self.mock_df
            mock_analyzer_instance.calculate_share_of_voice.return_value = self.mock_df
            mock_analyzer_instance.compute_sentiment_momentum.return_value = self.mock_df
            mock_analyzer_instance.generate_competitive_insights.return_value = {
                'insights': {'market_position': {'share_of_voice': 25.0}},
                'recommendations': []
            }
            mock_analyzer_instance.visualize_competitive_landscape.return_value = None
            
            # Mock collect to return brand data
            mock_row = Mock()
            mock_row.brand = 'Apple'
            self.mock_df.collect.return_value = [mock_row]
            
            # Test function call
            try:
                run_phase2_pipeline(sample_size=0.01)  # Very small sample
                success = True
            except Exception as e:
                success = False
                error = str(e)
                
            # Verify execution
            self.assertTrue(success, f"Pipeline failed with error: {error if not success else 'None'}")
            
            # Verify cleanup was called
            self.mock_spark.stop.assert_called_once()
    
    @patch('src.utils.path_utils.get_path', side_effect=lambda x: x)
    def test_brand_analysis_only_function(self, mock_get_path):
        """Test standalone brand analysis function"""
        from scripts.run_phase2_pipeline import run_brand_analysis_only
        
        with patch('src.ml.entity_recognition.create_brand_recognition_udf') as mock_brand_udf, \
             patch('src.ml.entity_recognition.create_product_extraction_udf') as mock_product_udf, \
             patch('pyspark.sql.functions.col'):
            
            mock_brand_udf.return_value = Mock()
            mock_product_udf.return_value = Mock()
            
            # Test function call
            result = run_brand_analysis_only(self.mock_spark, self.mock_df)
            
            # Verify it returns a DataFrame
            self.assertIsNotNone(result)
            
            # Verify UDFs were created
            mock_brand_udf.assert_called_once()
            mock_product_udf.assert_called_once()


class TestConfigurationHandling(unittest.TestCase):
    """Test configuration loading and handling"""
    
    def setUp(self):
        """Set up test config"""
        self.temp_dir = tempfile.mkdtemp()
        self.valid_config = {
            "industries": {
                "test": {
                    "brands": [{
                        "name": "TestBrand",
                        "aliases": ["TB"],
                        "products": ["TestProduct"],
                        "keywords": ["test"],
                        "competitors": ["CompetitorBrand"]
                    }]
                }
            }
        }
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_config_loading(self):
        """Test configuration file loading"""
        config_file = os.path.join(self.temp_dir, "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(self.valid_config, f)
        
        with patch('src.ml.entity_recognition.fuzz'), \
             patch('src.ml.entity_recognition.process'):
            
            from src.ml.entity_recognition import BrandRecognizer
            
            recognizer = BrandRecognizer(config_file, use_spacy=False)
            
            # Verify config was loaded
            self.assertIn('testbrand', recognizer.brands)
            brand_data = recognizer.brands['testbrand']
            self.assertEqual(brand_data['original_name'], 'TestBrand')
            self.assertIn('tb', brand_data['aliases'])
    
    def test_invalid_config_handling(self):
        """Test handling of invalid configuration"""
        invalid_config_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_config_file, 'w') as f:
            f.write("invalid json content")
        
        with patch('src.ml.entity_recognition.fuzz'), \
             patch('src.ml.entity_recognition.process'):
            
            from src.ml.entity_recognition import BrandRecognizer
            
            # Should raise exception for invalid config
            with self.assertRaises(Exception):
                BrandRecognizer(invalid_config_file, use_spacy=False)
    
    def test_spark_config_creation(self):
        """Test Spark configuration creation"""
        from config.spark_config import create_spark_session
        
        # Should be callable
        self.assertTrue(callable(create_spark_session))
        
        # Test import doesn't fail
        import inspect
        sig = inspect.signature(create_spark_session)
        self.assertIn('app_name', sig.parameters)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEntityRecognitionUnit,
        TestCompetitorAnalysisUnit, 
        TestPipelineIntegrationUnit,
        TestConfigurationHandling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("LIGHTWEIGHT UNIT TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if result.failures or result.errors:
        print("\n✗ Some tests failed")
        sys.exit(1)
    else:
        print("\n✓ All lightweight unit tests passed!")
        sys.exit(0) 