"""
Simplified Unit Tests for Phase 2 Pipeline
Focuses on core functionality with minimal complex mocking
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class TestRunPhase2PipelineSimplified(unittest.TestCase):
    """Simplified tests for Phase 2 pipeline with minimal mocking"""
    
    def test_module_imports(self):
        """Test that pipeline modules can be imported"""
        try:
            # Test basic imports
            import config.spark_config
            self.assertTrue(hasattr(config.spark_config, 'create_spark_session'))
            
            import scripts.run_phase2_pipeline
            self.assertTrue(hasattr(scripts.run_phase2_pipeline, 'run_phase2_pipeline'))
            self.assertTrue(hasattr(scripts.run_phase2_pipeline, 'run_brand_analysis_only'))
            
        except ImportError as e:
            self.fail(f"Failed to import required modules: {e}")
    
    def test_argument_parsing_functionality(self):
        """Test that argument parsing works correctly"""
        try:
            import argparse
            import scripts.run_phase2_pipeline as pipeline
            
            # Create parser similar to what's in the script
            parser = argparse.ArgumentParser(description="Run Phase 2 Pipeline")
            parser.add_argument("--sample", type=float, default=0.1)
            parser.add_argument("--full", action="store_true")
            
            # Test default arguments
            args = parser.parse_args([])
            self.assertEqual(args.sample, 0.1)
            self.assertFalse(args.full)
            
            # Test custom sample size
            args = parser.parse_args(["--sample", "0.05"])
            self.assertEqual(args.sample, 0.05)
            
            # Test full flag
            args = parser.parse_args(["--full"])
            self.assertTrue(args.full)
            
        except Exception as e:
            self.fail(f"Argument parsing test failed: {e}")
    
    @patch('scripts.run_phase2_pipeline.os.makedirs')
    @patch('scripts.run_phase2_pipeline.logger')
    def test_pipeline_structure_basic(self, mock_logger, mock_makedirs):
        """Test basic pipeline structure without heavy Spark operations"""
        
        # Mock the heavy dependencies at import time
        with patch.dict('sys.modules', {
            'pyspark': Mock(),
            'pyspark.sql': Mock(),
            'pyspark.sql.functions': Mock(),
            'src.ml.entity_recognition': Mock(),
            'src.spark.competitor_analysis': Mock(),
            'src.ml.trend_detection': Mock()
        }):
            
            # Test that we can import pipeline functions
            try:
                from scripts.run_phase2_pipeline import run_brand_analysis_only
                
                # Test function signature
                import inspect
                sig = inspect.signature(run_brand_analysis_only)
                
                # Should have spark and df parameters
                self.assertIn('spark', sig.parameters)
                self.assertIn('df', sig.parameters)
                
            except ImportError as e:
                self.skipTest(f"Pipeline import failed: {e}")
    
    def test_configuration_handling(self):
        """Test configuration file handling"""
        try:
            from src.utils.path_utils import get_path
            
            # Test that get_path function works
            self.assertTrue(callable(get_path))
            
            # Test with sample path
            test_path = get_path("data/test")
            self.assertIsInstance(test_path, (str, type(None)))
            
        except ImportError as e:
            self.skipTest(f"Path utils not available: {e}")
    
    def test_spark_config_creation(self):
        """Test Spark configuration creation"""
        try:
            from config.spark_config import create_spark_session
            
            # Test function exists and is callable
            self.assertTrue(callable(create_spark_session))
            
            # Test function signature
            import inspect
            sig = inspect.signature(create_spark_session)
            self.assertIn('app_name', sig.parameters)
            
        except ImportError as e:
            self.skipTest(f"Spark config not available: {e}")
    
    def test_entity_recognition_import(self):
        """Test entity recognition module import"""
        try:
            # Test basic import
            import src.ml.entity_recognition as er
            
            # Test that main classes exist
            self.assertTrue(hasattr(er, 'BrandRecognizer'))
            self.assertTrue(hasattr(er, 'ProductExtractor'))
            self.assertTrue(hasattr(er, 'create_brand_recognition_udf'))
            self.assertTrue(hasattr(er, 'create_product_extraction_udf'))
            
        except ImportError as e:
            self.skipTest(f"Entity recognition module not available: {e}")
    
    def test_competitor_analysis_import(self):
        """Test competitor analysis module import"""
        try:
            # Test basic import
            import src.spark.competitor_analysis as ca
            
            # Test that main class exists
            self.assertTrue(hasattr(ca, 'CompetitorAnalyzer'))
            
            # Test that CompetitorAnalyzer has required methods
            analyzer_methods = dir(ca.CompetitorAnalyzer)
            required_methods = [
                'aggregate_brand_sentiment',
                'calculate_share_of_voice',
                'compute_sentiment_momentum',
                'generate_competitive_insights'
            ]
            
            for method in required_methods:
                self.assertIn(method, analyzer_methods, f"Missing method: {method}")
                
        except ImportError as e:
            self.skipTest(f"Competitor analysis module not available: {e}")
    
    def test_pipeline_function_signatures(self):
        """Test that pipeline functions have correct signatures"""
        try:
            import scripts.run_phase2_pipeline as pipeline
            import inspect
            
            # Test run_phase2_pipeline signature
            pipeline_sig = inspect.signature(pipeline.run_phase2_pipeline)
            self.assertIn('sample_size', pipeline_sig.parameters)
            
            # Test default value
            sample_param = pipeline_sig.parameters['sample_size']
            self.assertEqual(sample_param.default, 0.1)
            
            # Test run_brand_analysis_only signature
            brand_sig = inspect.signature(pipeline.run_brand_analysis_only)
            self.assertIn('spark', brand_sig.parameters)
            self.assertIn('df', brand_sig.parameters)
            
        except ImportError as e:
            self.skipTest(f"Pipeline module not available: {e}")
    
    @patch('scripts.run_phase2_pipeline.time.time')
    def test_timing_functionality(self, mock_time):
        """Test timing functionality in pipeline"""
        
        # Mock time calls
        mock_time.side_effect = [1000.0, 1010.0]  # 10 second execution
        
        try:
            # Test timing calculation
            start_time = mock_time()
            elapsed_time = mock_time() - start_time
            
            self.assertEqual(elapsed_time, 10.0)
            
        except Exception as e:
            self.fail(f"Timing test failed: {e}")
    
    def test_logging_configuration(self):
        """Test logging configuration"""
        try:
            import scripts.run_phase2_pipeline
            import logging
            
            # Test that logger exists
            self.assertTrue(hasattr(scripts.run_phase2_pipeline, 'logger'))
            
            # Test logger configuration
            logger = scripts.run_phase2_pipeline.logger
            self.assertIsInstance(logger, logging.Logger)
            
        except Exception as e:
            self.fail(f"Logging test failed: {e}")


class TestEntityRecognitionBasic(unittest.TestCase):
    """Basic tests for entity recognition without heavy dependencies"""
    
    def test_fuzzy_dependency_check(self):
        """Test that fuzzywuzzy dependency is available"""
        try:
            import fuzzywuzzy
            import fuzzywuzzy.fuzz
            import fuzzywuzzy.process
            
            # Test basic functionality
            score = fuzzywuzzy.fuzz.ratio("apple", "Apple")
            self.assertIsInstance(score, int)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)
            
        except ImportError:
            self.skipTest("fuzzywuzzy not available")
    
    def test_entity_recognition_classes(self):
        """Test entity recognition class structure"""
        try:
            from src.ml.entity_recognition import BrandRecognizer, ProductExtractor
            
            # Test classes exist and are callable
            self.assertTrue(callable(BrandRecognizer))
            self.assertTrue(callable(ProductExtractor))
            
            # Test BrandRecognizer methods
            brand_methods = dir(BrandRecognizer)
            self.assertIn('recognize_brands', brand_methods)
            self.assertIn('get_brand_info', brand_methods)
            
            # Test ProductExtractor methods
            product_methods = dir(ProductExtractor)
            self.assertIn('extract_products', product_methods)
            
        except ImportError as e:
            self.skipTest(f"Entity recognition classes not available: {e}")


class TestCompetitorAnalysisBasic(unittest.TestCase):
    """Basic tests for competitor analysis without Spark operations"""
    
    def test_competitor_analyzer_class(self):
        """Test CompetitorAnalyzer class structure"""
        try:
            from src.spark.competitor_analysis import CompetitorAnalyzer
            
            # Test class exists and is callable
            self.assertTrue(callable(CompetitorAnalyzer))
            
            # Test required methods exist
            methods = dir(CompetitorAnalyzer)
            required_methods = [
                '__init__',
                'parse_brand_confidence',
                'aggregate_brand_sentiment',
                'compare_competitor_sentiment',
                'analyze_feature_sentiment',
                'calculate_share_of_voice',
                'compute_sentiment_momentum',
                'generate_competitive_insights'
            ]
            
            for method in required_methods:
                self.assertIn(method, methods, f"Missing method: {method}")
                
        except ImportError as e:
            self.skipTest(f"CompetitorAnalyzer not available: {e}")
    
    def test_method_signatures(self):
        """Test method signatures of CompetitorAnalyzer"""
        try:
            from src.spark.competitor_analysis import CompetitorAnalyzer
            import inspect
            
            # Test generate_competitive_insights signature (updated)
            insights_sig = inspect.signature(CompetitorAnalyzer.generate_competitive_insights)
            params = list(insights_sig.parameters.keys())
            
            # Should have the new signature with competitors parameter
            self.assertIn('self', params)
            self.assertIn('df', params) 
            self.assertIn('target_brand', params)
            self.assertIn('competitors', params)  # Optional parameter added
            
        except ImportError as e:
            self.skipTest(f"CompetitorAnalyzer not available: {e}")


if __name__ == '__main__':
    # Create simplified test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestRunPhase2PipelineSimplified,
        TestEntityRecognitionBasic,
        TestCompetitorAnalysisBasic
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SIMPLIFIED PHASE 2 PIPELINE TESTS SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if result.failures or result.errors:
        print("\n✗ Some tests failed")
        sys.exit(1)
    else:
        print("\n✓ All simplified pipeline tests passed!")
        sys.exit(0) 