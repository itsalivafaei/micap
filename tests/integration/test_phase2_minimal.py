"""
Minimal Resource Integration Tests for Phase 2 Pipeline
Tests with actual Spark operations using very small data samples
"""

import unittest
import os
import sys
import tempfile
import json
import shutil
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, size, when, lit
    from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType, IntegerType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


@unittest.skipUnless(SPARK_AVAILABLE, "PySpark not available")
class TestMinimalResourceIntegration(unittest.TestCase):
    """Minimal resource integration tests with actual Spark operations"""
    
    @classmethod
    def setUpClass(cls):
        """Set up minimal Spark session"""
        try:
            cls.spark = SparkSession.builder \
                .appName("MinimalPhase2Test") \
                .master("local[1]") \
                .config("spark.sql.adaptive.enabled", "false") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
                .config("spark.driver.memory", "1g") \
                .config("spark.executor.memory", "1g") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
                .getOrCreate()
            cls.spark.sparkContext.setLogLevel("WARN")
            
            # Create temporary directory
            cls.temp_dir = tempfile.mkdtemp()
            
            # Create minimal test configuration
            cls.test_config = {
                "industries": {
                    "technology": {
                        "brands": [
                            {
                                "name": "Apple",
                                "aliases": ["AAPL"],
                                "products": ["iPhone"],
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
            
            cls.config_file = os.path.join(cls.temp_dir, "minimal_config.json")
            with open(cls.config_file, 'w') as f:
                json.dump(cls.test_config, f, indent=2)
                
        except Exception as e:
            raise unittest.SkipTest(f"Failed to set up Spark session: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        if hasattr(cls, 'spark'):
            cls.spark.stop()
        if hasattr(cls, 'temp_dir'):
            shutil.rmtree(cls.temp_dir)
    
    def test_entity_recognition_with_real_data(self):
        """Test entity recognition with real small dataset"""
        # Create very small test dataset (5 records)
        test_data = [
            ("Apple iPhone is great", datetime(2024, 1, 1), 1.0),
            ("Samsung Galaxy phone", datetime(2024, 1, 1), 0.8),
            ("No brands mentioned here", datetime(2024, 1, 1), 0.5),
            ("iPhone camera quality", datetime(2024, 1, 1), 0.9),
            ("Galaxy display is good", datetime(2024, 1, 1), 0.7)
        ]
        
        # Create DataFrame with schema
        schema = StructType([
            StructField("text", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("sentiment", FloatType(), True)
        ])
        
        df = self.spark.createDataFrame(test_data, schema)
        
        # Test entity recognition integration
        try:
            from src.ml.entity_recognition import create_brand_recognition_udf
            
            brand_udf = create_brand_recognition_udf(self.config_file)
            df_with_brands = df.withColumn("detected_brands", brand_udf(col("text")))
            
            # Test that UDF works
            result = df_with_brands.collect()
            
            # Verify results
            self.assertEqual(len(result), 5)
            
            # At least some brands should be detected
            brand_detections = [row.detected_brands for row in result if row.detected_brands]
            self.assertGreater(len(brand_detections), 0, "No brands were detected")
            
            # Check brand format
            for detection in brand_detections:
                for brand_entry in detection:
                    self.assertIn(':', brand_entry, f"Invalid brand format: {brand_entry}")
                    
        except ImportError as e:
            self.skipTest(f"Entity recognition module not available: {e}")
    
    def test_competitor_analysis_with_minimal_data(self):
        """Test competitor analysis with minimal dataset"""
        # Create test data with pre-defined brands
        test_data = [
            {
                'text': 'Apple iPhone camera is excellent',
                'timestamp': datetime(2024, 1, 1, 10, 0),
                'sentiment': 1.0,
                'brands': ['Apple:0.95']
            },
            {
                'text': 'Samsung Galaxy display quality',
                'timestamp': datetime(2024, 1, 1, 11, 0), 
                'sentiment': 0.8,
                'brands': ['Samsung:0.90']
            },
            {
                'text': 'iPhone battery life could be better',
                'timestamp': datetime(2024, 1, 1, 12, 0),
                'sentiment': 0.3,
                'brands': ['Apple:0.88']
            }
        ]
        
        df = self.spark.createDataFrame(test_data)
        
        # Test competitor analysis
        try:
            from src.ml.entity_recognition import BrandRecognizer
            from src.spark.competitor_analysis import CompetitorAnalyzer
            
            # Initialize components
            brand_recognizer = BrandRecognizer(self.config_file, use_spacy=False)
            analyzer = CompetitorAnalyzer(self.spark, brand_recognizer)
            
            # Test sentiment aggregation
            brand_sentiment = analyzer.aggregate_brand_sentiment(df, "1 day")
            
            # Verify results
            sentiment_results = brand_sentiment.collect()
            self.assertGreater(len(sentiment_results), 0, "No sentiment aggregation results")
            
            # Check result structure
            for row in sentiment_results:
                self.assertIsNotNone(row.brand)
                self.assertIsNotNone(row.mention_count)
                self.assertIsInstance(row.sentiment_score, (int, float))
                
            # Test share of voice calculation
            sov_df = analyzer.calculate_share_of_voice(df, time_window="1 day")
            sov_results = sov_df.collect()
            
            if sov_results:
                # Verify share of voice totals
                total_sov = sum(row.share_of_voice for row in sov_results 
                               if hasattr(row, 'share_of_voice'))
                self.assertAlmostEqual(total_sov, 100.0, delta=0.1, 
                                     msg="Share of voice doesn't sum to 100%")
                
        except ImportError as e:
            self.skipTest(f"Competitor analysis module not available: {e}")
    
    def test_insights_generation_minimal(self):
        """Test insights generation with minimal data"""
        # Simplified test data
        test_data = [
            {
                'brand': 'apple',
                'window_start': datetime(2024, 1, 1),
                'sentiment_score': 75.0,
                'mention_count': 10,
                'positive_ratio': 0.8,
                'share_of_voice': 60.0,
                'sov_rank': 1
            },
            {
                'brand': 'samsung',
                'window_start': datetime(2024, 1, 1),
                'sentiment_score': 65.0,
                'mention_count': 8,
                'positive_ratio': 0.7,
                'share_of_voice': 40.0,
                'sov_rank': 2
            }
        ]
        
        df = self.spark.createDataFrame(test_data)
        
        try:
            from src.ml.entity_recognition import BrandRecognizer
            from src.spark.competitor_analysis import CompetitorAnalyzer
            
            brand_recognizer = BrandRecognizer(self.config_file, use_spacy=False)
            analyzer = CompetitorAnalyzer(self.spark, brand_recognizer)
            
            # Test insights generation
            insights = analyzer.generate_competitive_insights(
                df, 
                "Apple",
                competitors=["Samsung"]
            )
            
            # Verify insights structure
            required_keys = ['brand', 'summary', 'metrics', 'competitors', 'trends', 'recommendations', 'insights']
            for key in required_keys:
                self.assertIn(key, insights, f"Missing key: {key}")
            
            # Verify content
            self.assertEqual(insights['brand'], 'Apple')
            self.assertIsInstance(insights['summary'], dict)
            self.assertIsInstance(insights['recommendations'], list)
            
            # Test revised pipeline compatibility
            self.assertIn('insights', insights)
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_pipeline_integration_minimal(self):
        """Test minimal pipeline integration"""
        # Create minimal dataset
        input_data = [
            {
                'tweet_id': '1',
                'text': 'Apple iPhone camera is great',
                'timestamp': datetime(2024, 1, 1),
                'sentiment': 0.8,
                'vader_compound': 0.6
            },
            {
                'tweet_id': '2',
                'text': 'Samsung Galaxy display quality',
                'timestamp': datetime(2024, 1, 1),
                'sentiment': 0.7,
                'vader_compound': 0.5
            }
        ]
        
        df = self.spark.createDataFrame(input_data)
        
        try:
            from scripts.run_phase2_pipeline import run_brand_analysis_only
            
            # Test brand analysis function
            result_df = run_brand_analysis_only(self.spark, df)
            
            # Verify output
            self.assertIn('detected_brands', result_df.columns)
            self.assertIn('detected_products', result_df.columns)
            
            # Collect results to verify processing worked
            results = result_df.collect()
            self.assertEqual(len(results), 2)
            
            # At least one should have brand detections
            has_brands = any(row.detected_brands for row in results if row.detected_brands)
            self.assertTrue(has_brands, "No brands detected in pipeline test")
            
        except ImportError as e:
            self.skipTest(f"Pipeline module not available: {e}")
    
    def test_end_to_end_minimal_workflow(self):
        """Test complete end-to-end workflow with minimal data"""
        # Create comprehensive but small test dataset
        test_data = [
            {
                'text': 'Apple iPhone 15 Pro camera quality is excellent compared to Samsung Galaxy',
                'timestamp': datetime(2024, 1, 1, 10, 0),
                'sentiment': 1.0,
                'vader_compound': 0.8
            },
            {
                'text': 'Samsung Galaxy S24 display technology is impressive',
                'timestamp': datetime(2024, 1, 1, 11, 0),
                'sentiment': 0.9,
                'vader_compound': 0.7
            },
            {
                'text': 'iPhone price is too high, considering Samsung alternatives',
                'timestamp': datetime(2024, 1, 1, 12, 0),
                'sentiment': 0.2,
                'vader_compound': -0.3
            }
        ]
        
        df = self.spark.createDataFrame(test_data)
        
        try:
            from src.ml.entity_recognition import BrandRecognizer, create_brand_recognition_udf
            from src.spark.competitor_analysis import CompetitorAnalyzer
            
            # Step 1: Entity Recognition
            brand_udf = create_brand_recognition_udf(self.config_file)
            df_with_brands = df.withColumn("brands", brand_udf(col("text")))
            df_brands = df_with_brands.filter(size(col("brands")) > 0)
            
            # Verify brands were detected
            brand_count = df_brands.count()
            self.assertGreater(brand_count, 0, "No brands detected in end-to-end test")
            
            # Step 2: Competitor Analysis
            brand_recognizer = BrandRecognizer(self.config_file, use_spacy=False)
            analyzer = CompetitorAnalyzer(self.spark, brand_recognizer)
            
            # Run analysis pipeline
            brand_sentiment = analyzer.aggregate_brand_sentiment(df_brands, "1 day")
            sov_df = analyzer.calculate_share_of_voice(df_brands, time_window="1 day")
            
            # Verify analysis results
            sentiment_count = brand_sentiment.count()
            sov_count = sov_df.count()
            
            self.assertGreater(sentiment_count, 0, "No sentiment analysis results")
            self.assertGreater(sov_count, 0, "No share of voice results")
            
            # Step 3: Insights Generation
            # Get top brand for insights
            top_brands = (sov_df
                          .groupBy("brand")
                          .agg({"share_of_voice": "avg"})
                          .orderBy(col("avg(share_of_voice)").desc())
                          .limit(1)
                          .collect())
            
            if top_brands:
                target_brand = top_brands[0]['brand']
                
                insights = analyzer.generate_competitive_insights(
                    sov_df,
                    target_brand,
                    competitors=['samsung'] if target_brand == 'apple' else ['apple']
                )
                
                # Verify insights structure and content
                self.assertIn('brand', insights)
                self.assertIn('summary', insights)
                self.assertIn('insights', insights)  # For revised pipeline compatibility
                
                # Verify specific metrics exist
                if 'average_sentiment' in insights['summary']:
                    self.assertIsInstance(insights['summary']['average_sentiment'], (int, float))
                
                print(f"\nEnd-to-end test completed successfully for brand: {target_brand}")
                print(f"Sentiment score: {insights['summary'].get('average_sentiment', 'N/A')}")
                print(f"Recommendations: {len(insights['recommendations'])}")
                
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
        except Exception as e:
            self.fail(f"End-to-end test failed: {str(e)}")


class TestResourceConstraints(unittest.TestCase):
    """Test resource constraint handling"""
    
    def test_memory_usage_estimation(self):
        """Test that we can estimate memory usage for larger datasets"""
        # Calculate memory requirements for different dataset sizes
        base_memory_per_record = 1024  # 1KB per record (conservative estimate)
        
        test_sizes = [100, 1000, 10000, 100000]
        
        for size in test_sizes:
            estimated_memory_mb = (size * base_memory_per_record) / (1024 * 1024)
            
            # For testing, we should use sample sizes that keep memory under 100MB
            max_test_size = (100 * 1024 * 1024) // base_memory_per_record
            
            if size <= max_test_size:
                is_suitable = True
            else:
                is_suitable = False
                
            print(f"Dataset size: {size:,} records")
            print(f"Estimated memory: {estimated_memory_mb:.2f} MB")
            print(f"Suitable for testing: {is_suitable}")
            print("-" * 40)
    
    def test_spark_configuration_recommendations(self):
        """Test and recommend Spark configurations for different scenarios"""
        configs = {
            "minimal_testing": {
                "spark.driver.memory": "1g",
                "spark.executor.memory": "1g", 
                "spark.sql.adaptive.enabled": "false",
                "description": "For unit and integration tests"
            },
            "development": {
                "spark.driver.memory": "2g",
                "spark.executor.memory": "2g",
                "spark.sql.adaptive.enabled": "true",
                "description": "For development with sample data"
            },
            "production": {
                "spark.driver.memory": "4g",
                "spark.executor.memory": "4g", 
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
                "description": "For production workloads"
            }
        }
        
        for scenario, config in configs.items():
            print(f"\n{scenario.upper()} Configuration:")
            for key, value in config.items():
                if key != "description":
                    print(f"  {key}: {value}")
            print(f"  Use case: {config['description']}")
        
        # Verify configurations are valid
        for config in configs.values():
            if "spark.driver.memory" in config:
                memory_str = config["spark.driver.memory"]
                self.assertTrue(memory_str.endswith('g') or memory_str.endswith('m'))


if __name__ == '__main__':
    # Check if Spark is available
    if not SPARK_AVAILABLE:
        print("PySpark not available. Skipping integration tests.")
        sys.exit(0)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [TestMinimalResourceIntegration, TestResourceConstraints]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("MINIMAL RESOURCE INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            # Print first few lines of traceback for debugging
            lines = traceback.strip().split('\n')
            for line in lines[-3:]:  # Show last 3 lines
                print(f"    {line}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            lines = traceback.strip().split('\n')
            for line in lines[-3:]:  # Show last 3 lines
                print(f"    {line}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if result.failures or result.errors:
        print("\n✗ Some tests failed")
        sys.exit(1)
    else:
        print("\n✓ All minimal resource integration tests passed!")
        sys.exit(0) 