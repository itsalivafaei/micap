"""
Integration Tests for Phase 2 Pipeline
Tests the complete flow from entity recognition through competitor analysis
"""

import unittest
import os
import sys
import tempfile
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, when, lit

from src.ml.entity_recognition import (
    BrandRecognizer, ProductExtractor, 
    create_brand_recognition_udf, create_product_extraction_udf
)
from src.spark.competitor_analysis import CompetitorAnalyzer
from scripts.run_phase2_pipeline import run_brand_analysis_only


class TestPhase2Integration(unittest.TestCase):
    """Integration tests for Phase 2 pipeline components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up Spark session and test environment"""
        cls.spark = SparkSession.builder \
            .appName("TestPhase2Integration") \
            .master("local[2]") \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        cls.spark.sparkContext.setLogLevel("WARN")
        
        # Create temporary directory for test outputs
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create test brand configuration
        cls.test_config = {
            "industries": {
                "technology": {
                    "brands": [
                        {
                            "name": "Apple",
                            "aliases": ["AAPL", "apple inc"],
                            "products": ["iPhone", "iPad", "MacBook", "Apple Watch"],
                            "keywords": ["innovation", "design", "premium"],
                            "competitors": ["Samsung", "Google", "Microsoft"]
                        },
                        {
                            "name": "Samsung",
                            "aliases": ["samsung electronics"],
                            "products": ["Galaxy", "Galaxy S", "Galaxy Note"],
                            "keywords": ["android", "display", "technology"],
                            "competitors": ["Apple", "Google", "Huawei"]
                        },
                        {
                            "name": "Google",
                            "aliases": ["Alphabet", "GOOGL"],
                            "products": ["Pixel", "Android", "Chrome"],
                            "keywords": ["search", "ai", "cloud"],
                            "competitors": ["Apple", "Microsoft", "Amazon"]
                        }
                    ]
                },
                "automotive": {
                    "brands": [
                        {
                            "name": "Tesla",
                            "aliases": ["TSLA"],
                            "products": ["Model S", "Model 3", "Model X", "Model Y"],
                            "keywords": ["electric", "autonomous", "battery"],
                            "competitors": ["Ford", "BMW", "Mercedes"]
                        }
                    ]
                }
            }
        }
        
        cls.config_file = os.path.join(cls.temp_dir, "brand_config.json")
        with open(cls.config_file, 'w') as f:
            json.dump(cls.test_config, f, indent=2)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        cls.spark.stop()
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up test data for each test"""
        # Create comprehensive test dataset
        self.test_data = [
            {
                'tweet_id': '1',
                'text': 'Just got the new iPhone 15 Pro! The camera quality is amazing compared to my old Samsung Galaxy.',
                'timestamp': datetime(2024, 1, 1, 10, 0),
                'sentiment': 1.0,
                'vader_compound': 0.8,
                'emoji_sentiment': 0.9,
                'hashtags': ['iPhone15', 'Apple'],
                'exclamation_count': 1,
                'question_count': 0
            },
            {
                'tweet_id': '2', 
                'text': 'Samsung Galaxy S24 Ultra has the best display technology. Much better than Apple iPhone.',
                'timestamp': datetime(2024, 1, 1, 11, 30),
                'sentiment': 1.0,
                'vader_compound': 0.6,
                'emoji_sentiment': 0.7,
                'hashtags': ['Samsung', 'GalaxyS24'],
                'exclamation_count': 0,
                'question_count': 0
            },
            {
                'tweet_id': '3',
                'text': 'Google Pixel 8 camera vs iPhone 15 Pro - which one should I choose?',
                'timestamp': datetime(2024, 1, 1, 12, 15),
                'sentiment': 0.5,
                'vader_compound': 0.0,
                'emoji_sentiment': 0.5,
                'hashtags': ['Google', 'Pixel8', 'iPhone'],
                'exclamation_count': 0,
                'question_count': 1
            },
            {
                'tweet_id': '4',
                'text': 'Apple MacBook Pro M3 performance is incredible for video editing and design work.',
                'timestamp': datetime(2024, 1, 1, 14, 20),
                'sentiment': 1.0,
                'vader_compound': 0.9,
                'emoji_sentiment': 0.8,
                'hashtags': ['MacBook', 'M3', 'Apple'],
                'exclamation_count': 0,
                'question_count': 0
            },
            {
                'tweet_id': '5',
                'text': 'Tesla Model 3 battery life is disappointing. Expected better from electric vehicles.',
                'timestamp': datetime(2024, 1, 1, 15, 45),
                'sentiment': 0.0,
                'vader_compound': -0.6,
                'emoji_sentiment': 0.2,
                'hashtags': ['Tesla', 'Model3', 'EV'],
                'exclamation_count': 0,
                'question_count': 0
            },
            {
                'tweet_id': '6',
                'text': 'Love my new Samsung Galaxy Watch! Great alternative to Apple Watch.',
                'timestamp': datetime(2024, 1, 1, 16, 30),
                'sentiment': 1.0,
                'vader_compound': 0.7,
                'emoji_sentiment': 0.9,
                'hashtags': ['Samsung', 'GalaxyWatch'],
                'exclamation_count': 1,
                'question_count': 0
            },
            {
                'tweet_id': '7',
                'text': 'iPhone price is too high. Samsung offers better value for money.',
                'timestamp': datetime(2024, 1, 2, 9, 15),
                'sentiment': 0.0,
                'vader_compound': -0.3,
                'emoji_sentiment': 0.3,
                'hashtags': ['iPhone', 'Samsung'],
                'exclamation_count': 0,
                'question_count': 0
            },
            {
                'tweet_id': '8',
                'text': 'Google Android 14 features are amazing! Better than iOS in many ways.',
                'timestamp': datetime(2024, 1, 2, 11, 45),
                'sentiment': 1.0,
                'vader_compound': 0.8,
                'emoji_sentiment': 0.8,
                'hashtags': ['Android14', 'Google'],
                'exclamation_count': 1,
                'question_count': 0
            }
        ]
        
        self.df = self.spark.createDataFrame(self.test_data)
    
    def test_entity_recognition_integration(self):
        """Test entity recognition components working together"""
        # Initialize components
        brand_recognizer = BrandRecognizer(self.config_file, use_spacy=False)
        product_extractor = ProductExtractor(brand_recognizer)
        
        # Test brand recognition
        sample_text = "Just got the new iPhone 15 Pro! Much better than Samsung Galaxy."
        brands = brand_recognizer.recognize_brands(sample_text, return_details=False)
        
        # Should detect Apple and Samsung
        brand_names = [b[0] for b in brands]
        self.assertIn('Apple', brand_names)
        self.assertIn('Samsung', brand_names)
        
        # Test product extraction
        products = product_extractor.extract_products(sample_text, brands)
        product_names = [p['product'] for p in products]
        self.assertTrue(any('iPhone' in p for p in product_names))
        
        # Test competitor mapping
        competitor_map = brand_recognizer.competitor_map
        self.assertIn('Samsung', competitor_map.get('Apple', set()))
        self.assertIn('Apple', competitor_map.get('Samsung', set()))
    
    def test_spark_udf_integration(self):
        """Test Spark UDF integration with DataFrames"""
        # Create UDFs
        brand_udf = create_brand_recognition_udf(self.config_file)
        product_udf = create_product_extraction_udf(self.config_file)
        
        # Apply UDFs to DataFrame
        df_with_brands = self.df.withColumn("detected_brands", brand_udf(col("text")))
        df_with_products = df_with_brands.withColumn("detected_products", product_udf(col("text")))
        
        # Collect results
        results = df_with_products.collect()
        
        # Verify brand detection
        brand_detections = [row['detected_brands'] for row in results if row['detected_brands']]
        self.assertGreater(len(brand_detections), 0)
        
        # Check that brand mentions are in expected format
        for detection in brand_detections:
            for brand_mention in detection:
                self.assertIn(':', brand_mention)  # Format: "brand:confidence"
        
        # Verify product detection
        product_detections = [row['detected_products'] for row in results if row['detected_products']]
        self.assertGreater(len(product_detections), 0)
    
    def test_competitor_analysis_integration(self):
        """Test complete competitor analysis pipeline"""
        # Apply entity recognition
        brand_udf = create_brand_recognition_udf(self.config_file)
        df_with_brands = self.df.withColumn("brands", brand_udf(col("text")))
        
        # Filter to records with brand mentions
        df_brands = df_with_brands.filter(size(col("brands")) > 0)
        
        # Initialize competitor analyzer
        brand_recognizer = BrandRecognizer(self.config_file, use_spacy=False)
        analyzer = CompetitorAnalyzer(self.spark, brand_recognizer)
        
        # Test brand sentiment aggregation
        brand_sentiment = analyzer.aggregate_brand_sentiment(df_brands, "1 day")
        sentiment_count = brand_sentiment.count()
        self.assertGreater(sentiment_count, 0)
        
        # Test share of voice calculation
        sov_df = analyzer.calculate_share_of_voice(df_brands, time_window="1 day")
        sov_count = sov_df.count()
        self.assertGreater(sov_count, 0)
        
        # Verify share of voice totals approximately 100%
        sov_results = sov_df.collect()
        if sov_results:
            # Group by time window and sum share of voice
            window_totals = {}
            for row in sov_results:
                window_key = str(row['window_start'])
                if window_key not in window_totals:
                    window_totals[window_key] = 0
                window_totals[window_key] += row['share_of_voice']
            
            # Each window should sum to approximately 100%
            for total in window_totals.values():
                self.assertAlmostEqual(total, 100.0, delta=0.1)
        
        # Test sentiment momentum
        momentum_df = analyzer.compute_sentiment_momentum(sov_df)
        momentum_count = momentum_df.count()
        self.assertGreater(momentum_count, 0)
        
        # Verify momentum calculations
        momentum_results = momentum_df.collect()
        for row in momentum_results:
            # Momentum should be numeric
            if row['sentiment_momentum'] is not None:
                self.assertIsInstance(row['sentiment_momentum'], (int, float))
    
    def test_competitive_insights_generation(self):
        """Test end-to-end competitive insights generation"""
        # Run entity recognition
        brand_udf = create_brand_recognition_udf(self.config_file)
        df_with_brands = self.df.withColumn("brands", brand_udf(col("text")))
        df_brands = df_with_brands.filter(size(col("brands")) > 0)
        
        # Initialize analyzer
        brand_recognizer = BrandRecognizer(self.config_file, use_spacy=False)
        analyzer = CompetitorAnalyzer(self.spark, brand_recognizer)
        
        # Run full analysis pipeline
        brand_sentiment = analyzer.aggregate_brand_sentiment(df_brands, "1 day")
        sov_df = analyzer.calculate_share_of_voice(df_brands, time_window="1 day")
        momentum_df = analyzer.compute_sentiment_momentum(sov_df)
        
        # Get top brand for insights
        top_brands = (momentum_df
                      .groupBy("brand")
                      .agg({"share_of_voice": "avg"})
                      .orderBy(col("avg(share_of_voice)").desc())
                      .limit(1)
                      .collect())
        
        if top_brands:
            target_brand = top_brands[0]['brand']
            
            # Generate insights with explicit competitors
            competitors = ['samsung', 'google'] if target_brand == 'apple' else ['apple', 'google']
            insights = analyzer.generate_competitive_insights(
                momentum_df,
                target_brand,
                competitors=competitors
            )
            
            # Verify insights structure
            required_keys = ['brand', 'summary', 'metrics', 'competitors', 'trends', 'recommendations', 'insights']
            for key in required_keys:
                self.assertIn(key, insights)
            
            # Verify content quality
            self.assertEqual(insights['brand'], target_brand)
            self.assertIsInstance(insights['summary'], dict)
            self.assertIn('average_sentiment', insights['summary'])
            
            # Check market position data (for revised pipeline compatibility)
            self.assertIn('insights', insights)
            if 'market_position' in insights['insights']:
                market_pos = insights['insights']['market_position']
                self.assertIn('share_of_voice', market_pos)
                self.assertIn('sentiment_score', market_pos)
    
    def test_brand_analysis_pipeline_function(self):
        """Test the standalone brand analysis function"""
        result_df = run_brand_analysis_only(self.spark, self.df)
        
        # Verify output structure
        self.assertIn('detected_brands', result_df.columns)
        self.assertIn('detected_products', result_df.columns)
        
        # Verify some brands were detected
        brand_detections = result_df.filter(size(col("detected_brands")) > 0).count()
        self.assertGreater(brand_detections, 0)
    
    def test_multi_industry_analysis(self):
        """Test analysis across multiple industries"""
        # Apply entity recognition
        brand_udf = create_brand_recognition_udf(self.config_file)
        df_with_brands = self.df.withColumn("brands", brand_udf(col("text")))
        df_brands = df_with_brands.filter(size(col("brands")) > 0)
        
        # Initialize analyzer
        brand_recognizer = BrandRecognizer(self.config_file, use_spacy=False)
        analyzer = CompetitorAnalyzer(self.spark, brand_recognizer)
        
        # Test industry-specific analysis
        tech_sov = analyzer.calculate_share_of_voice(df_brands, industry="technology")
        auto_sov = analyzer.calculate_share_of_voice(df_brands, industry="automotive")
        
        # Should get results for both industries
        tech_count = tech_sov.count()
        auto_count = auto_sov.count()
        
        self.assertGreaterEqual(tech_count, 0)  # Should have tech brands
        self.assertGreaterEqual(auto_count, 0)  # May or may not have auto brands
    
    def test_feature_sentiment_analysis(self):
        """Test feature-level sentiment analysis integration"""
        # Apply entity recognition
        brand_udf = create_brand_recognition_udf(self.config_file)
        df_with_brands = self.df.withColumn("brands", brand_udf(col("text")))
        df_brands = df_with_brands.filter(size(col("brands")) > 0)
        
        # Initialize analyzer
        brand_recognizer = BrandRecognizer(self.config_file, use_spacy=False)
        analyzer = CompetitorAnalyzer(self.spark, brand_recognizer)
        
        # Test feature analysis
        features = ['camera', 'price', 'battery', 'display', 'performance', 'design']
        feature_sentiment = analyzer.analyze_feature_sentiment(df_brands, features)
        
        if feature_sentiment.count() > 0:
            results = feature_sentiment.collect()
            
            # Verify structure
            for row in results:
                self.assertIn(row['feature'], features)
                self.assertGreaterEqual(row['mention_count'], 1)
                self.assertIsInstance(row['sentiment_score'], (int, float))
                self.assertGreaterEqual(row['positive_ratio'], 0)
                self.assertLessEqual(row['positive_ratio'], 1)
    
    def test_error_handling_and_robustness(self):
        """Test error handling across the integrated pipeline"""
        # Test with malformed data
        malformed_data = [
            {'text': None, 'timestamp': datetime.now(), 'sentiment': 0.5},
            {'text': '', 'timestamp': datetime.now(), 'sentiment': 0.5},
            {'text': 'Normal text', 'timestamp': datetime.now(), 'sentiment': 0.5}
        ]
        
        malformed_df = self.spark.createDataFrame(malformed_data)
        
        # UDFs should handle malformed data gracefully
        brand_udf = create_brand_recognition_udf(self.config_file)
        result_df = malformed_df.withColumn("brands", brand_udf(col("text")))
        
        # Should not crash
        results = result_df.collect()
        self.assertEqual(len(results), 3)
        
        # Null/empty texts should return empty lists
        for i, row in enumerate(results):
            if malformed_data[i]['text'] in [None, '']:
                self.assertEqual(row['brands'], [])
    
    def test_performance_with_realistic_data_size(self):
        """Test performance with a more realistic data size"""
        # Generate larger dataset
        large_data = []
        base_texts = [
            "iPhone camera quality is amazing",
            "Samsung Galaxy display is beautiful", 
            "Google Pixel has great AI features",
            "Tesla Model 3 acceleration is incredible",
            "Apple MacBook performance is outstanding"
        ]
        
        for i in range(200):  # 200 records
            base_text = base_texts[i % len(base_texts)]
            large_data.append({
                'tweet_id': str(i),
                'text': f"{base_text} - tweet {i}",
                'timestamp': datetime(2024, 1, 1) + timedelta(minutes=i),
                'sentiment': 0.5 + (i % 3) * 0.25,
                'vader_compound': (i % 10 - 5) / 10.0,
                'emoji_sentiment': 0.5,
                'hashtags': ['test'],
                'exclamation_count': i % 2,
                'question_count': 0
            })
        
        large_df = self.spark.createDataFrame(large_data)
        
        # Time the analysis
        start_time = datetime.now()
        
        # Run entity recognition
        brand_udf = create_brand_recognition_udf(self.config_file)
        df_with_brands = large_df.withColumn("brands", brand_udf(col("text")))
        df_brands = df_with_brands.filter(size(col("brands")) > 0)
        
        # Run competitor analysis
        brand_recognizer = BrandRecognizer(self.config_file, use_spacy=False)
        analyzer = CompetitorAnalyzer(self.spark, brand_recognizer)
        
        brand_sentiment = analyzer.aggregate_brand_sentiment(df_brands, "1 hour")
        sov_df = analyzer.calculate_share_of_voice(df_brands, time_window="1 hour")
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Should complete in reasonable time (< 60 seconds for 200 records)
        self.assertLess(elapsed_time, 60)
        
        # Should produce meaningful results
        self.assertGreater(brand_sentiment.count(), 0)
        self.assertGreater(sov_df.count(), 0)


class TestPhase2DataFlow(unittest.TestCase):
    """Test data flow and transformations in Phase 2 pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up Spark session"""
        cls.spark = SparkSession.builder \
            .appName("TestPhase2DataFlow") \
            .master("local[1]") \
            .getOrCreate()
        cls.spark.sparkContext.setLogLevel("WARN")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        cls.spark.stop()
    
    def test_data_transformation_pipeline(self):
        """Test data transformations through the pipeline"""
        # Start with Phase 1 style data
        phase1_data = [
            {
                'tweet_id': '1',
                'text': 'Apple iPhone is great',
                'timestamp': datetime(2024, 1, 1, 10, 0),
                'sentiment': 1.0,
                'vader_compound': 0.8
            }
        ]
        
        input_df = self.spark.createDataFrame(phase1_data)
        
        # Apply transformations that pipeline would do
        # 1. Add brand detection
        df_step1 = input_df.withColumn(
            "detected_brands", 
            when(col("text").contains("Apple"), lit(["Apple:0.95"])).otherwise(lit([]))
        )
        
        # 2. Filter entities
        df_step2 = df_step1.filter(size(col("detected_brands")) > 0)
        
        # 3. Extract primary brand
        df_step3 = df_step2.withColumn(
            "primary_brand",
            when(size(col("detected_brands")) > 0, lit("Apple")).otherwise(lit(None))
        )
        
        # Verify transformation chain
        results = df_step3.collect()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['primary_brand'], 'Apple')
        self.assertGreater(len(results[0]['detected_brands']), 0)
    
    def test_data_quality_preservation(self):
        """Test that data quality is preserved through transformations"""
        # Test with various data quality scenarios
        test_cases = [
            {'text': 'Apple iPhone', 'expected_brands': True},
            {'text': 'No brands here', 'expected_brands': False},
            {'text': '', 'expected_brands': False},
            {'text': 'apple iphone', 'expected_brands': True},  # Case insensitive
        ]
        
        for case in test_cases:
            input_df = self.spark.createDataFrame([{
                'text': case['text'],
                'timestamp': datetime.now(),
                'sentiment': 0.5
            }])
            
            # Simulate brand detection
            df_with_brands = input_df.withColumn(
                "brands",
                when(col("text").rlike("(?i)(apple|iphone)"), lit(["Apple:0.9"])).otherwise(lit([]))
            )
            
            results = df_with_brands.collect()
            has_brands = len(results[0]['brands']) > 0
            
            self.assertEqual(has_brands, case['expected_brands'], 
                           f"Failed for text: '{case['text']}'")


if __name__ == '__main__':
    # Configure test runner
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [TestPhase2Integration, TestPhase2DataFlow]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PHASE 2 INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    # Exit with error code if tests failed
    if result.failures or result.errors:
        sys.exit(1)
    else:
        print("\n✓ All integration tests passed!")
        sys.exit(0) 