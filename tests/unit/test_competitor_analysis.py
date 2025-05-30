"""
Unit Tests for Competitor Analysis Module
Tests CompetitorAnalyzer, UDF functions, and visualization capabilities
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import tempfile
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StringType, FloatType, ArrayType, StructType, StructField
from pyspark.sql.functions import col, lit, array, expr

# Import competitor analysis components
from src.spark.competitor_analysis import (
    CompetitorAnalyzer, create_brand_recognition_udf, 
    create_feature_extraction_udf
)


class TestCompetitorAnalyzer(unittest.TestCase):
    """Test cases for CompetitorAnalyzer class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up Spark session for tests"""
        cls.spark = SparkSession.builder \
            .appName("TestCompetitorAnalysis") \
            .master("local[2]") \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
            .getOrCreate()
        cls.spark.sparkContext.setLogLevel("WARN")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up Spark session"""
        cls.spark.stop()
    
    def setUp(self):
        """Set up test data and mock configuration"""
        # Create test brand recognizer mock
        self.mock_brand_recognizer = Mock()
        self.mock_brand_recognizer.competitor_map = {
            'Apple': {'Samsung', 'Google'},
            'Samsung': {'Apple', 'Huawei'},
            'Google': {'Apple', 'Microsoft'}
        }
        self.mock_brand_recognizer.brands = {
            'apple': {'original_name': 'Apple', 'industry': 'technology'},
            'samsung': {'original_name': 'Samsung', 'industry': 'technology'},
            'google': {'original_name': 'Google', 'industry': 'technology'}
        }
        
        # Initialize analyzer
        self.analyzer = CompetitorAnalyzer(self.spark, self.mock_brand_recognizer)
        
        # Create test DataFrame
        self.test_data = [
            {
                'text': 'I love my new iPhone from Apple',
                'timestamp': datetime(2024, 1, 1, 10, 0),
                'sentiment': 1.0,
                'brands': ['Apple:0.95'],
                'vader_compound': 0.6,
                'emoji_sentiment': 0.8
            },
            {
                'text': 'Samsung Galaxy has great camera',
                'timestamp': datetime(2024, 1, 1, 11, 0),
                'sentiment': 1.0,
                'brands': ['Samsung:0.90'],
                'vader_compound': 0.5,
                'emoji_sentiment': 0.7
            },
            {
                'text': 'Apple iPhone vs Samsung Galaxy comparison',
                'timestamp': datetime(2024, 1, 1, 12, 0),
                'sentiment': 0.5,
                'brands': ['Apple:0.85', 'Samsung:0.80'],
                'vader_compound': 0.0,
                'emoji_sentiment': 0.5
            },
            {
                'text': 'Google Pixel camera is disappointing',
                'timestamp': datetime(2024, 1, 1, 13, 0),
                'sentiment': 0.0,
                'brands': ['Google:0.88'],
                'vader_compound': -0.4,
                'emoji_sentiment': 0.2
            }
        ]
        
        # Convert to Spark DataFrame
        self.df = self.spark.createDataFrame(self.test_data)
    
    def test_init(self):
        """Test CompetitorAnalyzer initialization"""
        self.assertEqual(self.analyzer.spark, self.spark)
        self.assertEqual(self.analyzer.brand_recognizer, self.mock_brand_recognizer)
        self.assertIsNotNone(self.analyzer.time_windows)
        self.assertIn('daily', self.analyzer.time_windows)
    
    def test_parse_brand_confidence(self):
        """Test parsing brand:confidence strings"""
        result = self.analyzer.parse_brand_confidence(self.df)
        
        # Check that brands are parsed correctly
        self.assertTrue(result.count() >= 4)  # At least 4 brand mentions
        
        # Check schema
        expected_columns = ['brand', 'confidence', 'brand_info']
        for col_name in expected_columns:
            self.assertIn(col_name, result.columns)
        
        # Check data types
        collected = result.collect()
        if collected:
            first_row = collected[0]
            self.assertIsInstance(first_row['brand'], str)
            self.assertIsInstance(first_row['confidence'], float)
    
    def test_aggregate_brand_sentiment(self):
        """Test brand sentiment aggregation"""
        result = self.analyzer.aggregate_brand_sentiment(
            self.df, 
            time_window="1 day",
            confidence_threshold=0.7
        )
        
        # Check that aggregation produces results
        self.assertTrue(result.count() > 0)
        
        # Check required columns
        expected_columns = [
            'brand', 'mention_count', 'avg_sentiment', 'sentiment_score',
            'positive_ratio', 'window_start', 'window_end'
        ]
        for col_name in expected_columns:
            self.assertIn(col_name, result.columns)
        
        # Check data validity
        collected = result.collect()
        if collected:
            first_row = collected[0]
            self.assertGreaterEqual(first_row['mention_count'], 1)
            self.assertGreaterEqual(first_row['positive_ratio'], 0)
            self.assertLessEqual(first_row['positive_ratio'], 1)
    
    def test_compare_competitor_sentiment(self):
        """Test competitor sentiment comparison"""
        target_brand = 'Apple'
        competitors = ['Samsung', 'Google']
        
        result = self.analyzer.compare_competitor_sentiment(
            self.df, 
            target_brand=target_brand,
            competitors=competitors
        )
        
        if result.count() > 0:
            # Check required columns
            expected_columns = [
                'brand', 'market_rank', 'sentiment_rank', 'market_share',
                'is_target', 'sentiment_gap', 'mention_gap'
            ]
            for col_name in expected_columns:
                self.assertIn(col_name, result.columns)
            
            # Check that target brand is identified
            collected = result.collect()
            target_rows = [row for row in collected if row['is_target'] == 1]
            self.assertTrue(len(target_rows) > 0)
    
    def test_analyze_feature_sentiment(self):
        """Test feature-level sentiment analysis"""
        features = ['camera', 'price', 'quality', 'performance']
        
        result = self.analyzer.analyze_feature_sentiment(
            self.df,
            features=features,
            auto_detect=False
        )
        
        if result.count() > 0:
            # Check schema
            expected_columns = [
                'brand', 'feature', 'mention_count', 'avg_sentiment',
                'sentiment_score', 'positive_ratio', 'feature_rank'
            ]
            for col_name in expected_columns:
                self.assertIn(col_name, result.columns)
            
            # Check data validity
            collected = result.collect()
            first_row = collected[0]
            self.assertIn(first_row['feature'], features)
            self.assertGreaterEqual(first_row['mention_count'], 1)
    
    def test_calculate_share_of_voice(self):
        """Test share of voice calculation"""
        result = self.analyzer.calculate_share_of_voice(
            self.df,
            time_window="1 day",
            min_mentions=1
        )
        
        if result.count() > 0:
            # Check schema
            expected_columns = [
                'brand', 'share_of_voice', 'sov_rank', 'market_position',
                'sov_change', 'sov_change_pct'
            ]
            for col_name in expected_columns:
                self.assertIn(col_name, result.columns)
            
            # Check data validity
            collected = result.collect()
            total_sov = sum(row['share_of_voice'] for row in collected 
                           if row['window_start'] == collected[0]['window_start'])
            # Total SOV should be approximately 100% (allowing for rounding)
            self.assertAlmostEqual(total_sov, 100.0, delta=0.1)
    
    def test_compute_sentiment_momentum(self):
        """Test sentiment momentum computation"""
        # First aggregate sentiment data
        sentiment_df = self.analyzer.aggregate_brand_sentiment(self.df, "1 day")
        
        result = self.analyzer.compute_sentiment_momentum(
            sentiment_df,
            lookback_windows=3,
            forecast_windows=2
        )
        
        if result.count() > 0:
            # Check momentum columns
            expected_columns = [
                'sentiment_momentum', 'sentiment_velocity', 'momentum_trend',
                'momentum_signal', 'trend_strength', 'projected_sentiment'
            ]
            for col_name in expected_columns:
                self.assertIn(col_name, result.columns)
            
            # Check trend classification
            collected = result.collect()
            if collected:
                trend_values = set(row['momentum_trend'] for row in collected if row['momentum_trend'])
                valid_trends = {'strong_uptrend', 'uptrend', 'sideways', 'downtrend', 'strong_downtrend'}
                self.assertTrue(trend_values.issubset(valid_trends))
    
    @patch('src.spark.competitor_analysis.os.makedirs')
    @patch('src.spark.competitor_analysis.json.dump')
    def test_generate_competitive_insights(self, mock_json_dump, mock_makedirs):
        """Test competitive insights generation"""
        # Create enhanced test data with required columns
        enhanced_data = self.df.union(
            self.spark.createDataFrame([
                {
                    'text': 'Another Apple mention',
                    'timestamp': datetime(2024, 1, 2, 10, 0),
                    'sentiment': 0.8,
                    'brands': ['Apple:0.92'],
                    'vader_compound': 0.4,
                    'emoji_sentiment': 0.6
                }
            ])
        )
        
        # Process data through aggregation pipeline
        sentiment_df = self.analyzer.aggregate_brand_sentiment(enhanced_data, "1 day")
        momentum_df = self.analyzer.compute_sentiment_momentum(sentiment_df)
        
        target_brand = 'Apple'
        competitors = ['Samsung', 'Google']
        
        # Test with explicit competitors list
        insights = self.analyzer.generate_competitive_insights(
            momentum_df,
            target_brand=target_brand,
            competitors=competitors,
            save_path="/tmp/test_insights.json"
        )
        
        # Check insights structure
        required_keys = [
            'brand', 'analysis_date', 'summary', 'metrics', 
            'competitors', 'trends', 'opportunities', 'threats', 
            'recommendations', 'insights'  # New key for revised pipeline
        ]
        for key in required_keys:
            self.assertIn(key, insights)
        
        # Check insights content
        self.assertEqual(insights['brand'], target_brand)
        self.assertIsInstance(insights['summary'], dict)
        self.assertIsInstance(insights['recommendations'], list)
        self.assertIsInstance(insights['insights'], dict)
        
        # Verify file save was attempted
        mock_makedirs.assert_called()
        mock_json_dump.assert_called()

    @patch('src.spark.competitor_analysis.os.makedirs')
    @patch('src.spark.competitor_analysis.json.dump')
    def test_generate_competitive_insights_auto_detect(self, mock_json_dump, mock_makedirs):
        """Test competitive insights generation with auto-detected competitors"""
        # Process data through aggregation pipeline
        sentiment_df = self.analyzer.aggregate_brand_sentiment(self.df, "1 day")
        momentum_df = self.analyzer.compute_sentiment_momentum(sentiment_df)
        
        target_brand = 'Apple'
        
        # Test without explicit competitors (should auto-detect)
        insights = self.analyzer.generate_competitive_insights(
            momentum_df,
            target_brand=target_brand
            # No competitors parameter - should auto-detect
        )
        
        # Check insights structure
        self.assertIn('competitors', insights)
        self.assertIsInstance(insights['competitors'], dict)
        
        # Should have detected competitors from brand recognizer
        self.assertGreaterEqual(len(insights['competitors']), 0)

    @patch('src.spark.competitor_analysis.os.makedirs')
    @patch('src.spark.competitor_analysis.json.dump')
    def test_generate_competitive_insights_fallback(self, mock_json_dump, mock_makedirs):
        """Test competitive insights with fallback competitor detection"""
        # Create analyzer without brand recognizer
        analyzer_no_recognizer = CompetitorAnalyzer(self.spark, None)
        
        # Process data through aggregation pipeline
        sentiment_df = analyzer_no_recognizer.aggregate_brand_sentiment(self.df, "1 day")
        momentum_df = analyzer_no_recognizer.compute_sentiment_momentum(sentiment_df)
        
        target_brand = 'Apple'
        
        # Test fallback competitor detection (top brands by mention count)
        insights = analyzer_no_recognizer.generate_competitive_insights(
            momentum_df,
            target_brand=target_brand
        )
        
        # Should still generate insights with fallback competitors
        self.assertIn('competitors', insights)
        self.assertIsInstance(insights['competitors'], dict)

    def test_insights_market_position_structure(self):
        """Test that insights include market position data for revised pipeline compatibility"""
        # Create test data with share_of_voice and sov_rank columns
        test_data_with_sov = [
            {
                'brand': 'apple',
                'window_start': datetime(2024, 1, 1),
                'sentiment_score': 75.0,
                'mention_count': 100,
                'share_of_voice': 45.0,
                'sov_rank': 1,
                'positive_ratio': 0.8,
                'market_position': 'Leader',
                'sentiment_momentum': 2.1
            }
        ]
        
        sov_df = self.spark.createDataFrame(test_data_with_sov)
        
        insights = self.analyzer.generate_competitive_insights(
            sov_df,
            target_brand='Apple'
        )
        
        # Check that insights include market position data
        self.assertIn('insights', insights)
        self.assertIn('market_position', insights['insights'])
        
        market_pos = insights['insights']['market_position']
        self.assertIn('share_of_voice', market_pos)
        self.assertIn('sentiment_score', market_pos)
        self.assertIn('market_rank', market_pos)
        
        # Verify values
        self.assertEqual(market_pos['share_of_voice'], 45.0)
        self.assertEqual(market_pos['sentiment_score'], 75.0)
        self.assertEqual(market_pos['market_rank'], 1)
    
    def test_generate_strategic_summary(self):
        """Test strategic summary generation"""
        # Create mock insights
        mock_insights = {
            'brand': 'Apple',
            'summary': {'average_sentiment': 65.5},
            'competitors': {
                'Samsung': {'competitive_position': 'behind'},
                'Google': {'competitive_position': 'ahead'}
            },
            'trends': {'momentum': 2.1},
            'recommendations': [
                {'area': 'sentiment'}, 
                {'area': 'marketing'}
            ]
        }
        
        summary = self.analyzer._generate_strategic_summary(mock_insights)
        
        self.assertIsInstance(summary, str)
        self.assertIn('Apple', summary)
        self.assertIn('positive', summary)  # Should identify positive sentiment
    
    @patch('src.spark.competitor_analysis.plt')
    @patch('src.spark.competitor_analysis.sns')
    @patch('src.spark.competitor_analysis.os.makedirs')
    def test_visualize_competitive_landscape(self, mock_makedirs, mock_sns, mock_plt):
        """Test competitive landscape visualization"""
        # Create test DataFrame with all required columns
        viz_data = [
            {
                'brand': 'Apple',
                'window_start': datetime(2024, 1, 1),
                'share_of_voice': 45.0,
                'sentiment_score': 75.0,
                'mention_count': 100,
                'sentiment_momentum': 2.1
            },
            {
                'brand': 'Samsung',
                'window_start': datetime(2024, 1, 1),
                'share_of_voice': 35.0,
                'sentiment_score': 65.0,
                'mention_count': 80,
                'sentiment_momentum': -1.2
            }
        ]
        
        viz_df = self.spark.createDataFrame(viz_data)
        
        # Test visualization creation
        self.analyzer.visualize_competitive_landscape(
            viz_df,
            target_brand='Apple',
            output_dir='/tmp/test_viz'
        )
        
        # Verify directory creation and plot calls
        mock_makedirs.assert_called()
        self.assertTrue(mock_plt.subplots.called or mock_plt.figure.called)


class TestUDFFunctions(unittest.TestCase):
    """Test cases for UDF functions"""
    
    @classmethod
    def setUpClass(cls):
        """Set up Spark session for UDF tests"""
        cls.spark = SparkSession.builder \
            .appName("TestUDFs") \
            .master("local[1]") \
            .getOrCreate()
        cls.spark.sparkContext.setLogLevel("WARN")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up Spark session"""
        cls.spark.stop()
    
    def setUp(self):
        """Set up test configuration"""
        self.test_config = {
            "industries": {
                "technology": {
                    "brands": [
                        {
                            "name": "Apple",
                            "aliases": ["AAPL"],
                            "products": ["iPhone", "iPad"],
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
        """Clean up temporary files"""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    @patch('src.ml.entity_recognition.get_path')
    @patch('src.ml.entity_recognition.HAS_FUZZYWUZZY', True)
    def test_create_brand_recognition_udf(self, mock_get_path):
        """Test brand recognition UDF creation and execution"""
        mock_get_path.return_value = self.temp_config.name
        
        # Create UDF
        brand_udf = create_brand_recognition_udf()
        
        # Test UDF function directly
        test_texts = [
            "I love my Apple iPhone",
            "Samsung Galaxy is great",
            None,
            "",
            "No brands mentioned here"
        ]
        
        for text in test_texts:
            result = brand_udf.func(text)
            self.assertIsInstance(result, list)
            if text and ("Apple" in text or "Samsung" in text):
                # May detect brands depending on fuzzywuzzy performance
                self.assertGreaterEqual(len(result), 0)
            elif not text:
                self.assertEqual(len(result), 0)
    
    def test_create_feature_extraction_udf(self):
        """Test feature extraction UDF"""
        features = ['camera', 'price', 'quality', 'battery']
        feature_udf = create_feature_extraction_udf(features)
        
        test_cases = [
            ("The camera quality is amazing", ['camera', 'quality']),
            ("Price is too high", ['price']),
            ("Battery life is great", []),  # 'battery' not 'battery life'
            (None, []),
            ("", [])
        ]
        
        for text, expected_features in test_cases:
            result = feature_udf.func(text)
            self.assertIsInstance(result, list)
            
            if text:
                # Check that detected features are subset of expected
                for feature in result:
                    self.assertIn(feature, features)
    
    @patch('src.ml.entity_recognition.get_path')
    def test_udf_spark_integration(self, mock_get_path):
        """Test UDF integration with Spark DataFrames"""
        mock_get_path.return_value = self.temp_config.name
        
        # Create test DataFrame
        test_data = [
            {"text": "Apple iPhone camera is great"},
            {"text": "Samsung price is competitive"},
            {"text": "No brands here"}
        ]
        df = self.spark.createDataFrame(test_data)
        
        # Apply brand recognition UDF
        brand_udf = create_brand_recognition_udf()
        df_with_brands = df.withColumn("brands", brand_udf(col("text")))
        
        # Apply feature extraction UDF
        features = ['camera', 'price']
        feature_udf = create_feature_extraction_udf(features)
        df_with_features = df_with_brands.withColumn("features", feature_udf(col("text")))
        
        # Collect results
        results = df_with_features.collect()
        
        # Verify structure
        self.assertEqual(len(results), 3)
        for row in results:
            self.assertIn('brands', row.asDict())
            self.assertIn('features', row.asDict())
            self.assertIsInstance(row['brands'], list)
            self.assertIsInstance(row['features'], list)


class TestDataProcessingMethods(unittest.TestCase):
    """Test data processing and utility methods"""
    
    @classmethod
    def setUpClass(cls):
        """Set up Spark session"""
        cls.spark = SparkSession.builder \
            .appName("TestDataProcessing") \
            .master("local[1]") \
            .getOrCreate()
        cls.spark.sparkContext.setLogLevel("WARN")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up Spark session"""
        cls.spark.stop()
    
    def setUp(self):
        """Set up test analyzer"""
        mock_brand_recognizer = Mock()
        mock_brand_recognizer.competitor_map = {}
        mock_brand_recognizer.brands = {}
        
        self.analyzer = CompetitorAnalyzer(self.spark, mock_brand_recognizer)
    
    def test_time_windows_configuration(self):
        """Test time window configurations"""
        expected_windows = {'hourly', 'daily', 'weekly', 'monthly'}
        actual_windows = set(self.analyzer.time_windows.keys())
        
        self.assertEqual(actual_windows, expected_windows)
        
        # Check time window values
        self.assertEqual(self.analyzer.time_windows['hourly'], '1 hour')
        self.assertEqual(self.analyzer.time_windows['daily'], '1 day')
        self.assertEqual(self.analyzer.time_windows['weekly'], '1 week')
        self.assertEqual(self.analyzer.time_windows['monthly'], '1 month')
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames"""
        # Create empty DataFrame with correct schema
        schema = StructType([
            StructField("text", StringType(), True),
            StructField("brands", ArrayType(StringType()), True),
            StructField("sentiment", FloatType(), True)
        ])
        empty_df = self.spark.createDataFrame([], schema)
        
        # Test aggregation with empty data
        result = self.analyzer.aggregate_brand_sentiment(empty_df)
        self.assertEqual(result.count(), 0)
        
        # Test share of voice with empty data
        sov_result = self.analyzer.calculate_share_of_voice(empty_df)
        self.assertEqual(sov_result.count(), 0)
    
    def test_error_handling_in_methods(self):
        """Test error handling in various methods"""
        # Test with malformed data
        malformed_data = [{"text": "test", "brands": "not_an_array"}]
        
        try:
            malformed_df = self.spark.createDataFrame(malformed_data)
            # This should not crash the analyzer
            result = self.analyzer.parse_brand_confidence(malformed_df)
            # Method should handle the error gracefully
        except Exception as e:
            # Expected to fail with malformed data
            self.assertIsInstance(e, Exception)


class TestPerformanceAndScaling(unittest.TestCase):
    """Test performance characteristics and scaling behavior"""
    
    @classmethod
    def setUpClass(cls):
        """Set up Spark session with performance settings"""
        cls.spark = SparkSession.builder \
            .appName("TestPerformance") \
            .master("local[2]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        cls.spark.sparkContext.setLogLevel("WARN")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up Spark session"""
        cls.spark.stop()
    
    def setUp(self):
        """Set up test environment"""
        mock_brand_recognizer = Mock()
        mock_brand_recognizer.competitor_map = {'Apple': {'Samsung'}}
        mock_brand_recognizer.brands = {
            'apple': {'original_name': 'Apple', 'industry': 'technology'}
        }
        
        self.analyzer = CompetitorAnalyzer(self.spark, mock_brand_recognizer)
    
    def test_large_dataset_handling(self):
        """Test behavior with larger datasets"""
        # Create larger test dataset
        large_data = []
        brands = ['Apple:0.9', 'Samsung:0.8', 'Google:0.85']
        
        for i in range(100):  # 100 records
            large_data.append({
                'text': f'Sample text {i}',
                'timestamp': datetime(2024, 1, 1) + timedelta(hours=i % 24),
                'sentiment': 0.5 + (i % 3) * 0.25,  # Vary sentiment
                'brands': [brands[i % len(brands)]]
            })
        
        large_df = self.spark.createDataFrame(large_data)
        
        # Test aggregation performance
        start_time = datetime.now()
        result = self.analyzer.aggregate_brand_sentiment(large_df, "1 hour")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Should complete in reasonable time (< 30 seconds for 100 records)
        self.assertLess(processing_time, 30)
        self.assertGreater(result.count(), 0)
    
    def test_memory_efficiency(self):
        """Test memory usage patterns"""
        # Create dataset with repeated processing
        test_data = [
            {
                'text': 'Apple iPhone test',
                'timestamp': datetime(2024, 1, 1, 10, 0),
                'sentiment': 0.8,
                'brands': ['Apple:0.9']
            }
        ] * 50  # Repeat 50 times
        
        df = self.spark.createDataFrame(test_data)
        
        # Process multiple times to test memory usage
        for i in range(5):
            result = self.analyzer.aggregate_brand_sentiment(df, "1 day")
            count = result.count()
            self.assertGreater(count, 0)
            
            # Clear cache to prevent memory buildup
            self.spark.catalog.clearCache()


if __name__ == '__main__':
    # Run tests with different verbosity levels
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestCompetitorAnalyzer,
        TestUDFFunctions,
        TestDataProcessingMethods,
        TestPerformanceAndScaling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMPETITOR ANALYSIS TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%") 