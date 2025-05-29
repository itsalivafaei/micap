import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Mock all heavy imports before any imports
sys.modules['pyspark'] = MagicMock()
sys.modules['pyspark.sql'] = MagicMock() 
sys.modules['pyspark.sql.functions'] = MagicMock()
sys.modules['pyspark.sql.types'] = MagicMock()
sys.modules['pyspark.ml'] = MagicMock()
sys.modules['pyspark.ml.feature'] = MagicMock()
sys.modules['pyspark.ml.clustering'] = MagicMock()
sys.modules['pyspark.ml.regression'] = MagicMock()
sys.modules['pyspark.ml.classification'] = MagicMock()
sys.modules['pyspark.ml.evaluation'] = MagicMock()
sys.modules['pyspark.ml.tuning'] = MagicMock()
sys.modules['pyspark.ml.linalg'] = MagicMock()

# Mock pandas and numpy
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Mock scikit-learn
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.ensemble'] = MagicMock()
sys.modules['sklearn.model_selection'] = MagicMock()
sys.modules['sklearn.preprocessing'] = MagicMock()
sys.modules['sklearn.metrics'] = MagicMock()

# Mock prophet
sys.modules['prophet'] = MagicMock()

# Mock other ML/data libraries that might be imported
sys.modules['spacy'] = MagicMock()
sys.modules['nltk'] = MagicMock()
sys.modules['textblob'] = MagicMock()


class TestRunPhase2Pipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up mocks for each test"""
        # Create comprehensive mocks
        self.mock_spark = MagicMock()
        self.mock_df = MagicMock()
        
        # Setup DataFrame method chaining
        self.mock_df.sample.return_value = self.mock_df
        self.mock_df.count.return_value = 100
        self.mock_df.withColumn.return_value = self.mock_df
        self.mock_df.filter.return_value = self.mock_df
        self.mock_df.select.return_value = self.mock_df
        self.mock_df.distinct.return_value = self.mock_df
        self.mock_df.limit.return_value = self.mock_df
        self.mock_df.collect.return_value = [{'brand': 'TestBrand'}]
        self.mock_df.groupBy.return_value = self.mock_df
        self.mock_df.agg.return_value = self.mock_df
        self.mock_df.coalesce.return_value = self.mock_df
        self.mock_df.write.mode.return_value = self.mock_df.write
        self.mock_df.write.parquet.return_value = None
        self.mock_df.show.return_value = None
        self.mock_df.explode.return_value = self.mock_df
        self.mock_df.withColumnRenamed.return_value = self.mock_df
        self.mock_df.drop.return_value = self.mock_df
        self.mock_df.alias.return_value = self.mock_df
        self.mock_df.__getitem__.return_value = self.mock_df
        self.mock_df.__iter__.return_value = iter([{'brand': 'TestBrand'}])
        
        # Setup spark session
        self.mock_spark.read.parquet.return_value = self.mock_df
        self.mock_spark.read.parquet.side_effect = None  # Reset any side effects from previous tests
        self.mock_spark.stop.return_value = None

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    @patch('json.dump')
    @patch('pyspark.sql.functions.size')
    @patch('pyspark.sql.functions.col')
    def test_pipeline_runs(self, mock_col, mock_size, mock_json_dump, mock_makedirs, mock_file):
        """Test that the pipeline runs without errors"""
        
        # Setup size function to return a mockable column that supports comparison
        mock_column = MagicMock()
        mock_column.__gt__ = MagicMock(return_value=mock_column)
        mock_column.__lt__ = MagicMock(return_value=mock_column) 
        mock_column.__eq__ = MagicMock(return_value=mock_column)
        mock_size.return_value = mock_column
        mock_col.return_value = mock_column
        
        with patch('src.utils.path_utils.get_path', side_effect=lambda x: x), \
             patch('config.spark_config.create_spark_session', return_value=self.mock_spark):
            
            # Import here to avoid import issues during test discovery
            import scripts.run_phase2_pipeline as pipeline
            
            # Patch all the class constructors and functions after import
            with patch.object(pipeline, 'BrandRecognizer') as mock_brand_recognizer, \
                 patch.object(pipeline, 'ProductExtractor') as mock_product_extractor, \
                 patch.object(pipeline, 'create_brand_recognition_udf') as mock_brand_udf, \
                 patch.object(pipeline, 'create_product_extraction_udf') as mock_product_udf, \
                 patch.object(pipeline, 'CompetitorAnalyzer') as mock_competitor_analyzer, \
                 patch.object(pipeline, 'TopicModeler') as mock_topic_modeler, \
                 patch.object(pipeline, 'TrendForecaster') as mock_trend_forecaster, \
                 patch.object(pipeline, 'AnomalyDetector') as mock_anomaly_detector, \
                 patch.object(pipeline, 'ViralityPredictor') as mock_virality_predictor:
                
                # Setup return values for constructor calls
                mock_brand_recognizer_instance = MagicMock()
                mock_product_extractor_instance = MagicMock()
                mock_competitor_analyzer_instance = MagicMock()
                mock_topic_modeler_instance = MagicMock()
                mock_trend_forecaster_instance = MagicMock()
                mock_anomaly_detector_instance = MagicMock()
                
                mock_brand_recognizer.return_value = mock_brand_recognizer_instance
                mock_product_extractor.return_value = mock_product_extractor_instance
                mock_competitor_analyzer.return_value = mock_competitor_analyzer_instance
                mock_topic_modeler.return_value = mock_topic_modeler_instance
                mock_trend_forecaster.return_value = mock_trend_forecaster_instance
                mock_anomaly_detector.return_value = mock_anomaly_detector_instance
                
                # Setup UDF returns
                mock_brand_udf.return_value = MagicMock()
                mock_product_udf.return_value = MagicMock()
                
                # Setup method returns
                mock_competitor_analyzer_instance.aggregate_brand_sentiment.return_value = self.mock_df
                mock_competitor_analyzer_instance.calculate_share_of_voice.return_value = self.mock_df
                mock_competitor_analyzer_instance.compute_sentiment_momentum.return_value = self.mock_df
                mock_competitor_analyzer_instance.generate_competitive_insights.return_value = {'insight': 1}
                
                mock_topic_modeler_instance.fit_topics.return_value = None
                mock_topic_modeler_instance.get_topics.return_value = {0: [('word', 0.5)]}
                mock_topic_modeler_instance.transform.return_value = self.mock_df
                
                mock_trend_forecaster_instance.forecast_sentiment_trends.return_value = self.mock_df
                mock_trend_forecaster_instance.forecast_topic_trends.return_value = self.mock_df
                
                mock_anomaly_detector_instance.detect_sentiment_anomalies.return_value = self.mock_df
                mock_anomaly_detector_instance.detect_volume_anomalies.return_value = self.mock_df
                
                # Run the pipeline
                pipeline.run_phase2_pipeline(sample_size=0.1)
                
                # Verify calls were made
                self.assertTrue(mock_makedirs.called)
                self.assertTrue(self.mock_spark.stop.called)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_pipeline_exception_handling(self, mock_makedirs, mock_file):
        """Test that the pipeline handles exceptions properly"""
        
        # Setup spark to raise exception
        self.mock_spark.read.parquet.side_effect = Exception('Test error')
        
        with patch('src.utils.path_utils.get_path', side_effect=lambda x: x), \
             patch('config.spark_config.create_spark_session', return_value=self.mock_spark):
            
            # Import here to avoid import issues during test discovery
            import scripts.run_phase2_pipeline as pipeline
            
            # Test that exception is raised and cleanup happens
            try:
                pipeline.run_phase2_pipeline(sample_size=0.1)
                self.fail("Expected exception was not raised")
            except Exception as e:
                # Verify it's the expected exception
                self.assertIn("Test error", str(e))
            
            # Verify cleanup was called
            self.assertTrue(self.mock_spark.stop.called)


if __name__ == '__main__':
    unittest.main() 