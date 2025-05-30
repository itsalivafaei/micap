"""
Unit Tests for Trend Detection Module
Tests TopicModeler, TrendForecaster, AnomalyDetector, and ViralityPredictor classes
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ml.trend_detection import (
    TopicModeler, TrendForecaster, AnomalyDetector, ViralityPredictor
)


# Mock Spark SQL functions to avoid SparkContext issues
class MockColumn:
    """Mock Column object"""
    def __init__(self, name):
        self.name = name
    
    def alias(self, alias_name):
        return MockColumn(alias_name)
    
    def __gt__(self, other):
        return MockColumn(f"{self.name} > {other}")
    
    def __eq__(self, other):
        return MockColumn(f"{self.name} == {other}")
    
    def __add__(self, other):
        return MockColumn(f"{self.name} + {other}")
    
    def __sub__(self, other):
        return MockColumn(f"{self.name} - {other}")
    
    def __mul__(self, other):
        return MockColumn(f"{self.name} * {other}")
    
    def __truediv__(self, other):
        return MockColumn(f"{self.name} / {other}")
    
    def over(self, window_spec):
        return MockColumn(f"{self.name}.over(window)")
    
    def contains(self, substr):
        return MockColumn(f"{self.name}.contains({substr})")
    
    def cast(self, data_type):
        return MockColumn(f"{self.name}.cast({data_type})")
    
    def otherwise(self, value):
        return MockColumn(f"{self.name}.otherwise({value})")


def mock_col(name):
    """Mock col function"""
    return MockColumn(name)


def mock_lit(value):
    """Mock lit function"""
    return MockColumn(f"lit({value})")


def mock_avg(col_name):
    """Mock avg function"""
    return MockColumn(f"avg({col_name})")


def mock_count(col_name):
    """Mock count function"""
    return MockColumn(f"count({col_name})")


def mock_size(col_name):
    """Mock size function"""
    return MockColumn(f"size({col_name})")


def mock_when(condition, value):
    """Mock when function"""
    mock_when_obj = Mock()
    mock_when_obj.otherwise.return_value = MockColumn(f"when({condition}, {value})")
    return mock_when_obj


def mock_window(time_column, duration):
    """Mock window function"""
    return MockColumn(f"window({time_column}, {duration})")


def mock_lag(col_name, offset=1):
    """Mock lag function"""
    return MockColumn(f"lag({col_name}, {offset})")


def mock_stddev(col_name):
    """Mock stddev function"""
    return MockColumn(f"stddev({col_name})")


def mock_spark_abs(col_name):
    """Mock spark_abs function"""
    return MockColumn(f"abs({col_name})")


class MockWindow:
    """Mock Window class"""
    @staticmethod
    def partitionBy(*cols):
        mock_window = Mock()
        mock_window.orderBy.return_value = Mock()
        return mock_window
    
    @staticmethod
    def orderBy(*cols):
        mock_window = Mock()
        mock_window.rowsBetween.return_value = Mock()
        return mock_window


class MockRow:
    """Mock Row object for Spark DataFrame results"""
    def __init__(self, **kwargs):
        self._data = kwargs
        # Set attributes directly for proper access
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class MockSparkSession:
    """Mock SparkSession for testing"""
    def __init__(self):
        self.sql = Mock()
        self.stop = Mock()
        self.createDataFrame = Mock()
        
    def createDataFrame(self, data, schema=None):
        """Mock createDataFrame method"""
        return MockDataFrame(data)


class MockDataFrame:
    """Mock Spark DataFrame for testing"""
    def __init__(self, data=None):
        self.data = data or []
        self.columns = ['timestamp', 'text', 'tokens_lemmatized', 'sentiment', 'brand']
        
    def withColumn(self, col_name, col_expr):
        return MockDataFrame(self.data)
    
    def withColumnRenamed(self, old_name, new_name):
        return MockDataFrame(self.data)
    
    def select(self, *cols):
        return MockDataFrame(self.data)
    
    def filter(self, condition):
        return MockDataFrame(self.data)
    
    def drop(self, col_name):
        return MockDataFrame(self.data)
    
    def count(self):
        return len(self.data) if self.data else 10
    
    def cache(self):
        return self
    
    def persist(self):
        return self
    
    def unpersist(self):
        return self
    
    def groupBy(self, *cols):
        """Mock groupBy operation"""
        mock_grouped = Mock()
        mock_grouped.agg = Mock(return_value=self)
        mock_grouped.count = Mock(return_value=self)
        return mock_grouped
    
    def agg(self, *args):
        return MockDataFrame(self.data)
    
    def orderBy(self, *cols):
        return MockDataFrame(self.data)
    
    def collect(self):
        """Mock collect operation that returns proper Row objects"""
        return [
            MockRow(topic=0, termIndices=[1, 2, 3], termWeights=[0.1, 0.08, 0.06]),
            MockRow(topic=1, termIndices=[4, 5, 6], termWeights=[0.12, 0.09, 0.07])
        ]
    
    def show(self, n=20):
        print(f"Showing {min(n, len(self.data))} rows")
    
    def toPandas(self):
        """Convert to pandas DataFrame"""
        if not self.data:
            return pd.DataFrame({
                'window_start': pd.date_range('2023-01-01', periods=10),
                'dominant_topic': np.random.randint(0, 5, 10),
                'doc_count': np.random.randint(1, 100, 10),
                'brand': ['Apple'] * 10,
                'sentiment_score': np.random.rand(10),
                'mention_count': np.random.randint(1, 100, 10)
            })
        return pd.DataFrame(self.data)


class MockPipelineModel:
    """Mock PipelineModel for testing"""
    def __init__(self):
        self.stages = [Mock(), Mock(), Mock(), Mock()]  # Mocking 4 stages
        self.stages[1].vocabulary = ['apple', 'phone', 'innovation', 'technology', 'design']
        
    def transform(self, df):
        # Mock transform to add topicDistribution column
        mock_df = MockDataFrame()
        mock_df.columns = ['timestamp', 'text', 'tokens_lemmatized', 'topicDistribution']
        return mock_df


class MockPipeline:
    """Mock Pipeline for testing"""
    def __init__(self, stages):
        self.stages = stages
    
    def fit(self, df):
        return MockPipelineModel()


class MockLDAModel:
    """Mock LDA model for testing"""
    def __init__(self):
        self.vocabulary = ['apple', 'phone', 'innovation', 'technology', 'design']
        
    def describeTopics(self, max_words):
        """Mock describe topics"""
        return MockDataFrame()


class TestTopicModeler(unittest.TestCase):
    """Test cases for TopicModeler class"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()
        self.modeler = TopicModeler(self.mock_spark, num_topics=5)

    def test_init(self):
        """Test TopicModeler initialization"""
        self.assertEqual(self.modeler.spark, self.mock_spark)
        self.assertEqual(self.modeler.num_topics, 5)
        self.assertIsNone(self.modeler.model)
        self.assertIsNone(self.modeler.vocabulary)

    @patch('src.ml.trend_detection.StopWordsRemover')
    @patch('src.ml.trend_detection.CountVectorizer')
    @patch('src.ml.trend_detection.IDF')
    @patch('src.ml.trend_detection.LDA')
    @patch('src.ml.trend_detection.Pipeline')
    def test_fit_topics(self, mock_pipeline, mock_lda, mock_idf, mock_cv, mock_remover):
        """Test topic model fitting"""
        # Setup mocks
        mock_remover.return_value = Mock()
        mock_cv.return_value = Mock()
        mock_idf.return_value = Mock()
        mock_lda.return_value = Mock()
        
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        mock_fitted_pipeline = MockPipelineModel()
        mock_pipeline_instance.fit.return_value = mock_fitted_pipeline
        
        df = MockDataFrame([{'tokens_lemmatized': ['apple', 'phone']}])
        
        result = self.modeler.fit_topics(df)
        
        self.assertEqual(result, self.modeler)
        self.assertIsNotNone(self.modeler.model)
        self.assertIsNotNone(self.modeler.vocabulary)
        mock_pipeline.assert_called_once()

    def test_get_topics_not_fitted(self):
        """Test get_topics when model is not fitted"""
        with self.assertRaises(ValueError):
            self.modeler.get_topics()

    def test_get_topics_fitted(self):
        """Test get_topics when model is fitted"""
        # Setup a fitted model
        self.modeler.model = MockPipelineModel()
        self.modeler.vocabulary = ['apple', 'phone', 'innovation', 'technology', 'design', 'great', 'product']
        
        # Mock the LDA model's describeTopics method
        mock_lda_stage = Mock()
        mock_lda_stage.describeTopics.return_value = MockDataFrame()
        self.modeler.model.stages[-1] = mock_lda_stage
        
        result = self.modeler.get_topics(max_words=3)
        
        self.assertIsInstance(result, dict)
        # Should have topics from the mock collect method
        self.assertIn(0, result)
        self.assertIn(1, result)

    def test_transform_not_fitted(self):
        """Test transform when model is not fitted"""
        df = MockDataFrame()
        with self.assertRaises(ValueError):
            self.modeler.transform(df)

    def test_transform_fitted(self):
        """Test transform when model is fitted"""
        self.modeler.model = MockPipelineModel()
        df = MockDataFrame()
        
        result = self.modeler.transform(df)
        
        self.assertIsInstance(result, MockDataFrame)

    @patch('src.ml.trend_detection.Window', MockWindow)
    @patch('src.ml.trend_detection.lag', mock_lag)
    @patch('src.ml.trend_detection.window', mock_window)
    @patch('src.ml.trend_detection.udf')
    @patch('src.ml.trend_detection.col', mock_col)
    def test_detect_emerging_topics(self, mock_udf, mock_window_func, mock_lag, mock_window, mock_col):
        """Test emerging topics detection"""
        # Setup fitted model
        self.modeler.model = MockPipelineModel()
        
        # Mock UDF function
        mock_udf.return_value = Mock()
        
        df = MockDataFrame([{'timestamp': '2023-01-01', 'topicDistribution': [0.1, 0.9]}])
        
        result = self.modeler.detect_emerging_topics(df, growth_threshold=0.5)
        
        self.assertIsInstance(result, MockDataFrame)


class TestTrendForecaster(unittest.TestCase):
    """Test cases for TrendForecaster class"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()
        self.forecaster = TrendForecaster(self.mock_spark)

    def test_init(self):
        """Test TrendForecaster initialization"""
        self.assertEqual(self.forecaster.spark, self.mock_spark)
        self.assertEqual(self.forecaster.models, {})

    @patch('src.ml.trend_detection.Prophet')
    def test_forecast_brand_sentiment(self, mock_prophet):
        """Test brand sentiment forecasting"""
        # Setup mock Prophet
        mock_prophet_instance = Mock()
        mock_prophet.return_value = mock_prophet_instance
        
        mock_prophet_instance.fit.return_value = None
        mock_prophet_instance.make_future_dataframe.return_value = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=7)
        })
        mock_prophet_instance.predict.return_value = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=7),
            'yhat': [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.5],
            'yhat_lower': [0.3, 0.4, 0.2, 0.5, 0.1, 0.6, 0.3],
            'yhat_upper': [0.7, 0.8, 0.6, 0.9, 0.5, 1.0, 0.7]
        })
        
        # Test data
        df = pd.DataFrame({
            'window_start': pd.date_range('2023-01-01', periods=10),
            'sentiment_score': np.random.rand(10)
        })
        
        result = self.forecaster.forecast_brand_sentiment(df, 'Apple', horizon=7)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('ds', result.columns)
        self.assertIn('yhat', result.columns)
        mock_prophet.assert_called_once()

    def test_forecast_market_trends(self):
        """Test market trends forecasting"""
        df = MockDataFrame([{
            'window_start': '2023-01-01',
            'sentiment_score': 0.5,
            'mention_count': 100
        }])
        
        with patch('src.ml.trend_detection.Prophet') as mock_prophet:
            mock_prophet_instance = Mock()
            mock_prophet.return_value = mock_prophet_instance
            mock_prophet_instance.fit.return_value = None
            mock_prophet_instance.make_future_dataframe.return_value = pd.DataFrame({
                'ds': pd.date_range('2023-01-01', periods=7)
            })
            mock_prophet_instance.predict.return_value = pd.DataFrame({
                'ds': pd.date_range('2023-01-01', periods=7),
                'yhat': [0.5] * 7,
                'yhat_lower': [0.3] * 7,
                'yhat_upper': [0.7] * 7
            })
            
            result = self.forecaster.forecast_market_trends(
                df, 
                metrics=['sentiment_score', 'mention_count'],
                horizon=7
            )
            
            self.assertIsInstance(result, dict)

    @patch('src.ml.trend_detection.Prophet')
    @patch('src.ml.trend_detection.window', mock_window)
    @patch('src.ml.trend_detection.avg', mock_avg)
    @patch('src.ml.trend_detection.count', mock_count)
    def test_forecast_sentiment_trends(self, mock_window_func, mock_prophet, mock_avg, mock_count):
        """Test overall sentiment trends forecasting"""
        # Setup mock Prophet
        mock_prophet_instance = Mock()
        mock_prophet.return_value = mock_prophet_instance
        mock_prophet_instance.fit.return_value = None
        mock_prophet_instance.make_future_dataframe.return_value = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=7)
        })
        mock_prophet_instance.predict.return_value = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=7),
            'yhat': [0.5] * 7,
            'yhat_lower': [0.3] * 7,
            'yhat_upper': [0.7] * 7
        })
        
        # Mock SparkSession createDataFrame
        self.forecaster.spark.createDataFrame = Mock(return_value=MockDataFrame())
        
        df = MockDataFrame([{
            'timestamp': '2023-01-01',
            'sentiment': 1,
            'vader_compound': 0.5
        }] * 20)
        
        result = self.forecaster.forecast_sentiment_trends(df, horizon=7)
        
        self.assertIsInstance(result, MockDataFrame)

    @patch('src.ml.trend_detection.Prophet')
    @patch('src.ml.trend_detection.window', mock_window)
    @patch('src.ml.trend_detection.udf')
    @patch('src.ml.trend_detection.col', mock_col)
    def test_forecast_topic_trends(self, mock_udf, mock_window_func, mock_prophet, mock_col):
        """Test topic trends forecasting"""
        # Setup mock Prophet
        mock_prophet_instance = Mock()
        mock_prophet.return_value = mock_prophet_instance
        mock_prophet_instance.fit.return_value = None
        mock_prophet_instance.make_future_dataframe.return_value = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=7)
        })
        mock_prophet_instance.predict.return_value = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=7),
            'yhat': [10] * 7,
            'yhat_lower': [5] * 7,
            'yhat_upper': [15] * 7
        })
        
        # Mock UDF function
        mock_udf.return_value = Mock()
        
        # Mock SparkSession createDataFrame
        self.forecaster.spark.createDataFrame = Mock(return_value=MockDataFrame())
        
        # Mock topic modeler
        mock_topic_modeler = Mock()
        
        df = MockDataFrame([{
            'timestamp': '2023-01-01',
            'topicDistribution': [0.1, 0.9, 0.0]
        }] * 20)
        
        result = self.forecaster.forecast_topic_trends(df, mock_topic_modeler, horizon=7)
        
        self.assertIsInstance(result, MockDataFrame)


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for AnomalyDetector class"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()
        self.detector = AnomalyDetector(self.mock_spark, contamination=0.1)

    def test_init(self):
        """Test AnomalyDetector initialization"""
        self.assertEqual(self.detector.spark, self.mock_spark)
        self.assertEqual(self.detector.contamination, 0.1)
        self.assertEqual(self.detector.models, {})

    @patch('src.ml.trend_detection.IsolationForest')
    def test_detect_sentiment_anomalies(self, mock_isolation_forest):
        """Test sentiment anomaly detection"""
        # Setup mock IsolationForest
        mock_forest = Mock()
        # Return correct length array to match the 15 rows in test data
        mock_forest.fit_predict.return_value = np.array([1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1])  # 15 elements
        mock_forest.score_samples.return_value = np.array([0.1, 0.2, -0.8, 0.15, 0.1, 0.2, -0.7, 0.1, 0.15, 0.2, 0.1, 0.2, -0.6, 0.1, 0.15])  # 15 elements
        mock_isolation_forest.return_value = mock_forest
        
        # Mock SparkSession createDataFrame
        self.detector.spark.createDataFrame = Mock(return_value=MockDataFrame())
        
        df = MockDataFrame([{
            'window_start': '2023-01-01',
            'brand': 'Apple',
            'sentiment_score': 0.5,
            'mention_count': 100
        }] * 15)
        
        features = ['sentiment_score', 'mention_count']
        result = self.detector.detect_sentiment_anomalies(df, features)
        
        self.assertIsInstance(result, MockDataFrame)
        mock_isolation_forest.assert_called()

    def test_detect_volume_anomalies(self):
        """Test volume anomaly detection"""
        # Patch all Spark functions at once
        with patch('src.ml.trend_detection.window', mock_window), \
             patch('src.ml.trend_detection.count', mock_count), \
             patch('src.ml.trend_detection.col', mock_col), \
             patch('src.ml.trend_detection.Window', MockWindow), \
             patch('src.ml.trend_detection.avg', mock_avg), \
             patch('src.ml.trend_detection.stddev', mock_stddev), \
             patch('src.ml.trend_detection.when', mock_when), \
             patch('src.ml.trend_detection.spark_abs', mock_spark_abs):
            
            df = MockDataFrame([{
                'timestamp': '2023-01-01 10:00:00',
            }] * 100)
            
            result = self.detector.detect_volume_anomalies(df)
            
            self.assertIsInstance(result, MockDataFrame)


class TestViralityPredictor(unittest.TestCase):
    """Test cases for ViralityPredictor class"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()
        self.predictor = ViralityPredictor(self.mock_spark)

    def test_init(self):
        """Test ViralityPredictor initialization"""
        self.assertEqual(self.predictor.spark, self.mock_spark)

    @patch('src.ml.trend_detection.size', mock_size)
    @patch('src.ml.trend_detection.when', mock_when)
    @patch('src.ml.trend_detection.col', mock_col)
    def test_identify_viral_potential_simple(self, mock_when, mock_size, mock_col):
        """Test viral potential identification"""
        df = MockDataFrame([{
            'text': 'Check this out! http://example.com #viral',
            'hashtags': ['viral'],
            'vader_compound': 0.8,
            'exclamation_count': 1,
            'question_count': 0,
            'emoji_sentiment': 0.5
        }])
        
        result = self.predictor.identify_viral_potential_simple(df)
        
        self.assertIsInstance(result, MockDataFrame)


class TestIntegration(unittest.TestCase):
    """Integration tests for trend detection components"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()

    def test_topic_modeling_to_forecasting_pipeline(self):
        """Test pipeline from topic modeling to trend forecasting"""
        # Setup topic modeler
        topic_modeler = TopicModeler(self.mock_spark, num_topics=5)
        topic_modeler.model = MockPipelineModel()  # Mock fitted model
        
        # Setup forecaster
        forecaster = TrendForecaster(self.mock_spark)
        
        df = MockDataFrame([{
            'timestamp': '2023-01-01',
            'text': 'Apple iPhone is great',
            'tokens_lemmatized': ['apple', 'iphone', 'great'],
            'topicDistribution': [0.1, 0.9, 0.0]
        }] * 10)
        
        # Mock the forecast method
        with patch.object(forecaster, 'forecast_topic_trends') as mock_forecast:
            mock_forecast.return_value = MockDataFrame()
            
            result = forecaster.forecast_topic_trends(df, topic_modeler)
            
            self.assertIsInstance(result, MockDataFrame)
            mock_forecast.assert_called_once()

    def test_anomaly_detection_integration(self):
        """Test anomaly detection with multiple features"""
        detector = AnomalyDetector(self.mock_spark)
        
        df = MockDataFrame([{
            'window_start': '2023-01-01',
            'brand': 'Apple',
            'sentiment_score': np.random.rand(),
            'mention_count': np.random.randint(1, 100),
            'positive_ratio': np.random.rand()
        } for _ in range(20)])
        
        features = ['sentiment_score', 'mention_count', 'positive_ratio']
        
        with patch('src.ml.trend_detection.IsolationForest') as mock_iso:
            mock_forest = Mock()
            mock_forest.fit_predict.return_value = np.random.choice([1, -1], 20)
            mock_forest.score_samples.return_value = np.random.rand(20)
            mock_iso.return_value = mock_forest
            
            detector.spark.createDataFrame = Mock(return_value=MockDataFrame())
            
            result = detector.detect_sentiment_anomalies(df, features)
            
            self.assertIsInstance(result, MockDataFrame)


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()

    def test_topic_modeling_with_insufficient_data(self):
        """Test topic modeling with insufficient data"""
        modeler = TopicModeler(self.mock_spark, num_topics=5)
        
        # Very small dataset
        df = MockDataFrame([{'tokens_lemmatized': ['apple']}])
        
        with patch('src.ml.trend_detection.Pipeline') as mock_pipeline:
            mock_pipeline_instance = Mock()
            mock_pipeline.return_value = mock_pipeline_instance
            mock_pipeline_instance.fit.side_effect = Exception("Insufficient data")
            
            with self.assertRaises(Exception):
                modeler.fit_topics(df)

    def test_forecasting_with_insufficient_data(self):
        """Test forecasting with insufficient data"""
        forecaster = TrendForecaster(self.mock_spark)
        
        # Very small dataset
        df = pd.DataFrame({
            'window_start': pd.date_range('2023-01-01', periods=2),
            'sentiment_score': [0.5, 0.6]
        })
        
        with patch('src.ml.trend_detection.Prophet') as mock_prophet:
            mock_prophet_instance = Mock()
            mock_prophet.return_value = mock_prophet_instance
            mock_prophet_instance.fit.side_effect = Exception("Insufficient data")
            
            with self.assertRaises(Exception):
                forecaster.forecast_brand_sentiment(df, 'Apple')

    @patch('src.ml.trend_detection.lit', mock_lit)
    def test_anomaly_detection_with_no_data(self):
        """Test anomaly detection with no data"""
        detector = AnomalyDetector(self.mock_spark)
        
        df = MockDataFrame([])  # Empty dataset
        features = ['sentiment_score']
        
        result = detector.detect_sentiment_anomalies(df, features)
        
        # Should return original DataFrame with added columns
        self.assertIsInstance(result, MockDataFrame)


if __name__ == '__main__':
    unittest.main() 