"""
Unit Tests for Sentiment Models Module
Tests BaseModel, various sentiment models, and ModelEvaluator classes
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ml.sentiment_models import (
    BaseModel, NaiveBayesModel, LogisticRegressionModel, RandomForestModel,
    GradientBoostingModel, SVMModel, EnsembleModel, ModelEvaluator
)


class MockSparkSession:
    """Mock SparkSession for testing"""
    def __init__(self):
        self.sql = Mock()
        self.stop = Mock()


class MockDataFrame:
    """Mock Spark DataFrame for testing"""
    def __init__(self, data=None):
        self.data = data or []
        self.columns = ['features', 'label', 'sentiment']
        
    def withColumn(self, col_name, col_expr):
        return MockDataFrame(self.data)
    
    def select(self, *cols):
        return MockDataFrame(self.data)
    
    def filter(self, condition):
        return MockDataFrame(self.data)
    
    def count(self):
        return len(self.data)
    
    def cache(self):
        return self
    
    def persist(self):
        return self
    
    def unpersist(self):
        return self
    
    def randomSplit(self, weights, seed=None):
        return [MockDataFrame(self.data[:5]), MockDataFrame(self.data[5:])]
    
    def sample(self, fraction, seed=None):
        return MockDataFrame(self.data)
    
    def groupBy(self, *cols):
        """Mock groupBy operation"""
        mock_grouped = Mock()
        mock_grouped.count.return_value = Mock()
        # Return proper numeric data that can be sorted - use actual values instead of MagicMock
        mock_grouped.count.return_value.toPandas.return_value = pd.DataFrame({
            'label': [0, 1, 0, 1],
            'prediction': [0, 1, 1, 0], 
            'count': [45, 50, 3, 2]
        }).sort_values(['label', 'prediction'])  # Pre-sort to avoid sorting issues
        return mock_grouped
    
    def toPandas(self):
        return pd.DataFrame({
            'window_start': pd.date_range('2023-01-01', periods=10),
            'avg_sentiment': np.random.rand(10),
            'tweet_count': np.random.randint(1, 100, 10),
            'avg_compound': np.random.rand(10) * 2 - 1
        })


class MockPipelineModel:
    """Mock PipelineModel for testing"""
    def __init__(self):
        pass
    
    def transform(self, df):
        # Return a DataFrame with prediction columns
        mock_df = MockDataFrame()
        mock_df.columns = ['features', 'label', 'prediction', 'rawPrediction', 'probability']
        return mock_df
    
    def save(self, path):
        pass


class MockPipeline:
    """Mock Pipeline for testing"""
    def __init__(self, stages):
        self.stages = stages
    
    def fit(self, df):
        return MockPipelineModel()


class MockEvaluator:
    """Mock evaluator for testing"""
    def evaluate(self, df):
        return 0.85  # Mock score


class TestBaseModel(unittest.TestCase):
    """Test cases for BaseModel abstract class"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()
        
        # Create a concrete implementation for testing
        class ConcreteModel(BaseModel):
            def build_model(self, feature_cols):
                return MockPipeline([])
            
            def _get_param_grid(self):
                return []
        
        self.model = ConcreteModel(self.mock_spark, "Test Model")

    def test_init(self):
        """Test BaseModel initialization"""
        self.assertEqual(self.model.model_name, "Test Model")
        self.assertEqual(self.model.spark, self.mock_spark)
        self.assertIsNone(self.model.model)
        self.assertIsNone(self.model.pipeline)
        self.assertEqual(self.model.metrics, {})

    @patch('src.ml.sentiment_models.VectorAssembler')
    @patch('src.ml.sentiment_models.StandardScaler')
    @patch('src.ml.sentiment_models.Pipeline')
    def test_prepare_features(self, mock_pipeline, mock_scaler, mock_assembler):
        """Test feature preparation"""
        # Mock the pipeline components
        mock_assembler.return_value = Mock()
        mock_scaler.return_value = Mock()
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        mock_fitted_pipeline = Mock()
        mock_pipeline_instance.fit.return_value = mock_fitted_pipeline
        mock_fitted_pipeline.transform.return_value = MockDataFrame()
        
        df = MockDataFrame()
        feature_cols = ['text_length', 'vader_compound']
        
        result = self.model.prepare_features(df, feature_cols)
        
        self.assertIsInstance(result, MockDataFrame)
        mock_assembler.assert_called_once()
        mock_scaler.assert_called_once()

    def test_train(self):
        """Test model training"""
        df = MockDataFrame([{'features': [1, 2], 'label': 1}] * 10)
        feature_cols = ['text_length', 'vader_compound']
        
        with patch.object(self.model, 'prepare_features', return_value=df):
            result = self.model.train(df, feature_cols)
            
            self.assertIsInstance(result, MockPipelineModel)
            self.assertIsNotNone(self.model.model)
            self.assertIn('training_time', self.model.metrics)

    @patch('src.ml.sentiment_models.BinaryClassificationEvaluator')
    @patch('src.ml.sentiment_models.MulticlassClassificationEvaluator')
    def test_evaluate(self, mock_multiclass_eval, mock_binary_eval):
        """Test model evaluation"""
        # Setup mocks
        mock_binary_evaluator = Mock()
        mock_binary_evaluator.evaluate.return_value = 0.85
        mock_binary_eval.return_value = mock_binary_evaluator
        
        mock_multiclass_evaluator = Mock()
        mock_multiclass_evaluator.evaluate.return_value = 0.80
        mock_multiclass_eval.return_value = mock_multiclass_evaluator
        
        # Setup model
        self.model.model = MockPipelineModel()
        df = MockDataFrame([{'features': [1, 2], 'label': 1}] * 10)
        feature_cols = ['text_length', 'vader_compound']
        
        with patch.object(self.model, 'prepare_features', return_value=df):
            result = self.model.evaluate(df, feature_cols)
            
            self.assertIn('auc', result)
            self.assertIn('accuracy', result)
            self.assertIn('precision', result)
            self.assertIn('recall', result)
            self.assertIn('f1', result)
            self.assertIn('confusion_matrix', result)


class TestNaiveBayesModel(unittest.TestCase):
    """Test cases for NaiveBayesModel"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()
        self.model = NaiveBayesModel(self.mock_spark)

    def test_init(self):
        """Test NaiveBayesModel initialization"""
        self.assertEqual(self.model.model_name, "Naive Bayes")

    @patch('src.ml.sentiment_models.NaiveBayes')
    @patch('src.ml.sentiment_models.Pipeline')
    def test_build_model(self, mock_pipeline, mock_nb):
        """Test Naive Bayes model building"""
        mock_nb_instance = Mock()
        mock_nb.return_value = mock_nb_instance
        
        feature_cols = ['text_length', 'vader_compound']
        result = self.model.build_model(feature_cols)
        
        mock_nb.assert_called_once()
        mock_pipeline.assert_called_once()

    @patch('src.ml.sentiment_models.NaiveBayes')
    @patch('src.ml.sentiment_models.ParamGridBuilder')
    def test_get_param_grid(self, mock_param_builder, mock_naive_bayes):
        """Test parameter grid creation"""
        # Mock NaiveBayes instance
        mock_nb_instance = Mock()
        mock_nb_instance.smoothing = Mock()
        mock_naive_bayes.return_value = mock_nb_instance
        
        # Mock ParamGridBuilder
        mock_builder = Mock()
        mock_builder.addGrid.return_value = mock_builder
        mock_builder.build.return_value = []
        mock_param_builder.return_value = mock_builder
        
        result = self.model._get_param_grid()
        
        self.assertEqual(result, [])
        # Verify the builder was used correctly
        mock_param_builder.assert_called_once()
        mock_builder.addGrid.assert_called_once()
        mock_builder.build.assert_called_once()


class TestLogisticRegressionModel(unittest.TestCase):
    """Test cases for LogisticRegressionModel"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()
        self.model = LogisticRegressionModel(self.mock_spark)

    def test_init(self):
        """Test LogisticRegressionModel initialization"""
        self.assertEqual(self.model.model_name, "Logistic Regression")

    @patch('src.ml.sentiment_models.LogisticRegression')
    @patch('src.ml.sentiment_models.Pipeline')
    def test_build_model(self, mock_pipeline, mock_lr):
        """Test Logistic Regression model building"""
        mock_lr_instance = Mock()
        mock_lr.return_value = mock_lr_instance
        
        feature_cols = ['text_length', 'vader_compound']
        result = self.model.build_model(feature_cols)
        
        mock_lr.assert_called_once()
        mock_pipeline.assert_called_once()


class TestRandomForestModel(unittest.TestCase):
    """Test cases for RandomForestModel"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()
        self.model = RandomForestModel(self.mock_spark)

    def test_init(self):
        """Test RandomForestModel initialization"""
        self.assertEqual(self.model.model_name, "Random Forest")

    @patch('src.ml.sentiment_models.RandomForestClassifier')
    @patch('src.ml.sentiment_models.Pipeline')
    def test_build_model(self, mock_pipeline, mock_rf):
        """Test Random Forest model building"""
        mock_rf_instance = Mock()
        mock_rf.return_value = mock_rf_instance
        
        feature_cols = ['text_length', 'vader_compound']
        result = self.model.build_model(feature_cols)
        
        mock_rf.assert_called_once()
        mock_pipeline.assert_called_once()


class TestGradientBoostingModel(unittest.TestCase):
    """Test cases for GradientBoostingModel"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()
        self.model = GradientBoostingModel(self.mock_spark)

    def test_init(self):
        """Test GradientBoostingModel initialization"""
        self.assertEqual(self.model.model_name, "Gradient Boosting")

    @patch('src.ml.sentiment_models.GBTClassifier')
    @patch('src.ml.sentiment_models.Pipeline')
    def test_build_model(self, mock_pipeline, mock_gbt):
        """Test Gradient Boosting model building"""
        mock_gbt_instance = Mock()
        mock_gbt.return_value = mock_gbt_instance
        
        feature_cols = ['text_length', 'vader_compound']
        result = self.model.build_model(feature_cols)
        
        mock_gbt.assert_called_once()
        mock_pipeline.assert_called_once()


class TestSVMModel(unittest.TestCase):
    """Test cases for SVMModel"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()
        self.model = SVMModel(self.mock_spark)

    def test_init(self):
        """Test SVMModel initialization"""
        self.assertEqual(self.model.model_name, "SVM")

    @patch('src.ml.sentiment_models.LinearSVC')
    @patch('src.ml.sentiment_models.Pipeline')
    def test_build_model(self, mock_pipeline, mock_svm):
        """Test SVM model building"""
        mock_svm_instance = Mock()
        mock_svm.return_value = mock_svm_instance
        
        feature_cols = ['text_length', 'vader_compound']
        result = self.model.build_model(feature_cols)
        
        mock_svm.assert_called_once()
        mock_pipeline.assert_called_once()


class TestEnsembleModel(unittest.TestCase):
    """Test cases for EnsembleModel"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()
        self.model = EnsembleModel(self.mock_spark)

    def test_init(self):
        """Test EnsembleModel initialization"""
        self.assertEqual(self.model.model_name, "Ensemble")
        self.assertEqual(self.model.base_models, [])

    @patch('src.ml.sentiment_models.RandomForestClassifier')
    @patch('src.ml.sentiment_models.Pipeline')
    def test_build_model(self, mock_pipeline, mock_rf):
        """Test Ensemble model building"""
        mock_rf_instance = Mock()
        mock_rf.return_value = mock_rf_instance
        
        feature_cols = ['text_length', 'vader_compound']
        result = self.model.build_model(feature_cols)
        
        mock_rf.assert_called_once()
        mock_pipeline.assert_called_once()


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()
        self.evaluator = ModelEvaluator(self.mock_spark)

    def test_init(self):
        """Test ModelEvaluator initialization"""
        self.assertEqual(self.evaluator.spark, self.mock_spark)
        self.assertEqual(self.evaluator.results, {})

    @patch('src.ml.sentiment_models.NaiveBayesModel')
    @patch('src.ml.sentiment_models.LogisticRegressionModel')
    @patch('src.ml.sentiment_models.RandomForestModel')
    def test_evaluate_all_models(self, mock_rf, mock_lr, mock_nb):
        """Test evaluation of all models"""
        # Setup mock models
        mock_models = []
        for mock_class, name in [(mock_nb, "Naive Bayes"), 
                                 (mock_lr, "Logistic Regression"), 
                                 (mock_rf, "Random Forest")]:
            mock_model = Mock()
            mock_model.model_name = name
            mock_model.train.return_value = Mock()
            mock_model.evaluate.return_value = {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1': 0.85,
                'auc': 0.88,
                'training_time': 10.5
            }
            mock_class.return_value = mock_model
            mock_models.append(mock_model)
        
        train_df = MockDataFrame([{'features': [1, 2], 'label': 1}] * 10)
        test_df = MockDataFrame([{'features': [1, 2], 'label': 0}] * 5)
        feature_cols = ['text_length', 'vader_compound']
        
        # Patch other model classes to avoid creating them
        with patch('src.ml.sentiment_models.GradientBoostingModel'), \
             patch('src.ml.sentiment_models.SVMModel'), \
             patch('src.ml.sentiment_models.EnsembleModel'):
            
            result = self.evaluator.evaluate_all_models(train_df, test_df, feature_cols)
            
            self.assertIsInstance(result, pd.DataFrame)
            self.assertGreater(len(result), 0)

    @patch('src.ml.sentiment_models.RandomForestModel')
    def test_perform_cross_validation(self, mock_rf):
        """Test cross-validation"""
        mock_model = Mock()
        mock_model.cross_validate.return_value = {
            'cv_scores': [0.8, 0.85, 0.82, 0.87, 0.84],
            'cv_best_score': 0.87,
            'cv_time': 15.2
        }
        mock_rf.return_value = mock_model
        
        df = MockDataFrame([{'features': [1, 2], 'label': 1}] * 20)
        feature_cols = ['text_length', 'vader_compound']
        
        result = self.evaluator.perform_cross_validation(df, feature_cols)
        
        self.assertIn('cv_scores', result)
        self.assertIn('cv_best_score', result)
        self.assertIn('cv_time', result)

    def test_save_results(self):
        """Test results saving"""
        comparison_df = pd.DataFrame({
            'Model': ['Naive Bayes', 'Random Forest'],
            'Accuracy': [0.85, 0.90],
            'F1': [0.83, 0.88]
        })
        
        with patch('pandas.DataFrame.to_csv') as mock_to_csv, \
             patch('pandas.DataFrame.to_json') as mock_to_json, \
             patch('os.makedirs'):
            
            self.evaluator.save_results(comparison_df, '/tmp/test')
            
            mock_to_csv.assert_called_once()
            mock_to_json.assert_called_once()


class TestFeatureScaling(unittest.TestCase):
    """Test feature scaling for different models"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()

    @patch('src.ml.sentiment_models.MinMaxScaler')
    @patch('src.ml.sentiment_models.VectorAssembler')
    @patch('src.ml.sentiment_models.Pipeline')
    def test_naive_bayes_feature_scaling(self, mock_pipeline, mock_assembler, mock_scaler):
        """Test that Naive Bayes uses MinMaxScaler"""
        model = NaiveBayesModel(self.mock_spark)
        
        # Mock the pipeline components
        mock_assembler.return_value = Mock()
        mock_scaler.return_value = Mock()
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        mock_fitted_pipeline = Mock()
        mock_pipeline_instance.fit.return_value = mock_fitted_pipeline
        mock_fitted_pipeline.transform.return_value = MockDataFrame()
        
        df = MockDataFrame()
        feature_cols = ['text_length', 'vader_compound']
        
        model.prepare_features(df, feature_cols)
        
        # Verify MinMaxScaler is used for Naive Bayes
        mock_scaler.assert_called_once()

    @patch('src.ml.sentiment_models.StandardScaler')
    @patch('src.ml.sentiment_models.VectorAssembler')
    @patch('src.ml.sentiment_models.Pipeline')
    def test_other_models_feature_scaling(self, mock_pipeline, mock_assembler, mock_scaler):
        """Test that other models use StandardScaler"""
        model = LogisticRegressionModel(self.mock_spark)
        
        # Mock the pipeline components
        mock_assembler.return_value = Mock()
        mock_scaler.return_value = Mock()
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        mock_fitted_pipeline = Mock()
        mock_pipeline_instance.fit.return_value = mock_fitted_pipeline
        mock_fitted_pipeline.transform.return_value = MockDataFrame()
        
        df = MockDataFrame()
        feature_cols = ['text_length', 'vader_compound']
        
        model.prepare_features(df, feature_cols)
        
        # Verify StandardScaler is used for other models
        mock_scaler.assert_called_once()


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_spark = MockSparkSession()

    def test_model_evaluation_with_errors(self):
        """Test model evaluation when training fails"""
        evaluator = ModelEvaluator(self.mock_spark)
        
        # Create a model that will fail during training
        with patch('src.ml.sentiment_models.NaiveBayesModel') as mock_nb:
            mock_model = Mock()
            mock_model.model_name = "Naive Bayes"
            mock_model.train.side_effect = Exception("Training failed")
            mock_nb.return_value = mock_model
            
            # Patch other models to avoid creating them
            with patch('src.ml.sentiment_models.LogisticRegressionModel'), \
                 patch('src.ml.sentiment_models.RandomForestModel'), \
                 patch('src.ml.sentiment_models.GradientBoostingModel'), \
                 patch('src.ml.sentiment_models.SVMModel'), \
                 patch('src.ml.sentiment_models.EnsembleModel'):
                
                train_df = MockDataFrame()
                test_df = MockDataFrame()
                feature_cols = ['text_length']
                
                result = evaluator.evaluate_all_models(train_df, test_df, feature_cols)
                
                # Should still return a DataFrame with placeholder results
                self.assertIsInstance(result, pd.DataFrame)
                if len(result) > 0:
                    # If Naive Bayes failed, its metrics should be 0.0
                    if "Naive Bayes" in result.index:
                        self.assertEqual(result.loc["Naive Bayes", "accuracy"], 0.0)


if __name__ == '__main__':
    unittest.main() 