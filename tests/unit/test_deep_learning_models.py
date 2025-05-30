import unittest
from unittest.mock import patch, MagicMock, PropertyMock, call
import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Mock external dependencies before importing
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['tensorflow.keras.layers'] = MagicMock()
sys.modules['tensorflow.keras.models'] = MagicMock()
sys.modules['tensorflow.keras.callbacks'] = MagicMock()
sys.modules['tensorflow.keras.preprocessing'] = MagicMock()
sys.modules['tensorflow.keras.preprocessing.text'] = MagicMock()
sys.modules['tensorflow.keras.preprocessing.sequence'] = MagicMock()
sys.modules['tensorflow.keras.optimizers'] = MagicMock()
sys.modules['tensorflow.keras.metrics'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.model_selection'] = MagicMock()
sys.modules['pyspark'] = MagicMock()
sys.modules['pyspark.sql'] = MagicMock()
sys.modules['pyspark.sql.functions'] = MagicMock()


class TestDeepLearningModelBase(unittest.TestCase):
    """Test suite for DeepLearningModel base class"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock Spark session and DataFrame
        self.mock_spark = MagicMock()
        self.mock_df = MagicMock()
        
        # Mock DataFrame methods
        self.mock_df.sample.return_value = self.mock_df
        self.mock_df.select.return_value = self.mock_df
        self.mock_df.repartition.return_value = self.mock_df
        self.mock_df.columns = ["text", "sentiment"]
        
        # Mock toLocalIterator for streaming
        self.sample_data = [
            MagicMock(text="This is a positive tweet", sentiment=1),
            MagicMock(text="This is a negative tweet", sentiment=0),
            MagicMock(text="Another positive example", sentiment=1)
        ]
        self.mock_df.toLocalIterator.return_value = iter(self.sample_data)

    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_deep_learning_model_initialization(self, mock_tokenizer):
        """Test DeepLearningModel initialization"""
        from src.ml.deep_learning_models import DeepLearningModel
        
        # Create instance
        model = DeepLearningModel(
            spark=self.mock_spark,
            max_words=5000,
            max_length=50
        )
        
        # Verify initialization
        self.assertEqual(model.spark, self.mock_spark)
        self.assertEqual(model.max_words, 5000)
        self.assertEqual(model.max_length, 50)
        self.assertIsNone(model.model)
        self.assertIsNone(model.history)
        mock_tokenizer.assert_called_once_with(num_words=5000)

    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_spark_to_pandas_stream_basic(self, mock_tokenizer):
        """Test basic spark_to_pandas_stream functionality"""
        from src.ml.deep_learning_models import DeepLearningModel
        
        model = DeepLearningModel(self.mock_spark)
        
        # Test with small batch size
        result_df = model.spark_to_pandas_stream(self.mock_df, batch_size=2)
        
        # Verify result is pandas DataFrame
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), len(self.sample_data))

    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_spark_to_pandas_stream_empty_data(self, mock_tokenizer):
        """Test spark_to_pandas_stream with empty data"""
        from src.ml.deep_learning_models import DeepLearningModel
        
        model = DeepLearningModel(self.mock_spark)
        
        # Mock empty iterator
        self.mock_df.toLocalIterator.return_value = iter([])
        
        result_df = model.spark_to_pandas_stream(self.mock_df)
        
        # Verify empty DataFrame with correct columns
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 0)
        self.assertListEqual(list(result_df.columns), ["text", "sentiment"])

    @patch('src.ml.deep_learning_models.pad_sequences')
    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_prepare_text_data(self, mock_tokenizer, mock_pad_sequences):
        """Test text data preparation"""
        from src.ml.deep_learning_models import DeepLearningModel
        
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.texts_to_sequences.return_value = [[1, 2], [2, 3], [1, 3]]
        mock_tokenizer_instance.word_index = {"the": 1, "is": 2, "good": 3}
        
        mock_pad_sequences.return_value = np.array([[1, 2, 0], [2, 3, 0], [1, 3, 0]])
        
        model = DeepLearningModel(self.mock_spark, max_length=3)
        
        X, y = model.prepare_text_data(self.mock_df)
        
        # Verify shapes and calls
        self.assertEqual(X.shape, (3, 3))
        self.assertEqual(len(y), 3)
        mock_tokenizer_instance.fit_on_texts.assert_called_once()
        mock_tokenizer_instance.texts_to_sequences.assert_called_once()
        mock_pad_sequences.assert_called_once()


class TestLSTMModel(unittest.TestCase):
    """Test suite for LSTM model"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_spark = MagicMock()
        
    @patch('src.ml.deep_learning_models.models')
    @patch('src.ml.deep_learning_models.layers')
    @patch('src.ml.deep_learning_models.tf')
    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_lstm_build_model(self, mock_tokenizer, mock_tf, mock_layers, mock_models):
        """Test LSTM model building"""
        from src.ml.deep_learning_models import LSTMModel
        
        # Setup mocks
        mock_model = MagicMock()
        mock_models.Sequential.return_value = mock_model
        
        lstm = LSTMModel(self.mock_spark, max_words=1000, max_length=50)
        result_model = lstm.build_model(embedding_dim=64)
        
        # Verify model creation
        mock_models.Sequential.assert_called_once()
        mock_model.compile.assert_called_once()
        self.assertEqual(lstm.model, mock_model)
        self.assertEqual(result_model, mock_model)

    @patch('src.ml.deep_learning_models.callbacks')
    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_lstm_train(self, mock_tokenizer, mock_callbacks):
        """Test LSTM model training"""
        from src.ml.deep_learning_models import LSTMModel
        
        # Setup mock model
        mock_model = MagicMock()
        mock_history = MagicMock()
        mock_history.history = {"loss": [0.5, 0.3], "accuracy": [0.7, 0.8]}
        mock_model.fit.return_value = mock_history
        
        lstm = LSTMModel(self.mock_spark)
        lstm.model = mock_model
        
        # Test data
        X = np.random.random((100, 50))
        y = np.random.randint(0, 2, 100)
        
        # Train model
        history = lstm.train(X, y, epochs=2, batch_size=32)
        
        # Verify training
        mock_model.fit.assert_called_once()
        self.assertEqual(history, mock_history.history)
        self.assertEqual(lstm.history, mock_history)


class TestCNNModel(unittest.TestCase):
    """Test suite for CNN model"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_spark = MagicMock()

    @patch('src.ml.deep_learning_models.models')
    @patch('src.ml.deep_learning_models.layers')
    @patch('src.ml.deep_learning_models.tf')
    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_cnn_build_model(self, mock_tokenizer, mock_tf, mock_layers, mock_models):
        """Test CNN model building"""
        from src.ml.deep_learning_models import CNNModel
        
        # Setup mocks
        mock_model = MagicMock()
        mock_models.Sequential.return_value = mock_model
        
        cnn = CNNModel(self.mock_spark, max_words=1000, max_length=50)
        result_model = cnn.build_model(embedding_dim=64)
        
        # Verify model creation
        mock_models.Sequential.assert_called_once()
        mock_model.compile.assert_called_once()
        self.assertEqual(cnn.model, mock_model)
        self.assertEqual(result_model, mock_model)

    @patch('src.ml.deep_learning_models.callbacks')
    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_cnn_train(self, mock_tokenizer, mock_callbacks):
        """Test CNN model training"""
        from src.ml.deep_learning_models import CNNModel
        
        # Setup mock model
        mock_model = MagicMock()
        mock_history = MagicMock()
        mock_history.history = {"loss": [0.6, 0.4], "accuracy": [0.6, 0.75]}
        mock_model.fit.return_value = mock_history
        
        cnn = CNNModel(self.mock_spark)
        cnn.model = mock_model
        
        # Test data
        X = np.random.random((100, 50))
        y = np.random.randint(0, 2, 100)
        
        # Train model
        history = cnn.train(X, y, epochs=2, batch_size=16)
        
        # Verify training
        mock_model.fit.assert_called_once()
        self.assertEqual(history, mock_history.history)


class TestTransformerModel(unittest.TestCase):
    """Test suite for Transformer model"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_spark = MagicMock()

    @patch('src.ml.deep_learning_models.models')
    @patch('src.ml.deep_learning_models.layers')
    @patch('src.ml.deep_learning_models.tf')
    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_transformer_build_model(self, mock_tokenizer, mock_tf, mock_layers, mock_models):
        """Test Transformer model building"""
        from src.ml.deep_learning_models import TransformerModel
        
        # Setup mocks
        mock_model = MagicMock()
        mock_models.Model.return_value = mock_model
        
        # Mock layers
        mock_input = MagicMock()
        mock_layers.Input.return_value = mock_input
        
        transformer = TransformerModel(self.mock_spark, max_words=1000, max_length=50)
        result_model = transformer.build_model(embedding_dim=64, num_heads=4)
        
        # Verify model creation
        mock_layers.Input.assert_called_once()
        mock_model.compile.assert_called_once()
        self.assertEqual(transformer.model, mock_model)

    @patch('src.ml.deep_learning_models.callbacks')
    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_transformer_train(self, mock_tokenizer, mock_callbacks):
        """Test Transformer model training"""
        from src.ml.deep_learning_models import TransformerModel
        
        # Setup mock model
        mock_model = MagicMock()
        mock_history = MagicMock()
        mock_history.history = {"loss": [0.7, 0.5], "accuracy": [0.65, 0.8]}
        mock_model.fit.return_value = mock_history
        
        transformer = TransformerModel(self.mock_spark)
        transformer.model = mock_model
        
        # Test data
        X = np.random.random((100, 50))
        y = np.random.randint(0, 2, 100)
        
        # Train model
        history = transformer.train(X, y, epochs=2, batch_size=16)
        
        # Verify training
        mock_model.fit.assert_called_once()
        self.assertEqual(history, mock_history.history)


class TestEvaluateDeepLearningModels(unittest.TestCase):
    """Test suite for the main evaluation function"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_spark = MagicMock()
        self.mock_df = MagicMock()
        
        # Mock DataFrame operations
        self.mock_df.count.return_value = 1000
        self.mock_df.sample.return_value = self.mock_df

    @patch('src.ml.deep_learning_models.train_test_split')
    @patch('src.ml.deep_learning_models.TransformerModel')
    @patch('src.ml.deep_learning_models.CNNModel')
    @patch('src.ml.deep_learning_models.LSTMModel')
    def test_evaluate_deep_learning_models(self, mock_lstm_class, mock_cnn_class, 
                                         mock_transformer_class, mock_train_test_split):
        """Test the main evaluation function"""
        from src.ml.deep_learning_models import evaluate_deep_learning_models
        
        # Mock model instances
        mock_lstm = MagicMock()
        mock_cnn = MagicMock()
        mock_transformer = MagicMock()
        
        mock_lstm_class.return_value = mock_lstm
        mock_cnn_class.return_value = mock_cnn
        mock_transformer_class.return_value = mock_transformer
        
        # Mock data preparation
        X = np.random.random((100, 50))
        y = np.random.randint(0, 2, 100)
        mock_lstm.prepare_text_data.return_value = (X, y)
        
        # Mock train/test split
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        mock_train_test_split.return_value = (X_train, X_test, y_train, y_test)
        
        # Mock model training and evaluation
        mock_lstm.train.return_value = {"loss": [0.5], "accuracy": [0.8]}
        mock_cnn.train.return_value = {"loss": [0.4], "accuracy": [0.85]}
        mock_transformer.train.return_value = {"loss": [0.3], "accuracy": [0.9]}
        
        mock_lstm.model.evaluate.return_value = [0.45, 0.82, 0.88]
        mock_cnn.model.evaluate.return_value = [0.38, 0.87, 0.91]
        mock_transformer.model.evaluate.return_value = [0.28, 0.92, 0.95]
        
        # Mock Spark read operation
        with patch('src.ml.deep_learning_models.spark.read.parquet') as mock_read:
            mock_read.return_value = self.mock_df
            
            # Run evaluation
            results = evaluate_deep_learning_models(self.mock_spark, sample_size=0.1)
        
        # Verify results structure
        self.assertIn('LSTM', results)
        self.assertIn('CNN', results)
        self.assertIn('Transformer', results)
        
        # Verify LSTM results
        self.assertIn('loss', results['LSTM'])
        self.assertIn('accuracy', results['LSTM'])
        self.assertIn('auc', results['LSTM'])
        
        # Verify model building and training calls
        mock_lstm.build_model.assert_called_once()
        mock_cnn.build_model.assert_called_once()
        mock_transformer.build_model.assert_called_once()
        
        mock_lstm.train.assert_called_once()
        mock_cnn.train.assert_called_once()
        mock_transformer.train.assert_called_once()


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_spark = MagicMock()

    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_empty_text_handling(self, mock_tokenizer):
        """Test handling of empty text data"""
        from src.ml.deep_learning_models import DeepLearningModel
        
        # Mock empty DataFrame
        mock_df = MagicMock()
        mock_df.sample.return_value = mock_df
        mock_df.select.return_value = mock_df
        mock_df.repartition.return_value = mock_df
        mock_df.columns = ["text", "sentiment"]
        mock_df.toLocalIterator.return_value = iter([])
        
        model = DeepLearningModel(self.mock_spark)
        
        with patch.object(model, 'spark_to_pandas_stream') as mock_stream:
            mock_stream.return_value = pd.DataFrame(columns=["text", "sentiment"])
            
            # Should handle empty data gracefully
            X, y = model.prepare_text_data(mock_df)
            self.assertEqual(len(X), 0)
            self.assertEqual(len(y), 0)

    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_invalid_spark_session(self, mock_tokenizer):
        """Test handling of invalid Spark session"""
        from src.ml.deep_learning_models import DeepLearningModel
        
        # Test with None spark session
        with self.assertRaises(AttributeError):
            model = DeepLearningModel(None)
            model.prepare_text_data(None)

    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_model_training_without_build(self, mock_tokenizer):
        """Test training model without building it first"""
        from src.ml.deep_learning_models import LSTMModel
        
        lstm = LSTMModel(self.mock_spark)
        X = np.random.random((10, 50))
        y = np.random.randint(0, 2, 10)
        
        # Should handle None model gracefully
        with self.assertRaises(AttributeError):
            lstm.train(X, y)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios"""

    @patch('src.ml.deep_learning_models.create_spark_session')
    def test_main_function(self, mock_create_session):
        """Test the main function execution"""
        mock_spark = MagicMock()
        mock_create_session.return_value = mock_spark
        
        # Mock the evaluate function
        with patch('src.ml.deep_learning_models.evaluate_deep_learning_models') as mock_eval:
            mock_eval.return_value = {
                'LSTM': {'accuracy': 0.8, 'loss': 0.5, 'auc': 0.85},
                'CNN': {'accuracy': 0.82, 'loss': 0.48, 'auc': 0.87}
            }
            
            # Import and test main
            import src.ml.deep_learning_models
            
            # Mock the if __name__ == "__main__" block
            with patch.object(src.ml.deep_learning_models, '__name__', '__main__'):
                # This would normally run the main block
                # We'll just verify the components work
                results = src.ml.deep_learning_models.evaluate_deep_learning_models(
                    mock_spark, sample_size=0.01
                )
                
                self.assertIsInstance(results, dict)
                mock_spark.stop.assert_called_once()


class TestModelConfigurationValidation(unittest.TestCase):
    """Test model configuration validation"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_spark = MagicMock()

    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_lstm_parameter_validation(self, mock_tokenizer):
        """Test LSTM parameter validation"""
        from src.ml.deep_learning_models import LSTMModel
        
        # Test valid parameters
        lstm = LSTMModel(self.mock_spark, max_words=5000, max_length=100)
        self.assertEqual(lstm.max_words, 5000)
        self.assertEqual(lstm.max_length, 100)
        
        # Test default parameters
        lstm_default = LSTMModel(self.mock_spark)
        self.assertEqual(lstm_default.max_words, 10000)
        self.assertEqual(lstm_default.max_length, 100)

    @patch('src.ml.deep_learning_models.Tokenizer')
    def test_model_architecture_parameters(self, mock_tokenizer):
        """Test model architecture parameters"""
        from src.ml.deep_learning_models import TransformerModel
        
        transformer = TransformerModel(self.mock_spark)
        
        # Mock the layers to test parameter passing
        with patch('src.ml.deep_learning_models.layers') as mock_layers:
            with patch('src.ml.deep_learning_models.models') as mock_models:
                transformer.build_model(embedding_dim=256, num_heads=8)
                
                # Verify embedding layer was called with correct parameters
                mock_layers.Embedding.assert_called()
                mock_layers.MultiHeadAttention.assert_called()


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDeepLearningModelBase,
        TestLSTMModel,
        TestCNNModel,
        TestTransformerModel,
        TestEvaluateDeepLearningModels,
        TestErrorHandling,
        TestIntegrationScenarios,
        TestModelConfigurationValidation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1) 