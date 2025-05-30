# Deep Learning Models Test Suite

This document describes the comprehensive unit test suite for the `deep_learning_models.py` module.

## Overview

The test suite provides complete coverage for all deep learning models and their components:
- LSTM Model
- CNN Model 
- Transformer Model
- Base DeepLearningModel class
- Main evaluation function
- Error handling scenarios

## Test Structure

### 1. TestDeepLearningModelBase
Tests the base `DeepLearningModel` class functionality:
- **Initialization**: Validates proper setup of Spark session, tokenizer, and parameters
- **Data Streaming**: Tests `spark_to_pandas_stream` method with various batch sizes
- **Text Preparation**: Tests `prepare_text_data` method including tokenization and padding
- **Edge Cases**: Empty data handling, memory optimization

### 2. TestLSTMModel
Tests LSTM-specific functionality:
- **Model Architecture**: Validates model building with correct layers and configuration
- **Training**: Tests training process with callbacks and history tracking
- **Parameter Validation**: Ensures proper parameter passing to TensorFlow layers

### 3. TestCNNModel
Tests CNN-specific functionality:
- **Model Architecture**: Validates Conv1D and pooling layers
- **Training**: Tests training with appropriate batch sizes
- **Compilation**: Ensures model is compiled with correct optimizer and metrics

### 4. TestTransformerModel
Tests Transformer-specific functionality:
- **Multi-Head Attention**: Validates attention mechanism setup
- **Layer Normalization**: Tests add & norm operations
- **Feed Forward**: Validates dense layer configuration

### 5. TestEvaluateDeepLearningModels
Tests the main evaluation function:
- **Model Comparison**: Tests evaluation of all three models
- **Data Loading**: Validates data loading from parquet files
- **Results Structure**: Ensures proper result dictionary format
- **Integration**: Tests full pipeline execution

### 6. TestErrorHandling
Tests error scenarios and edge cases:
- **Empty Data**: Handling of empty text datasets
- **Invalid Sessions**: Response to None or invalid Spark sessions
- **Model State**: Training without proper model initialization
- **Memory Management**: Large dataset handling

### 7. TestIntegrationScenarios
Tests integration with other components:
- **Main Function**: Tests if __name__ == "__main__" block
- **Spark Integration**: Tests with mocked Spark sessions
- **Cross-Component**: Integration with other modules

### 8. TestModelConfigurationValidation
Tests configuration validation:
- **Parameter Validation**: Tests max_words, max_length parameters
- **Architecture Parameters**: Tests embedding dimensions, attention heads
- **Default Values**: Validates default parameter behavior

## Mocking Strategy

The test suite uses comprehensive mocking to isolate the code under test:

### External Dependencies Mocked:
- **TensorFlow**: All Keras components (models, layers, callbacks)
- **PySpark**: SparkSession, DataFrame, functions
- **scikit-learn**: train_test_split and other utilities
- **NumPy/Pandas**: Where needed for controlled data

### Mock Patterns:
```python
# Mock TensorFlow components
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()

# Mock model building
@patch('src.ml.deep_learning_models.models')
@patch('src.ml.deep_learning_models.layers')
def test_model_building(self, mock_layers, mock_models):
    # Test implementation
```

## Test Data Patterns

### Sample Data Creation:
```python
self.sample_data = [
    MagicMock(text="This is a positive tweet", sentiment=1),
    MagicMock(text="This is a negative tweet", sentiment=0),
    MagicMock(text="Another positive example", sentiment=1)
]
```

### DataFrame Mocking:
```python
self.mock_df.toLocalIterator.return_value = iter(self.sample_data)
self.mock_df.sample.return_value = self.mock_df
```

## Running the Tests

### Individual Test Classes:
```bash
# Run specific test class
python -m pytest tests/unit/test_deep_learning_models.py::TestLSTMModel -v

# Run with coverage
python -m pytest tests/unit/test_deep_learning_models.py --cov=src.ml.deep_learning_models
```

### Full Test Suite:
```bash
# Run all deep learning tests
python tests/unit/test_deep_learning_models.py

# Run with unittest
python -m unittest tests.unit.test_deep_learning_models -v
```

### Integration with Project Tests:
```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run with coverage report
python -m pytest tests/unit/ --cov=src --cov-report=html
```

## Test Coverage

The test suite provides comprehensive coverage across multiple dimensions:

### Functional Coverage:
- ✅ Model initialization and configuration
- ✅ Data preprocessing and streaming
- ✅ Model architecture building
- ✅ Training process and callbacks
- ✅ Evaluation and metrics
- ✅ Error handling and edge cases

### Code Coverage:
- ✅ All public methods tested
- ✅ Private methods tested through public interfaces
- ✅ Exception paths covered
- ✅ Configuration variations tested

### Integration Coverage:
- ✅ Spark DataFrame integration
- ✅ TensorFlow model integration
- ✅ Cross-component data flow
- ✅ Main function execution

## Expected Test Output

When running the tests, you should see output similar to:
```
test_deep_learning_model_initialization ... ok
test_spark_to_pandas_stream_basic ... ok
test_lstm_build_model ... ok
test_cnn_build_model ... ok
test_transformer_build_model ... ok
test_evaluate_deep_learning_models ... ok
test_empty_text_handling ... ok
test_model_configuration ... ok

----------------------------------------------------------------------
Ran 24 tests in 0.045s

OK
```

## Maintenance Guidelines

### Adding New Tests:
1. Follow the existing naming convention: `test_<functionality>`
2. Use appropriate mocking for external dependencies
3. Include both positive and negative test cases
4. Document test purpose in docstrings

### Updating Existing Tests:
1. Maintain backward compatibility where possible
2. Update mocks when external APIs change
3. Ensure test isolation is preserved
4. Update documentation accordingly

### Best Practices:
- **Isolation**: Each test should be independent
- **Clarity**: Test names should clearly indicate what is being tested
- **Coverage**: Aim for both line and branch coverage
- **Performance**: Tests should run quickly (< 1 second each)
- **Reliability**: Tests should be deterministic and repeatable

## Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure all external dependencies are properly mocked
2. **Mock Failures**: Verify mock setup matches actual function signatures
3. **Assertion Errors**: Check that mocked return values match expected formats
4. **Environment Issues**: Ensure Python path includes project root

### Debug Tips:
```python
# Add debug prints in tests
print(f"Mock called with: {mock_function.call_args}")

# Check mock call count
self.assertEqual(mock_function.call_count, 1)

# Verify specific arguments
mock_function.assert_called_with(expected_arg)
```

## Future Enhancements

Potential areas for test expansion:
- **Performance Tests**: Memory usage and execution time validation
- **Integration Tests**: Full end-to-end model training
- **Property-Based Tests**: Using hypothesis for edge case generation
- **Stress Tests**: Large dataset handling validation 