# Preprocessing Module Test Suite

This document describes the comprehensive unit test suite for the `src/spark/preprocessing.py` module.

## Overview

The test suite provides complete coverage of the text preprocessing functionality used in the MICAP sentiment analysis pipeline. It includes tests for all major components:

- Text cleaning and normalization
- Emoji sentiment extraction
- Language detection and filtering
- Tokenization and stop word removal
- Lemmatization
- Complete preprocessing pipeline
- Edge cases and error handling

## Test Structure

### Test Classes

#### 1. `TestTextPreprocessor`
Tests the main `TextPreprocessor` class and its core methods:

- **`test_preprocessor_initialization`**: Verifies proper initialization with Spark session
- **`test_clean_text_basic_functionality`**: Tests basic text cleaning (URLs, mentions, hashtags)
- **`test_handle_emojis`**: Tests emoji processing and sentiment extraction
- **`test_detect_and_filter_language`**: Tests language detection and filtering
- **`test_tokenize_text`**: Tests text tokenization
- **`test_remove_stopwords`**: Tests stop word removal
- **`test_lemmatize_tokens`**: Tests token lemmatization
- **`test_create_processed_text`**: Tests final text processing
- **`test_full_preprocess_pipeline`**: Tests complete preprocessing pipeline

#### 2. `TestPreprocessingUDFs`
Tests User Defined Functions (UDFs) used in preprocessing:

- **`test_emoji_sentiment_extraction`**: Tests emoji sentiment scoring
  - Positive emojis (😊❤️👍) → positive scores
  - Negative emojis (😡💔👎) → negative scores
  - Neutral text → zero score
  - Empty text → zero score

- **`test_enhanced_lemmatization`**: Tests rule-based lemmatization
  - Regular verbs: "running walking" → "run walk"
  - Irregular verbs: "was going" → "be go"
  - Plurals: "cats dogs" → "cat dog"
  - Comparatives: "better faster" → "good fast"

- **`test_language_detection`**: Tests language detection
  - English text detection
  - Spanish text detection
  - Short text handling
  - Special character handling (Cyrillic, Chinese, etc.)

#### 3. `TestTextCleaningFunctions`
Tests pattern matching for text cleaning:

- **`test_url_pattern_matching`**: Tests URL pattern recognition
- **`test_mention_pattern_matching`**: Tests @mention pattern recognition
- **`test_hashtag_pattern_matching`**: Tests #hashtag pattern recognition

#### 4. `TestEdgeCasesAndErrorHandling`
Tests edge cases and error conditions:

- **`test_empty_dataframe_handling`**: Tests behavior with empty DataFrames
- **`test_null_text_handling`**: Tests handling of null/None text values
- **`test_unicode_text_handling`**: Tests Unicode character processing
  - Mixed scripts (English + Chinese)
  - Accented characters
  - Mathematical symbols
  - Compound emojis

#### 5. `TestCreateSparkSession`
Tests Spark session creation:

- **`test_create_spark_session`**: Tests proper Spark configuration
  - Environment variable setup
  - Python executable configuration
  - Builder pattern usage

#### 6. `TestIntegrationScenarios`
Tests integration scenarios:

- **`test_main_function_execution`**: Tests main function execution flow

## Running Tests

### Run All Tests
```bash
# From project root
cd tests
python run_preprocessing_tests.py
```

### Run Specific Test Class
```bash
# From tests directory
python run_preprocessing_tests.py TestTextPreprocessor
python run_preprocessing_tests.py TestPreprocessingUDFs
python run_preprocessing_tests.py TestEdgeCasesAndErrorHandling
```

### Run Individual Test File
```bash
# From project root
python -m pytest tests/unit/test_preprocessing.py -v
```

Or using unittest:
```bash
python -m unittest tests.unit.test_preprocessing -v
```

## Test Coverage

### Functional Coverage
- ✅ Text cleaning (URLs, mentions, hashtags)
- ✅ Emoji sentiment extraction
- ✅ Language detection and filtering
- ✅ Tokenization
- ✅ Stop word removal
- ✅ Lemmatization
- ✅ Complete preprocessing pipeline
- ✅ Spark session management

### Edge Case Coverage
- ✅ Empty/null input handling
- ✅ Unicode character support
- ✅ Empty DataFrames
- ✅ Error conditions
- ✅ Invalid input types

### Pattern Coverage
- ✅ URL patterns (http/https)
- ✅ Mention patterns (@username)
- ✅ Hashtag patterns (#hashtag)
- ✅ Emoji patterns (sentiment mapping)

## Mock Strategy

The tests use comprehensive mocking to avoid dependencies:

### PySpark Mocking
- All PySpark modules are mocked at import time
- DataFrame operations are mocked with method chaining
- Spark session creation is mocked

### UDF Testing
- UDFs are tested by accessing their underlying functions directly
- This allows testing the actual logic without Spark overhead

### External Dependencies
- No external dependencies (spaCy, NLTK) are required
- All functionality is tested using built-in preprocessing logic

## Test Data

### Sample Inputs
The tests use realistic sample data:

```python
# Emoji testing
positive_text = "I love this! 😊❤️👍"
negative_text = "This is terrible 😡💔👎"

# Language detection
english_text = "This is a simple English sentence with common words"
spanish_text = "Este es un texto en español con palabras comunes"

# Unicode testing
unicode_texts = [
    "Hello 世界",  # Mixed English and Chinese
    "Café naïve résumé",  # Accented characters
    "👨‍💻👩‍🔬",  # Compound emojis
]
```

## Expected Outputs

### Emoji Sentiment Scores
- Positive emojis: score > 0
- Negative emojis: score < 0
- No emojis: score = 0

### Lemmatization Examples
- "running walking" → "run walk"
- "was going" → "be go"
- "cats dogs" → "cat dog"

### Language Detection
- English text → ("en", confidence > 0.0)
- Spanish text → ("es", confidence > 0.0)
- Unknown/short text → ("unknown", 0.0)

## Performance Considerations

### Test Speed
- Tests use mocks to avoid Spark startup overhead
- UDF functions are tested directly for faster execution
- No actual file I/O operations

### Memory Usage
- Minimal memory footprint due to mocking
- No large datasets loaded during testing

## Debugging Tips

### Running Individual Tests
```bash
# Run a specific test method
python -m unittest tests.unit.test_preprocessing.TestTextPreprocessor.test_clean_text_basic_functionality -v
```

### Viewing Test Output
```bash
# Run with maximum verbosity
python -m unittest tests.unit.test_preprocessing -v

# Or use the custom runner for formatted output
python tests/run_preprocessing_tests.py
```

### Common Issues
1. **Import Errors**: Ensure project root is in Python path
2. **Mock Failures**: Check that PySpark modules are properly mocked
3. **UDF Errors**: Verify UDF functions are accessible via `.func` attribute

## Extending the Tests

### Adding New Test Cases
1. Create new test methods in appropriate test class
2. Follow naming convention: `test_<functionality>`
3. Use descriptive docstrings
4. Include both positive and negative test cases

### Adding New Test Classes
1. Inherit from `unittest.TestCase`
2. Add appropriate `setUp` method for fixtures
3. Group related functionality together
4. Update the test runner script

### Mock Updates
When adding tests for new functionality:
1. Ensure all external dependencies are mocked
2. Mock at the module level if needed
3. Use realistic return values for mocks
4. Chain DataFrame operations properly

## Maintenance

### Regular Updates
- Update tests when preprocessing logic changes
- Add tests for new features
- Maintain mock compatibility with PySpark updates
- Review and update test data periodically

### Quality Checks
- Ensure all tests pass before committing
- Maintain >95% test coverage
- Keep test execution time under 30 seconds
- Validate that tests catch real bugs 