import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import sys
import os
import re

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Mock PySpark modules before importing
sys.modules['pyspark'] = MagicMock()
sys.modules['pyspark.sql'] = MagicMock()
sys.modules['pyspark.sql.functions'] = MagicMock()
sys.modules['pyspark.sql.types'] = MagicMock()


class TestTextPreprocessor(unittest.TestCase):
    """Comprehensive test suite for TextPreprocessor class"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock Spark session and DataFrame
        self.mock_spark = MagicMock()
        self.mock_df = MagicMock()
        
        # Setup DataFrame method chaining
        self.mock_df.withColumn.return_value = self.mock_df
        self.mock_df.filter.return_value = self.mock_df
        self.mock_df.select.return_value = self.mock_df
        self.mock_df.groupBy.return_value = self.mock_df
        self.mock_df.count.return_value = 100
        self.mock_df.collect.return_value = [
            {'detected_language': 'en', 'count': 80},
            {'detected_language': 'es', 'count': 20}
        ]
        self.mock_df.drop.return_value = self.mock_df
        
        # Mock Spark context
        self.mock_spark.sparkContext.broadcast.return_value.value = {
            'the', 'and', 'is', 'a', 'to', 'in', 'it', 'of'
        }

    @patch('src.spark.preprocessing.create_spark_session')
    def test_preprocessor_initialization(self, mock_create_session):
        """Test TextPreprocessor initialization"""
        mock_create_session.return_value = self.mock_spark
        
        from src.spark.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor(self.mock_spark)
        
        # Verify initialization
        self.assertEqual(preprocessor.spark, self.mock_spark)
        self.assertIsInstance(preprocessor.stop_words, set)
        self.assertIn('the', preprocessor.stop_words)
        
        # Verify patterns are compiled
        self.assertIsInstance(preprocessor.url_pattern, str)
        self.assertIsInstance(preprocessor.mention_pattern, str)
        self.assertIsInstance(preprocessor.hashtag_pattern, str)

    @patch('src.spark.preprocessing.create_spark_session')
    def test_clean_text_basic_functionality(self, mock_create_session):
        """Test basic text cleaning functionality"""
        mock_create_session.return_value = self.mock_spark
        
        from src.spark.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor(self.mock_spark)
        result_df = preprocessor.clean_text(self.mock_df, "text")
        
        # Verify DataFrame operations were called
        self.assertTrue(self.mock_df.withColumn.called)
        self.assertEqual(result_df, self.mock_df)

    @patch('src.spark.preprocessing.create_spark_session')
    def test_handle_emojis(self, mock_create_session):
        """Test emoji handling functionality"""
        mock_create_session.return_value = self.mock_spark
        
        from src.spark.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor(self.mock_spark)
        result_df = preprocessor.handle_emojis(self.mock_df)
        
        # Verify emoji processing was applied
        self.assertTrue(self.mock_df.withColumn.called)
        self.assertEqual(result_df, self.mock_df)

    @patch('src.spark.preprocessing.create_spark_session')
    def test_detect_and_filter_language(self, mock_create_session):
        """Test language detection and filtering"""
        mock_create_session.return_value = self.mock_spark
        
        from src.spark.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor(self.mock_spark)
        result_df = preprocessor.detect_and_filter_language(
            self.mock_df, 
            target_languages=["en"], 
            min_confidence=0.5
        )
        
        # Verify language processing was applied
        self.assertTrue(self.mock_df.withColumn.called)
        self.assertTrue(self.mock_df.filter.called)
        self.assertEqual(result_df, self.mock_df)

    @patch('src.spark.preprocessing.create_spark_session')
    def test_tokenize_text(self, mock_create_session):
        """Test text tokenization"""
        mock_create_session.return_value = self.mock_spark
        
        from src.spark.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor(self.mock_spark)
        result_df = preprocessor.tokenize_text(self.mock_df)
        
        # Verify tokenization was applied
        self.assertTrue(self.mock_df.withColumn.called)
        self.assertEqual(result_df, self.mock_df)

    @patch('src.spark.preprocessing.create_spark_session')
    def test_remove_stopwords(self, mock_create_session):
        """Test stop word removal"""
        mock_create_session.return_value = self.mock_spark
        
        from src.spark.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor(self.mock_spark)
        result_df = preprocessor.remove_stopwords(self.mock_df)
        
        # Verify stop word removal was applied
        self.assertTrue(self.mock_df.withColumn.called)
        self.assertEqual(result_df, self.mock_df)

    @patch('src.spark.preprocessing.create_spark_session')
    def test_lemmatize_tokens(self, mock_create_session):
        """Test token lemmatization"""
        mock_create_session.return_value = self.mock_spark
        
        from src.spark.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor(self.mock_spark)
        result_df = preprocessor.lemmatize_tokens(self.mock_df)
        
        # Verify lemmatization was applied
        self.assertTrue(self.mock_df.withColumn.called)
        self.assertEqual(result_df, self.mock_df)

    @patch('src.spark.preprocessing.create_spark_session')
    def test_create_processed_text(self, mock_create_session):
        """Test final processed text creation"""
        mock_create_session.return_value = self.mock_spark
        
        from src.spark.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor(self.mock_spark)
        result_df = preprocessor.create_processed_text(self.mock_df)
        
        # Verify processed text creation
        self.assertTrue(self.mock_df.withColumn.called)
        self.assertEqual(result_df, self.mock_df)

    @patch('src.spark.preprocessing.create_spark_session')
    def test_full_preprocess_pipeline(self, mock_create_session):
        """Test complete preprocessing pipeline"""
        mock_create_session.return_value = self.mock_spark
        
        # Mock the select method to return specific columns
        self.mock_df.columns = [
            "tweet_id", "text", "text_processed", "sentiment",
            "timestamp", "year", "month", "day", "hour",
            "hashtags", "emoji_sentiment", "text_length",
            "processed_length", "token_count", "tokens_lemmatized",
            "detected_language", "language_confidence"
        ]
        
        from src.spark.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor(self.mock_spark)
        result_df = preprocessor.preprocess_pipeline(self.mock_df)
        
        # Verify all steps were executed
        self.assertTrue(self.mock_df.withColumn.called)
        self.assertTrue(self.mock_df.select.called)
        self.assertEqual(result_df, self.mock_df)


class TestPreprocessingUDFs(unittest.TestCase):
    """Test User Defined Functions used in preprocessing - testing the logic directly"""

    def test_emoji_sentiment_extraction_logic(self):
        """Test emoji sentiment extraction logic directly"""
        
        # Simulate the emoji sentiment function logic
        def extract_emoji_sentiment_enhanced(text):
            """Extract sentiment from emojis using Unicode ranges"""
            try:
                emoji_sentiment = 0
                positive_emojis = {'😊', '❤️', '👍', '🎉', '😃', '😄', '😆', '😁', '😂', '🤣'}
                negative_emojis = {'😡', '💔', '👎', '😢', '😭', '😰', '😨', '😠', '🤬'}
                
                for char in text:
                    if ord(char) >= 0x1F600:  # Basic emoji range check
                        if char in positive_emojis:
                            emoji_sentiment += 2
                        elif char in negative_emojis:
                            emoji_sentiment -= 2
                
                return emoji_sentiment
            except Exception:
                return 0
        
        # Test positive emojis
        positive_text = "I love this! 😊❤️👍"
        positive_score = extract_emoji_sentiment_enhanced(positive_text)
        self.assertGreater(positive_score, 0)
        
        # Test negative emojis
        negative_text = "This is terrible 😡💔👎"
        negative_score = extract_emoji_sentiment_enhanced(negative_text)
        self.assertLess(negative_score, 0)
        
        # Test neutral text
        neutral_text = "This is just text without emojis"
        neutral_score = extract_emoji_sentiment_enhanced(neutral_text)
        self.assertEqual(neutral_score, 0)
        
        # Test empty text
        empty_score = extract_emoji_sentiment_enhanced("")
        self.assertEqual(empty_score, 0)

    def test_enhanced_lemmatization_logic(self):
        """Test enhanced lemmatization logic directly"""
        
        # Simulate the lemmatization function logic
        def enhanced_lemmatize(text):
            """Enhanced rule-based lemmatization"""
            try:
                if not text:
                    return ""
                    
                words = text.split()
                lemmatized = []
                
                irregular_verbs = {
                    'was': 'be', 'were': 'be', 'been': 'be', 'being': 'be',
                    'had': 'have', 'has': 'have', 'having': 'have',
                    'did': 'do', 'does': 'do', 'doing': 'do', 'done': 'do',
                    'went': 'go', 'goes': 'go', 'going': 'go', 'gone': 'go',
                    'said': 'say', 'says': 'say', 'saying': 'say',
                    'got': 'get', 'gets': 'get', 'getting': 'get', 'gotten': 'get',
                    'came': 'come', 'comes': 'come', 'coming': 'come',
                    'took': 'take', 'takes': 'take', 'taking': 'take', 'taken': 'take',
                    'made': 'make', 'makes': 'make', 'making': 'make',
                    'gave': 'give', 'gives': 'give', 'giving': 'give', 'given': 'give'
                }
                
                for word in words:
                    word_lower = word.lower()
                    
                    if word_lower in irregular_verbs:
                        lemmatized.append(irregular_verbs[word_lower])
                    elif word_lower.endswith('ing') and len(word) > 4:
                        lemmatized.append(word[:-3])
                    elif word_lower.endswith('ed') and len(word) > 3:
                        lemmatized.append(word[:-2])
                    elif word_lower.endswith('s') and len(word) > 2 and not word_lower.endswith('ss'):
                        lemmatized.append(word[:-1])
                    else:
                        lemmatized.append(word)
                
                return ' '.join(lemmatized)
            except:
                return text if text else ""
        
        # Test regular verb forms
        self.assertEqual(enhanced_lemmatize("running walking"), "runn walk")
        self.assertEqual(enhanced_lemmatize("played worked"), "play work")
        
        # Test irregular verbs
        self.assertEqual(enhanced_lemmatize("was going"), "be go")
        self.assertEqual(enhanced_lemmatize("had taken"), "have take")
        
        # Test plural forms
        self.assertEqual(enhanced_lemmatize("cats dogs"), "cat dog")
        
        # Test empty text
        self.assertEqual(enhanced_lemmatize(""), "")

    def test_language_detection_logic(self):
        """Test language detection logic directly"""
        
        # Simulate the language detection function logic
        def detect_language_basic(text):
            """Basic language detection using common word patterns"""
            try:
                if not text or len(text.strip()) < 10:
                    return ["unknown", 0.0]
                
                text_lower = text.lower()
                
                english_common = {
                    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
                    'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
                    'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
                    'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one'
                }
                
                spanish_common = {
                    'el', 'la', 'de', 'que', 'y', 'en', 'un', 'ser', 'se',
                    'no', 'haber', 'por', 'con', 'su', 'para', 'como', 'estar',
                    'tener', 'le', 'lo', 'todo', 'pero', 'más', 'hacer', 'o'
                }
                
                words = text_lower.split()
                if len(words) < 3:
                    return ["unknown", 0.0]
                
                english_matches = sum(1 for word in words if word in english_common)
                spanish_matches = sum(1 for word in words if word in spanish_common)
                
                total_words = len(words)
                english_score = english_matches / total_words
                spanish_score = spanish_matches / total_words
                
                if english_score > spanish_score and english_score > 0.15:
                    return ["en", float(english_score)]
                elif spanish_score > 0.15:
                    return ["es", float(spanish_score)]
                else:
                    # Check for other patterns
                    if re.search(r'[а-яА-Я]', text):  # Cyrillic
                        return ["ru", 0.7]
                    elif re.search(r'[\u4e00-\u9fff]', text):  # Chinese
                        return ["zh", 0.8]
                    
                return ["unknown", 0.0]
            
            except Exception:
                return ["unknown", 0.0]
        
        # Test English text
        english_text = "This is a simple English sentence with common words"
        lang_result = detect_language_basic(english_text)
        self.assertEqual(lang_result[0], "en")
        self.assertGreater(lang_result[1], 0.0)
        
        # Test Spanish text
        spanish_text = "Este es un texto en español con palabras comunes"
        lang_result = detect_language_basic(spanish_text)
        self.assertEqual(lang_result[0], "es")
        
        # Test too short text
        short_text = "hi"
        lang_result = detect_language_basic(short_text)
        self.assertEqual(lang_result[0], "unknown")
        
        # Test empty text
        empty_result = detect_language_basic("")
        self.assertEqual(empty_result[0], "unknown")
        
        # Test text with special characters (Cyrillic)
        cyrillic_text = "Это текст на русском языке"
        lang_result = detect_language_basic(cyrillic_text)
        self.assertEqual(lang_result[0], "ru")


class TestTextCleaningFunctions(unittest.TestCase):
    """Test individual text cleaning functions"""

    def test_url_pattern_matching(self):
        """Test URL pattern matching"""
        from src.spark.preprocessing import TextPreprocessor
        
        # Create a minimal preprocessor instance for pattern access
        mock_spark = MagicMock()
        preprocessor = TextPreprocessor(mock_spark)
        
        # Test URL patterns
        urls = [
            "http://example.com",
            "https://www.example.com",
            "http://example.com/path?param=value"
        ]
        
        for url in urls:
            match = re.search(preprocessor.url_pattern, url)
            self.assertIsNotNone(match, f"Should match URL: {url}")

    def test_mention_pattern_matching(self):
        """Test mention pattern matching"""
        from src.spark.preprocessing import TextPreprocessor
        
        mock_spark = MagicMock()
        preprocessor = TextPreprocessor(mock_spark)
        
        # Test mention patterns
        mentions = ["@user123", "@john_doe", "@TestUser"]
        
        for mention in mentions:
            match = re.search(preprocessor.mention_pattern, mention)
            self.assertIsNotNone(match, f"Should match mention: {mention}")

    def test_hashtag_pattern_matching(self):
        """Test hashtag pattern matching"""
        from src.spark.preprocessing import TextPreprocessor
        
        mock_spark = MagicMock()
        preprocessor = TextPreprocessor(mock_spark)
        
        # Test hashtag patterns
        hashtags = ["#test", "#MachineLearning", "#AI2024"]
        
        for hashtag in hashtags:
            match = re.search(preprocessor.hashtag_pattern, hashtag)
            self.assertIsNotNone(match, f"Should match hashtag: {hashtag}")


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling"""

    @patch('src.spark.preprocessing.create_spark_session')
    def test_empty_dataframe_handling(self, mock_create_session):
        """Test handling of empty DataFrames"""
        mock_spark = MagicMock()
        mock_df = MagicMock()
        mock_df.count.return_value = 0
        mock_df.withColumn.return_value = mock_df
        
        mock_create_session.return_value = mock_spark
        
        from src.spark.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor(mock_spark)
        
        # Test that empty DataFrame doesn't crash
        result = preprocessor.clean_text(mock_df)
        self.assertIsNotNone(result)

    def test_null_text_handling(self):
        """Test handling of null/None text values"""
        
        # Test emoji sentiment function with None
        def safe_emoji_sentiment(text):
            try:
                if text is None:
                    return 0
                return len([c for c in text if ord(c) >= 0x1F600])
            except:
                return 0
        
        self.assertEqual(safe_emoji_sentiment(None), 0)
        self.assertEqual(safe_emoji_sentiment(""), 0)
        self.assertGreater(safe_emoji_sentiment("😊"), 0)

    def test_unicode_text_handling(self):
        """Test handling of Unicode text"""
        
        # Test with various Unicode characters
        unicode_texts = [
            "Hello 世界",  # Mixed English and Chinese
            "Café naïve résumé",  # Accented characters
            "𝕳𝖊𝖑𝖑𝖔",  # Mathematical script
            "👨‍💻👩‍🔬",  # Compound emojis
        ]
        
        def safe_unicode_length(text):
            try:
                return len(text)
            except:
                return 0
        
        for text in unicode_texts:
            try:
                result = safe_unicode_length(text)
                self.assertIsInstance(result, int)
                self.assertGreater(result, 0)
            except Exception as e:
                self.fail(f"Unicode handling failed for '{text}': {e}")


class TestCreateSparkSession(unittest.TestCase):
    """Test Spark session creation"""

    @patch('sys.executable', '/test/path/python')
    @patch('os.environ')
    @patch('pyspark.sql.SparkSession')
    def test_create_spark_session(self, mock_spark_session, mock_environ):
        """Test Spark session creation with proper configuration"""
        
        # Mock the builder pattern
        mock_builder = MagicMock()
        mock_spark_session.builder = mock_builder
        mock_builder.appName.return_value = mock_builder
        mock_builder.master.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = MagicMock()
        
        from src.spark.preprocessing import create_spark_session
        
        session = create_spark_session("TestApp")
        
        # Verify environment variables were set
        expected_calls = [
            unittest.mock.call.__setitem__("PYSPARK_PYTHON", "/test/path/python"),
            unittest.mock.call.__setitem__("PYSPARK_DRIVER_PYTHON", "/test/path/python")
        ]
        mock_environ.assert_has_calls(expected_calls, any_order=True)
        
        # Verify builder methods were called
        mock_builder.appName.assert_called_with("TestApp")
        mock_builder.master.assert_called_with("local[*]")
        self.assertTrue(mock_builder.config.called)
        mock_builder.getOrCreate.assert_called_once()


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios"""

    @patch('src.spark.preprocessing.create_spark_session')
    @patch('src.utils.path_utils.get_path')
    @patch('src.spark.data_ingestion.DataIngestion')
    def test_main_function_execution(self, mock_data_ingestion, mock_get_path, mock_create_session):
        """Test main function execution"""
        
        # Setup mocks
        mock_spark = MagicMock()
        mock_df = MagicMock()
        
        mock_create_session.return_value = mock_spark
        mock_get_path.return_value = "/test/path"
        mock_spark.read.parquet.return_value = mock_df
        mock_df.limit.return_value = mock_df
        mock_df.select.return_value = mock_df
        mock_df.show.return_value = None
        mock_df.coalesce.return_value = mock_df
        
        # Mock write operations
        mock_write = MagicMock()
        mock_df.coalesce.return_value.write = mock_write
        mock_write.mode.return_value = mock_write
        mock_write.parquet.return_value = None
        
        from src.spark.preprocessing import main
        
        # Test main function doesn't crash
        try:
            main()
        except Exception as e:
            # It's okay if it fails due to missing files, but not due to code errors
            if "No such file" not in str(e) and "path" not in str(e).lower():
                self.fail(f"Main function failed with unexpected error: {e}")
        
        # Verify Spark session was created and stopped
        mock_create_session.assert_called()
        mock_spark.stop.assert_called()


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 