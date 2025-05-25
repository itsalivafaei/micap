"""
Text Preprocessing Module for MICAP - No external dependencies
Handles all text cleaning and normalization tasks
Optimized for parallel processing on M4 Mac
Compatible replacement for emoji/spacy dependencies
"""

import re
import string
import sys, os
from typing import List, Dict, Optional
import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, lower, regexp_replace, trim, split, array_distinct, array_join,
    udf, when, length, expr, size, regexp_extract_all, lit
)
from pyspark.sql.types import StringType, ArrayType, IntegerType

# No external dependencies - emoji and spacy removed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_spark_session(app_name: str) -> SparkSession:
    # Guarantee all workers use the same venv Python
    python_exec = sys.executable          # …/micap/.venv2/bin/python

    os.environ["PYSPARK_PYTHON"] = python_exec
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec

    return (
        SparkSession.builder
            .appName(app_name)
            .master("local[*]")
            .config("spark.pyspark.python",  python_exec)
            .config("spark.pyspark.driver.python", python_exec)
            .config("spark.executorEnv.PYSPARK_PYTHON", python_exec)
            .getOrCreate()
    )

class TextPreprocessor:
    """
    Comprehensive text preprocessing for tweet sentiment analysis
    Handles cleaning, normalization, and feature preparation
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize preprocessor with Spark session

        Args:
            spark: Active SparkSession
        """
        self.spark = spark

        self.punct_pattern = "[" + re.escape(string.punctuation) + "]"

        # Define patterns for text cleaning
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        self.mention_pattern = r'@[\w]+'
        self.hashtag_pattern = r'#([\w]+)'
        self.number_pattern = r'\d+'

        # Common English stop words (expanded set)
        self.stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                           'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                           'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                           'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'been',
                           'be', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
                           'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
                           'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                           'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                           'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
                           'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                           'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'}

        # No spaCy initialization - using built-in methods only
        logger.info("Using built-in text processing (no external dependencies)")

        # Register UDFs
        self._register_udfs()

    def _register_udfs(self):
        """Register User Defined Functions for complex preprocessing"""

        # Enhanced emoji sentiment without external dependencies
        def extract_emoji_sentiment_enhanced(text):
            """Extract sentiment from emojis using Unicode ranges and comprehensive patterns"""
            try:
                emoji_sentiment = 0

                # Comprehensive emoji sentiment mapping
                positive_emojis = {
                    # Smiling faces
                    '😊', '😃', '😄', '😆', '😁', '😂', '🤣', '😍', '🥰', '😘', '😗', '😙', '😚',
                    '🙂', '🤗', '🤩', '🥳', '😋', '😛', '😜', '🤪', '😇', '🥲',
                    # Hearts and love
                    '❤️', '💕', '💖', '💗', '💓', '💝', '💘', '💞', '💟', '❣️', '💔', '🧡', '💛',
                    '💚', '💙', '💜', '🤍', '🖤', '🤎', '❤️‍🔥', '❤️‍🩹',
                    # Thumbs and hands
                    '👍', '👌', '✌️', '🤞', '🤟', '🤘', '🤙', '👏', '🙌', '👐', '🤲', '🙏',
                    '✊', '👊', '🤛', '🤜', '💪', '🦾', '🦿',
                    # Celebration and positive symbols
                    '🎉', '🎊', '🎈', '🎁', '🎂', '🍰', '🧁', '🥳', '🎆', '🎇', '✨', '⭐', '🌟',
                    '💫', '🔆', '☀️', '🌞', '🌝', '🌛', '🌜', '🌚', '🌈', '☁️', '⛅', '⛈️',
                    # Success and achievement
                    '🏆', '🥇', '🏅', '🎖️', '🏵️', '🎗️', '🎯', '🎪', '🎭', '🎨', '🎬', '🎤',
                    '🎧', '🎼', '🎵', '🎶', '🎺', '🎸', '🎻', '🥁', '🎹'
                }

                negative_emojis = {
                    # Sad and crying faces
                    '😢', '😭', '😿', '😾', '🙀', '😰', '😨', '😧', '😦', '😮', '😯', '😲', '😱',
                    '🤯', '😳', '🥵', '🥶', '😞', '😔', '😟', '😕', '🙁', '☹️', '😣', '😖',
                    '😫', '😩', '🥺', '😤', '😠', '😡', '🤬', '😈', '👿', '💀', '☠️',
                    # Negative symbols
                    '👎', '✋', '🛑', '⛔', '📵', '🚫', '❌', '❎', '💢', '💥', '💫', '💦',
                    '💨', '🕳️', '💣', '💔', '⚡', '🔥', '❄️', '💤', '💭', '🗯️', '💬',
                    # Sickness and pain
                    '🤢', '🤮', '🤧', '😷', '🤒', '🤕', '🩹', '🩺', '💊', '💉', '🦠', '🦷',
                    '🦴', '💀', '☠️'
                }

                neutral_positive = {
                    '😊', '🙂', '😌', '😉', '😏', '🤨', '🤓', '😎', '🥸', '🤯', '🥱', '🤤'
                }

                # Check each character for emoji sentiment
                for char in text:
                    if ord(char) >= 0x1F600:  # Basic emoji range check
                        if char in positive_emojis:
                            emoji_sentiment += 2  # Strong positive
                        elif char in neutral_positive:
                            emoji_sentiment += 1  # Mild positive
                        elif char in negative_emojis:
                            emoji_sentiment -= 2  # Strong negative

                return emoji_sentiment
            except Exception:
                return 0

        self.emoji_sentiment_udf = udf(extract_emoji_sentiment_enhanced, IntegerType())

        # Enhanced lemmatization with better rules
        def enhanced_lemmatize(text):
            """Enhanced rule-based lemmatization with more comprehensive patterns"""
            try:
                words = text.split()
                lemmatized = []

                # Common irregular verb mappings
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

                    # Check irregular verbs first
                    if word_lower in irregular_verbs:
                        lemmatized.append(irregular_verbs[word_lower])
                    # Apply suffix rules
                    elif word_lower.endswith('ies') and len(word) > 4:
                        lemmatized.append(word[:-3] + 'y')
                    elif word_lower.endswith('ied') and len(word) > 4:
                        lemmatized.append(word[:-3] + 'y')
                    elif word_lower.endswith('ing') and len(word) > 4:
                        # Handle double consonant (running -> run)
                        if word[-4] == word[-5] and word[-4] not in 'aeiou':
                            lemmatized.append(word[:-4])
                        else:
                            lemmatized.append(word[:-3])
                    elif word_lower.endswith('ed') and len(word) > 3:
                        # Handle double consonant (stopped -> stop)
                        if len(word) > 4 and word[-3] == word[-4] and word[-3] not in 'aeiou':
                            lemmatized.append(word[:-3])
                        else:
                            lemmatized.append(word[:-2])
                    elif word_lower.endswith('er') and len(word) > 3:
                        lemmatized.append(word[:-2])
                    elif word_lower.endswith('est') and len(word) > 4:
                        lemmatized.append(word[:-3])
                    elif word_lower.endswith('ly') and len(word) > 3:
                        lemmatized.append(word[:-2])
                    elif word_lower.endswith('ness') and len(word) > 5:
                        lemmatized.append(word[:-4])
                    elif word_lower.endswith('ment') and len(word) > 5:
                        lemmatized.append(word[:-4])
                    elif word_lower.endswith('tion') and len(word) > 5:
                        lemmatized.append(word[:-4])
                    elif word_lower.endswith('sion') and len(word) > 5:
                        lemmatized.append(word[:-4])
                    elif word_lower.endswith('s') and len(word) > 2 and not word_lower.endswith('ss') and not word_lower.endswith('us'):
                        lemmatized.append(word[:-1])
                    else:
                        lemmatized.append(word)

                return ' '.join(lemmatized)
            except:
                return text

        self.lemmatize_udf = udf(enhanced_lemmatize, StringType())

    def clean_text(self, df: DataFrame, text_col: str = "text") -> DataFrame:
        """
        Perform basic text cleaning

        Args:
            df: Input DataFrame
            text_col: Name of text column

        Returns:
            DataFrame with cleaned text
        """
        logger.info("Starting text cleaning...")

        # Convert to lowercase
        df = df.withColumn(f"{text_col}_clean", lower(col(text_col)))

        # Remove URLs
        df = df.withColumn(
            f"{text_col}_clean",
            regexp_replace(col(f"{text_col}_clean"), lit(self.url_pattern), " URL ")
        )

        # Handle @mentions (replace with generic token)
        df = df.withColumn(
            f"{text_col}_clean",
            regexp_replace(col(f"{text_col}_clean"), lit(self.mention_pattern), " USER ")
        )

        # Extract hashtags before removing
        df = df.withColumn(
            "hashtags",
            regexp_extract_all(col(text_col), lit(self.hashtag_pattern), 1)
        )

        # Remove hashtag symbols but keep the text
        df = df.withColumn(
            f"{text_col}_clean",
            regexp_replace(col(f"{text_col}_clean"), r'#', ' ')
        )

        # Remove extra whitespace
        df = df.withColumn(
            f"{text_col}_clean",
            trim(regexp_replace(col(f"{text_col}_clean"), r'\s+', ' '))
        )

        logger.info("Text cleaning completed")
        return df

    def handle_emojis(self, df: DataFrame, text_col: str = "text_clean") -> DataFrame:
        """
        Handle emojis - extract sentiment and clean text

        Args:
            df: Input DataFrame
            text_col: Name of text column

        Returns:
            DataFrame with emoji features
        """
        logger.info("Processing emojis...")

        # Extract emoji sentiment
        df = df.withColumn("emoji_sentiment", self.emoji_sentiment_udf(col(text_col)))

        # Simple emoji removal/replacement without external library
        def clean_emojis(text):
            """Remove or replace emojis with simple text markers"""
            try:
                # Define common emoji to text mappings
                emoji_replacements = {
                    '😊': ' happy ', '😃': ' happy ', '😄': ' happy ', '😆': ' laugh ',
                    '😂': ' laugh ', '🤣': ' laugh ', '😍': ' love ', '🥰': ' love ',
                    '❤️': ' love ', '💕': ' love ', '👍': ' good ', '👌': ' good ',
                    '🎉': ' celebrate ', '✨': ' sparkle ', '🌟': ' star ',
                    '😢': ' sad ', '😭': ' cry ', '😡': ' angry ', '😠': ' angry ',
                    '💔': ' broken heart ', '👎': ' bad ', '😰': ' worried ', '😨': ' scared '
                }

                cleaned_text = text
                for emoji, replacement in emoji_replacements.items():
                    cleaned_text = cleaned_text.replace(emoji, replacement)

                # Remove remaining emoji characters (basic Unicode range)
                import re
                # Remove most common emoji ranges
                emoji_pattern = re.compile(
                    "["
                    "\U0001F600-\U0001F64F"  # emoticons
                    "\U0001F300-\U0001F5FF"  # symbols & pictographs
                    "\U0001F680-\U0001F6FF"  # transport & map symbols
                    "\U0001F700-\U0001F77F"  # alchemical symbols
                    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                    "\U0001FA00-\U0001FA6F"  # Chess Symbols
                    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                    "\U00002702-\U000027B0"  # Dingbats
                    "\U000024C2-\U0001F251"  # Enclosed characters
                    "]+", flags=re.UNICODE)

                cleaned_text = emoji_pattern.sub(' ', cleaned_text)

                # Clean up extra spaces
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

                return cleaned_text
            except:
                return text

        clean_emojis_udf = udf(clean_emojis, StringType())
        df = df.withColumn(f"{text_col}_demojized", clean_emojis_udf(col(text_col)))

        logger.info("Emoji processing completed")
        return df

    def tokenize_text(self, df: DataFrame, text_col: str = "text_clean") -> DataFrame:
        """
        Tokenize text into words

        Args:
            df: Input DataFrame
            text_col: Name of text column

        Returns:
            DataFrame with tokenized text
        """
        logger.info("Tokenizing text...")

        # Remove punctuation and tokenize
        df = df.withColumn(
            "tokens",
            split(
                regexp_replace(col(f"{text_col}_demojized"), lit(self.punct_pattern), " "),
                r'\s+'
            )
        )

        # Remove empty tokens
        df = df.withColumn(
            "tokens",
            array_distinct(
                expr("filter(tokens, x -> length(x) > 0)")
            )
        )

        logger.info("Tokenization completed")
        return df

    def remove_stopwords(self, df: DataFrame, tokens_col: str = "tokens") -> DataFrame:
        """
        Remove stop words from tokenized text

        Args:
            df: Input DataFrame
            tokens_col: Name of tokens column

        Returns:
            DataFrame with stop words removed
        """
        logger.info("Removing stop words...")

        # Broadcast stop words for efficiency
        stop_words_broadcast = self.spark.sparkContext.broadcast(self.stop_words)

        def filter_stop_words(tokens):
            """Filter out stop words"""
            if tokens is None:
                return []
            return [word for word in tokens if word.lower() not in stop_words_broadcast.value]

        filter_stop_words_udf = udf(filter_stop_words, ArrayType(StringType()))

        df = df.withColumn(
            "tokens_filtered",
            filter_stop_words_udf(col(tokens_col))
        )

        logger.info("Stop word removal completed")
        return df

    def lemmatize_tokens(self, df: DataFrame, tokens_col: str = "tokens_filtered") -> DataFrame:
        """
        Lemmatize tokens to their base form

        Args:
            df: Input DataFrame
            tokens_col: Name of tokens column

        Returns:
            DataFrame with lemmatized tokens
        """
        logger.info("Lemmatizing tokens...")

        # Join tokens back to text for lemmatization
        df = df.withColumn(
            "text_for_lemma",
            array_join(col(tokens_col), " ")
        )

        # Apply lemmatization
        df = df.withColumn(
            "text_lemmatized",
            self.lemmatize_udf(col("text_for_lemma"))
        )

        # Split back to tokens
        df = df.withColumn(
            "tokens_lemmatized",
            split(col("text_lemmatized"), " ")
        )

        logger.info("Lemmatization completed")
        return df

    def create_processed_text(self, df: DataFrame) -> DataFrame:
        """
        Create final processed text from tokens

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with processed text
        """
        logger.info("Creating final processed text...")

        # Join lemmatized tokens
        df = df.withColumn(
            "text_processed",
            array_join(col("tokens_lemmatized"), " ")
        )

        # Add text length features
        df = df.withColumn("text_length", length(col("text")))
        df = df.withColumn("processed_length", length(col("text_processed")))
        df = df.withColumn("token_count", size(col("tokens_lemmatized")))

        logger.info("Processed text creation completed")
        return df

    def preprocess_pipeline(self, df: DataFrame) -> DataFrame:
        """
        Run complete preprocessing pipeline

        Args:
            df: Input DataFrame with 'text' column

        Returns:
            DataFrame with all preprocessing steps applied
        """
        logger.info("Running complete preprocessing pipeline...")

        # Execute pipeline steps
        df = self.clean_text(df)
        df = self.handle_emojis(df)
        df = self.tokenize_text(df)
        df = self.remove_stopwords(df)
        df = self.lemmatize_tokens(df)
        df = self.create_processed_text(df)

        # Select relevant columns
        columns_to_keep = [
            "tweet_id", "text", "text_processed", "sentiment",
            "timestamp", "year", "month", "day", "hour",
            "hashtags", "emoji_sentiment", "text_length",
            "processed_length", "token_count", "tokens_lemmatized"
        ]

        df_final = df.select([col for col in columns_to_keep if col in df.columns])

        logger.info("Preprocessing pipeline completed")
        return df_final


def main():
    """
    Demonstrate preprocessing functionality
    """
    from config.spark_config import create_spark_session
    from src.spark.data_ingestion import DataIngestion

    # Create Spark session
    spark = create_spark_session("Preprocessing")

    # Load sample data
    logger.info("Loading sample data...")
    sample_path = "/Users/ali/Documents/Projects/micap/data/processed/sentiment140_sample"
    df = spark.read.parquet(sample_path)

    # Initialize preprocessor
    preprocessor = TextPreprocessor(spark)

    # Run preprocessing
    df_processed = preprocessor.preprocess_pipeline(df.limit(1000))

    # Show results
    logger.info("Sample preprocessed records:")
    df_processed.select(
        "text", "text_processed", "emoji_sentiment", "token_count"
    ).show(5, truncate=False)

    # Save processed sample
    output_path = "/Users/ali/Documents/Projects/micap/data/processed/sentiment140_preprocessed_sample"
    df_processed.coalesce(1).write.mode("overwrite").parquet(output_path)
    logger.info(f"Processed data saved to: {output_path}")

    # Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()