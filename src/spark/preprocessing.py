"""
Text Preprocessing Module for MICAP - Updated with spaCy
Handles all text cleaning and normalization tasks
Optimized for parallel processing on M4 Mac
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

import emoji
import spacy

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

        # Common English stop words
        self.stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                           'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                           'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                           'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'been',
                           'be', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
                           'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
                           'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                           'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under'}

        # Initialize spaCy model (load once for efficiency)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Using basic lemmatization.")
            self.nlp = None

        # Register UDFs
        self._register_udfs()

    def _register_udfs(self):
        """Register User Defined Functions for complex preprocessing"""

        # UDF for emoji handling
        def extract_emoji_sentiment(text):
            """Extract sentiment from emojis"""
            try:
                emoji_list = []
                emoji_sentiment = 0

                # Common positive emojis
                positive_emojis = ['😊', '😃', '😄', '❤️', '👍', '🎉', '✨', '🌟', '💕']
                # Common negative emojis
                negative_emojis = ['😢', '😭', '😡', '😠', '👎', '💔', '😰', '😨']

                for char in text:
                    if emoji.is_emoji(char):
                        emoji_list.append(char)
                        if char in positive_emojis:
                            emoji_sentiment += 1
                        elif char in negative_emojis:
                            emoji_sentiment -= 1

                return emoji_sentiment
            except:
                return 0

        self.emoji_sentiment_udf = udf(extract_emoji_sentiment, IntegerType())

        # UDF for lemmatization with spaCy or fallback
        def lemmatize_text(text):
            """Lemmatize text using spaCy or basic fallback"""
            try:
                if self.nlp is not None:
                    # Use spaCy for lemmatization
                    doc = self.nlp(text)
                    return ' '.join([token.lemma_ for token in doc if not token.is_space])
                else:
                    # Basic fallback - just return the text
                    # You could implement simple stemming here if needed
                    return text
            except:
                return text

        self.lemmatize_udf = udf(lemmatize_text, StringType())

        # Simple lemmatization UDF without external dependencies
        def simple_lemmatize(text):
            """Simple rule-based lemmatization for common cases"""
            try:
                words = text.split()
                lemmatized = []

                for word in words:
                    # Simple rules for common suffixes
                    if word.endswith('ing') and len(word) > 4:
                        lemmatized.append(word[:-3])
                    elif word.endswith('ed') and len(word) > 3:
                        lemmatized.append(word[:-2])
                    elif word.endswith('s') and len(word) > 2 and not word.endswith('ss'):
                        lemmatized.append(word[:-1])
                    else:
                        lemmatized.append(word)

                return ' '.join(lemmatized)
            except:
                return text

        self.simple_lemmatize_udf = udf(simple_lemmatize, StringType())

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
        Handle emojis - extract sentiment and convert to text

        Args:
            df: Input DataFrame
            text_col: Name of text column

        Returns:
            DataFrame with emoji features
        """
        logger.info("Processing emojis...")

        # Extract emoji sentiment
        df = df.withColumn("emoji_sentiment", self.emoji_sentiment_udf(col(text_col)))

        # Convert emojis to text description
        def demojize_text(text):
            """Convert emojis to text descriptions"""
            try:
                return emoji.demojize(text, delimiters=(" ", " "))
            except:
                return text

        demojize_udf = udf(demojize_text, StringType())
        df = df.withColumn(f"{text_col}_demojized", demojize_udf(col(text_col)))

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

        # Apply lemmatization (use simple lemmatizer if spaCy fails)
        if self.nlp is not None:
            df = df.withColumn(
                "text_lemmatized",
                self.lemmatize_udf(col("text_for_lemma"))
            )
        else:
            df = df.withColumn(
                "text_lemmatized",
                self.simple_lemmatize_udf(col("text_for_lemma"))
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