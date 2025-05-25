"""
Feature Engineering Module for MICAP
Creates ML-ready features from preprocessed text data
Optimized for M4 Mac local processing
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import (
    col, count, sum as spark_sum, mean, stddev,
    explode, array, collect_list, size, when,
    regexp_extract, length, udf, desc, row_number, lit, date_format,
    regexp_replace, col, date_format, hour, when, lit, sin, cos, udf
)
from pyspark.sql.types import ArrayType, DoubleType, StringType
from pyspark.ml.feature import (
    HashingTF, IDF, Word2Vec, NGram,
    CountVectorizer, StringIndexer
)
from pyspark.ml import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- module-level lazy singleton (executor local) ----------
_sentiment_analyzer = None

def _vader():
    """Return one analyzer per executor (lazy-init)."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentIntensityAnalyzer()
    return _sentiment_analyzer

def get_vader_scores(text: str):
    """Pure function used inside UDF."""
    if text is None:
        return [0.0, 0.0, 0.0, 0.0]
    scores = _vader().polarity_scores(text)
    return [
        float(scores["compound"]),
        float(scores["pos"]),
        float(scores["neg"]),
        float(scores["neu"]),
    ]

# pre-declare the UDF once so Spark can reuse it
VADER_UDF = udf(get_vader_scores, ArrayType(DoubleType()))


class FeatureEngineer:
    """
    Comprehensive feature engineering for sentiment analysis
    Creates various text and metadata features
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize feature engineer

        Args:
            spark: Active SparkSession
        """
        self.spark = spark

        # Initialize sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()

        # Feature configuration
        self.tfidf_features = 1000  # Number of TF-IDF features
        self.word2vec_size = 100  # Word2Vec embedding size
        self.min_word_count = 5  # Minimum word frequency

    def create_tfidf_features(self, df: DataFrame,
                              text_col: str = "tokens_lemmatized",
                              num_features: int = 1000) -> DataFrame:
        """
        Create TF-IDF features from tokenized text

        Args:
            df: Input DataFrame
            text_col: Column containing tokens
            num_features: Number of TF-IDF features

        Returns:
            DataFrame with TF-IDF features
        """
        logger.info(f"Creating TF-IDF features with {num_features} dimensions...")

        # Configure TF-IDF pipeline
        hashingTF = HashingTF(
            inputCol=text_col,
            outputCol="raw_features",
            numFeatures=num_features
        )

        idf = IDF(
            inputCol="raw_features",
            outputCol="tfidf_features",
            minDocFreq=5  # Ignore terms that appear in less than 5 documents
        )

        # Build pipeline
        pipeline = Pipeline(stages=[hashingTF, idf])

        # Fit and transform
        model = pipeline.fit(df)
        df_tfidf = model.transform(df)

        logger.info("TF-IDF features created")
        return df_tfidf

    def extract_ngram_features(self, df: DataFrame,
                               tokens_col: str = "tokens_lemmatized",
                               n_values: List[int] = [2, 3]) -> DataFrame:
        """
        Extract n-gram features

        Args:
            df: Input DataFrame
            tokens_col: Column containing tokens
            n_values: List of n values for n-grams

        Returns:
            DataFrame with n-gram features
        """
        logger.info(f"Extracting n-gram features for n={n_values}...")

        for n in n_values:
            ngram = NGram(n=n, inputCol=tokens_col, outputCol=f"{n}grams")
            df = ngram.transform(df)

            # Count n-grams
            df = df.withColumn(f"{n}gram_count", size(col(f"{n}grams")))

        logger.info("N-gram extraction completed")
        return df

    def generate_word_embeddings(self, df: DataFrame,
                                 tokens_col: str = "tokens_lemmatized",
                                 vector_size: int = 100) -> DataFrame:
        """
        Generate Word2Vec embeddings

        Args:
            df: Input DataFrame
            tokens_col: Column containing tokens
            vector_size: Size of word vectors

        Returns:
            DataFrame with word embeddings
        """
        logger.info(f"Generating Word2Vec embeddings with size {vector_size}...")

        # Configure Word2Vec
        word2vec = Word2Vec(
            vectorSize=vector_size,
            minCount=self.min_word_count,
            inputCol=tokens_col,
            outputCol="word2vec_features",
            windowSize=5,
            maxIter=10,
            seed=42
        )

        # Fit and transform
        model = word2vec.fit(df)
        df_w2v = model.transform(df)

        # Also save the word vectors for later use
        word_vectors = model.getVectors()
        logger.info(f"Learned vectors for {word_vectors.count()} words")

        logger.info("Word2Vec generation completed")
        return df_w2v

    def extract_lexicon_features(self, df: DataFrame,
                                 text_col: str = "text") -> DataFrame:
        """
        Extract sentiment lexicon features using VADER

        Args:
            df: Input DataFrame
            text_col: Column containing text

        Returns:
            DataFrame with lexicon features
        """
        logger.info("Extracting sentiment lexicon features...")

        # Define UDF for VADER sentiment
        # def get_vader_scores(text):
        #     """Get VADER sentiment scores"""
        #     try:
        #         scores = self.vader.polarity_scores(text)
        #         return [
        #             float(scores['compound']),
        #             float(scores['pos']),
        #             float(scores['neg']),
        #             float(scores['neu'])
        #         ]
        #     except:
        #         return [0.0, 0.0, 0.0, 0.0]

        # vader_udf = udf(get_vader_scores, ArrayType(DoubleType()))

        # Apply VADER
        # df = df.withColumn("vader_scores", vader_udf(col(text_col)))
        df = df.withColumn("vader_scores", VADER_UDF(col(text_col)))

        # Extract individual scores
        df = df.withColumn("vader_compound", col("vader_scores")[0]) \
            .withColumn("vader_positive", col("vader_scores")[1]) \
            .withColumn("vader_negative", col("vader_scores")[2]) \
            .withColumn("vader_neutral", col("vader_scores")[3]) \
            .drop("vader_scores")

        logger.info("Lexicon feature extraction completed")
        return df

    def extract_temporal_features(self, df: DataFrame) -> DataFrame:
        """
        Extract temporal features from timestamp

        Args:
            df: Input DataFrame with timestamp column

        Returns:
            DataFrame with temporal features
        """
        logger.info("Extracting temporal features...")

        df = df.withColumn("hour", hour(col("timestamp")))

        # Hour of day features (already have hour column)
        df = df.withColumn(
            "hour_sin",
            sin(lit(2 * np.pi) * col("hour") / lit(24))
        )
        df = df.withColumn(
            "hour_cos",
            cos(lit(2 * np.pi) * col("hour") / lit(24))
        )

        # Day of week features
        df = df.withColumn("day_of_week", date_format(col("timestamp"), "E"))

        # Weekend indicator
        df = df.withColumn(
            "is_weekend",
            when(col("day_of_week").isin(["Sat", "Sun"]), 1).otherwise(0)
        )

        # Time of day categories
        df = df.withColumn(
            "time_of_day",
            when(col("hour").cast("int").between(6, 11), "morning")
            .when(col("hour").cast("int").between(12, 17), "afternoon")
            .when(col("hour").cast("int").between(18, 23), "evening")
            .otherwise("night")
        )

        logger.info("Temporal feature extraction completed")
        return df

    def extract_text_statistics(self, df: DataFrame) -> DataFrame:
        """
        Extract statistical features from text

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with text statistics
        """
        logger.info("Extracting text statistics...")

        # Exclamation marks
        df = df.withColumn(
            "exclamation_count",
            length(col("text")) - length(regexp_replace(col("text"), "!", ""))
        )

        # Question marks
        df = df.withColumn(
            "question_count",
            length(col("text")) - length(regexp_replace(col("text"), "\\?", ""))
        )

        # Uppercase ratio
        def uppercase_ratio(text):
            """Calculate ratio of uppercase characters"""
            if not text:
                return 0.0
            uppercase_count = sum(1 for c in text if c.isupper())
            return uppercase_count / len(text) if len(text) > 0 else 0.0

        uppercase_ratio_udf = udf(uppercase_ratio, DoubleType())
        df = df.withColumn("uppercase_ratio", uppercase_ratio_udf(col("text")))

        # Punctuation density
        def punctuation_density(text):
            """Calculate punctuation density"""
            if not text:
                return 0.0
            punct_count = sum(1 for c in text if c in "!?.,;:'-\"")
            return punct_count / len(text) if len(text) > 0 else 0.0

        punct_density_udf = udf(punctuation_density, DoubleType())
        df = df.withColumn("punctuation_density", punct_density_udf(col("text")))

        logger.info("Text statistics extraction completed")
        return df

    def create_all_features(self, df: DataFrame) -> DataFrame:
        """
        Create all features using the complete pipeline

        Args:
            df: Preprocessed DataFrame

        Returns:
            DataFrame with all features
        """
        logger.info("Creating all features...")

        # Apply all feature extraction methods
        df = self.extract_text_statistics(df)
        df = self.extract_temporal_features(df)
        df = self.extract_lexicon_features(df)
        df = self.extract_ngram_features(df)
        df = self.create_tfidf_features(df, num_features=self.tfidf_features)
        df = self.generate_word_embeddings(df, vector_size=self.word2vec_size)

        # Create feature summary
        feature_cols = [
            "text_length", "processed_length", "token_count",
            "emoji_sentiment", "exclamation_count", "question_count",
            "uppercase_ratio", "punctuation_density",
            "vader_compound", "vader_positive", "vader_negative", "vader_neutral",
            "hour_sin", "hour_cos", "is_weekend",
            "2gram_count", "3gram_count"
        ]

        logger.info(f"Created {len(feature_cols)} basic features + TF-IDF + Word2Vec")

        # Add feature column list to DataFrame metadata
        df = df.withColumn("feature_columns", array([lit(col) for col in feature_cols]))

        logger.info("All features created successfully")
        return df

    def save_feature_stats(self, df: DataFrame, output_path: str):
        """
        Calculate and save feature statistics

        Args:
            df: DataFrame with features
            output_path: Path to save statistics
        """
        logger.info("Calculating feature statistics...")

        # Select numeric features
        numeric_cols = [
            "text_length", "processed_length", "token_count",
            "emoji_sentiment", "exclamation_count", "question_count",
            "uppercase_ratio", "punctuation_density",
            "vader_compound", "vader_positive", "vader_negative", "vader_neutral"
        ]

        # Calculate statistics
        stats_df = df.select(numeric_cols).describe()

        # Save statistics
        stats_df.coalesce(1).write.mode("overwrite").json(output_path)
        logger.info(f"Feature statistics saved to: {output_path}")

        return stats_df


def main():
    """
    Demonstrate feature engineering functionality
    """
    from config.spark_config import create_spark_session

    # Create Spark session
    spark = create_spark_session("FeatureEngineering")

    # Load preprocessed sample data
    logger.info("Loading preprocessed data...")
    input_path = "/Users/ali/Documents/Projects/micap/data/processed/sentiment140_preprocessed_sample"
    df = spark.read.parquet(input_path)

    # Initialize feature engineer
    feature_engineer = FeatureEngineer(spark)

    # Create features
    df_features = feature_engineer.create_all_features(df)

    # Show feature summary
    logger.info("Feature summary:")
    df_features.select(
        "tweet_id", "sentiment", "text_length", "token_count",
        "vader_compound", "emoji_sentiment", "exclamation_count"
    ).show(10)

    # Save featured data
    output_path = "data/processed/sentiment140_features_sample"
    df_features.coalesce(4).write.mode("overwrite").parquet(output_path)
    logger.info(f"Featured data saved to: {output_path}")

    # Save feature statistics
    stats_path = "data/processed/feature_statistics"
    feature_engineer.save_feature_stats(df_features, stats_path)

    # Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()