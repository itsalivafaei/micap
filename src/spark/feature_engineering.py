"""
Feature Engineering Module for MICAP - FIXED for distributed execution
Creates ML-ready features from preprocessed text data
FIXED: Uses lazy imports to avoid serialization issues with Spark workers
"""

import logging
import math
import os
from typing import List, Dict, Optional, Tuple
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import (
    col, count, sum as spark_sum, mean, stddev,
    explode, array, collect_list, size, when,
    regexp_extract, length, udf, desc, row_number, lit, date_format,
    regexp_replace, col, date_format, hour, when, lit, sin, cos, udf, array_contains
)
from pyspark.sql.types import ArrayType, DoubleType, StringType, IntegerType

# FIX: Import only basic Spark SQL functions, not ML features at module level
# ML features will be imported lazily inside functions when needed

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils.path_utils import get_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global analyzer - lazy initialization
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


# Pre-declare the UDF once so Spark can reuse it
VADER_UDF = udf(get_vader_scores, ArrayType(DoubleType()))


class FeatureEngineer:
    """
    Comprehensive feature engineering for sentiment analysis
    FIXED: Uses lazy imports to avoid worker serialization issues
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
        self.tfidf_features = 1000
        self.word2vec_size = 100
        self.min_word_count = 2

    def _lazy_import_ml_features(self):
        """
        Lazy import of PySpark ML features to avoid serialization issues
        This ensures imports happen within the Spark context
        """
        try:
            from pyspark.ml.feature import (
                HashingTF, IDF, Word2Vec, NGram,
                CountVectorizer, StringIndexer
            )
            from pyspark.ml import Pipeline

            return {
                'HashingTF': HashingTF,
                'IDF': IDF,
                'Word2Vec': Word2Vec,
                'NGram': NGram,
                'CountVectorizer': CountVectorizer,
                'StringIndexer': StringIndexer,
                'Pipeline': Pipeline
            }
        except ImportError as e:
            logger.error(f"Failed to import PySpark ML features: {e}")
            return None

    def create_tfidf_features(self, df: DataFrame,
                              text_col: str = "tokens_lemmatized",
                              num_features: int = 1000) -> DataFrame:
        """
        Create TF-IDF features from tokenized text with lazy imports

        Args:
            df: Input DataFrame
            text_col: Column containing tokens
            num_features: Number of TF-IDF features

        Returns:
            DataFrame with TF-IDF features
        """
        logger.info(f"Creating TF-IDF features with {num_features} dimensions...")

        # Lazy import ML features
        ml_imports = self._lazy_import_ml_features()
        if ml_imports is None:
            logger.error("Could not import ML features. Skipping TF-IDF.")
            return df.withColumn("tfidf_features", lit(None))

        HashingTF = ml_imports['HashingTF']
        IDF = ml_imports['IDF']
        Pipeline = ml_imports['Pipeline']

        # Check if we have enough data
        row_count = df.count()
        logger.info(f"Dataset has {row_count} rows")

        # Adjust minDocFreq based on dataset size
        min_doc_freq = max(1, min(2, row_count // 5))

        try:
            # Configure TF-IDF pipeline
            hashingTF = HashingTF(
                inputCol=text_col,
                outputCol="raw_features",
                numFeatures=num_features
            )

            idf = IDF(
                inputCol="raw_features",
                outputCol="tfidf_features",
                minDocFreq=min_doc_freq
            )

            # Build and fit pipeline
            pipeline = Pipeline(stages=[hashingTF, idf])
            model = pipeline.fit(df)
            df_tfidf = model.transform(df)

            logger.info("TF-IDF features created")
            return df_tfidf

        except Exception as e:
            logger.error(f"TF-IDF creation failed: {e}")
            return df.withColumn("tfidf_features", lit(None))

    def extract_ngram_features(self, df: DataFrame,
                               tokens_col: str = "tokens_lemmatized",
                               n_values: List[int] = [2, 3]) -> DataFrame:
        """
        Extract n-gram features with lazy imports
        """
        logger.info(f"Extracting n-gram features for n={n_values}...")

        ml_imports = self._lazy_import_ml_features()
        if ml_imports is None:
            logger.error("Could not import ML features. Skipping n-grams.")
            # Add dummy n-gram columns
            for n in n_values:
                df = df.withColumn(f"{n}grams", lit(None)) \
                    .withColumn(f"{n}gram_count", lit(0))
            return df

        NGram = ml_imports['NGram']

        try:
            for n in n_values:
                ngram = NGram(n=n, inputCol=tokens_col, outputCol=f"{n}grams")
                df = ngram.transform(df)
                df = df.withColumn(f"{n}gram_count", size(col(f"{n}grams")))

            logger.info("N-gram extraction completed")
            return df

        except Exception as e:
            logger.error(f"N-gram extraction failed: {e}")
            # Add dummy columns
            for n in n_values:
                df = df.withColumn(f"{n}grams", lit(None)) \
                    .withColumn(f"{n}gram_count", lit(0))
            return df

    def generate_word_embeddings_optimized(self, df: DataFrame,
                                         tokens_col: str = "tokens_lemmatized",
                                         vector_size: int = 100) -> DataFrame:
        """
        Generate Word2Vec embeddings with memory optimization and checkpointing
        """
        logger.info(f"Generating optimized Word2Vec embeddings with size {vector_size}...")

        ml_imports = self._lazy_import_ml_features()
        if ml_imports is None:
            logger.error("Could not import ML features. Using dummy embeddings.")
            return self._create_dummy_embeddings(df, vector_size)

        Word2Vec = ml_imports['Word2Vec']
        
        # Cache the dataframe to avoid recomputation
        df.cache()
        row_count = df.count()
        logger.info(f"Dataset has {row_count} rows")

        try:
            # Checkpoint the data to disk to break lineage
            checkpoint_dir = str(get_path("data/checkpoints/word2vec"))
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.spark.sparkContext.setCheckpointDir(checkpoint_dir)
            
            # Filter and checkpoint data
            df_filtered = df.filter(size(col(tokens_col)) > 0)
            df_filtered.checkpoint()
            
            # Count vocabulary more efficiently
            vocab_sample = df_filtered.sample(0.1).select(explode(col(tokens_col)).alias("word"))
            estimated_vocab = vocab_sample.distinct().count() * 10
            logger.info(f"Estimated vocabulary size: {estimated_vocab} unique words")

            # Dynamic parameter adjustment based on data size
            if row_count < 100:
                min_count = 1
                max_iter = 3
                window_size = 3
            elif row_count < 1000:
                min_count = 2
                max_iter = 5
                window_size = 4
            else:
                min_count = max(2, row_count // 1000)
                max_iter = min(10, max(3, row_count // 100))
                window_size = 5

            logger.info(f"Word2Vec params: minCount={min_count}, maxIter={max_iter}, windowSize={window_size}")

            # Create Word2Vec with optimized parameters
            word2vec = Word2Vec(
                vectorSize=vector_size,
                minCount=min_count,
                inputCol=tokens_col,
                outputCol="word2vec_features",
                windowSize=window_size,
                maxIter=max_iter,
                seed=42,
                numPartitions=4,  # Reduce parallelism to save memory
                stepSize=0.025    # Smaller step size for stability
            )

            # Train with retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    logger.info(f"Word2Vec training attempt {attempt + 1}/{max_retries}")
                    model = word2vec.fit(df_filtered)
                    df_w2v = model.transform(df_filtered)
                    
                    # Verify success
                    word_vectors = model.getVectors()
                    vector_count = word_vectors.count()
                    logger.info(f"Successfully learned vectors for {vector_count} words")
                    
                    # Unpersist cached data
                    df.unpersist()
                    return df_w2v
                    
                except Exception as e:
                    logger.warning(f"Word2Vec attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise e
                    # Force garbage collection and try again
                    import gc
                    gc.collect()

        except Exception as e:
            logger.warning(f"Word2Vec training failed after all attempts: {e}. Using dummy embeddings.")
            df.unpersist()
            return self._create_dummy_embeddings(df, vector_size)

    def _create_dummy_embeddings(self, df: DataFrame, vector_size: int) -> DataFrame:
        """Create dummy zero embeddings when Word2Vec fails"""
        dummy_vector = [0.0] * vector_size
        return df.withColumn("word2vec_features",
                             lit(dummy_vector).cast(ArrayType(DoubleType())))

    def extract_lexicon_features(self, df: DataFrame,
                                 text_col: str = "text") -> DataFrame:
        """
        Extract sentiment lexicon features using VADER
        """
        logger.info("Extracting sentiment lexicon features...")

        try:
            # Apply VADER UDF
            df = df.withColumn("vader_scores", VADER_UDF(col(text_col)))

            # Extract individual scores
            df = df.withColumn("vader_compound", col("vader_scores")[0]) \
                .withColumn("vader_positive", col("vader_scores")[1]) \
                .withColumn("vader_negative", col("vader_scores")[2]) \
                .withColumn("vader_neutral", col("vader_scores")[3]) \
                .drop("vader_scores")

            logger.info("Lexicon feature extraction completed")
            return df

        except Exception as e:
            logger.error(f"VADER feature extraction failed: {e}")
            # Add dummy VADER columns
            return df.withColumn("vader_compound", lit(0.0)) \
                .withColumn("vader_positive", lit(0.0)) \
                .withColumn("vader_negative", lit(0.0)) \
                .withColumn("vader_neutral", lit(1.0))

    def extract_temporal_features(self, df: DataFrame) -> DataFrame:
        """
        Extract temporal features from timestamp
        """
        logger.info("Extracting temporal features...")

        try:
            # Check if timestamp column exists
            if "timestamp" not in df.columns:
                logger.warning("No timestamp column found. Adding dummy temporal features.")
                return df.withColumn("hour_sin", lit(0.0)) \
                    .withColumn("hour_cos", lit(1.0)) \
                    .withColumn("is_weekend", lit(0)) \
                    .withColumn("time_of_day", lit("unknown"))

            # Extract hour if not already present
            if "hour" not in df.columns:
                df = df.withColumn("hour", hour(col("timestamp")))

            # Hour of day features (cyclical encoding)
            df = df.withColumn("hour_sin", sin(lit(2 * math.pi) * col("hour") / lit(24))) \
                .withColumn("hour_cos", cos(lit(2 * math.pi) * col("hour") / lit(24)))

            # Day of week features
            df = df.withColumn("day_of_week", date_format(col("timestamp"), "E"))

            # Weekend indicator
            df = df.withColumn("is_weekend",
                               when(col("day_of_week").isin(["Sat", "Sun"]), 1).otherwise(0))

            # Time of day categories
            df = df.withColumn("time_of_day",
                               when(col("hour").cast("int").between(6, 11), "morning")
                               .when(col("hour").cast("int").between(12, 17), "afternoon")
                               .when(col("hour").cast("int").between(18, 23), "evening")
                               .otherwise("night"))

            logger.info("Temporal feature extraction completed")
            return df

        except Exception as e:
            logger.error(f"Temporal feature extraction failed: {e}")
            return df.withColumn("hour_sin", lit(0.0)) \
                .withColumn("hour_cos", lit(1.0)) \
                .withColumn("is_weekend", lit(0)) \
                .withColumn("time_of_day", lit("unknown"))

    def extract_text_statistics(self, df: DataFrame) -> DataFrame:
        """
        Extract statistical features from text
        """
        logger.info("Extracting text statistics...")

        try:
            # Exclamation marks
            df = df.withColumn("exclamation_count",
                               length(col("text")) - length(regexp_replace(col("text"), "!", "")))

            # Question marks
            df = df.withColumn("question_count",
                               length(col("text")) - length(regexp_replace(col("text"), "\\?", "")))

            # Uppercase ratio
            def uppercase_ratio(text):
                if not text:
                    return 0.0
                uppercase_count = sum(1 for c in text if c.isupper())
                return uppercase_count / len(text) if len(text) > 0 else 0.0

            uppercase_ratio_udf = udf(uppercase_ratio, DoubleType())
            df = df.withColumn("uppercase_ratio", uppercase_ratio_udf(col("text")))

            # Punctuation density
            def punctuation_density(text):
                if not text:
                    return 0.0
                punct_count = sum(1 for c in text if c in "!?.,;:'-\"")
                return punct_count / len(text) if len(text) > 0 else 0.0

            punct_density_udf = udf(punctuation_density, DoubleType())
            df = df.withColumn("punctuation_density", punct_density_udf(col("text")))

            logger.info("Text statistics extraction completed")
            return df

        except Exception as e:
            logger.error(f"Text statistics extraction failed: {e}")
            return df.withColumn("exclamation_count", lit(0)) \
                .withColumn("question_count", lit(0)) \
                .withColumn("uppercase_ratio", lit(0.0)) \
                .withColumn("punctuation_density", lit(0.0))

    def create_all_features(self, df: DataFrame) -> DataFrame:
        """
        Create all features using the complete pipeline with error handling
        """
        logger.info("Creating all features...")

        # Log dataset info
        row_count = df.count()
        logger.info(f"Processing {row_count} rows")

        # Apply all feature extraction methods with error handling
        try:
            df = self.extract_text_statistics(df)
            df = self.extract_temporal_features(df)
            df = self.extract_lexicon_features(df)
            df = self.extract_entity_features(df)  # Add this line
            df = self.extract_ngram_features(df)
            df = self.create_tfidf_features(df, num_features=self.tfidf_features)
            df = self.generate_word_embeddings_optimized(df, vector_size=self.word2vec_size)

            # Update feature summary to include entity features
            feature_cols = [
                "text_length", "processed_length", "token_count",
                "emoji_sentiment", "exclamation_count", "question_count",
                "uppercase_ratio", "punctuation_density",
                "vader_compound", "vader_positive", "vader_negative", "vader_neutral",
                "hour_sin", "hour_cos", "is_weekend",
                "2gram_count", "3gram_count",
                "brand_count", "category_count", "has_competitor_comparison",  # Add these
                "has_apple", "has_samsung", "has_google", "has_microsoft",  # Add these
                "has_amazon", "has_tesla"  # Add these
            ]

            logger.info(f"Created {len(feature_cols)} basic features + TF-IDF + Word2Vec")
            logger.info("All features created successfully")
            return df

        except Exception as e:
            logger.error(f"Feature creation failed: {e}")
            raise

    def save_feature_stats(self, df: DataFrame, output_path: str):
        """
        Calculate and save feature statistics with error handling
        """
        logger.info("Calculating feature statistics...")

        try:
            # Select numeric features that exist
            numeric_cols = [
                "text_length", "processed_length", "token_count",
                "emoji_sentiment", "exclamation_count", "question_count",
                "uppercase_ratio", "punctuation_density",
                "vader_compound", "vader_positive", "vader_negative", "vader_neutral"
            ]

            existing_cols = [col for col in numeric_cols if col in df.columns]

            if not existing_cols:
                logger.warning("No numeric columns found for statistics")
                stats_data = [("count", "0"), ("mean", "0"), ("stddev", "0"), ("min", "0"), ("max", "0")]
                stats_df = self.spark.createDataFrame(stats_data, ["summary", "dummy"])
            else:
                stats_df = df.select(existing_cols).describe()

            # Save statistics
            stats_df.coalesce(1).write.mode("overwrite").json(output_path)
            logger.info(f"Feature statistics saved to: {output_path}")
            return stats_df

        except Exception as e:
            logger.warning(f"Could not save statistics: {e}")
            stats_data = [("error", str(e))]
            return self.spark.createDataFrame(stats_data, ["summary", "message"])

    def extract_entity_features(self, df: DataFrame,
                                text_col: str = "text") -> DataFrame:
        """
        Extract named entity features for brand/product recognition
        Uses pattern matching and dictionary-based approach for distributed processing

        Args:
            df: Input DataFrame
            text_col: Column containing text

        Returns:
            DataFrame with entity features
        """
        logger.info("Extracting named entity features...")

        try:
            # Define common tech brands and products for pattern matching
            # This is a simplified approach suitable for distributed processing
            tech_brands = [
                'apple', 'iphone', 'ipad', 'macbook', 'airpods', 'ios',
                'samsung', 'galaxy', 'android',
                'google', 'pixel', 'chrome', 'gmail',
                'microsoft', 'windows', 'xbox', 'surface',
                'amazon', 'alexa', 'kindle', 'aws',
                'tesla', 'model s', 'model 3', 'model x', 'model y',
                'facebook', 'meta', 'instagram', 'whatsapp',
                'twitter', 'tweet',
                'netflix', 'spotify', 'youtube',
                'uber', 'lyft', 'airbnb',
                'playstation', 'nintendo', 'switch'
            ]

            # Broadcast the brand list for efficiency
            brands_broadcast = self.spark.sparkContext.broadcast(tech_brands)

            # UDF to extract brand mentions
            def extract_brands(text):
                """Extract brand mentions from text"""
                if not text:
                    return []

                text_lower = text.lower()
                found_brands = []

                # Check for each brand in the text
                for brand in brands_broadcast.value:
                    # Use word boundaries to avoid partial matches
                    import re
                    pattern = r'\b' + re.escape(brand) + r'\b'
                    if re.search(pattern, text_lower):
                        found_brands.append(brand)

                return found_brands

            # UDF to count brand mentions
            def count_brand_mentions(text):
                """Count total brand mentions in text"""
                if not text:
                    return 0

                text_lower = text.lower()
                mention_count = 0

                for brand in brands_broadcast.value:
                    import re
                    pattern = r'\b' + re.escape(brand) + r'\b'
                    matches = re.findall(pattern, text_lower)
                    mention_count += len(matches)

                return mention_count

            # Pattern-based entity extraction for product categories
            def extract_product_categories(text):
                """Extract product category mentions"""
                if not text:
                    return []

                text_lower = text.lower()
                categories = []

                # Product category patterns
                phone_pattern = r'\b(phone|smartphone|mobile|cellphone)\b'
                laptop_pattern = r'\b(laptop|notebook|computer|pc|mac)\b'
                tablet_pattern = r'\b(tablet|ipad)\b'
                car_pattern = r'\b(car|vehicle|automobile|suv|sedan)\b'
                app_pattern = r'\b(app|application|software)\b'

                import re
                if re.search(phone_pattern, text_lower):
                    categories.append('phone')
                if re.search(laptop_pattern, text_lower):
                    categories.append('laptop')
                if re.search(tablet_pattern, text_lower):
                    categories.append('tablet')
                if re.search(car_pattern, text_lower):
                    categories.append('car')
                if re.search(app_pattern, text_lower):
                    categories.append('app')

                return categories

            # Create UDFs
            extract_brands_udf = udf(extract_brands, ArrayType(StringType()))
            count_mentions_udf = udf(count_brand_mentions, IntegerType())
            extract_categories_udf = udf(extract_product_categories, ArrayType(StringType()))

            # Apply entity extraction
            df = df.withColumn("brand_mentions", extract_brands_udf(col(text_col)))
            df = df.withColumn("brand_count", count_mentions_udf(col(text_col)))
            df = df.withColumn("product_categories", extract_categories_udf(col(text_col)))
            df = df.withColumn("category_count", size(col("product_categories")))

            # Create binary features for top brands (presence indicators)
            top_brands = ['apple', 'samsung', 'google', 'microsoft', 'amazon', 'tesla']
            for brand in top_brands:
                df = df.withColumn(
                    f"has_{brand}",
                    when(array_contains(col("brand_mentions"), brand), 1).otherwise(0)
                )

            # Add competitor co-mention feature
            def has_competitor_mentions(brands):
                """Check if multiple competing brands are mentioned"""
                if not brands or len(brands) < 2:
                    return 0

                # Define competitor groups
                competitors = [
                    {'apple', 'samsung', 'google'},  # Phone competitors
                    {'microsoft', 'google', 'apple'},  # Tech giants
                    {'uber', 'lyft'},  # Rideshare
                    {'netflix', 'spotify', 'youtube'}  # Entertainment
                ]

                brand_set = set(brands)
                for comp_group in competitors:
                    if len(brand_set.intersection(comp_group)) >= 2:
                        return 1
                return 0

            competitor_udf = udf(has_competitor_mentions, IntegerType())
            df = df.withColumn("has_competitor_comparison",
                               competitor_udf(col("brand_mentions")))

            logger.info("Entity feature extraction completed")
            return df

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            # Add dummy columns on failure
            df = df.withColumn("brand_mentions", array().cast(ArrayType(StringType()))) \
                .withColumn("brand_count", lit(0)) \
                .withColumn("product_categories", array().cast(ArrayType(StringType()))) \
                .withColumn("category_count", lit(0)) \
                .withColumn("has_competitor_comparison", lit(0))

            # Add dummy brand presence features
            for brand in ['apple', 'samsung', 'google', 'microsoft', 'amazon', 'tesla']:
                df = df.withColumn(f"has_{brand}", lit(0))

            return df

    def can_train_word2vec(self, df: DataFrame, tokens_col: str = "tokens_lemmatized") -> bool:
        """
        Check if Word2Vec training is feasible given memory constraints
        """
        try:
            # Get system info
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Sample data to estimate requirements
            sample_df = df.sample(0.01).select(tokens_col)
            sample_count = sample_df.count()
            
            if sample_count == 0:
                return False
            
            # Estimate vocabulary size from sample
            vocab_sample = sample_df.select(explode(col(tokens_col)).alias("word"))
            unique_words_sample = vocab_sample.distinct().count()
            estimated_vocab = unique_words_sample * 100  # Scale up
            
            # Rough memory estimation (vocab_size * vector_size * 8 bytes * safety_factor)
            estimated_memory_gb = (estimated_vocab * 100 * 8 * 3) / (1024**3)
            
            logger.info(f"Available memory: {available_memory_gb:.1f}GB")
            logger.info(f"Estimated Word2Vec memory needed: {estimated_memory_gb:.1f}GB")
            
            return estimated_memory_gb < available_memory_gb * 0.7  # 70% safety margin
            
        except Exception as e:
            logger.warning(f"Could not estimate Word2Vec feasibility: {e}")
            return True  # Default to trying


def main():
    """
    Demonstrate feature engineering functionality
    """
    from config.spark_config import create_spark_session

    # Create Spark session
    spark = create_spark_session("FeatureEngineering")

    try:
        # Load preprocessed sample data
        logger.info("Loading preprocessed data...")
        input_path = str(get_path("data/processed/sentiment140_sample"))
        df = spark.read.parquet(input_path)

        # Initialize feature engineer
        feature_engineer = FeatureEngineer(spark)

        # Create features
        df_features = feature_engineer.create_all_features(df)

        # Save featured data
        output_path = str(get_path("data/processed/sentiment140_sample"))
        df_features.coalesce(4).write.mode("overwrite").parquet(output_path)
        logger.info(f"Featured data saved to: {output_path}")

        # Save feature statistics
        stats_path = str(get_path("data/processed/feature_statistics"))
        feature_engineer.save_feature_stats(df_features, stats_path)

    except Exception as e:
        logger.error(f"Feature engineering demo failed: {e}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()