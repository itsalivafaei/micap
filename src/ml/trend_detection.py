"""
Trend Detection Module for Market Intelligence
Implements LDA topic modeling, anomaly detection, and trend forecasting
"""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.path_utils import get_path

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, udf, count, avg, stddev, collect_list,
    array_distinct, size, when, lit, explode,
    window, desc, rank, dense_rank, percent_rank,
    sum as spark_sum, max as spark_max, min as spark_min, lag, expr
)
from pyspark.sql.types import ArrayType, StringType, FloatType, StructType, StructField, IntegerType
from pyspark.ml.feature import CountVectorizer, IDF, StopWordsRemover
from pyspark.ml.clustering import LDA, KMeans
from pyspark.ml import Pipeline

from sklearn.ensemble import IsolationForest
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicModeler:
    """
    Implements LDA topic modeling for trend detection
    """

    def __init__(self, spark: SparkSession, num_topics: int = 20):
        """
        Initialize topic modeler

        Args:
            spark: Active SparkSession
            num_topics: Number of topics to extract
        """
        self.spark = spark
        self.num_topics = num_topics
        self.model = None
        self.vocabulary = None

    def fit_topics(self, df: DataFrame, text_col: str = "tokens_lemmatized") -> 'TopicModeler':
        """
        Fit LDA model on text data

        Args:
            df: DataFrame with tokenized text
            text_col: Column containing tokens

        Returns:
            Self for chaining
        """
        logger.info(f"Fitting LDA model with {self.num_topics} topics")

        # Remove stop words
        remover = StopWordsRemover(
            inputCol=text_col,
            outputCol="tokens_filtered"
        )

        # Count vectorizer
        cv = CountVectorizer(
            inputCol="tokens_filtered",
            outputCol="custom_raw_features",
            maxDF=0.95,
            minDF=0.01,
            vocabSize=5000
        )

        # IDF
        idf = IDF(
            inputCol="custom_raw_features",
            outputCol="features"
        )

        # LDA
        lda = LDA(
            k=self.num_topics,
            maxIter=20,
            optimizer="online",
            learningOffset=1024.0,
            learningDecay=0.51,
            subsamplingRate=0.05,
            optimizeDocConcentration=True,
            seed=42
        )

        # Build pipeline
        pipeline = Pipeline(stages=[remover, cv, idf, lda])

        # Fit model
        self.model = pipeline.fit(df)

        # Extract vocabulary
        cv_model = self.model.stages[1]
        self.vocabulary = cv_model.vocabulary

        return self

    def get_topics(self, max_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract topic descriptions

        Args:
            max_words: Maximum words per topic

        Returns:
            Dictionary of topic ID to word-weight tuples
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")

        lda_model = self.model.stages[-1]
        topics_matrix = lda_model.describeTopics(max_words)

        topics = {}
        for row in topics_matrix.collect():
            topic_id = row['topic']
            term_indices = row['termIndices']
            term_weights = row['termWeights']

            topic_words = []
            for idx, weight in zip(term_indices, term_weights):
                if idx < len(self.vocabulary):
                    topic_words.append((self.vocabulary[idx], float(weight)))

            topics[topic_id] = topic_words

        return topics

    def transform_topics(self, df: DataFrame) -> DataFrame:
        """
        Transform documents to topic distributions

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with topic distributions
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")

        return self.model.transform(df)

    def detect_emerging_topics(self, df: DataFrame,
                               window_size: str = "7 days",
                               growth_threshold: float = 0.5) -> DataFrame:
        """
        Detect emerging topics based on growth rate

        Args:
            df: DataFrame with topic distributions over time
            window_size: Time window for analysis
            growth_threshold: Minimum growth rate to consider emerging

        Returns:
            DataFrame of emerging topics
        """
        logger.info("Detecting emerging topics")

        # Get topic distributions
        df_topics = self.transform_topics(df)

        # Extract dominant topic for each document
        def get_dominant_topic(distribution):
            if distribution is None:
                return -1
            return int(np.argmax(distribution))

        dominant_topic_udf = udf(get_dominant_topic, IntegerType())

        df_topics = df_topics.withColumn(
            "dominant_topic",
            dominant_topic_udf(col("topicDistribution"))
        )

        # Aggregate by time window and topic
        topic_trends = df_topics.groupBy(
            window("timestamp", window_size),
            "dominant_topic"
        ).agg(
            count("*").alias("doc_count")
        )

        # Calculate growth rate
        window_spec = Window.partitionBy("dominant_topic").orderBy("window")

        topic_trends = topic_trends.withColumn(
            "prev_count",
            lag("doc_count", 1).over(window_spec)
        ).withColumn(
            "growth_rate",
            when(col("prev_count") > 0,
                 (col("doc_count") - col("prev_count")) / col("prev_count")
                 ).otherwise(0)
        )

        # Filter emerging topics
        emerging = topic_trends.filter(
            col("growth_rate") > growth_threshold
        )

        return emerging.orderBy(desc("growth_rate"))


class TrendForecaster:
    """
    Forecasts sentiment and mention trends using Prophet
    """

    def __init__(self):
        """Initialize trend forecaster"""
        self.models = {}

    def forecast_brand_sentiment(self, df: pd.DataFrame,
                                 brand: str,
                                 horizon: int = 7) -> pd.DataFrame:
        """
        Forecast sentiment trend for a brand

        Args:
            df: Pandas DataFrame with historical data
            brand: Brand name
            horizon: Forecast horizon in days

        Returns:
            DataFrame with forecast
        """
        logger.info(f"Forecasting sentiment for {brand}")

        # Prepare data for Prophet
        prophet_df = df[['window_start', 'sentiment_score']].rename(
            columns={'window_start': 'ds', 'sentiment_score': 'y'}
        )

        # Initialize and fit model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05
        )

        # Add custom seasonalities if enough data
        if len(prophet_df) > 30:
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )

        model.fit(prophet_df)

        # Make forecast
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)

        # Store model
        self.models[brand] = model

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def forecast_market_trends(self, df: DataFrame,
                               metrics: List[str],
                               horizon: int = 7) -> Dict[str, pd.DataFrame]:
        """
        Forecast multiple market metrics

        Args:
            df: Spark DataFrame with historical data
            metrics: List of metrics to forecast
            horizon: Forecast horizon

        Returns:
            Dictionary of forecasts by metric
        """
        logger.info(f"Forecasting market trends for {metrics}")

        # Convert to pandas
        pdf = df.toPandas()

        forecasts = {}

        for metric in metrics:
            if metric in pdf.columns:
                # Prepare data
                metric_df = pdf[['window_start', metric]].rename(
                    columns={'window_start': 'ds', metric: 'y'}
                )

                # Remove nulls
                metric_df = metric_df.dropna()

                if len(metric_df) > 10:  # Need minimum data
                    # Fit and forecast
                    model = Prophet()
                    model.fit(metric_df)

                    future = model.make_future_dataframe(periods=horizon)
                    forecast = model.predict(future)

                    forecasts[metric] = forecast

        return forecasts


class AnomalyDetector:
    """
    Detects anomalies in sentiment patterns
    """

    def __init__(self, contamination: float = 0.05):
        """
        Initialize anomaly detector

        Args:
            contamination: Expected proportion of anomalies
        """
        self.contamination = contamination
        self.models = {}

    def detect_sentiment_anomalies(self, df: DataFrame,
                                   features: List[str]) -> DataFrame:
        """
        Detect anomalies in sentiment patterns

        Args:
            df: DataFrame with sentiment data
            features: Features to use for anomaly detection

        Returns:
            DataFrame with anomaly labels
        """
        logger.info("Detecting sentiment anomalies")

        # Convert to pandas for sklearn
        pdf = df.select(features + ["window_start", "brand"]).toPandas()

        # Group by brand for brand-specific models
        anomalies_list = []

        for brand in pdf['brand'].unique():
            brand_df = pdf[pdf['brand'] == brand].copy()

            if len(brand_df) > 10:  # Need minimum data
                # Prepare features
                X = brand_df[features].fillna(0)

                # Fit Isolation Forest
                iso_forest = IsolationForest(
                    contamination=self.contamination,
                    random_state=42
                )

                # Predict anomalies
                anomalies = iso_forest.fit_predict(X)
                brand_df['is_anomaly'] = (anomalies == -1).astype(int)

                # Calculate anomaly scores
                scores = iso_forest.score_samples(X)
                brand_df['anomaly_score'] = -scores  # Higher score = more anomalous

                # Store model
                self.models[brand] = iso_forest

                anomalies_list.append(brand_df)

        # Combine results
        if anomalies_list:
            result_df = pd.concat(anomalies_list, ignore_index=True)

            # Convert back to Spark
            return self.spark.createDataFrame(result_df)
        else:
            return df.withColumn("is_anomaly", lit(0)).withColumn("anomaly_score", lit(0.0))

    def detect_volume_anomalies(self, df: DataFrame) -> DataFrame:
        """
        Detect anomalies in mention volumes

        Args:
            df: DataFrame with mention counts

        Returns:
            DataFrame with volume anomaly flags
        """
        logger.info("Detecting volume anomalies")

        # Calculate rolling statistics
        window_spec = Window.partitionBy("brand").orderBy("window_start").rowsBetween(-7, -1)

        df = df.withColumn(
            "volume_mean",
            avg("mention_count").over(window_spec)
        ).withColumn(
            "volume_std",
            stddev("mention_count").over(window_spec)
        )

        # Calculate z-score
        df = df.withColumn(
            "volume_zscore",
            when(col("volume_std") > 0,
                 (col("mention_count") - col("volume_mean")) / col("volume_std")
                 ).otherwise(0)
        )

        # Flag anomalies (|z-score| > 3)
        df = df.withColumn(
            "is_volume_anomaly",
            when(abs(col("volume_zscore")) > 3, 1).otherwise(0)
        ).withColumn(
            "volume_anomaly_type",
            when(col("is_volume_anomaly") == 1,
                 when(col("volume_zscore") > 0, "spike").otherwise("drop")
                 ).otherwise("normal")
        )

        return df


class ViralityPredictor:
    """
    Predicts viral potential of content
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize virality predictor

        Args:
            spark: Active SparkSession
        """
        self.spark = spark

    def calculate_virality_score(self, df: DataFrame) -> DataFrame:
        """
        Calculate virality potential score

        Args:
            df: DataFrame with tweet data

        Returns:
            DataFrame with virality scores
        """
        logger.info("Calculating virality scores")

        # Define virality features
        df = df.withColumn(
            "engagement_score",
            (col("retweet_count") * 2 + col("favorite_count")) /
            (col("follower_count") + 1)  # Normalize by followers
        )

        # Text features that correlate with virality
        df = df.withColumn(
            "has_hashtag",
            when(size(col("hashtags")) > 0, 1).otherwise(0)
        ).withColumn(
            "has_mention",
            when(col("text").contains("@"), 1).otherwise(0)
        ).withColumn(
            "has_url",
            when(col("text").contains("http"), 1).otherwise(0)
        )

        # Sentiment extremity (very positive/negative tends to go viral)
        df = df.withColumn(
            "sentiment_extremity",
            abs(col("vader_compound"))
        )

        # Calculate virality score
        df = df.withColumn(
            "virality_score",
            (
                    col("engagement_score") * 0.4 +
                    col("sentiment_extremity") * 0.2 +
                    col("has_hashtag") * 0.15 +
                    col("has_mention") * 0.15 +
                    col("has_url") * 0.1
            )
        )

        # Classify virality potential
        df = df.withColumn(
            "virality_potential",
            when(col("virality_score") > 0.8, "high")
            .when(col("virality_score") > 0.5, "medium")
            .otherwise("low")
        )

        return df

    def identify_viral_topics(self, df: DataFrame,
                              threshold: float = 0.7) -> DataFrame:
        """
        Identify topics with viral potential

        Args:
            df: DataFrame with topic and virality data
            threshold: Virality score threshold

        Returns:
            DataFrame of viral topics
        """
        logger.info("Identifying viral topics")

        # Filter high virality content
        viral_df = df.filter(col("virality_score") > threshold)

        # Aggregate by topic
        viral_topics = viral_df.groupBy("dominant_topic").agg(
            count("*").alias("viral_count"),
            avg("virality_score").alias("avg_virality_score"),
            collect_list("text").alias("sample_texts")
        )

        # Limit sample texts
        viral_topics = viral_topics.withColumn(
            "sample_texts",
            expr("slice(sample_texts, 1, 5)")
        )

        return viral_topics.orderBy(desc("viral_count"))


# Main function for testing
def main():
    """Test trend detection functionality"""
    from config.spark_config import create_spark_session

    # Create Spark session
    spark = create_spark_session("TrendDetection")

    # Load data
    logger.info("Loading data...")
    df = spark.read.parquet(str(get_path("data/processed/pipeline_features")))

    # Sample for testing
    df_sample = df.sample(0.95)

    # Initialize components
    topic_modeler = TopicModeler(spark, num_topics=15)
    trend_forecaster = TrendForecaster()
    anomaly_detector = AnomalyDetector()
    virality_predictor = ViralityPredictor(spark)

    # 1. Topic modeling
    logger.info("Fitting topic model...")
    topic_modeler.fit_topics(df_sample)
    topics = topic_modeler.get_topics()

    logger.info("Extracted topics:")
    for topic_id, words in topics.items():
        top_words = [w[0] for w in words[:5]]
        logger.info(f"Topic {topic_id}: {', '.join(top_words)}")

    # 2. Detect emerging topics
    emerging = topic_modeler.detect_emerging_topics(df_sample)
    logger.info("Emerging topics:")
    emerging.show(10)

    # 3. Forecast trends (need aggregated data)
    # This would normally use the competitor analysis output

    # 4. Detect anomalies
    features = ["sentiment_score", "mention_count", "positive_ratio"]
    anomalies = anomaly_detector.detect_sentiment_anomalies(aggregated_df, features)

    # 5. Predict virality
    # Note: This assumes additional features like retweet_count
    virality_df = virality_predictor.calculate_virality_score(df_sample)

    logger.info("Trend detection analysis complete")

    spark.stop()


if __name__ == "__main__":
    main()