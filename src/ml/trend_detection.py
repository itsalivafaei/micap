"""Trend Detection Module for Market Intelligence.

.

Implements LDA topic modeling, anomaly detection, and trend forecasting.
."""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from src.utils.path_utils import get_path

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, udf, count, avg, stddev, collect_list, size, when, lit,
    window, desc, lag, expr, abs as spark_abs
)
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import CountVectorizer, IDF, StopWordsRemover
from pyspark.ml.clustering import LDA
from pyspark.ml import Pipeline

from sklearn.ensemble import IsolationForest
from prophet import Prophet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicModeler:
    """Implements LDA topic modeling for trend detection."""def __init__(self, spark: SparkSession, num_topics: int = 20):."""Initialize topic modeler.

        Args:
            spark: Active SparkSession
            num_topics: Number of topics to extract.
        ."""self.spark = spark.
        self.num_topics = num_topics
        self.model = None
        self.vocabulary = None

    def fit_topics(self, df: DataFrame, text_col: str = "tokens_lemmatized") -> 'TopicModeler':
        """Fit LDA model on text data.

        Args:
            df: DataFrame with tokenized text
            text_col: Column containing tokens

        Returns:
            Self for chaining.
        ."""
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
        """Extract topic descriptions.

        Args:
            max_words: Maximum words per topic

        Returns:
            Dictionary of topic ID to word-weight tuples.
        ."""
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

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform documents to topic distributions.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with topic distributions.
        ."""
        if self.model is None:
            raise ValueError("Model not fitted yet")

        return self.model.transform(df)

    def detect_emerging_topics(self, df: DataFrame,
                               window_size: str = "7 days",
                               growth_threshold: float = 0.5) -> DataFrame:
        """Detect emerging topics based on growth rate.

        Args:
            df: DataFrame with topic distributions over time
            window_size: Time window for analysis
            growth_threshold: Minimum growth rate to consider emerging

        Returns:
            DataFrame of emerging topics.
        ."""
        logger.info("Detecting emerging topics")

        # Get topic distributions
        df_topics = self.transform(df)

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
    """Forecasts sentiment and mention trends using Prophet.
    ."""def __init__(self, spark: SparkSession):."""Initialize trend forecaster."""self.spark = spark.
        self.models = {}

    def forecast_brand_sentiment(self, df: pd.DataFrame,
                                 brand: str,
                                 horizon: int = 7) -> pd.DataFrame:
        """Forecast sentiment trend for a brand

        Args:
            df: Pandas DataFrame with historical data
            brand: Brand name
            horizon: Forecast horizon in days

        Returns:
            DataFrame with forecast.
        ."""
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
        """Forecast multiple market metrics.

        Args:
            df: Spark DataFrame with historical data
            metrics: List of metrics to forecast
            horizon: Forecast horizon

        Returns:
            Dictionary of forecasts by metric.
        ."""
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

    def forecast_sentiment_trends(self, df: DataFrame,
                                  horizon: int = 7,
                                  granularity: str = "daily") -> DataFrame:
        """Forecast overall sentiment trends.

        Args:
            df: Spark DataFrame with sentiment data
            horizon: Forecast horizon in days
            granularity: Time granularity (hourly/daily)

        Returns:
            DataFrame with sentiment forecasts.
        ."""
        logger.info("Forecasting overall sentiment trends")

        # Aggregate sentiment by time window
        time_window = "1 hour" if granularity == "hourly" else "1 day"

        agg_df = df.groupBy(
            window("timestamp", time_window)
        ).agg(
            avg("sentiment").alias("avg_sentiment"),
            count("*").alias("tweet_count"),
            avg("vader_compound").alias("avg_compound")
        ).withColumn(
            "window_start", col("window.start")
        ).select("window_start", "avg_sentiment", "tweet_count", "avg_compound")

        # Convert to pandas for Prophet
        pdf = agg_df.toPandas()

        # Forecast sentiment
        sentiment_df = pdf[['window_start', 'avg_sentiment']].rename(
            columns={'window_start': 'ds', 'avg_sentiment': 'y'}
        )

        if len(sentiment_df) > 10:  # Need minimum data
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(sentiment_df)

            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)

            # Convert back to Spark DataFrame
            forecast_df = self.spark.createDataFrame(
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            ).withColumnRenamed('ds', 'window_start') \
                .withColumnRenamed('yhat', 'forecast') \
                .withColumnRenamed('yhat_lower', 'lower') \
                .withColumnRenamed('yhat_upper', 'upper')

            return forecast_df
        else:
            # Return empty DataFrame with schema
            return self.spark.createDataFrame([],
                                              "window_start: timestamp, forecast: double, lower: double, upper: double")

    def forecast_topic_trends(self, df: DataFrame,
                              topic_modeler: 'TopicModeler',
                              horizon: int = 7) -> DataFrame:
        """Forecast topic popularity trends.

        Args:
            df: DataFrame with topic distributions (already transformed)
            topic_modeler: Fitted TopicModeler instance
            horizon: Forecast horizon in days

        Returns:
            DataFrame with topic trend forecasts.
        ."""
        logger.info("Forecasting topic trends")

        # Check if the DataFrame already has topicDistribution column
        if "topicDistribution" not in df.columns:
            # Only transform if not already transformed
            df_with_topics = topic_modeler.transform(df)
        else:
            # Already transformed, use as is
            df_with_topics = df

        # Get dominant topic for each document
        def get_dominant_topic(distribution):
            if distribution is None:
                return -1
            return int(np.argmax(distribution))

        dominant_topic_udf = udf(get_dominant_topic, IntegerType())

        df_topics = df_with_topics.withColumn(
            "dominant_topic",
            dominant_topic_udf(col("topicDistribution"))
        ).filter(col("dominant_topic") >= 0)

        # Aggregate by day and topic
        topic_trends = df_topics.groupBy(
            window("timestamp", "1 day"),
            "dominant_topic"
        ).agg(
            count("*").alias("doc_count")
        ).withColumn(
            "window_start", col("window.start")
        )

        # Convert to pandas for forecasting
        pdf = topic_trends.toPandas()

        # Forecast for each topic
        all_forecasts = []

        for topic_id in pdf['dominant_topic'].unique():
            topic_df = pdf[pdf['dominant_topic'] == topic_id][['window_start', 'doc_count']].rename(
                columns={'window_start': 'ds', 'doc_count': 'y'}
            )

            if len(topic_df) > 5:  # Need minimum data
                try:
                    model = Prophet(yearly_seasonality=False)
                    model.fit(topic_df)

                    future = model.make_future_dataframe(periods=horizon)
                    forecast = model.predict(future)

                    # Add topic ID
                    forecast['topic_id'] = topic_id
                    all_forecasts.append(
                        forecast[['ds', 'topic_id', 'yhat', 'yhat_lower', 'yhat_upper']]
                    )
                except Exception as e:
                    logger.warning(f"Could not forecast topic {topic_id}: {e}")

        if all_forecasts:
            # Combine all forecasts
            combined_forecast = pd.concat(all_forecasts, ignore_index=True)

            # Convert back to Spark
            forecast_df = self.spark.createDataFrame(combined_forecast) \
                .withColumnRenamed('ds', 'window_start') \
                .withColumnRenamed('yhat', 'forecast') \
                .withColumnRenamed('yhat_lower', 'lower') \
                .withColumnRenamed('yhat_upper', 'upper')

            return forecast_df
        else:
            return self.spark.createDataFrame([],
                                              "window_start: timestamp, topic_id: int, forecast: double, lower: double, upper: double")


class AnomalyDetector:
    """Detects anomalies in sentiment patterns.
    ."""def __init__(self, spark: SparkSession, contamination: float = 0.05):."""Initialize anomaly detector.

        Args:
            contamination: Expected proportion of anomalies.
        ."""
        self.spark = spark
        self.contamination = contamination
        self.models = {}

    def detect_sentiment_anomalies(self, df: DataFrame,
                                   features: List[str]) -> DataFrame:
        """Detect anomalies in sentiment patterns.

        Args:
            df: DataFrame with sentiment data
            features: Features to use for anomaly detection

        Returns:
            DataFrame with anomaly labels.
        ."""
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
        """Detect anomalies in tweet volumes.

        Args:
            df: DataFrame with tweet counts

        Returns:
            DataFrame with volume anomaly flags.
        ."""
        logger.info("Detecting volume anomalies")

        # First aggregate by time window to get volume counts
        volume_df = df.groupBy(
            window("timestamp", "1 hour").alias("time_window")
        ).agg(
            count("*").alias("tweet_volume")
        ).withColumn(
            "window_start", col("time_window.start")
        ).drop("time_window")

        # Calculate rolling statistics
        window_spec = Window.orderBy("window_start").rowsBetween(-24, -1)  # 24 hour window

        volume_df = volume_df.withColumn(
            "volume_mean",
            avg("tweet_volume").over(window_spec)
        ).withColumn(
            "volume_std",
            stddev("tweet_volume").over(window_spec)
        )

        # Calculate z-score
        volume_df = volume_df.withColumn(
            "volume_zscore",
            when(col("volume_std") > 0,
                 (col("tweet_volume") - col("volume_mean")) / col("volume_std")
                 ).otherwise(0)
        )

        # Flag anomalies (|z-score| > 3)
        volume_df = volume_df.withColumn(
            "is_volume_anomaly",
            when(spark_abs(col("volume_zscore")) > 3, 1).otherwise(0)
        ).withColumn(
            "volume_anomaly_type",
            when(col("is_volume_anomaly") == 1,
                 when(col("volume_zscore") > 0, "spike").otherwise("drop")
                 ).otherwise("normal")
        )

        return volume_df


class ViralityPredictor:
    """Predicts viral potential of content.
    ."""def __init__(self, spark: SparkSession):."""Initialize virality predictor.

        Args:
            spark: Active SparkSession.
        ."""
        self.spark = spark

    def identify_viral_potential_simple(self, df: DataFrame) -> DataFrame:
        """Identify viral potential without social media metrics.
        Uses text features as proxy for virality

        Args:
            df: DataFrame with text features

        Returns:
            DataFrame with viral potential scores.
        ."""
        logger.info("Identifying viral potential from text features")

        # Create virality score from available features
        df_viral = df.withColumn(
            "has_hashtag",
            when(size(col("hashtags")) > 0, 1).otherwise(0)
        ).withColumn(
            "has_url",
            when(col("text").contains("http"), 1).otherwise(0)
        ).withColumn(
            "sentiment_extremity",
            spark_abs(col("vader_compound"))
        ).withColumn(
            "engagement_features",
            col("exclamation_count") + col("question_count") +
            col("emoji_sentiment").cast("double")
        )

        # Calculate viral potential score
        df_viral = df_viral.withColumn(
            "virality_score",
            (
                    col("sentiment_extremity") * 0.3 +
                    col("has_hashtag") * 0.2 +
                    col("has_url") * 0.1 +
                    col("engagement_features") * 0.1 / 10  # Normalize
            )
        ).withColumn(
            "virality_potential",
            when(col("virality_score") > 0.7, "high")
            .when(col("virality_score") > 0.4, "medium")
            .otherwise("low")
        )

        return df_viral

    def calculate_virality_score(self, df: DataFrame) -> DataFrame:
        """Calculate virality potential score.

        Args:
            df: DataFrame with tweet data

        Returns:
            DataFrame with virality scores.
        ."""
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
            spark_abs(col("vader_compound"))
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
        """Identify topics with viral potential.

        Args:
            df: DataFrame with topic and virality data
            threshold: Virality score threshold

        Returns:
            DataFrame of viral topics.
        ."""
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
    """Test trend detection functionality."""
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
    trend_forecaster = TrendForecaster(spark)
    anomaly_detector = AnomalyDetector(spark)
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

    aggregated_df = df_sample.groupBy(
        window("timestamp", "1 hour")
    ).agg(
        count("*").alias("mention_count"),
        avg("sentiment").alias("sentiment_score"),
        avg(when(col("sentiment") == 1, 1).otherwise(0)).alias("positive_ratio")
    ).withColumn("window_start", col("window.start")).drop("window")

    # Add brand column for compatibility
    aggregated_df = aggregated_df.withColumn("brand", lit("general"))

    # 4. Detect anomalies
    features = ["sentiment_score", "mention_count", "positive_ratio"]
    anomalies = anomaly_detector.detect_sentiment_anomalies(aggregated_df, features)
    logger.info(f"Found {anomalies.filter(col('is_anomaly') == 1).count()} anomalies")

    # 5. Predict virality
    # Note: This assumes additional features like retweet_count
    df_virality = df_sample
    for col_name in ["retweet_count", "favorite_count", "follower_count"]:
        if col_name not in df_virality.columns:
            df_virality = df_virality.withColumn(col_name, lit(0))
            
    virality_df = virality_predictor.calculate_virality_score(df_virality)
    logger.info(f"Calculated virality scores for {virality_df.count()} tweets")

    logger.info("Trend detection analysis complete")

    spark.stop()


if __name__ == "__main__":
    main()