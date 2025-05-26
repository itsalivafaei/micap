"""
Competitor Analysis Module for MICAP
Implements comparative sentiment analysis and market intelligence
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from src.utils.path_utils import get_path

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, avg, count, sum as spark_sum, stddev, when, lit,
    window, collect_list, explode, array_contains, size,
    first, last, min as spark_min, max as spark_max,
    percentile_approx, udf, struct, create_map, expr, lag
)
from pyspark.sql.types import StringType, FloatType, ArrayType, StructType, StructField

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_brand_recognition_udf():
    """
    Returns a UDF that initialises BrandRecognizer **once per executor**.
    The driver never touches spaCy / TensorFlow-Metal.
    """
    def _recognize_brands(text):
        if not hasattr(_recognize_brands, "_model"):
            # heavy imports happen here, inside the worker
            from src.ml.entity_recognition import BrandRecognizer
            # full pipeline incl. spaCy
            _recognize_brands._model = BrandRecognizer()

        pairs = _recognize_brands._model.recognize_brands(text) or []
        return [f"{b}:{c:.2f}" for b, c in pairs]

    return udf(_recognize_brands, ArrayType(StringType()))

class CompetitorAnalyzer:
    """
    Performs competitive analysis on sentiment data
    """

    def __init__(self, spark: SparkSession, brand_recognizer):
        """
        Initialize competitor analyzer

        Args:
            spark: Active SparkSession
            brand_recognizer: Initialized BrandRecognizer instance
        """
        self.spark = spark
        self.brand_recognizer = brand_recognizer

    def aggregate_brand_sentiment(self, df: DataFrame,
                                  time_window: str = "1 day") -> DataFrame:
        """
        Aggregate sentiment metrics by brand and time window

        Args:
            df: DataFrame with brand recognition results
            time_window: Time window for aggregation

        Returns:
            DataFrame with aggregated brand sentiment
        """
        logger.info(f"Aggregating brand sentiment with {time_window} window")

        # Explode brands array to individual rows
        df_brands = df.select(
            col("timestamp"),
            col("sentiment"),
            col("vader_compound"),
            col("text"),
            explode(col("brands")).alias("brand_info")
        )

        # Parse brand and confidence
        df_brands = df_brands.withColumn(
            "brand",
            expr("split(brand_info, ':')[0]")
        ).withColumn(
            "confidence",
            expr("cast(split(brand_info, ':')[1] as float)")
        )

        # Filter by confidence threshold
        df_brands = df_brands.filter(col("confidence") >= 0.7)

        # Aggregate by brand and time window
        brand_sentiment = df_brands.groupBy(
            window("timestamp", time_window).alias("time_window"),
            "brand"
        ).agg(
            count("*").alias("mention_count"),
            avg("sentiment").alias("avg_sentiment"),
            stddev("sentiment").alias("sentiment_stddev"),
            avg("vader_compound").alias("avg_vader_compound"),
            spark_sum(when(col("sentiment") == 1, 1).otherwise(0)).alias("positive_mentions"),
            spark_sum(when(col("sentiment") == 0, 1).otherwise(0)).alias("negative_mentions"),
            avg("confidence").alias("avg_confidence"),
            collect_list("text").alias("sample_texts")
        )

        # Calculate sentiment metrics
        brand_sentiment = brand_sentiment.withColumn(
            "positive_ratio",
            col("positive_mentions") / col("mention_count")
        ).withColumn(
            "sentiment_score",
            (col("avg_sentiment") * 2 - 1) * 100  # Convert to -100 to 100 scale
        )

        # Extract window times
        brand_sentiment = brand_sentiment.withColumn(
            "window_start", col("time_window.start")
        ).withColumn(
            "window_end", col("time_window.end")
        ).drop("time_window")

        # Limit sample texts
        brand_sentiment = brand_sentiment.withColumn(
            "sample_texts", expr("slice(sample_texts, 1, 5)")
        )

        return brand_sentiment.orderBy("window_start", "brand")

    def compare_competitor_sentiment(self, df: DataFrame,
                                     target_brand: str,
                                     time_window: str = "1 day") -> DataFrame:
        """
        Compare sentiment between target brand and competitors

        Args:
            df: DataFrame with brand sentiment
            target_brand: Target brand for comparison
            time_window: Time window for comparison

        Returns:
            DataFrame with competitor comparison
        """
        logger.info(f"Comparing {target_brand} with competitors")

        # Get competitors for target brand
        competitors = self.brand_recognizer.competitor_map.get(target_brand.lower(), set())

        if not competitors:
            logger.warning(f"No competitors found for {target_brand}")
            return self.spark.createDataFrame([], StructType([]))

        # Filter for target brand and competitors
        brands_to_compare = [target_brand.lower()] + list(competitors)

        # Aggregate brand sentiment
        brand_sentiment = self.aggregate_brand_sentiment(df, time_window)

        # Filter for relevant brands
        comparison_df = brand_sentiment.filter(
            col("brand").isin(brands_to_compare)
        )

        # Pivot for side-by-side comparison
        pivot_df = comparison_df.groupBy("window_start").pivot("brand").agg(
            first("sentiment_score").alias("sentiment_score"),
            first("mention_count").alias("mention_count"),
            first("positive_ratio").alias("positive_ratio")
        )

        # Calculate relative performance
        for competitor in competitors:
            if competitor in pivot_df.columns:
                # Sentiment difference
                pivot_df = pivot_df.withColumn(
                    f"{target_brand}_vs_{competitor}_sentiment",
                    col(f"{target_brand}_sentiment_score") - col(f"{competitor}_sentiment_score")
                )

                # Mention share
                pivot_df = pivot_df.withColumn(
                    f"{target_brand}_mention_share",
                    col(f"{target_brand}_mention_count") /
                    (col(f"{target_brand}_mention_count") + col(f"{competitor}_mention_count"))
                )

        return pivot_df.orderBy("window_start")

    def analyze_feature_sentiment(self, df: DataFrame,
                                  features: List[str]) -> DataFrame:
        """
        Analyze sentiment for specific product features

        Args:
            df: DataFrame with brand and product data
            features: List of features to analyze

        Returns:
            DataFrame with feature-level sentiment
        """
        logger.info(f"Analyzing sentiment for features: {features}")

        # Create feature detection UDF
        def detect_features(text):
            if text is None:
                return []
            text_lower = text.lower()
            detected = []
            for feature in features:
                if feature.lower() in text_lower:
                    detected.append(feature)
            return detected

        detect_features_udf = udf(detect_features, ArrayType(StringType()))

        # Detect features in text
        df_features = df.withColumn(
            "detected_features",
            detect_features_udf(col("text"))
        ).filter(size(col("detected_features")) > 0)

        # Explode features and brands
        df_analysis = df_features.select(
            col("sentiment"),
            col("vader_compound"),
            explode(col("brands")).alias("brand_info"),
            explode(col("detected_features")).alias("feature")
        )

        # Parse brand
        df_analysis = df_analysis.withColumn(
            "brand",
            expr("split(brand_info, ':')[0]")
        )

        # Aggregate by brand and feature
        feature_sentiment = df_analysis.groupBy("brand", "feature").agg(
            count("*").alias("mention_count"),
            avg("sentiment").alias("avg_sentiment"),
            avg("vader_compound").alias("avg_vader_compound"),
            spark_sum(when(col("sentiment") == 1, 1).otherwise(0)).alias("positive_mentions"),
            spark_sum(when(col("sentiment") == 0, 1).otherwise(0)).alias("negative_mentions")
        )

        # Calculate sentiment score
        feature_sentiment = feature_sentiment.withColumn(
            "sentiment_score",
            (col("avg_sentiment") * 2 - 1) * 100
        ).withColumn(
            "positive_ratio",
            col("positive_mentions") / col("mention_count")
        )

        return feature_sentiment.orderBy("brand", "feature")

    def calculate_share_of_voice(self, df: DataFrame,
                                 industry: Optional[str] = None,
                                 time_window: str = "1 day") -> DataFrame:
        """
        Calculate share of voice (mention share) for brands

        Args:
            df: DataFrame with brand mentions
            industry: Optional industry filter
            time_window: Time window for calculation

        Returns:
            DataFrame with share of voice metrics
        """
        logger.info("Calculating share of voice")

        # Get brands to analyze
        if industry:
            brands = [b for b, data in self.brand_recognizer.brands.items()
                      if data['industry'] == industry]
        else:
            brands = list(self.brand_recognizer.brands.keys())

        # Aggregate brand mentions
        brand_mentions = self.aggregate_brand_sentiment(df, time_window)

        # Calculate total mentions per window
        window_totals = brand_mentions.groupBy("window_start").agg(
            spark_sum("mention_count").alias("total_mentions")
        )

        # Join and calculate share
        sov_df = brand_mentions.join(
            window_totals,
            on="window_start"
        ).withColumn(
            "share_of_voice",
            col("mention_count") / col("total_mentions") * 100
        )

        # Add rank within window
        window_spec = Window.partitionBy("window_start").orderBy(col("mention_count").desc())
        sov_df = sov_df.withColumn(
            "rank",
            expr("row_number() over (partition by window_start order by mention_count desc)")
        )

        return sov_df.select(
            "window_start", "brand", "mention_count",
            "share_of_voice", "rank", "sentiment_score"
        ).orderBy("window_start", "rank")

    def compute_sentiment_momentum(self, df: DataFrame,
                                   lookback_windows: int = 7) -> DataFrame:
        """
        Compute sentiment momentum and velocity for brands

        Args:
            df: DataFrame with brand sentiment over time
            lookback_windows: Number of windows for momentum calculation

        Returns:
            DataFrame with momentum indicators
        """
        logger.info("Computing sentiment momentum")

        # Aggregate daily sentiment
        daily_sentiment = self.aggregate_brand_sentiment(df, "1 day")

        # Define window for calculations
        window_spec = Window.partitionBy("brand").orderBy("window_start")

        # Calculate moving averages
        for i in [3, 7, 14]:
            daily_sentiment = daily_sentiment.withColumn(
                f"sentiment_ma_{i}",
                avg("sentiment_score").over(
                    window_spec.rowsBetween(-(i - 1), 0)
                )
            )

        # Calculate momentum (rate of change)
        daily_sentiment = daily_sentiment.withColumn(
            "sentiment_prev",
            lag("sentiment_score", lookback_windows).over(window_spec)
        ).withColumn(
            "sentiment_momentum",
            when(col("sentiment_prev").isNotNull(),
                 (col("sentiment_score") - col("sentiment_prev")) / lookback_windows
                 ).otherwise(0)
        )

        # Calculate velocity (acceleration)
        daily_sentiment = daily_sentiment.withColumn(
            "momentum_prev",
            lag("sentiment_momentum", 1).over(window_spec)
        ).withColumn(
            "sentiment_velocity",
            when(col("momentum_prev").isNotNull(),
                 col("sentiment_momentum") - col("momentum_prev")
                 ).otherwise(0)
        )

        # Classify trend
        daily_sentiment = daily_sentiment.withColumn(
            "trend",
            when(col("sentiment_momentum") > 0.5, "strong_positive")
            .when(col("sentiment_momentum") > 0, "positive")
            .when(col("sentiment_momentum") < -0.5, "strong_negative")
            .when(col("sentiment_momentum") < 0, "negative")
            .otherwise("neutral")
        )

        # Add signals
        daily_sentiment = daily_sentiment.withColumn(
            "momentum_signal",
            when(
                (col("sentiment_momentum") > 0) &
                (col("sentiment_velocity") > 0),
                "accelerating_positive"
            ).when(
                (col("sentiment_momentum") > 0) &
                (col("sentiment_velocity") < 0),
                "decelerating_positive"
            ).when(
                (col("sentiment_momentum") < 0) &
                (col("sentiment_velocity") < 0),
                "accelerating_negative"
            ).when(
                (col("sentiment_momentum") < 0) &
                (col("sentiment_velocity") > 0),
                "decelerating_negative"
            ).otherwise("stable")
        )

        return daily_sentiment.orderBy("brand", "window_start")

    def generate_competitive_insights(self, df: DataFrame,
                                      target_brand: str) -> Dict:
        """
        Generate comprehensive competitive insights

        Args:
            df: DataFrame with competitor analysis results
            target_brand: Brand to analyze

        Returns:
            Dictionary of insights
        """
        logger.info(f"Generating competitive insights for {target_brand}")

        insights = {
            "brand": target_brand,
            "analysis_date": datetime.now().isoformat(),
            "metrics": {},
            "competitors": {},
            "opportunities": [],
            "threats": []
        }

        # Get overall metrics
        brand_metrics = df.filter(col("brand") == target_brand.lower()).agg(
            avg("sentiment_score").alias("avg_sentiment"),
            avg("mention_count").alias("avg_mentions"),
            avg("share_of_voice").alias("avg_sov")
        ).collect()[0]

        insights["metrics"] = {
            "average_sentiment": float(brand_metrics["avg_sentiment"]),
            "average_mentions": float(brand_metrics["avg_mentions"]),
            "average_share_of_voice": float(brand_metrics["avg_sov"])
        }

        # Analyze competitors
        competitors = self.brand_recognizer.competitor_map.get(target_brand.lower(), set())

        for competitor in competitors:
            comp_metrics = df.filter(col("brand") == competitor).agg(
                avg("sentiment_score").alias("avg_sentiment"),
                avg("mention_count").alias("avg_mentions")
            ).collect()

            safe_float = lambda x: float(x) if x is not None else 0.0

            if comp_metrics:
                comp_data = comp_metrics[0]
                insights["competitors"][competitor] = {
                    "average_sentiment": safe_float(comp_data["avg_sentiment"]),
                    "average_mentions": safe_float(comp_data["avg_mentions"]),
                    "sentiment_gap": safe_float(brand_metrics["avg_sentiment"]) - safe_float(comp_data["avg_sentiment"])
                }

                # Identify opportunities and threats
                if safe_float(comp_data["avg_sentiment"]) < safe_float(brand_metrics["avg_sentiment"]) - 10:
                    insights["opportunities"].append({
                        "type": "sentiment_advantage",
                        "competitor": competitor,
                        "gap": safe_float(brand_metrics["avg_sentiment"]) - safe_float(comp_data["avg_sentiment"])
                    })
                elif safe_float(comp_data["avg_sentiment"]) > safe_float(brand_metrics["avg_sentiment"]) + 10:
                    insights["threats"].append({
                        "type": "sentiment_disadvantage",
                        "competitor": competitor,
                        "gap": safe_float(comp_data["avg_sentiment"]) - safe_float(brand_metrics["avg_sentiment"])
                    })

        return insights


# Main function for testing
def main():
    """Test competitor analysis functionality"""
    from config.spark_config import create_spark_session
    # >>> keep heavy ML libs out of the driver <<<
    # (we will import BrandRecognizer later, after Spark has forked)
    # from src.ml.entity_recognition import BrandRecognizer, create_brand_recognition_udf

    # Create Spark session
    spark = create_spark_session("CompetitorAnalysis")

    # Initialize brand recognizer
    # brand_recognizer = BrandRecognizer()
    # brand_udf = create_brand_recognition_udf(brand_recognizer)
    # Build a *lazy* UDF (no heavy import yet) instead
    brand_udf = create_brand_recognition_udf()

    # Load data
    logger.info("Loading data...")
    df = spark.read.parquet(str(get_path("data/processed/pipeline_features")))

    # Add brand recognition
    df = df.withColumn("brands", brand_udf(col("text")))

    # Filter to records with brands
    df_brands = df.filter(size(col("brands")) > 0)

    '''
        Now (after Spark has already forked its workers) we can
        safely create a lightweight BrandRecognizer for the driver.
    '''
    from src.ml.entity_recognition import BrandRecognizer

    # Initialize analyzer
    brand_recognizer = BrandRecognizer(use_spacy=False)
    analyzer = CompetitorAnalyzer(spark, brand_recognizer)

    # We only need the competitor_map & brand list on the driver.
    # BrandRecognizer has a flag that skips spaCy ⇒ no MPSGraph.


    # Test functions
    # 1. Aggregate brand sentiment
    brand_sentiment = analyzer.aggregate_brand_sentiment(df_brands, "1 day")
    logger.info("Brand sentiment aggregation:")
    brand_sentiment.show(20)

    # 2. Compare competitors
    comparison = analyzer.compare_competitor_sentiment(df_brands, "apple", "1 day")
    logger.info("Competitor comparison:")
    comparison.show(10)

    # 3. Share of voice
    sov = analyzer.calculate_share_of_voice(df_brands, industry="technology")
    logger.info("Share of voice:")
    sov.show(20)

    # 4. Sentiment momentum
    momentum = analyzer.compute_sentiment_momentum(df_brands)
    logger.info("Sentiment momentum:")
    momentum.filter(col("brand") == "apple").show(10)

    # 5. Generate insights
    insights = analyzer.generate_competitive_insights(sov, "apple")
    logger.info("Competitive insights:")
    print(json.dumps(insights, indent=2))

    # Save results
    output_path = str(get_path("data/analytics/competitor_analysis"))
    brand_sentiment.coalesce(1).write.mode("overwrite").parquet(f"{output_path}/brand_sentiment")
    sov.coalesce(1).write.mode("overwrite").parquet(f"{output_path}/share_of_voice")
    momentum.coalesce(1).write.mode("overwrite").parquet(f"{output_path}/sentiment_momentum")

    spark.stop()


if __name__ == "__main__":
    main()