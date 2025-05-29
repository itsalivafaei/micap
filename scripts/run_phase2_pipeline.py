"""
Complete Phase 2 Pipeline Runner
Executes competitor analysis and trend detection
"""

import os
import sys
import time
import json
from src.utils.path_utils import get_path

import logging
from pyspark.sql.functions import (
    col, size, count, avg, when, window, lit,
    sum as spark_sum, abs as spark_abs, explode, expr
)
from config.spark_config import create_spark_session
from src.ml.entity_recognition import (
    BrandRecognizer, create_brand_recognition_udf,
    create_product_extraction_udf, ProductExtractor
)
from src.spark.competitor_analysis import CompetitorAnalyzer
from src.ml.trend_detection import (
    TopicModeler, TrendForecaster, AnomalyDetector, ViralityPredictor
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_phase2_pipeline(sample_size: float = 0.01):
    """
    Run complete Phase 2 pipeline

    Args:
        sample_size: Fraction of data to process
    """
    start_time = time.time()

    # Create output directories
    os.makedirs(str(get_path("data/analytics/competitor_analysis")), exist_ok=True)
    os.makedirs(str(get_path("data/analytics/trends")), exist_ok=True)
    os.makedirs(str(get_path("data/analytics/insights")), exist_ok=True)
    os.makedirs(str(get_path("data/visualizations/phase2")), exist_ok=True)

    # Initialize Spark
    spark = create_spark_session("Phase2Pipeline")

    try:
        # Load processed data from Phase 1
        logger.info("Loading Phase 1 data...")
        df = spark.read.parquet(str(get_path("data/processed/pipeline_features")))

        # Sample for testing
        df_sample = df.sample(sample_size)
        # print(sample_size)
        logger.info(f"Processing {df_sample.count()} records")

        # Step 1: Entity Recognition
        logger.info("=" * 50)
        logger.info("Step 1: Entity Recognition")
        logger.info("=" * 50)

        brand_recognizer = BrandRecognizer()
        product_extractor = ProductExtractor(brand_recognizer)

        brand_udf = create_brand_recognition_udf(brand_recognizer)
        product_udf = create_product_extraction_udf(product_extractor)

        df_entities = df_sample.withColumn(
            "brands", brand_udf(col("text"))
        ).withColumn(
            "products", product_udf(col("text"))
        )

        # Filter to tweets with brands
        df_brands = df_entities.filter(size(col("brands")) > 0)
        brand_count = df_brands.count()
        logger.info(f"Found {brand_count} tweets with brand mentions")

        # Step 2: Competitor Analysis
        logger.info("=" * 50)
        logger.info("Step 2: Competitor Analysis")
        logger.info("=" * 50)

        analyzer = CompetitorAnalyzer(spark, brand_recognizer)

        # Aggregate brand sentiment
        brand_sentiment = analyzer.aggregate_brand_sentiment(df_brands)
        brand_sentiment.coalesce(1).write.mode("overwrite").parquet(
            "data/analytics/competitor_analysis/brand_sentiment"
        )

        # Calculate share of voice
        sov = analyzer.calculate_share_of_voice(df_brands)
        sov.coalesce(1).write.mode("overwrite").parquet(
            str(get_path("data/analytics/competitor_analysis/share_of_voice"))
        )

        # Compute sentiment momentum
        momentum = analyzer.compute_sentiment_momentum(df_brands)
        momentum.coalesce(1).write.mode("overwrite").parquet(
            str(get_path("data/analytics/competitor_analysis/sentiment_momentum"))
        )

        # Generate insights for top brands
        top_brands = sov.select("brand").distinct().limit(5).collect()
        insights = {}

        for row in top_brands:
            brand = row["brand"]
            brand_insights = analyzer.generate_competitive_insights(sov, brand)
            insights[brand] = brand_insights

        # Save insights
        with open(str(get_path("data/analytics/insights/competitive_insights.json")), "w") as f:
            json.dump(insights, f, indent=2)

        # Step 3: Trend Detection
        logger.info("=" * 50)
        logger.info("Step 3: Trend Detection")
        logger.info("=" * 50)

        # Topic modeling
        topic_modeler = TopicModeler(spark, num_topics=15)
        topic_modeler.fit_topics(df_sample)

        # Get topic descriptions
        topics = topic_modeler.get_topics()
        with open("data/analytics/trends/topics.json", "w") as f:
            # Convert to serializable format - topics is already a dict of lists
            topics_serializable = {
                str(k): [(w, float(score)) for w, score in v[:20]]
                for k, v in topics.items()
            }
            json.dump(topics_serializable, f, indent=2)

        # Transform data with topics
        df_topics = topic_modeler.transform(df_sample)

        # Step 4: Trend Forecasting
        logger.info("=" * 50)
        logger.info("Step 4: Trend Forecasting")
        logger.info("=" * 50)

        # Initialize trend forecaster
        forecaster = TrendForecaster(spark)

        # Forecast sentiment trends
        sentiment_forecast = forecaster.forecast_sentiment_trends(df_sample)
        sentiment_forecast.coalesce(1).write.mode("overwrite").parquet(
            str(get_path("data/analytics/trends/sentiment_forecast"))
        )

        # Forecast topic trends
        topic_forecast = forecaster.forecast_topic_trends(df_topics, topic_modeler)
        topic_forecast.coalesce(1).write.mode("overwrite").parquet(
            str(get_path("data/analytics/trends/topic_forecast"))
        )

        # Step 5: Anomaly Detection
        logger.info("=" * 50)
        logger.info("Step 5: Anomaly Detection")
        logger.info("=" * 50)

        # Initialize anomaly detector
        anomaly_detector = AnomalyDetector(spark)

        # For sentiment anomalies, we need to:
        # 1. Explode the brands array to get individual brand entries
        # 2. Parse the brand name from the "brand:confidence" format
        from pyspark.sql.functions import explode, expr

        # Explode brands and extract brand name
        df_brands_exploded = df_brands.select(
            "*",
            explode(col("brands")).alias("brand_info")
        ).withColumn(
            "brand",
            expr("split(brand_info, ':')[0]")
        ).withColumn(
            "brand_confidence",
            expr("cast(split(brand_info, ':')[1] as float)")
        )

        # Now aggregate by brand with the properly structured data
        sentiment_agg = df_brands_exploded.groupBy(
            window("timestamp", "1 hour"),
            "brand"
        ).agg(
            count("*").alias("mention_count"),
            avg("sentiment").alias("sentiment_score"),
            avg("vader_compound").alias("avg_vader_compound"),
            spark_sum(when(col("sentiment") == 1, 1).otherwise(0)).alias("positive_count"),
            spark_sum(when(col("sentiment") == 0, 1).otherwise(0)).alias("negative_count")
        ).withColumn(
            "window_start", col("window.start")
        ).withColumn(
            "positive_ratio",
            when(col("mention_count") > 0,
                 col("positive_count") / col("mention_count")
                 ).otherwise(0)
        ).drop("window")

        # Detect sentiment anomalies with the properly aggregated data
        sentiment_anomalies = anomaly_detector.detect_sentiment_anomalies(
            sentiment_agg,
            features=["sentiment_score", "mention_count", "positive_ratio"]
        )

        # Save sentiment anomalies
        sentiment_anomalies.coalesce(1).write.mode("overwrite").parquet(
            str(get_path("data/analytics/trends/sentiment_anomalies"))
        )

        # Detect volume anomalies using the sample data directly
        volume_anomalies = anomaly_detector.detect_volume_anomalies(df_sample)
        volume_anomalies.coalesce(1).write.mode("overwrite").parquet(
            str(get_path("data/analytics/trends/volume_anomalies"))
        )

        # Step 6: Virality Prediction
        logger.info("=" * 50)
        logger.info("Step 6: Virality Prediction")
        logger.info("=" * 50)

        # Initialize virality predictor
        virality_predictor = ViralityPredictor(spark)

        # Since Sentiment140 doesn't have engagement metrics, we'll simulate based on content
        # Create proxy metrics based on text features
        df_virality = df_sample.withColumn(
            "engagement_proxy",
            (col("exclamation_count") + col("question_count") +
             size(col("hashtags")) + col("emoji_sentiment").cast("double"))
        ).withColumn(
            "viral_potential_score",
            (col("engagement_proxy") * 0.3 +
             spark_abs(col("vader_compound")) * 0.4 +
             when(size(col("hashtags")) > 0, 0.3).otherwise(0))
        )

        # Filter potential viral content
        viral_content = df_virality.filter(col("viral_potential_score") > 0.7)
        viral_count = viral_content.count()
        logger.info(f"Identified {viral_count} potentially viral tweets")

        # Save viral content
        viral_content.select(
            "tweet_id", "text", "sentiment", "viral_potential_score",
            "hashtags", "vader_compound", "emoji_sentiment"
        ).coalesce(1).write.mode("overwrite").parquet(
            str(get_path("data/analytics/trends/viral_content"))
        )

        # Create summary instead of calling analyze_viral_patterns
        viral_patterns = {
            "total_viral_content": viral_count,
            "avg_viral_score": viral_content.agg(
                avg("viral_potential_score")
            ).collect()[0][0] if viral_count > 0 else 0,
            "viral_by_sentiment": viral_content.groupBy("sentiment").count().collect()
        }

        # Step 7: Generate Summary Report
        logger.info("=" * 50)
        logger.info("Step 7: Generating Summary Report")
        logger.info("=" * 50)

        summary = {
            "pipeline_run": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sample_size": sample_size,
                "total_records_processed": df_sample.count(),
                "execution_time_seconds": time.time() - start_time
            },
            "entity_recognition": {
                "tweets_with_brands": brand_count,
                "unique_brands": df_brands.select("brands").distinct().count(),
                "brand_coverage": brand_count / df_sample.count()
            },
            "competitor_analysis": {
                "brands_analyzed": len(top_brands),
                "sentiment_metrics_generated": True,
                "share_of_voice_calculated": True,
                "momentum_tracked": True
            },
            # Update the trend_detection part of summary
            "trend_detection": {
                "topics_discovered": len(topics),
                "sentiment_anomalies": sentiment_anomalies.filter(col("is_anomaly") == 1).count(),
                "volume_anomalies": volume_anomalies.filter(col("is_volume_anomaly") == 1).count(),
                "viral_content_identified": viral_count
            },
            "outputs": {
                "competitor_analysis": str(get_path("data/analytics/competitor_analysis/")),
                "trend_data": str(get_path("data/analytics/trends/")),
                "insights": str(get_path("data/analytics/insights/"))
            }
        }

        # Save summary
        with open("data/analytics/phase2_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Display key metrics
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2 PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total execution time: {summary['pipeline_run']['execution_time_seconds']:.2f} seconds")
        logger.info(f"Records processed: {summary['pipeline_run']['total_records_processed']:,}")
        logger.info(f"Brand mentions found: {summary['entity_recognition']['tweets_with_brands']:,}")
        logger.info(f"Topics discovered: {summary['trend_detection']['topics_discovered']}")
        logger.info(f"Anomalies detected: {summary['trend_detection']['sentiment_anomalies']}")
        logger.info(f"Viral content identified: {summary['trend_detection']['viral_content_identified']}")

        # Sample outputs for verification
        logger.info("\nSample Brand Sentiment:")
        # brand_sentiment.select("brand", "avg_sentiment", "tweet_count").show(5)
        brand_sentiment.select("brand", "avg_sentiment", "mention_count").show(5)

        logger.info("\nTop Viral Content:")
        viral_content.select("text", "viral_potential_score", "sentiment").show(3, truncate=False)

        logger.info("\nPhase 2 pipeline completed successfully!")
        logger.info(f"Results saved to data/analytics/")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Stop Spark session
        spark.stop()


if __name__ == "__main__":
    # Run with a configurable sample size
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 2 Pipeline")
    parser.add_argument(
        "--sample-size",
        type=float,
        default=0.95,
        help="Fraction of data to process (default: 0.95)"
    )

    args = parser.parse_args()

    # Validate sample size
    if not 0 < args.sample_size <= 1.0:
        logger.error("Sample size must be between 0 and 1")
        sys.exit(1)

    # Run pipeline
    run_phase2_pipeline(sample_size=args.sample_size)