"""
Complete Phase 2 Pipeline Runner
Executes competitor analysis and trend detection
"""

import os
import sys
import time
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging
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


def run_phase2_pipeline(sample_size: float = 0.95):
    """
    Run complete Phase 2 pipeline

    Args:
        sample_size: Fraction of data to process
    """
    start_time = time.time()

    # Create output directories
    os.makedirs("data/analytics/competitor_analysis", exist_ok=True)
    os.makedirs("data/analytics/trends", exist_ok=True)
    os.makedirs("data/analytics/insights", exist_ok=True)

    # Initialize Spark
    spark = create_spark_session("Phase2Pipeline")

    try:
        # Load processed data from Phase 1
        logger.info("Loading Phase 1 data...")
        df = spark.read.parquet("data/processed/pipeline_features")

        # Sample for testing
        df_sample = df.sample(sample_size)
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
            "data/analytics/competitor_analysis/share_of_voice"
        )

        # Compute sentiment momentum
        momentum = analyzer.compute_sentiment_momentum(df_brands)
        momentum.coalesce(1).write.mode("overwrite").parquet(
            "data/analytics/competitor_analysis/sentiment_momentum"
        )

        # Generate insights for top brands
        top_brands = sov.select("brand").distinct().limit(5).collect()
        insights = {}

        for row in top_brands:
            brand = row["brand"]
            brand_insights = analyzer.generate_competitive_insights(sov, brand)
            insights[brand] = brand_insights

        # Save insights
        with open("data/analytics/insights/competitive_insights.json", "w") as f:
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
            # Convert to serializable format
            topics_serializable = {
                str(k): [(w, float(s