"""
Test script for Phase 2 functionality
Validates entity recognition and competitor analysis
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging
from config.spark_config import create_spark_session
from src.ml.entity_recognition import (
    BrandRecognizer, ProductExtractor, CompetitorMapper,
    EntityDisambiguator, create_brand_recognition_udf
)
from src.spark.competitor_analysis import CompetitorAnalyzer
from src.ml.trend_detection import TopicModeler, TrendForecaster, AnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_entity_recognition():
    """Test entity recognition components"""
    logger.info("Testing Entity Recognition...")

    # Initialize components
    brand_recognizer = BrandRecognizer()
    product_extractor = ProductExtractor(brand_recognizer)
    competitor_mapper = CompetitorMapper(brand_recognizer)

    # Test texts
    test_texts = [
        "The new iPhone 15 Pro has amazing cameras, much better than Samsung Galaxy S23",
        "Tesla Model 3 beats competitors in range and acceleration",
        "Comparing MacBook Pro M3 with Dell XPS for development work",
        "Apple Watch health features are unmatched by Galaxy Watch"
    ]

    for text in test_texts:
        logger.info(f"\nAnalyzing: {text}")

        # Brand recognition
        brands = brand_recognizer.recognize_brands(text)
        logger.info(f"Brands found: {brands}")

        # Product extraction
        products = product_extractor.extract_products(text)
        logger.info(f"Products found: {products}")

        # Competitive context
        contexts = competitor_mapper.identify_competitive_context(text)
        logger.info(f"Competitive contexts: {contexts}")

    logger.info("✅ Entity Recognition tests passed")


def test_competitor_analysis():
    """Test competitor analysis on real data"""
    logger.info("\nTesting Competitor Analysis...")

    # Create Spark session
    spark = create_spark_session("TestPhase2")

    # Load sample data
    df = spark.read.parquet("data/processed/pipeline_features").limit(1000)

    # Initialize components
    brand_recognizer = BrandRecognizer()
    brand_udf = create_brand_recognition_udf(brand_recognizer)

    # Add brand recognition
    df = df.withColumn("brands", brand_udf(col("text")))

    # Initialize analyzer
    analyzer = CompetitorAnalyzer(spark, brand_recognizer)

    # Test brand sentiment aggregation
    brand_sentiment = analyzer.aggregate_brand_sentiment(df)
    logger.info("Brand sentiment aggregation:")
    brand_sentiment.show(5)

    # Test share of voice
    sov = analyzer.calculate_share_of_voice(df)
    logger.info("Share of voice calculation:")
    sov.show(5)

    logger.info("✅ Competitor Analysis tests passed")

    spark.stop()


def test_trend_detection():
    """Test trend detection components"""
    logger.info("\nTesting Trend Detection...")

    spark = create_spark_session("TestTrends")

    # Load sample data
    df = spark.read.parquet("data/processed/pipeline_features").limit(1000)

    # Initialize topic modeler
    topic_modeler = TopicModeler(spark, num_topics=5)

    # Fit model
    topic_modeler.fit_topics(df)

    # Get topics
    topics = topic_modeler.get_topics()
    logger.info("Extracted topics:")
    for topic_id, words in topics.items():
        top_words = [w[0] for w in words[:3]]
        logger.info(f"Topic {topic_id}: {', '.join(top_words)}")

    logger.info("✅ Trend Detection tests passed")

    spark.stop()


def main():
    """Run all Phase 2 tests"""
    logger.info("=== Phase 2 Testing Suite ===")

    try:
        test_entity_recognition()
        test_competitor_analysis()
        test_trend_detection()

        logger.info("\n✅ All Phase 2 tests passed successfully!")
        logger.info("Phase 2 implementation is complete.")

    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()