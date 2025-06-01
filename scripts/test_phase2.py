"""
Test Phase 2 Pipeline Components
Quick tests to verify functionality before full run.
."""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, count, avg, when, window, lit, sum as spark_sum, abs as spark_abs, explode, expr
from config.spark_config import create_spark_session
from src.utils.path_utils import get_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading(spark, limit=100):
    """Test if we can load the data."""
    logger.info("Testing data loading...")
    try:
        df = spark.read.parquet(str(get_path("data/processed/pipeline_features")))
        logger.info(f"✓ Data loaded successfully. Schema:")
        df.printSchema()
        logger.info(f"✓ Total records: {df.count()}")
        return df.limit(limit)  # Return small sample
    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        return None

def test_brand_recognition(spark, df_sample):
    """Test brand recognition."""
    logger.info("\nTesting brand recognition...")
    try:
        from src.ml.entity_recognition import BrandRecognizer, create_brand_recognition_udf
        
        brand_recognizer = BrandRecognizer()
        brand_udf = create_brand_recognition_udf(brand_recognizer)
        
        # Test on a few records
        df_brands = df_sample.withColumn("brands", brand_udf(col("text")))
        
        # Show results
        logger.info("Sample brand detection:")
        df_brands.select("text", "brands").filter(size(col("brands")) > 0).show(5, truncate=False)
        
        brand_count = df_brands.filter(size(col("brands")) > 0).count()
        logger.info(f"✓ Found {brand_count} tweets with brands out of {df_sample.count()}")
        
        return df_brands
    except Exception as e:
        logger.error(f"✗ Brand recognition failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_anomaly_detection_data_prep(spark, df_brands):
    """Test the problematic anomaly detection data preparation."""
    logger.info("\nTesting anomaly detection data preparation...")
    try:
        # This is the part that was failing
        df_brands_exploded = df_brands.filter(size(col("brands")) > 0).select(
            "*",
            explode(col("brands")).alias("brand_info")
        ).withColumn(
            "brand",
            expr("split(brand_info, ':')[0]")
        ).withColumn(
            "brand_confidence", 
            expr("cast(split(brand_info, ':')[1] as float)")
        )
        
        logger.info("✓ Brand explosion successful. Sample:")
        df_brands_exploded.select("brand", "brand_confidence", "sentiment").show(10)
        
        # Test aggregation
        sentiment_agg = df_brands_exploded.groupBy(
            window("timestamp", "1 hour"),
            "brand"
        ).agg(
            count("*").alias("mention_count"),
            avg("sentiment").alias("sentiment_score")
        ).withColumn(
            "window_start", col("window.start")
        )
        
        logger.info("✓ Aggregation successful. Sample:")
        sentiment_agg.show(5)
        
        return True
    except Exception as e:
        logger.error(f"✗ Anomaly detection prep failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_topic_modeling(spark, df_sample):
    """Test topic modeling with tiny dataset."""
    logger.info("\nTesting topic modeling...")
    try:
        from src.ml.trend_detection import TopicModeler
        
        # Use very small parameters for testing
        topic_modeler = TopicModeler(spark, num_topics=3)
        topic_modeler.fit_topics(df_sample)
        topics = topic_modeler.get_topics()
        
        logger.info(f"✓ Topic modeling successful. Found {len(topics)} topics")
        for topic_id, words in topics.items():
            logger.info(f"  Topic {topic_id}: {[w[0] for w in words[:5]]}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Topic modeling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all component tests."""
    spark = create_spark_session("Phase2Testing")
    
    # Test 1: Load data
    df_sample = test_data_loading(spark, limit=100)
    if df_sample is None:
        logger.error("Cannot proceed without data")
        spark.stop()
        return
    
    # Test 2: Brand recognition
    df_brands = test_brand_recognition(spark, df_sample)
    if df_brands is None:
        logger.error("Brand recognition failed")
        spark.stop()
        return
    
    # Test 3: Anomaly detection prep (the problematic part)
    success = test_anomaly_detection_data_prep(spark, df_brands)
    if not success:
        logger.error("Anomaly detection data prep failed")
    
    # Test 4: Topic modeling
    test_topic_modeling(spark, df_sample)
    
    logger.info("\n" + "="*50)
    logger.info("Test Summary Complete")
    logger.info("="*50)
    
    spark.stop()

if __name__ == "__main__":
    run_all_tests()