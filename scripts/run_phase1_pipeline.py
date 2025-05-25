"""
Complete Phase 1 Pipeline Runner
Executes all data processing steps in sequence
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.spark_config import create_spark_session
from src.spark.data_ingestion import DataIngestion
from src.spark.preprocessing import TextPreprocessor
from src.spark.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(sample_fraction: float = 0.01):
    """
    Run complete Phase 1 pipeline

    Args:
        sample_fraction: Fraction of data to process (0.01 = 1%)
    """
    start_time = time.time()

    # Create Spark session
    logger.info("Initializing Spark session...")
    spark = create_spark_session("Phase1_Pipeline")

    try:
        # Step 1: Data Ingestion
        logger.info("=" * 50)
        logger.info("STEP 1: Data Ingestion")
        logger.info("=" * 50)

        ingestion = DataIngestion(spark)
        data_path = "/Users/ali/Documents/Projects/micap/data/raw/testdata.manual.2009.06.14.csv"

        # Load full dataset
        df_raw = ingestion.load_sentiment140_data(data_path)
        logger.info(f"Loaded {df_raw.count()} total records")

        # Validate data quality
        df_clean, quality_metrics = ingestion.validate_data_quality(df_raw)

        # Create sample for processing
        df_sample = ingestion.create_sample_dataset(df_clean, sample_size=sample_fraction)

        # Save clean sample
        sample_path = "/Users/ali/Documents/Projects/micap/data/processed/pipeline_sample"
        ingestion.save_to_local_storage(df_sample, sample_path)

        # Step 2: Text Preprocessing
        logger.info("=" * 50)
        logger.info("STEP 2: Text Preprocessing")
        logger.info("=" * 50)

        preprocessor = TextPreprocessor(spark)
        df_preprocessed = preprocessor.preprocess_pipeline(df_sample)

        # Save preprocessed data
        preprocessed_path = "/Users/ali/Documents/Projects/micap/data/processed/pipeline_preprocessed"
        df_preprocessed.coalesce(4).write.mode("overwrite").parquet(preprocessed_path)
        logger.info(f"Preprocessed {df_preprocessed.count()} records")

        # Step 3: Feature Engineering
        logger.info("=" * 50)
        logger.info("STEP 3: Feature Engineering")
        logger.info("=" * 50)

        feature_engineer = FeatureEngineer(spark)
        df_features = feature_engineer.create_all_features(df_preprocessed)

        # Save featured data
        features_path = "/Users/ali/Documents/Projects/micap/data/processed/pipeline_features"
        df_features.coalesce(4).write.mode("overwrite").parquet(features_path)
        logger.info(f"Created features for {df_features.count()} records")

        # Save feature statistics
        stats_path = "data/processed/pipeline_feature_stats"
        stats_df = feature_engineer.save_feature_stats(df_features, stats_path)

        # Display summary
        logger.info("=" * 50)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        logger.info(f"Records processed: {df_features.count()}")
        logger.info(f"Features created: Multiple feature sets including TF-IDF and Word2Vec")

        # Show sample results
        logger.info("\nSample processed records:")
        df_features.select(
            "tweet_id", "sentiment", "text_length",
            "vader_compound", "emoji_sentiment"
        ).show(5)

        # Show feature statistics
        logger.info("\nFeature statistics:")
        stats_df.show()

        logger.info("\nPhase 1 pipeline completed successfully!")
        logger.info(f"Results saved to: {features_path}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        # Stop Spark session
        spark.stop()


if __name__ == "__main__":
    # Run with 1% sample for testing
    # Increase to 1.0 for full dataset processing
    run_pipeline(sample_fraction=0.01)