"""
Complete Phase 1 Pipeline Runner - ENHANCED VERSION
Executes all data processing steps in sequence with better error handling
FIXED: Added environment setup and numpy import debugging
"""

import os
import sys
import time
import logging
from pathlib import Path
from src.utils.path_utils import get_path

# FIX: Set environment variables before any imports
python_exec = sys.executable
os.environ["PYSPARK_PYTHON"] = python_exec
os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# FIX: Test numpy import before anything else
try:
    import numpy as np
    print(f"✓ Numpy imported successfully. Version: {np.__version__}")
    print(f"✓ Numpy location: {np.__file__}")
except ImportError as e:
    print(f"✗ Numpy import failed: {e}")
    print("Please fix numpy installation before continuing.")
    sys.exit(1)

from config.spark_config import create_spark_session
from src.spark.data_ingestion import DataIngestion
from src.spark.preprocessing import TextPreprocessor

# FIX: Import feature engineering with better error handling
try:
    from src.spark.feature_engineering import FeatureEngineer
    print("✓ Feature engineering module imported successfully")
except ImportError as e:
    print(f"✗ Feature engineering import failed: {e}")
    print("This is likely due to PySpark ML import issues.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(sample_fraction: float = 0.01):
    """
    Run complete Phase 1 pipeline with enhanced error handling

    Args:
        sample_fraction: Fraction of data to process (0.01 = 1%)
    """
    start_time = time.time()

    # Create Spark session with fixed configuration
    logger.info("Initializing Spark session...")
    logger.info(f"Using Python: {sys.executable}")

    try:
        spark = create_spark_session("Phase1_Pipeline")
        logger.info("✓ Spark session created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create Spark session: {e}")
        return

    try:
        # Step 1: Data Ingestion
        logger.info("=" * 50)
        logger.info("STEP 1: Data Ingestion")
        logger.info("=" * 50)

        ingestion = DataIngestion(spark)
        # data_path = "/Users/ali/Documents/Projects/micap/data/raw/testdata.manual.2009.06.14.csv"
        data_path = str(get_path("data/raw/training.1600000.processed.noemoticon.csv"))

        # Load full dataset
        df_raw = ingestion.load_sentiment140_data(data_path)
        logger.info(f"Loaded {df_raw.count()} total records")

        # Validate data quality
        df_clean, quality_metrics = ingestion.validate_data_quality(df_raw)

        # Create sample for processing
        df_sample = ingestion.create_sample_dataset(df_clean, sample_size=sample_fraction)

        # Save clean sample
        sample_path = str(get_path("data/processed/pipeline_sample"))
        ingestion.save_to_local_storage(df_sample, sample_path)
        logger.info("✓ Step 1 completed successfully")

        # Step 2: Text Preprocessing
        logger.info("=" * 50)
        logger.info("STEP 2: Text Preprocessing")
        logger.info("=" * 50)

        preprocessor = TextPreprocessor(spark)
        df_preprocessed = preprocessor.preprocess_pipeline(df_sample)

        # Save preprocessed data
        preprocessed_path = str(get_path("data/processed/pipeline_preprocessed"))
        df_preprocessed.coalesce(4).write.mode("overwrite").parquet(preprocessed_path)
        logger.info(f"Preprocessed {df_preprocessed.count()} records")
        logger.info("✓ Step 2 completed successfully")

        # Step 3: Feature Engineering (the problem area)
        logger.info("=" * 50)
        logger.info("STEP 3: Feature Engineering")
        logger.info("=" * 50)

        try:
            # Test if we can create the feature engineer without issues
            feature_engineer = FeatureEngineer(spark)
            logger.info("✓ FeatureEngineer created successfully")

            # Try creating features
            df_features = feature_engineer.create_all_features(df_preprocessed)
            logger.info("✓ Features created successfully")

            # Save featured data
            features_path = str(get_path("data/processed/pipeline_features"))

            # FIX: Use explicit write with error handling
            try:
                df_features.coalesce(4).write.mode("overwrite").parquet(features_path)
                logger.info("✓ Features saved successfully")
            except Exception as save_error:
                logger.error(f"✗ Error saving features: {save_error}")
                # Try alternative save method
                logger.info("Trying alternative save method...")
                df_features.write.mode("overwrite").format("parquet").save(features_path)
                logger.info("✓ Features saved with alternative method")

            logger.info(f"Created features for {df_features.count()} records")

            # Save feature statistics (optional - skip if problematic)
            try:
                stats_path = str(get_path("data/processed/pipeline_feature_stats"))
                stats_df = feature_engineer.save_feature_stats(df_features, stats_path)
                logger.info("✓ Feature statistics saved")
            except Exception as stats_error:
                logger.warning(f"Could not save feature statistics: {stats_error}")

        except Exception as fe_error:
            logger.error(f"✗ Feature engineering failed: {fe_error}")
            logger.info("Attempting to continue with preprocessed data only...")

            # Fallback: save preprocessed data as final output
            fallback_path = str(get_path("data/processed/pipeline_features_fallback"))
            df_preprocessed.coalesce(4).write.mode("overwrite").parquet(fallback_path)
            logger.info(f"✓ Fallback data saved to: {fallback_path}")

        # Display summary
        logger.info("=" * 50)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")

        try:
            if 'df_features' in locals():
                logger.info(f"Records processed: {df_features.count()}")
                logger.info("Phase 1 pipeline completed successfully!")
                logger.info(f"Results saved to: {features_path}")

                # Show sample results
                logger.info("\nSample processed records:")
                df_features.select(
                    "tweet_id", "sentiment", "text_length",
                    "vader_compound", "emoji_sentiment"
                ).show(5)
            else:
                logger.info(f"Records preprocessed: {df_preprocessed.count()}")
                logger.info("Pipeline completed with preprocessing only")
        except Exception as summary_error:
            logger.warning(f"Could not display summary: {summary_error}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Stop Spark session
        try:
            spark.stop()
            logger.info("✓ Spark session stopped")
        except:
            pass


if __name__ == "__main__":
    # Run with 10% sample for testing
    # Increase to 1.0 for full dataset processing
    run_pipeline(sample_fraction=0.1)