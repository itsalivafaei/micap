"""
Complete Phase 1 Pipeline Runner - ENHANCED VERSION
Executes all data processing steps in sequence with improved error handling
FIXED: Enhanced serialization handling and recovery mechanisms
"""

import os
import sys
import time
import logging
import traceback
from pathlib import Path
from typing import Optional, Tuple, Any

# FIX: Set environment variables before any imports
python_exec = sys.executable
os.environ["PYSPARK_PYTHON"] = python_exec
os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.path_utils import get_path

# FIX: Test numpy import before anything else
try:
    import numpy as np
    print(f"✓ Numpy imported successfully. Version: {np.__version__}")
    print(f"✓ Numpy location: {np.__file__}")
except ImportError as e:
    print(f"✗ Numpy import failed: {e}")
    print("Please fix numpy installation before continuing.")
    sys.exit(1)

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase1_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import modules with better error handling
def safe_import(module_name: str, description: str) -> Any:
    """Safely import modules with descriptive error handling"""
    try:
        if module_name == "config.spark_config":
            from config.spark_config import create_spark_session
            return create_spark_session
        elif module_name == "data_ingestion":
            from src.spark.data_ingestion import DataIngestion
            return DataIngestion
        elif module_name == "preprocessing":
            from src.spark.preprocessing import TextPreprocessor
            return TextPreprocessor
        elif module_name == "feature_engineering":
            from src.spark.feature_engineering import FeatureEngineer
            return FeatureEngineer
    except ImportError as e:
        logger.error(f"✗ Failed to import {description}: {e}")
        logger.error(f"  This likely indicates missing dependencies or configuration issues.")
        raise ImportError(f"Critical dependency {description} not available: {e}")

# Import required components
try:
    create_spark_session = safe_import("config.spark_config", "Spark configuration")
    DataIngestion = safe_import("data_ingestion", "Data ingestion module")
    TextPreprocessor = safe_import("preprocessing", "Text preprocessing module")
    logger.info("✓ Core modules imported successfully")
except ImportError as e:
    logger.error(f"✗ Critical import failure: {e}")
    sys.exit(1)

# Try to import feature engineering with fallback
try:
    FeatureEngineer = safe_import("feature_engineering", "Feature engineering module")
    FEATURE_ENGINEERING_AVAILABLE = True
    logger.info("✓ Feature engineering module imported successfully")
except ImportError as e:
    logger.warning(f"⚠️  Feature engineering not available: {e}")
    logger.warning("  Pipeline will run without advanced feature generation")
    FEATURE_ENGINEERING_AVAILABLE = False


class PipelineRecovery:
    """Handles pipeline recovery and checkpoint management"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.checkpoints = {}
    
    def save_checkpoint(self, stage: str, df: Any, metadata: dict = None) -> bool:
        """Save pipeline checkpoint"""
        try:
            checkpoint_path = self.base_path / f"checkpoint_{stage}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Save DataFrame
            df.coalesce(4).write.mode("overwrite").parquet(str(checkpoint_path / "data"))
            
            # Save metadata
            if metadata:
                import json
                with open(checkpoint_path / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            self.checkpoints[stage] = str(checkpoint_path)
            logger.info(f"✓ Checkpoint saved for stage: {stage}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to save checkpoint for {stage}: {e}")
            return False
    
    def load_checkpoint(self, stage: str, spark) -> Optional[Any]:
        """Load pipeline checkpoint"""
        try:
            if stage in self.checkpoints:
                checkpoint_path = Path(self.checkpoints[stage])
                if (checkpoint_path / "data").exists():
                    df = spark.read.parquet(str(checkpoint_path / "data"))
                    logger.info(f"✓ Checkpoint loaded for stage: {stage}")
                    return df
            return None
        except Exception as e:
            logger.error(f"✗ Failed to load checkpoint for {stage}: {e}")
            return None


def validate_environment() -> bool:
    """Validate environment setup before pipeline execution"""
    logger.info("Validating environment setup...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("✗ Python 3.8+ required")
        return False
    logger.info(f"✓ Python version: {sys.version}")
    
    # Check required directories
    required_dirs = ['data', 'logs', 'config']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            logger.warning(f"⚠️  Creating missing directory: {dir_name}")
            dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directory exists: {dir_name}")
    
    return True


def run_pipeline(sample_fraction: float = 0.01, 
                enable_checkpoints: bool = True,
                resume_from_checkpoint: bool = False) -> Tuple[bool, dict]:
    """
    Run complete Phase 1 pipeline with enhanced error handling and recovery
    
    Args:
        sample_fraction: Fraction of data to process (0.01 = 1%)
        enable_checkpoints: Enable checkpoint saving for recovery
        resume_from_checkpoint: Try to resume from existing checkpoints
        
    Returns:
        Tuple of (success: bool, results: dict)
    """
    start_time = time.time()
    results = {
        'success': False,
        'stages_completed': [],
        'processing_time': 0,
        'records_processed': 0,
        'errors': []
    }
    
    # Validate environment
    if not validate_environment():
        results['errors'].append("Environment validation failed")
        return False, results
    
    logger.info("=" * 60)
    logger.info("STARTING PHASE 1 PIPELINE - ENHANCED VERSION")
    logger.info("=" * 60)
    logger.info(f"Sample fraction: {sample_fraction}")
    logger.info(f"Checkpoints enabled: {enable_checkpoints}")
    logger.info(f"Resume from checkpoint: {resume_from_checkpoint}")
    
    # Initialize recovery manager
    recovery = PipelineRecovery("data/checkpoints") if enable_checkpoints else None
    
    # Create Spark session with better error handling
    logger.info("Initializing Spark session...")
    logger.info(f"Using Python: {sys.executable}")
    
    try:
        spark = create_spark_session("Phase1_Pipeline_Enhanced")
        logger.info("✓ Spark session created successfully")
    except Exception as e:
        error_msg = f"Failed to create Spark session: {e}"
        logger.error(f"✗ {error_msg}")
        results['errors'].append(error_msg)
        return False, results

    try:
        # ========================================
        # STEP 1: Data Ingestion
        # ========================================
        logger.info("\n" + "=" * 50)
        logger.info("STEP 1: Data Ingestion")
        logger.info("=" * 50)
        
        df_sample = None
        if resume_from_checkpoint and recovery:
            df_sample = recovery.load_checkpoint("ingestion", spark)
        
        if df_sample is None:
            try:
                ingestion = DataIngestion(spark)
                data_path = str(get_path("data/raw/training.1600000.processed.noemoticon.csv"))
                
                if not Path(data_path).exists():
                    # Fallback to test data
                    test_data_path = str(get_path("data/raw/testdata.manual.2009.06.14.csv"))
                    if Path(test_data_path).exists():
                        data_path = test_data_path
                        logger.warning(f"⚠️  Using test dataset: {test_data_path}")
                    else:
                        raise FileNotFoundError("No dataset found for processing")
                
                # Load and validate data
                logger.info(f"Loading data from: {data_path}")
                df_raw = ingestion.load_sentiment140_data(data_path)
                total_records = df_raw.count()
                logger.info(f"Loaded {total_records:,} total records")
                
                # Validate data quality
                df_clean, quality_metrics = ingestion.validate_data_quality(df_raw)
                logger.info(f"Data quality validation completed")
                
                # Create sample for processing
                df_sample = ingestion.create_sample_dataset(df_clean, sample_size=sample_fraction)
                sample_count = df_sample.count()
                logger.info(f"Created sample with {sample_count:,} records")
                
                # Save sample data with better error handling
                try:
                    sample_path = str(get_path("data/processed/pipeline_sample"))
                    ingestion.save_to_local_storage(df_sample, sample_path)
                    logger.info(f"✓ Sample data saved to: {sample_path}")
                except Exception as save_error:
                    logger.warning(f"⚠️  Could not save sample data: {save_error}")
                
                # Save checkpoint
                if recovery:
                    recovery.save_checkpoint("ingestion", df_sample, {
                        'total_records': total_records,
                        'sample_records': sample_count,
                        'sample_fraction': sample_fraction
                    })
                    
            except Exception as e:
                error_msg = f"Data ingestion failed: {e}"
                logger.error(f"✗ {error_msg}")
                logger.error(traceback.format_exc())
                results['errors'].append(error_msg)
                raise
        
        results['stages_completed'].append('ingestion')
        results['records_processed'] = df_sample.count()
        logger.info("✓ Step 1 completed successfully")

        # ========================================
        # STEP 2: Text Preprocessing  
        # ========================================
        logger.info("\n" + "=" * 50)
        logger.info("STEP 2: Text Preprocessing")
        logger.info("=" * 50)
        
        df_preprocessed = None
        if resume_from_checkpoint and recovery:
            df_preprocessed = recovery.load_checkpoint("preprocessing", spark)
        
        if df_preprocessed is None:
            try:
                preprocessor = TextPreprocessor(spark)
                
                # Apply preprocessing with better error handling
                logger.info("Applying text preprocessing pipeline...")
                df_preprocessed = preprocessor.preprocess_pipeline(df_sample)
                
                # Validate preprocessing results
                preprocessed_count = df_preprocessed.count()
                logger.info(f"Preprocessed {preprocessed_count:,} records")
                
                if preprocessed_count == 0:
                    raise ValueError("Preprocessing resulted in empty dataset")
                
                # Save preprocessed data with retry logic
                try:
                    preprocessed_path = str(get_path("data/processed/pipeline_preprocessed"))
                    df_preprocessed.coalesce(4).write.mode("overwrite").parquet(preprocessed_path)
                    logger.info(f"✓ Preprocessed data saved to: {preprocessed_path}")
                except Exception as save_error:
                    logger.warning(f"⚠️  Primary save failed, trying alternative method: {save_error}")
                    df_preprocessed.write.mode("overwrite").format("parquet").save(preprocessed_path)
                    logger.info("✓ Preprocessed data saved with alternative method")
                
                # Save checkpoint
                if recovery:
                    recovery.save_checkpoint("preprocessing", df_preprocessed, {
                        'preprocessed_records': preprocessed_count
                    })
                    
            except Exception as e:
                error_msg = f"Text preprocessing failed: {e}"
                logger.error(f"✗ {error_msg}")
                logger.error(traceback.format_exc())
                results['errors'].append(error_msg)
                raise
        
        results['stages_completed'].append('preprocessing')
        logger.info("✓ Step 2 completed successfully")

        # ========================================
        # STEP 3: Feature Engineering (with fallback)
        # ========================================
        logger.info("\n" + "=" * 50)
        logger.info("STEP 3: Feature Engineering")
        logger.info("=" * 50)
        
        df_features = None
        if resume_from_checkpoint and recovery:
            df_features = recovery.load_checkpoint("feature_engineering", spark)
        
        if df_features is None and FEATURE_ENGINEERING_AVAILABLE:
            try:
                logger.info("Attempting advanced feature engineering...")
                feature_engineer = FeatureEngineer(spark)
                
                # Apply feature engineering with careful error handling
                df_features = feature_engineer.create_all_features(df_preprocessed)
                feature_count = df_features.count()
                logger.info(f"✓ Features created for {feature_count:,} records")
                
                # Save featured data with multiple attempts
                features_path = str(get_path("data/processed/pipeline_features"))
                
                try:
                    df_features.coalesce(4).write.mode("overwrite").parquet(features_path)
                    logger.info("✓ Feature data saved successfully")
                except Exception as save_error:
                    logger.warning(f"⚠️  Primary save failed: {save_error}")
                    try:
                        df_features.write.mode("overwrite").format("parquet").save(features_path)
                        logger.info("✓ Feature data saved with alternative method")
                    except Exception as alt_save_error:
                        logger.error(f"✗ All save methods failed: {alt_save_error}")
                        raise alt_save_error
                
                # Save feature statistics (optional)
                try:
                    stats_path = str(get_path("data/processed/pipeline_feature_stats"))
                    feature_engineer.save_feature_stats(df_features, stats_path)
                    logger.info("✓ Feature statistics saved")
                except Exception as stats_error:
                    logger.warning(f"⚠️  Could not save feature statistics: {stats_error}")
                
                # Save checkpoint
                if recovery:
                    recovery.save_checkpoint("feature_engineering", df_features, {
                        'feature_records': feature_count
                    })
                    
            except Exception as fe_error:
                logger.error(f"✗ Advanced feature engineering failed: {fe_error}")
                logger.error(traceback.format_exc())
                logger.info("Falling back to basic features...")
                df_features = None
        
        # Fallback: use preprocessed data if feature engineering failed
        if df_features is None:
            logger.warning("⚠️  Using preprocessed data as final output (feature engineering unavailable)")
            df_features = df_preprocessed
            
            # Save fallback data
            fallback_path = str(get_path("data/processed/pipeline_features"))
            try:
                df_features.coalesce(4).write.mode("overwrite").parquet(fallback_path)
                logger.info(f"✓ Fallback data saved to: {fallback_path}")
            except Exception as fallback_error:
                logger.error(f"✗ Failed to save fallback data: {fallback_error}")
                raise fallback_error
        
        results['stages_completed'].append('feature_engineering')
        logger.info("✓ Step 3 completed (with or without advanced features)")

        # ========================================
        # Pipeline Summary and Validation
        # ========================================
        elapsed_time = time.time() - start_time
        results['processing_time'] = elapsed_time
        results['success'] = True
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1 PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        logger.info(f"Records processed: {results['records_processed']:,}")
        logger.info(f"Stages completed: {', '.join(results['stages_completed'])}")
        
        if FEATURE_ENGINEERING_AVAILABLE and 'feature_engineering' in results['stages_completed']:
            logger.info("✓ Advanced feature engineering completed")
        else:
            logger.info("⚠️  Basic preprocessing completed (no advanced features)")
        
        # Show sample results
        try:
            logger.info("\nSample processed records:")
            sample_cols = ["tweet_id", "sentiment", "text_length"]
            if "vader_compound" in df_features.columns:
                sample_cols.append("vader_compound")
            if "emoji_sentiment" in df_features.columns:
                sample_cols.append("emoji_sentiment")
            
            df_features.select(*sample_cols).show(5, truncate=False)
        except Exception as display_error:
            logger.warning(f"⚠️  Could not display sample results: {display_error}")
        
        logger.info("\n✓ Phase 1 pipeline completed successfully!")
        
        return True, results

    except Exception as e:
        error_msg = f"Pipeline execution failed: {str(e)}"
        logger.error(f"✗ {error_msg}")
        logger.error(traceback.format_exc())
        results['errors'].append(error_msg)
        results['processing_time'] = time.time() - start_time
        
        return False, results
        
    finally:
        # Clean up Spark session
        try:
            if 'spark' in locals():
                spark.stop()
                logger.info("✓ Spark session stopped")
        except Exception as cleanup_error:
            logger.warning(f"⚠️  Error during cleanup: {cleanup_error}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MICAP Phase 1 Pipeline")
    parser.add_argument("--sample", type=float, default=0.1, 
                       help="Sample fraction to process (default: 0.1)")
    parser.add_argument("--no-checkpoints", action="store_true",
                       help="Disable checkpoint saving")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing checkpoints")
    
    args = parser.parse_args()
    
    # Run pipeline with arguments
    success, results = run_pipeline(
        sample_fraction=args.sample,
        enable_checkpoints=not args.no_checkpoints,
        resume_from_checkpoint=args.resume
    )
    
    # Exit with appropriate code
    if success:
        logger.info("🎉 Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Pipeline failed!")
        logger.error(f"Errors: {'; '.join(results['errors'])}")
        sys.exit(1)