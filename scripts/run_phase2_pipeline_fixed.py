#!/usr/bin/env python3
"""
Phase 2 Pipeline Runner - Fixed Version
Brand Recognition and Competitor Analysis without problematic imports
Author: AI Assistant
Date: 2024
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import fuzzywuzzy for optimized UDFs
try:
    from fuzzywuzzy import fuzz, process
    HAS_FUZZYWUZZY = True
except ImportError:
    HAS_FUZZYWUZZY = False
    print("Warning: fuzzywuzzy not available. Install with: pip install fuzzywuzzy[speedup]")

from src.utils.path_utils import get_path
from config.spark_config import create_spark_session

from pyspark.sql.functions import (
    col, size, explode, collect_set, array_distinct,
    first, count, avg, when, lit, expr, desc, udf
)
from pyspark.sql.types import ArrayType, StringType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase2_pipeline_fixed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_optimized_brand_udf(broadcast_config):
    """
    Create optimized brand recognition UDF using broadcast configuration.
    This avoids recreating the recognizer for each row.
    """
    def optimized_brand_recognition(text):
        if not text or not text.strip():
            return []
        
        try:
            # Get config from broadcast variable
            config = broadcast_config.value
            
            # Simple but efficient brand detection using fuzzywuzzy
            detected_brands = []
            text_lower = text.lower()
            
            # Extract all brand names and aliases from config
            brand_terms = {}
            for industry, industry_data in config.get('industries', {}).items():
                for brand_data in industry_data.get('brands', []):
                    brand_name = brand_data['name']
                    terms = [brand_name.lower()]
                    terms.extend([alias.lower() for alias in brand_data.get('aliases', [])])
                    
                    for term in terms:
                        if len(term) > 2:  # Skip very short terms
                            brand_terms[term] = brand_name
            
            # Quick fuzzy matching with reasonable threshold
            for term, brand_name in brand_terms.items():
                if term in text_lower:
                    # Exact match
                    detected_brands.append(f"{brand_name}:1.00")
                elif len(term) > 4:  # Only fuzzy match longer terms
                    # Use fuzzywuzzy for partial matching
                    ratio = fuzz.partial_ratio(term, text_lower)
                    if ratio >= 80:  # High threshold for quality
                        confidence = ratio / 100.0
                        detected_brands.append(f"{brand_name}:{confidence:.2f}")
            
            # Remove duplicates and keep highest confidence
            brand_scores = {}
            for detection in detected_brands:
                brand, score_str = detection.split(':')
                score = float(score_str)
                if brand not in brand_scores or score > brand_scores[brand]:
                    brand_scores[brand] = score
            
            # Return top detections
            result = [f"{brand}:{score:.2f}" for brand, score in brand_scores.items()]
            return result[:5]  # Limit to top 5 brands
            
        except Exception as e:
            # Log error but don't fail the entire job
            return []
    
    return udf(optimized_brand_recognition, ArrayType(StringType()))


def create_optimized_product_udf(broadcast_config):
    """
    Create optimized product extraction UDF using broadcast configuration.
    """
    def optimized_product_extraction(text):
        if not text or not text.strip():
            return []
        
        try:
            # Get config from broadcast variable
            config = broadcast_config.value
            
            detected_products = []
            text_lower = text.lower()
            
            # Extract all products from config
            product_terms = {}
            for industry, industry_data in config.get('industries', {}).items():
                for brand_data in industry_data.get('brands', []):
                    brand_name = brand_data['name']
                    for product in brand_data.get('products', []):
                        if len(product) > 3:  # Skip very short product names
                            product_terms[product.lower()] = (product, brand_name)
            
            # Quick product matching
            for term, (product_name, brand_name) in product_terms.items():
                if term in text_lower:
                    # Exact match
                    detected_products.append(f"{product_name}|{brand_name}:1.00")
                elif len(term) > 5:  # Only fuzzy match longer product names
                    ratio = fuzz.partial_ratio(term, text_lower)
                    if ratio >= 85:  # High threshold for products
                        confidence = ratio / 100.0
                        detected_products.append(f"{product_name}|{brand_name}:{confidence:.2f}")
            
            # Remove duplicates
            unique_products = list(set(detected_products))
            return unique_products[:3]  # Limit to top 3 products
            
        except Exception as e:
            return []
    
    return udf(optimized_product_extraction, ArrayType(StringType()))


def monitor_udf_progress(df, operation_name):
    """
    Monitor UDF execution progress with periodic logging.
    """
    logger.info(f"Starting {operation_name}...")
    start_time = time.time()
    
    # Force evaluation with progress monitoring
    result_df = df.cache()
    count = result_df.count()
    
    elapsed = time.time() - start_time
    logger.info(f"✓ {operation_name} completed: {count} records in {elapsed:.2f}s")
    
    return result_df


def run_phase2_pipeline_fixed(sample_size: float = 0.1):
    """
    Run complete Phase 2 pipeline: Brand Recognition and Competitor Analysis (Fixed Version)
    
    Args:
        sample_size: Fraction of data to process (0.1 = 10%)
    """
    # Check dependencies first
    if not HAS_FUZZYWUZZY:
        logger.error("fuzzywuzzy is required for Phase 2 pipeline")
        logger.error("Install with: pip install fuzzywuzzy[speedup]")
        raise ImportError("fuzzywuzzy is required but not available")
    
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Starting Phase 2 Pipeline (Fixed) - Brand Recognition & Competitor Analysis")
    logger.info("=" * 60)
    
    # Create Spark session with optimized configuration
    logger.info("Initializing Spark session...")
    spark = create_spark_session("Phase2_Pipeline_Fixed")
    
    # Optimize Spark configuration for UDF performance
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    
    try:
        # Load Phase 1 processed data
        logger.info("Loading Phase 1 processed data...")
        input_path = str(get_path("data/processed/pipeline_features"))
        df = spark.read.parquet(input_path)
        
        total_records = df.count()
        logger.info(f"Loaded {total_records} records from Phase 1")
        
        # Sample data for processing
        df_sample = df.sample(sample_size, seed=42)
        sample_count = df_sample.cache().count()
        logger.info(f"Processing {sample_count} records ({sample_size*100:.1f}% sample)")
        
        # Optimize partitioning for better performance
        optimal_partitions = max(4, min(sample_count // 1000, 200))
        df_sample = df_sample.repartition(optimal_partitions)
        logger.info(f"Repartitioned data into {optimal_partitions} partitions")
        
        # ========================================
        # STEP 1: Brand/Entity Recognition (Optimized)
        # ========================================
        logger.info("\n" + "=" * 50)
        logger.info("STEP 1: Brand/Entity Recognition (Optimized)")
        logger.info("=" * 50)
        
        # Create optimized UDFs with broadcasting
        logger.info("Creating optimized UDFs with broadcasting...")
        config_path = str(get_path("config/brands/brand_config.json"))
        
        # Load configuration once and broadcast it
        with open(config_path, 'r') as f:
            brand_config = json.load(f)
        
        # Broadcast the configuration to all workers
        broadcast_config = spark.sparkContext.broadcast(brand_config)
        logger.info("Brand configuration broadcasted to all workers")
        
        # Create optimized UDFs
        brand_udf = create_optimized_brand_udf(broadcast_config)
        product_udf = create_optimized_product_udf(broadcast_config)
        
        # Apply brand recognition
        logger.info("Recognizing brands in tweets...")
        df_brands = df_sample.withColumn(
            "detected_brands",
            brand_udf(col("text"))
        )
        
        # Monitor progress and cache intermediate result
        df_brands = monitor_udf_progress(df_brands, "Brand Recognition")
        
        # Apply product extraction
        logger.info("Extracting product mentions...")
        df_entities = df_brands.withColumn(
            "detected_products",
            product_udf(col("text"))
        )
        
        # Monitor progress and force evaluation
        df_entities = monitor_udf_progress(df_entities, "Product Extraction")
        
        # Filter to records with detected entities
        df_with_entities = df_entities.filter(
            (size(col("detected_brands")) > 0) | 
            (size(col("detected_products")) > 0)
        )
        
        entity_count = df_with_entities.count()
        logger.info(f"Found {entity_count} tweets with brand/product mentions")
        
        # Extract primary brand for analysis
        df_with_entities = df_with_entities.withColumn(
            "primary_brand",
            when(size(col("detected_brands")) > 0, 
                 expr("split(detected_brands[0], ':')[0]"))
            .otherwise(None)
        )
        
        # Show sample results
        logger.info("\nSample brand detections:")
        df_with_entities.select(
            "text", "detected_brands", "detected_products"
        ).show(5, truncate=False)
        
        # ========================================
        # STEP 2: Basic Analytics
        # ========================================
        logger.info("\n" + "=" * 50)
        logger.info("STEP 2: Basic Analytics")
        logger.info("=" * 50)
        
        # Brand distribution analysis
        logger.info("Analyzing brand distribution...")
        brand_dist = (df_with_entities
                      .groupBy("primary_brand")
                      .count()
                      .orderBy(col("count").desc())
                      .limit(10))
        
        logger.info("Top brands by mention count:")
        brand_dist.show()
        
        # Sentiment analysis by brand
        if "sentiment_score" in df_with_entities.columns:
            logger.info("Analyzing sentiment by brand...")
            brand_sentiment = (df_with_entities
                              .groupBy("primary_brand")
                              .agg(
                                  avg("sentiment_score").alias("avg_sentiment"),
                                  count("*").alias("mention_count")
                              )
                              .filter(col("mention_count") >= 5)  # Filter brands with enough mentions
                              .orderBy(col("avg_sentiment").desc()))
            
            logger.info("Brand sentiment analysis:")
            brand_sentiment.show()
        
        # ========================================
        # STEP 3: Save Results
        # ========================================
        logger.info("\n" + "=" * 50)
        logger.info("STEP 3: Saving Results")
        logger.info("=" * 50)
        
        # Create output directories
        output_base = str(get_path("data/phase2_results_fixed"))
        os.makedirs(output_base, exist_ok=True)
        
        # Save entity recognition results
        entity_path = f"{output_base}/entity_recognition"
        df_with_entities.coalesce(4).write.mode("overwrite").parquet(entity_path)
        logger.info(f"✓ Entity recognition results saved to: {entity_path}")
        
        # Save brand distribution
        brand_dist_path = f"{output_base}/brand_distribution"
        brand_dist.coalesce(1).write.mode("overwrite").parquet(brand_dist_path)
        logger.info(f"✓ Brand distribution saved to: {brand_dist_path}")
        
        # ========================================
        # Pipeline Summary
        # ========================================
        elapsed_time = time.time() - start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2 PIPELINE (FIXED) SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        logger.info(f"Records processed: {sample_count:,}")
        logger.info(f"Entities detected: {entity_count:,}")
        logger.info(f"Detection rate: {entity_count/sample_count*100:.1f}%")
        logger.info(f"\nResults saved to: {output_base}")
        logger.info("\n✓ Phase 2 pipeline (fixed) completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up broadcast variables and cached DataFrames
        try:
            if 'broadcast_config' in locals():
                broadcast_config.unpersist()
                logger.info("Broadcast variables cleaned up")
        except:
            pass
        
        try:
            if 'df_sample' in locals():
                df_sample.unpersist()
            if 'df_brands' in locals():
                df_brands.unpersist()
            if 'df_entities' in locals():
                df_entities.unpersist()
            logger.info("Cached DataFrames cleaned up")
        except:
            pass
        
        spark.stop()
        logger.info("Spark session closed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Phase 2 Pipeline (Fixed)")
    parser.add_argument("--sample", type=float, default=0.1,
                       help="Sample size (default: 0.1)")
    parser.add_argument("--full", action="store_true",
                       help="Process full dataset")
    
    args = parser.parse_args()
    
    sample_size = 1.0 if args.full else args.sample
    
    # Ensure log directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Run pipeline
    success = run_phase2_pipeline_fixed(sample_size=sample_size)
    
    if success:
        print(f"\n✓ Phase 2 pipeline completed successfully!")
        print(f"Check logs/phase2_pipeline_fixed.log for details")
    else:
        print(f"\n✗ Phase 2 pipeline failed!")
        sys.exit(1) 