"""
Phase 2 Pipeline Runner - Brand Recognition and Competitor Analysis
Executes brand/competitor detection and competitive analysis
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

from src.utils.path_utils import get_path
from config.spark_config import create_spark_session
from src.ml.entity_recognition import (
    BrandRecognizer, ProductExtractor, CompetitorMapper,
    EntityDisambiguator, create_brand_recognition_udf,
    create_product_extraction_udf
)
from src.spark.competitor_analysis import CompetitorAnalyzer

# Additional imports for Phase 2.2
from src.ml.trend_detection import (
    TopicModeler, TrendForecaster, AnomalyDetector, ViralityPredictor
)

from pyspark.sql.functions import (
    col, size, explode, collect_set, array_distinct,
    first, count, avg, when, lit, expr, desc
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase2_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_phase2_pipeline(sample_size: float = 0.1):
    """
    Run complete Phase 2 pipeline: Brand Recognition and Competitor Analysis
    
    Args:
        sample_size: Fraction of data to process (0.1 = 10%)
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Starting Phase 2 Pipeline - Brand Recognition & Competitor Analysis")
    logger.info("=" * 60)
    
    # Create Spark session
    logger.info("Initializing Spark session...")
    spark = create_spark_session("Phase2_Pipeline")
    
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
        
        # ========================================
        # STEP 1: Brand/Entity Recognition
        # ========================================
        logger.info("\n" + "=" * 50)
        logger.info("STEP 1: Brand/Entity Recognition")
        logger.info("=" * 50)
        
        # Initialize entity recognition components
        config_path = str(get_path("config/brands/brand_config.json"))
        brand_recognizer = BrandRecognizer(config_path)
        product_extractor = ProductExtractor(brand_recognizer)
        
        # Create UDFs for distributed processing
        brand_udf = create_brand_recognition_udf(config_path)
        product_udf = create_product_extraction_udf(config_path)
        
        # Apply brand recognition
        logger.info("Recognizing brands in tweets...")
        df_brands = df_sample.withColumn(
            "detected_brands",
            brand_udf(col("text"))
        )
        
        # Apply product extraction
        logger.info("Extracting product mentions...")
        df_entities = df_brands.withColumn(
            "detected_products",
            product_udf(col("text"))
        )
        
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
        # STEP 2: Competitor Analysis
        # ========================================
        logger.info("\n" + "=" * 50)
        logger.info("STEP 2: Competitor Analysis")
        logger.info("=" * 50)
        
        # Initialize competitor analyzer
        competitor_analyzer = CompetitorAnalyzer(spark, brand_recognizer)
        
        # Prepare data for competitor analysis - rename column for compatibility
        df_for_analysis = df_with_entities.withColumnRenamed("detected_brands", "brands")
        
        # Aggregate brand sentiment
        logger.info("Aggregating sentiment by brand...")
        brand_sentiment = competitor_analyzer.aggregate_brand_sentiment(
            df_for_analysis, 
            time_window='1 day'
        )
        
        # Calculate share of voice
        logger.info("Calculating market share of voice...")
        sov_df = competitor_analyzer.calculate_share_of_voice(
            df_for_analysis,
            time_window='1 day'
        )
        
        # Compute sentiment momentum
        logger.info("Computing sentiment momentum...")
        momentum_df = competitor_analyzer.compute_sentiment_momentum(sov_df)
        
        # Get top brands for detailed analysis
        top_brands = (momentum_df
                      .groupBy("brand")
                      .agg(avg("share_of_voice").alias("avg_sov"))
                      .orderBy(col("avg_sov").desc())
                      .limit(5)
                      .select("brand")
                      .collect())
        
        top_brand_names = [row.brand for row in top_brands]
        logger.info(f"Top brands by share of voice: {top_brand_names}")
        
        # Generate competitive insights for top brand
        if top_brand_names:
            target_brand = top_brand_names[0]
            competitors = top_brand_names[1:4] if len(top_brand_names) > 1 else []
            
            logger.info(f"\nGenerating insights for {target_brand} vs {competitors}")
            insights = competitor_analyzer.generate_competitive_insights(
                momentum_df,
                target_brand,
                competitors
            )
            
            # Log key insights
            logger.info("\nKey Competitive Insights:")
            if 'market_position' in insights['insights']:
                pos = insights['insights']['market_position']
                logger.info(f"  - Share of Voice: {pos.get('share_of_voice', 0):.1f}%")
                logger.info(f"  - Sentiment Score: {pos.get('sentiment_score', 0):.3f}")
            
            logger.info(f"\nRecommendations ({len(insights['recommendations'])}):")
            for rec in insights['recommendations'][:3]:  # Show top 3
                logger.info(f"  - {rec['recommendation']}")
        
        # ========================================
        # STEP 3: Market Intelligence Analytics
        # ========================================
        logger.info("\n" + "=" * 50)
        logger.info("STEP 3: Market Intelligence Analytics")
        logger.info("=" * 50)
        
        # Topic modeling for market insights
        logger.info("Performing topic modeling...")
        topic_modeler = TopicModeler(spark, num_topics=10)
        topic_modeler.fit_topics(df_with_entities)
        
        # Get topic descriptions
        topics = topic_modeler.get_topics()
        logger.info(f"Discovered {len(topics)} topics")
        
        # Apply topic assignments
        df_topics = topic_modeler.transform(df_with_entities)
        
        # Trend forecasting
        logger.info("Forecasting sentiment trends...")
        trend_forecaster = TrendForecaster(spark)
        sentiment_forecast = trend_forecaster.forecast_sentiment_trends(
            momentum_df
        )
        
        # Anomaly detection
        logger.info("Detecting anomalies...")
        anomaly_detector = AnomalyDetector(spark)
        anomalies = anomaly_detector.detect_sentiment_anomalies(momentum_df)
        
        anomaly_count = anomalies.filter(col("is_anomaly") == 1).count()
        logger.info(f"Detected {anomaly_count} anomalies in sentiment patterns")
        
        # ========================================
        # STEP 4: Save Results
        # ========================================
        logger.info("\n" + "=" * 50)
        logger.info("STEP 4: Saving Results")
        logger.info("=" * 50)
        
        # Create output directories
        output_base = str(get_path("data/phase2_results"))
        os.makedirs(output_base, exist_ok=True)
        
        # Save entity recognition results
        entity_path = f"{output_base}/entity_recognition"
        df_with_entities.coalesce(4).write.mode("overwrite").parquet(entity_path)
        logger.info(f"✓ Entity recognition results saved to: {entity_path}")
        
        # Save competitor analysis results
        competitor_path = f"{output_base}/competitor_analysis"
        momentum_df.coalesce(4).write.mode("overwrite").parquet(competitor_path)
        logger.info(f"✓ Competitor analysis saved to: {competitor_path}")
        
        # Save market intelligence results
        intelligence_path = f"{output_base}/market_intelligence"
        os.makedirs(intelligence_path, exist_ok=True)
        
        # Save insights as JSON
        if top_brand_names and 'insights' in locals():
            with open(f"{intelligence_path}/competitive_insights.json", 'w') as f:
                json.dump(insights, f, indent=2)
            logger.info(f"✓ Competitive insights saved")
        
        # Save topic descriptions
        with open(f"{intelligence_path}/topics.json", 'w') as f:
            json.dump(topics, f, indent=2)
        logger.info(f"✓ Topic descriptions saved")
        
        # Create visualizations
        logger.info("\nCreating visualizations...")
        viz_path = f"{output_base}/visualizations"
        competitor_analyzer.visualize_competitive_landscape(momentum_df, target_brand if top_brand_names else None, viz_path)
        logger.info(f"✓ Visualizations saved to: {viz_path}")
        
        # ========================================
        # Pipeline Summary
        # ========================================
        elapsed_time = time.time() - start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2 PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        logger.info(f"Records processed: {sample_count:,}")
        logger.info(f"Entities detected: {entity_count:,}")
        logger.info(f"Brands identified: {len(top_brand_names)}")
        logger.info(f"Anomalies detected: {anomaly_count}")
        logger.info(f"\nResults saved to: {output_base}")
        logger.info("\n✓ Phase 2 pipeline completed successfully!")
        
        # Display final statistics
        logger.info("\nBrand Distribution:")
        brand_dist = (df_with_entities
                      .groupBy("primary_brand")
                      .count()
                      .orderBy(col("count").desc())
                      .limit(10))
        brand_dist.show()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up
        try:
            df_sample.unpersist()
        except:
            pass
        spark.stop()
        logger.info("Spark session closed")


def run_brand_analysis_only(spark, df):
    """
    Run only brand/entity recognition part of the pipeline.
    Useful for testing or partial processing.
    
    Args:
        spark: Active SparkSession
        df: Input DataFrame
        
    Returns:
        DataFrame with brand/product entities
    """
    logger.info("Running brand analysis only...")
    
    # Initialize components
    config_path = str(get_path("config/brands/brand_config.json"))
    brand_udf = create_brand_recognition_udf(config_path)
    product_udf = create_product_extraction_udf(config_path)
    
    # Apply recognition
    df_analyzed = df.withColumn(
        "detected_brands", brand_udf(col("text"))
    ).withColumn(
        "detected_products", product_udf(col("text"))
    )
    
    return df_analyzed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Phase 2 Pipeline")
    parser.add_argument("--sample", type=float, default=0.1,
                       help="Sample size (default: 0.1)")
    parser.add_argument("--full", action="store_true",
                       help="Process full dataset")
    
    args = parser.parse_args()
    
    sample_size = 1.0 if args.full else args.sample
    
    # Ensure log directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Run pipeline
    run_phase2_pipeline(sample_size=sample_size)