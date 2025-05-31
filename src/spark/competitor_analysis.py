"""
Enhanced Competitor Analysis Module for MICAP
Combines fork safety with comprehensive competitive intelligence features
Author: AI Assistant (Enhanced Hybrid Version)
Date: 2024
"""

import os
import logging
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, avg, count, sum as spark_sum, stddev, when, lit,
    window, collect_list, explode, array_contains, size,
    first, last, min as spark_min, max as spark_max,
    percentile_approx, udf, struct, create_map, expr, lag,
    lead, abs as spark_abs, concat_ws, date_format, datediff,
    array_distinct, row_number, desc, asc
)
from pyspark.sql.types import (
    StringType, FloatType, ArrayType, StructType, 
    StructField, IntegerType, DoubleType
)

from src.utils.path_utils import get_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/competitor_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_brand_recognition_udf():
    """
    Returns a fork-safe UDF that initializes BrandRecognizer once per executor.
    Updated for fuzzywuzzy-first approach.
    This avoids the macOS fork safety issue with Metal/MPS.
    """
    def _recognize_brands(text):
        if not hasattr(_recognize_brands, "_model"):
            # Heavy imports happen here, inside the worker
            from src.ml.entity_recognition import BrandRecognizer
            # Initialize with fuzzywuzzy-first approach and spaCy disabled for Spark workers
            _recognize_brands._model = BrandRecognizer(
                use_spacy=False,
                fuzzy_threshold=65,  # Use fuzzywuzzy scoring (0-100)
                exact_threshold=90
            )
        
        if not text:
            return []
            
        try:
            # Get brand detections (returns list of tuples when return_details=False)
            pairs = _recognize_brands._model.recognize_brands(text, return_details=False) or []
            # Return in "brand:confidence" format - confidence is already 0-1 range
            return [f"{brand}:{conf:.2f}" for brand, conf in pairs]
        except Exception as e:
            logger.error(f"Error in brand recognition: {e}")
            return []
    
    return udf(_recognize_brands, ArrayType(StringType()))


def create_feature_extraction_udf(features: List[str]):
    """
    Create a UDF for feature extraction with predefined features.
    """
    features_broadcast = features
    
    def extract_features(text):
        if not text:
            return []
        text_lower = text.lower()
        detected = []
        for feature in features_broadcast:
            if feature.lower() in text_lower:
                detected.append(feature)
        return detected
    
    return udf(extract_features, ArrayType(StringType()))


class CompetitorAnalyzer:
    """
    Enhanced competitor analysis with fork safety and comprehensive features.
    
    Key capabilities:
    - Fork-safe brand recognition
    - Advanced sentiment momentum tracking
    - Feature-level competitive analysis
    - Market positioning insights
    - Automated opportunity/threat detection
    - Comprehensive visualizations
    """
    
    def __init__(self, spark: SparkSession, brand_recognizer=None):
        """
        Initialize competitor analyzer.
        
        Args:
            spark: Active SparkSession
            brand_recognizer: Optional BrandRecognizer for driver-side operations
        """
        self.spark = spark
        self.brand_recognizer = brand_recognizer
        
        # Time window configurations
        self.time_windows = {
            'hourly': '1 hour',
            'daily': '1 day',
            'weekly': '1 week',
            'monthly': '1 month'
        }
        
        logger.info("Initialized CompetitorAnalyzer")
        
    def parse_brand_confidence(self, df: DataFrame) -> DataFrame:
        """
        Parse brand:confidence strings into structured columns.
        
        Args:
            df: DataFrame with brands array column
            
        Returns:
            DataFrame with parsed brand and confidence columns
        """
        # Explode brands array to individual rows
        df_brands = df.select(
            col("*"),
            explode(col("brands")).alias("brand_info")
        ).filter(col("brand_info").isNotNull())
        
        # Parse brand and confidence
        df_brands = df_brands.withColumn(
            "brand",
            expr("lower(split(brand_info, ':')[0])")
        ).withColumn(
            "confidence",
            expr("cast(split(brand_info, ':')[1] as float)")
        )
        
        return df_brands
        
    def aggregate_brand_sentiment(self, df: DataFrame,
                                  time_window: str = "1 day",
                                  confidence_threshold: float = 0.7) -> DataFrame:
        """
        Aggregate sentiment metrics by brand and time window with enhanced metrics.
        
        Args:
            df: DataFrame with brand recognition results
            time_window: Time window for aggregation
            confidence_threshold: Minimum confidence for inclusion
            
        Returns:
            DataFrame with comprehensive brand sentiment metrics
        """
        logger.info(f"Aggregating brand sentiment with {time_window} window")
        
        # Parse brand data
        df_brands = self.parse_brand_confidence(df)
        
        # Filter by confidence
        df_brands = df_brands.filter(col("confidence") >= confidence_threshold)
        
        # Check if required columns exist
        has_vader = "vader_compound" in df.columns
        has_emoji = "emoji_sentiment" in df.columns
        
        # Build aggregation expressions
        agg_exprs = [
            count("*").alias("mention_count"),
            avg("sentiment").alias("avg_sentiment"),
            stddev("sentiment").alias("sentiment_stddev"),
            spark_sum(when(col("sentiment") == 1, 1).otherwise(0)).alias("positive_mentions"),
            spark_sum(when(col("sentiment") == 0, 1).otherwise(0)).alias("negative_mentions"),
            spark_sum(when(col("sentiment") == 0.5, 1).otherwise(0)).alias("neutral_mentions"),
            avg("confidence").alias("avg_confidence"),
            collect_list("text").alias("sample_texts"),
            # Time-based metrics
            spark_min("timestamp").alias("first_mention"),
            spark_max("timestamp").alias("last_mention")
        ]
        
        # Add optional columns
        if has_vader:
            agg_exprs.append(avg("vader_compound").alias("avg_vader_compound"))
        if has_emoji:
            agg_exprs.append(avg("emoji_sentiment").alias("avg_emoji_sentiment"))
            
        # Aggregate by brand and time window
        brand_sentiment = df_brands.groupBy(
            window("timestamp", time_window).alias("time_window"),
            "brand"
        ).agg(*agg_exprs)
        
        # Calculate derived metrics
        brand_sentiment = brand_sentiment.withColumn(
            "positive_ratio",
            col("positive_mentions") / col("mention_count")
        ).withColumn(
            "negative_ratio",
            col("negative_mentions") / col("mention_count")
        ).withColumn(
            "sentiment_score",
            (col("avg_sentiment") * 2 - 1) * 100  # -100 to 100 scale
        ).withColumn(
            "net_sentiment",
            col("positive_ratio") - col("negative_ratio")
        ).withColumn(
            "mention_duration_hours",
            (col("last_mention").cast("long") - col("first_mention").cast("long")) / 3600
        )
        
        # Extract window times
        brand_sentiment = brand_sentiment.withColumn(
            "window_start", col("time_window.start")
        ).withColumn(
            "window_end", col("time_window.end")
        ).drop("time_window")
        
        # Limit sample texts and add text diversity metric
        brand_sentiment = brand_sentiment.withColumn(
            "sample_texts", expr("slice(sample_texts, 1, 5)")
        ).withColumn(
            "unique_text_count", size(array_distinct(col("sample_texts")))
        )
        
        # Add time features
        brand_sentiment = brand_sentiment.withColumn(
            "hour", date_format("window_start", "HH")
        ).withColumn(
            "day_of_week", date_format("window_start", "EEEE")
        ).withColumn(
            "is_weekend",
            when(col("day_of_week").isin(["Saturday", "Sunday"]), 1).otherwise(0)
        )
        
        return brand_sentiment.orderBy("window_start", "brand")
        
    def compare_competitor_sentiment(self, df: DataFrame,
                                     target_brand: str,
                                     time_window: str = "1 day",
                                     competitors: List[str] = None) -> DataFrame:
        """
        Enhanced competitor comparison with relative performance metrics.
        
        Args:
            df: DataFrame with brand sentiment
            target_brand: Target brand for comparison
            time_window: Time window for comparison
            competitors: Optional list of competitors (auto-detected if None)
            
        Returns:
            DataFrame with detailed competitor comparison
        """
        logger.info(f"Comparing {target_brand} with competitors")
        
        # Get competitors if not provided
        if competitors is None and self.brand_recognizer:
            competitors = list(self.brand_recognizer.competitor_map.get(target_brand.lower(), set()))
            
        if not competitors:
            logger.warning(f"No competitors found for {target_brand}")
            # Return empty DataFrame with schema
            return self.spark.createDataFrame([], StructType([]))
            
        # Aggregate brand sentiment
        brand_sentiment = self.aggregate_brand_sentiment(df, time_window)
        
        # Filter for target and competitors
        brands_to_compare = [target_brand.lower()] + [c.lower() for c in competitors]
        comparison_df = brand_sentiment.filter(col("brand").isin(brands_to_compare))
        
        # Add ranking within each time window
        window_spec = Window.partitionBy("window_start").orderBy(desc("mention_count"))
        
        comparison_df = comparison_df.withColumn(
            "market_rank",
            row_number().over(window_spec)
        ).withColumn(
            "sentiment_rank",
            row_number().over(
                Window.partitionBy("window_start").orderBy(desc("sentiment_score"))
            )
        )
        
        # Calculate market share within competitive set
        window_totals = comparison_df.groupBy("window_start").agg(
            spark_sum("mention_count").alias("total_mentions")
        )
        
        comparison_df = comparison_df.join(window_totals, "window_start").withColumn(
            "market_share",
            col("mention_count") / col("total_mentions") * 100
        )
        
        # Add target brand flag and calculate gaps
        comparison_df = comparison_df.withColumn(
            "is_target",
            when(col("brand") == target_brand.lower(), 1).otherwise(0)
        )
        
        # Get target metrics for gap calculation
        target_window = Window.partitionBy("window_start")
        comparison_df = comparison_df.withColumn(
            "target_sentiment",
            spark_sum(when(col("is_target") == 1, col("sentiment_score")).otherwise(0)).over(target_window)
        ).withColumn(
            "target_mentions",
            spark_sum(when(col("is_target") == 1, col("mention_count")).otherwise(0)).over(target_window)
        ).withColumn(
            "sentiment_gap",
            when(col("is_target") == 0, col("sentiment_score") - col("target_sentiment")).otherwise(0)
        ).withColumn(
            "mention_gap",
            when(col("is_target") == 0, col("mention_count") - col("target_mentions")).otherwise(0)
        )
        
        return comparison_df.orderBy("window_start", "market_rank")
        
    def analyze_feature_sentiment(self, df: DataFrame,
                                  features: List[str] = None,
                                  auto_detect: bool = True) -> DataFrame:
        """
        Enhanced feature-level sentiment analysis with auto-detection.
        
        Args:
            df: DataFrame with brand and text data
            features: List of features to analyze
            auto_detect: Whether to auto-detect features from common patterns
            
        Returns:
            DataFrame with feature-level sentiment analysis
        """
        logger.info("Analyzing feature-level sentiment")
        
        # Default features if none provided
        if features is None:
            features = [
                # Product attributes
                "price", "quality", "performance", "design", "battery",
                "camera", "screen", "speed", "reliability", "durability",
                # Service attributes  
                "service", "support", "delivery", "warranty", "customer service",
                # Experience attributes
                "easy", "difficult", "simple", "complex", "intuitive",
                "user-friendly", "convenient", "fast", "slow"
            ]
            
        # Create feature extraction UDF
        feature_udf = create_feature_extraction_udf(features)
        
        # Parse brands and extract features
        df_brands = self.parse_brand_confidence(df)
        df_features = df_brands.withColumn(
            "detected_features",
            feature_udf(col("text"))
        ).filter(size(col("detected_features")) > 0)
        
        # Explode features
        df_analysis = df_features.select(
            col("brand"),
            col("sentiment"),
            col("confidence"),
            explode(col("detected_features")).alias("feature")
        )
        
        # Add optional columns if available
        if "vader_compound" in df.columns:
            df_analysis = df_analysis.join(
                df_features.select("text", "vader_compound"),
                "text"
            )
            
        # Aggregate by brand and feature
        agg_exprs = [
            count("*").alias("mention_count"),
            avg("sentiment").alias("avg_sentiment"),
            stddev("sentiment").alias("sentiment_stddev"),
            spark_sum(when(col("sentiment") == 1, 1).otherwise(0)).alias("positive_mentions"),
            spark_sum(when(col("sentiment") == 0, 1).otherwise(0)).alias("negative_mentions"),
            avg("confidence").alias("avg_brand_confidence")
        ]
        
        if "vader_compound" in df_analysis.columns:
            agg_exprs.append(avg("vader_compound").alias("avg_vader_compound"))
            
        feature_sentiment = df_analysis.groupBy("brand", "feature").agg(*agg_exprs)
        
        # Calculate sentiment metrics
        feature_sentiment = feature_sentiment.withColumn(
            "sentiment_score",
            (col("avg_sentiment") * 2 - 1) * 100
        ).withColumn(
            "positive_ratio",
            col("positive_mentions") / col("mention_count")
        ).withColumn(
            "confidence_weighted_sentiment",
            col("sentiment_score") * col("avg_brand_confidence")
        )
        
        # Rank features within each brand
        brand_window = Window.partitionBy("brand").orderBy(desc("mention_count"))
        feature_sentiment = feature_sentiment.withColumn(
            "feature_rank",
            row_number().over(brand_window)
        ).withColumn(
            "is_strength",
            when(col("positive_ratio") > 0.7, 1).otherwise(0)
        ).withColumn(
            "is_weakness", 
            when(col("positive_ratio") < 0.3, 1).otherwise(0)
        )
        
        return feature_sentiment.orderBy("brand", "feature_rank")
        
    def calculate_share_of_voice(self, df: DataFrame,
                                 industry: Optional[str] = None,
                                 time_window: str = "1 day",
                                 min_mentions: int = 5) -> DataFrame:
        """
        Enhanced share of voice calculation with trend analysis.
        
        Args:
            df: DataFrame with brand mentions
            industry: Optional industry filter
            time_window: Time window for calculation
            min_mentions: Minimum mentions to include brand
            
        Returns:
            DataFrame with detailed share of voice metrics
        """
        logger.info("Calculating share of voice")
        
        # Get brands to analyze
        if industry and self.brand_recognizer:
            brands = [b for b, data in self.brand_recognizer.brands.items()
                     if data.get('industry') == industry]
        else:
            brands = None  # Analyze all brands
            
        # Aggregate brand mentions
        brand_mentions = self.aggregate_brand_sentiment(df, time_window)
        
        # Filter by brand list if provided
        if brands:
            brand_mentions = brand_mentions.filter(col("brand").isin(brands))
            
        # Filter by minimum mentions
        brand_mentions = brand_mentions.filter(col("mention_count") >= min_mentions)
        
        # Calculate total mentions per window
        window_totals = brand_mentions.groupBy("window_start").agg(
            spark_sum("mention_count").alias("total_mentions"),
            count("brand").alias("active_brands"),
            avg("sentiment_score").alias("market_avg_sentiment")
        )
        
        # Join and calculate share metrics
        sov_df = brand_mentions.join(window_totals, "window_start")
        
        # Calculate share of voice and relative metrics
        sov_df = sov_df.withColumn(
            "share_of_voice",
            col("mention_count") / col("total_mentions") * 100
        ).withColumn(
            "relative_sentiment",
            col("sentiment_score") - col("market_avg_sentiment")
        ).withColumn(
            "mention_concentration",
            col("mention_count") / col("active_brands")
        )
        
        # Add ranking
        window_spec = Window.partitionBy("window_start").orderBy(desc("share_of_voice"))
        sov_df = sov_df.withColumn(
            "sov_rank",
            row_number().over(window_spec)
        )
        
        # Calculate SOV momentum
        brand_window = Window.partitionBy("brand").orderBy("window_start")
        sov_df = sov_df.withColumn(
            "prev_sov",
            lag("share_of_voice", 1).over(brand_window)
        ).withColumn(
            "sov_change",
            col("share_of_voice") - col("prev_sov")
        ).withColumn(
            "sov_change_pct",
            when(col("prev_sov") > 0,
                 (col("sov_change") / col("prev_sov")) * 100
            ).otherwise(0)
        )
        
        # Categorize market position
        sov_df = sov_df.withColumn(
            "market_position",
            when(col("share_of_voice") > 40, "Dominant")
            .when(col("share_of_voice") > 20, "Leader")
            .when(col("share_of_voice") > 10, "Challenger")
            .when(col("share_of_voice") > 5, "Niche")
            .otherwise("Emerging")
        )
        
        return sov_df.orderBy("window_start", "sov_rank")
        
    def compute_sentiment_momentum(self, df: DataFrame,
                                   lookback_windows: int = 7,
                                   forecast_windows: int = 3) -> DataFrame:
        """
        Enhanced momentum calculation with velocity and acceleration.
        
        Args:
            df: DataFrame with time-series brand sentiment
            lookback_windows: Number of windows for momentum calculation
            forecast_windows: Number of windows for trend projection
            
        Returns:
            DataFrame with comprehensive momentum indicators
        """
        logger.info(f"Computing sentiment momentum with {lookback_windows} lookback")
        
        # Ensure we have daily aggregation
        if "window_start" not in df.columns:
            df = self.aggregate_brand_sentiment(df, "1 day")
            
        # Define window specifications
        brand_window = Window.partitionBy("brand").orderBy("window_start")
        
        # Calculate multiple moving averages
        for period in [3, 7, 14, 30]:
            if period <= lookback_windows * 2:
                ma_window = brand_window.rowsBetween(-(period-1), 0)
                df = df.withColumn(
                    f"sentiment_ma_{period}",
                    avg("sentiment_score").over(ma_window)
                ).withColumn(
                    f"volume_ma_{period}",
                    avg("mention_count").over(ma_window)
                )
                
        # Calculate momentum (first derivative)
        df = df.withColumn(
            "sentiment_lag_1",
            lag("sentiment_score", 1).over(brand_window)
        ).withColumn(
            "sentiment_lag_n",
            lag("sentiment_score", lookback_windows).over(brand_window)
        ).withColumn(
            "sentiment_momentum",
            when(col("sentiment_lag_n").isNotNull(),
                 (col("sentiment_score") - col("sentiment_lag_n")) / lookback_windows
            ).otherwise(0)
        ).withColumn(
            "sentiment_roc",  # Rate of change
            when(col("sentiment_lag_n").isNotNull() & (col("sentiment_lag_n") != 0),
                 ((col("sentiment_score") - col("sentiment_lag_n")) / spark_abs(col("sentiment_lag_n"))) * 100
            ).otherwise(0)
        )
        
        # Calculate velocity (second derivative)
        df = df.withColumn(
            "momentum_lag_1",
            lag("sentiment_momentum", 1).over(brand_window)
        ).withColumn(
            "sentiment_velocity",
            when(col("momentum_lag_1").isNotNull(),
                 col("sentiment_momentum") - col("momentum_lag_1")
            ).otherwise(0)
        )
        
        # Calculate volatility
        volatility_window = brand_window.rowsBetween(-(lookback_windows-1), 0)
        df = df.withColumn(
            "sentiment_volatility",
            stddev("sentiment_score").over(volatility_window)
        ).withColumn(
            "volume_volatility",
            stddev("mention_count").over(volatility_window)
        )
        
        # Trend classification
        df = df.withColumn(
            "momentum_trend",
            when(col("sentiment_momentum") > 2, "strong_uptrend")
            .when(col("sentiment_momentum") > 0.5, "uptrend")
            .when(col("sentiment_momentum") < -2, "strong_downtrend")
            .when(col("sentiment_momentum") < -0.5, "downtrend")
            .otherwise("sideways")
        ).withColumn(
            "momentum_signal",
            when(
                (col("sentiment_momentum") > 0) & (col("sentiment_velocity") > 0),
                "accelerating_positive"
            ).when(
                (col("sentiment_momentum") > 0) & (col("sentiment_velocity") < 0),
                "decelerating_positive"
            ).when(
                (col("sentiment_momentum") < 0) & (col("sentiment_velocity") < 0),
                "accelerating_negative"
            ).when(
                (col("sentiment_momentum") < 0) & (col("sentiment_velocity") > 0),
                "decelerating_negative"
            ).otherwise("neutral")
        )
        
        # Trend strength indicator
        df = df.withColumn(
            "trend_strength",
            spark_abs(col("sentiment_momentum")) / (col("sentiment_volatility") + 1)
        )
        
        # Simple trend projection
        df = df.withColumn(
            "projected_sentiment",
            col("sentiment_score") + (col("sentiment_momentum") * forecast_windows)
        ).withColumn(
            "projection_confidence",
            when(col("sentiment_volatility") < 5, "high")
            .when(col("sentiment_volatility") < 10, "medium")
            .otherwise("low")
        )
        
        return df.orderBy("brand", "window_start")
        
    def generate_competitive_insights(self, df: DataFrame,
                                      target_brand: str,
                                      competitors: Optional[List[str]] = None,
                                      save_path: Optional[str] = None) -> Dict:
        """
        Generate comprehensive competitive insights with recommendations.
        
        Args:
            df: DataFrame with analysis results  
            target_brand: Brand to analyze
            competitors: Optional list of specific competitors to analyze
            save_path: Optional path to save insights JSON
            
        Returns:
            Dictionary of insights and recommendations
        """
        logger.info(f"Generating competitive insights for {target_brand}")
        
        insights = {
            "brand": target_brand,
            "analysis_date": datetime.now().isoformat(),
            "summary": {},
            "metrics": {},
            "competitors": {},
            "trends": {},
            "opportunities": [],
            "threats": [],
            "recommendations": [],
            "insights": {}  # Added for compatibility with revised pipeline
        }
        
        # Ensure brand is lowercase for matching
        target_brand_lower = target_brand.lower()
        
        # Helper function for safe float conversion
        def safe_float(value):
            """Safely convert value to float, handling None and string values."""
            if value is None:
                return 0.0
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
                
        def safe_row_get(row, column_name, default_value=None):
            """Safely get value from Spark Row, with default fallback."""
            try:
                if hasattr(row, column_name):
                    value = getattr(row, column_name, default_value)
                    return value if value is not None else default_value
                else:
                    return default_value
            except:
                return default_value
                
        # Get overall metrics
        brand_data = df.filter(col("brand") == target_brand_lower)
        
        if brand_data.count() == 0:
            logger.warning(f"No data found for brand: {target_brand}")
            return insights
            
        # Summary metrics
        summary_metrics = brand_data.agg(
            avg("sentiment_score").alias("avg_sentiment"),
            avg("mention_count").alias("avg_mentions"),
            spark_sum("mention_count").alias("total_mentions"),
            stddev("sentiment_score").alias("sentiment_volatility"),
            avg("share_of_voice").alias("avg_sov")
        ).collect()[0]
        
        insights["summary"] = {
            "average_sentiment": safe_float(summary_metrics["avg_sentiment"]),
            "average_mentions_per_period": safe_float(summary_metrics["avg_mentions"]),
            "total_mentions": int(summary_metrics["total_mentions"] or 0),
            "sentiment_stability": "stable" if safe_float(summary_metrics["sentiment_volatility"]) < 10 else "volatile"
        }
        
        # Latest metrics
        latest_data = brand_data.orderBy(desc("window_start")).first()
        if latest_data:
            insights["metrics"]["current"] = {
                "sentiment_score": safe_float(latest_data["sentiment_score"]),
                "mention_count": int(latest_data["mention_count"] or 0),
                "positive_ratio": safe_float(safe_row_get(latest_data, "positive_ratio", 0)),
                "market_position": safe_row_get(latest_data, "market_position", "Unknown")
            }
            
            # Add market position to insights for revised pipeline compatibility
            insights["insights"]["market_position"] = {
                "share_of_voice": safe_float(safe_row_get(latest_data, "share_of_voice", 0)),
                "sentiment_score": safe_float(latest_data["sentiment_score"]),
                "market_rank": int(safe_row_get(latest_data, "sov_rank", 0))
            }
            
        # Trend analysis
        if "sentiment_momentum" in brand_data.columns:
            trend_data = brand_data.orderBy(desc("window_start")).first()
            if trend_data:
                insights["trends"] = {
                    "momentum": safe_float(safe_row_get(trend_data, "sentiment_momentum", 0)),
                    "trend_direction": safe_row_get(trend_data, "momentum_trend", "unknown"),
                    "trend_signal": safe_row_get(trend_data, "momentum_signal", "neutral"),
                    "projected_sentiment": safe_float(safe_row_get(trend_data, "projected_sentiment", 0))
                }
                
        # Competitor analysis - use provided list or auto-detect
        competitor_list = competitors
        if competitor_list is None and self.brand_recognizer:
            competitor_list = list(self.brand_recognizer.competitor_map.get(target_brand_lower, set()))
        elif competitor_list is None:
            # Get top brands by mention count as fallback
            top_brands_df = df.groupBy("brand").agg(
                spark_sum("mention_count").alias("total_mentions")
            ).orderBy(desc("total_mentions")).limit(10)
            competitor_list = [row["brand"] for row in top_brands_df.collect() 
                             if row["brand"].lower() != target_brand_lower][:5]
            
        if competitor_list:
            for competitor in competitor_list:
                comp_data = df.filter(col("brand") == competitor.lower())
                if comp_data.count() > 0:
                    comp_metrics = comp_data.agg(
                        avg("sentiment_score").alias("avg_sentiment"),
                        avg("mention_count").alias("avg_mentions")
                    ).collect()[0]
                    
                    sentiment_gap = safe_float(summary_metrics["avg_sentiment"]) - safe_float(comp_metrics["avg_sentiment"])
                    
                    insights["competitors"][competitor] = {
                        "average_sentiment": safe_float(comp_metrics["avg_sentiment"]),
                        "average_mentions": safe_float(comp_metrics["avg_mentions"]),
                        "sentiment_gap": sentiment_gap,
                        "competitive_position": "ahead" if sentiment_gap > 0 else "behind"
                    }
                    
                    # Identify opportunities and threats
                    if sentiment_gap > 10:
                        insights["opportunities"].append({
                            "type": "sentiment_advantage",
                            "competitor": competitor,
                            "description": f"Strong sentiment lead over {competitor} ({sentiment_gap:.1f} points)",
                            "action": "Leverage positive perception in marketing"
                        })
                    elif sentiment_gap < -10:
                        insights["threats"].append({
                            "type": "sentiment_deficit",
                            "competitor": competitor,
                            "description": f"Sentiment trailing {competitor} by {abs(sentiment_gap):.1f} points",
                            "action": "Analyze competitor strengths and address gaps"
                        })
                        
        # Generate recommendations based on insights
        recommendations = []
        
        # Sentiment-based recommendations
        current_sentiment = insights["metrics"].get("current", {}).get("sentiment_score", 0)
        if current_sentiment < -20:
            recommendations.append({
                "priority": "critical",
                "area": "sentiment",
                "recommendation": "Implement immediate reputation management strategy",
                "rationale": f"Current sentiment score ({current_sentiment:.1f}) indicates significant negative perception"
            })
        elif current_sentiment < 0:
            recommendations.append({
                "priority": "high",
                "area": "sentiment",
                "recommendation": "Focus on addressing negative feedback drivers",
                "rationale": "Negative sentiment trend requires intervention"
            })
            
        # Momentum-based recommendations
        momentum = insights["trends"].get("momentum", 0)
        if momentum < -2:
            recommendations.append({
                "priority": "high",
                "area": "trend",
                "recommendation": "Investigate and address causes of declining sentiment",
                "rationale": "Strong negative momentum detected"
            })
        elif momentum > 2:
            recommendations.append({
                "priority": "medium",
                "area": "trend", 
                "recommendation": "Capitalize on positive momentum with increased engagement",
                "rationale": "Strong positive trend provides growth opportunity"
            })
            
        # Volume-based recommendations
        avg_mentions = insights["summary"].get("average_mentions_per_period", 0)
        if avg_mentions < 100:
            recommendations.append({
                "priority": "medium",
                "area": "visibility",
                "recommendation": "Increase brand visibility through targeted campaigns",
                "rationale": "Low mention volume limits market presence"
            })
            
        insights["recommendations"] = recommendations
        
        # Add strategic summary
        insights["strategic_summary"] = self._generate_strategic_summary(insights)
        
        # Save insights if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(insights, f, indent=2)
            logger.info(f"Insights saved to {save_path}")
            
        return insights
        
    def _generate_strategic_summary(self, insights: Dict) -> str:
        """Generate executive summary from insights."""
        brand = insights["brand"]
        sentiment = insights["summary"].get("average_sentiment", 0)
        position = "positive" if sentiment > 0 else "negative"
        
        competitors_ahead = sum(1 for c in insights["competitors"].values() 
                               if c["competitive_position"] == "behind")
        total_competitors = len(insights["competitors"])
        
        summary = f"{brand} maintains {position} market perception with average sentiment of {sentiment:.1f}. "
        
        if total_competitors > 0:
            summary += f"Competitively positioned ahead of {competitors_ahead}/{total_competitors} rivals. "
            
        if insights["trends"].get("momentum", 0) > 0:
            summary += "Positive momentum indicates improving brand perception. "
        else:
            summary += "Negative momentum requires strategic intervention. "
            
        summary += f"Key focus areas: {', '.join(r['area'] for r in insights['recommendations'][:3])}."
        
        return summary
        
    def visualize_competitive_landscape(self, df: DataFrame,
                                        target_brand: str = None,
                                        output_dir: str = None):
        """
        Create comprehensive competitive analysis visualizations.
        
        Args:
            df: DataFrame with competitive analysis data
            target_brand: Optional brand to highlight
            output_dir: Directory to save visualizations
        """
        if output_dir is None:
            output_dir = str(get_path("data/visualizations/competitive_analysis"))
            
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Creating competitive visualizations in {output_dir}")
        
        # Convert to pandas for visualization
        pdf = df.toPandas()
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Share of Voice Timeline
        if "share_of_voice" in pdf.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Get top brands by average SOV
            top_brands = pdf.groupby('brand')['share_of_voice'].mean().nlargest(5).index
            
            for brand in top_brands:
                brand_data = pdf[pdf['brand'] == brand]
                linewidth = 3 if brand == target_brand else 2
                alpha = 1.0 if brand == target_brand else 0.7
                
                ax.plot(brand_data['window_start'], brand_data['share_of_voice'],
                       marker='o', label=brand.title(), linewidth=linewidth, alpha=alpha)
                
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Share of Voice (%)', fontsize=12)
            ax.set_title('Market Share of Voice Over Time', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/share_of_voice_timeline.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        # 2. Sentiment Comparison Heatmap
        if "sentiment_score" in pdf.columns:
            # Get latest data for each brand
            latest_date = pdf['window_start'].max()
            latest_data = pdf[pdf['window_start'] == latest_date]
            
            if len(latest_data) > 1:
                # Create comparison matrix
                brands = latest_data['brand'].unique()[:10]  # Top 10 brands
                metrics = ['sentiment_score', 'mention_count', 'positive_ratio']
                
                # Create pivot table
                pivot_data = latest_data[latest_data['brand'].isin(brands)].pivot_table(
                    values=metrics,
                    index='brand',
                    aggfunc='mean'
                )
                
                # Normalize for heatmap
                pivot_norm = (pivot_data - pivot_data.mean()) / pivot_data.std()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(pivot_norm.T, annot=True, fmt='.2f', cmap='RdYlGn',
                           center=0, cbar_kws={'label': 'Normalized Score'},
                           xticklabels=[b.title() for b in pivot_norm.index],
                           yticklabels=['Sentiment', 'Volume', 'Positivity'])
                
                ax.set_title('Competitive Metrics Comparison (Normalized)', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/competitive_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close()
                
        # 3. Sentiment Momentum Chart
        if "sentiment_momentum" in pdf.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            # Get top brands
            top_brands = pdf.groupby('brand')['mention_count'].sum().nlargest(5).index
            
            for brand in top_brands:
                brand_data = pdf[pdf['brand'] == brand].sort_values('window_start')
                linewidth = 3 if brand == target_brand else 2
                alpha = 1.0 if brand == target_brand else 0.7
                
                # Sentiment over time
                ax1.plot(brand_data['window_start'], brand_data['sentiment_score'],
                        marker='o', label=brand.title(), alpha=alpha, linewidth=linewidth)
                
                # Momentum
                ax2.plot(brand_data['window_start'], brand_data['sentiment_momentum'],
                        marker='s', label=brand.title(), alpha=alpha, linewidth=linewidth)
                
            ax1.set_ylabel('Sentiment Score', fontsize=12)
            ax1.set_title('Brand Sentiment Trends', fontsize=14, fontweight='bold')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax1.grid(True, alpha=0.3)
            
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Sentiment Momentum', fontsize=12)
            ax2.set_title('Sentiment Momentum (Rate of Change)', fontsize=14, fontweight='bold')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/sentiment_momentum.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        # 4. Competitive Positioning Matrix
        if all(col in pdf.columns for col in ['share_of_voice', 'sentiment_score', 'mention_count']):
            latest_data = pdf[pdf['window_start'] == pdf['window_start'].max()]
            
            if len(latest_data) > 1:
                fig, ax = plt.subplots(figsize=(12, 9))
                
                # Create scatter plot
                scatter_data = latest_data.groupby('brand').agg({
                    'share_of_voice': 'mean',
                    'sentiment_score': 'mean',
                    'mention_count': 'sum'
                }).reset_index()
                
                # Limit to top brands
                scatter_data = scatter_data.nlargest(15, 'mention_count')
                
                # Create scatter
                scatter = ax.scatter(
                    scatter_data['share_of_voice'],
                    scatter_data['sentiment_score'],
                    s=scatter_data['mention_count'] / 10,  # Scale bubble size
                    alpha=0.6,
                    c=range(len(scatter_data)),
                    cmap='viridis'
                )
                
                # Add brand labels
                for idx, row in scatter_data.iterrows():
                    ax.annotate(
                        row['brand'].title(),
                        (row['share_of_voice'], row['sentiment_score']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=10,
                        alpha=0.8,
                        fontweight='bold' if row['brand'] == target_brand else 'normal'
                    )
                    
                # Add quadrant lines
                ax.axvline(x=scatter_data['share_of_voice'].median(), 
                          color='gray', linestyle='--', alpha=0.5)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                
                # Add quadrant labels
                ax.text(0.95, 0.95, 'Market Leaders', transform=ax.transAxes,
                       fontsize=12, ha='right', va='top', alpha=0.5,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(0.05, 0.95, 'Niche Players', transform=ax.transAxes,
                       fontsize=12, ha='left', va='top', alpha=0.5,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(0.95, 0.05, 'Volume Leaders', transform=ax.transAxes,
                       fontsize=12, ha='right', va='bottom', alpha=0.5,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(0.05, 0.05, 'Struggling', transform=ax.transAxes,
                       fontsize=12, ha='left', va='bottom', alpha=0.5,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                ax.set_xlabel('Share of Voice (%)', fontsize=12)
                ax.set_ylabel('Sentiment Score', fontsize=12)
                ax.set_title('Competitive Positioning Matrix', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add note about bubble size
                ax.text(0.02, 0.02, 'Bubble size = Total mentions', transform=ax.transAxes,
                       fontsize=9, alpha=0.5, style='italic')
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/competitive_positioning.png", dpi=300, bbox_inches='tight')
                plt.close()
                
        logger.info("Competitive visualizations completed")


def main():
    """Test enhanced competitor analysis functionality."""
    from config.spark_config import create_spark_session
    
    # Create Spark session
    spark = create_spark_session("CompetitorAnalysis")
    
    # Create fork-safe brand recognition UDF
    brand_udf = create_brand_recognition_udf()
    
    # Load data
    logger.info("Loading data...")
    data_path = str(get_path("data/processed/pipeline_features"))
    
    try:
        df = spark.read.parquet(data_path)
        logger.info(f"Loaded {df.count()} records")
        
        # Add brand recognition
        logger.info("Applying brand recognition...")
        df_with_brands = df.withColumn("brands", brand_udf(col("text")))
        
        # Filter to records with detected brands
        df_brands = df_with_brands.filter(size(col("brands")) > 0)
        brand_count = df_brands.count()
        logger.info(f"Found {brand_count} records with brand mentions")
        
        if brand_count == 0:
            logger.warning("No brands detected. Check brand configuration.")
            spark.stop()
            return
            
        # Now safe to import BrandRecognizer for driver-side operations
        from src.ml.entity_recognition import BrandRecognizer
        brand_recognizer = BrandRecognizer(
            use_spacy=False,
            fuzzy_threshold=65,  # Use fuzzywuzzy scoring
            exact_threshold=90
        )
        
        # Initialize analyzer
        analyzer = CompetitorAnalyzer(spark, brand_recognizer)
        
        # 1. Test brand sentiment aggregation
        logger.info("\n" + "="*50)
        logger.info("Testing brand sentiment aggregation...")
        brand_sentiment = analyzer.aggregate_brand_sentiment(df_brands, "1 day")
        
        logger.info("Sample brand sentiment results:")
        brand_sentiment.select(
            "brand", "window_start", "mention_count", 
            "sentiment_score", "positive_ratio"
        ).show(20, truncate=False)
        
        # 2. Test competitor comparison
        # Get a brand with data
        top_brand = brand_sentiment.groupBy("brand").count().orderBy(desc("count")).first()
        if top_brand:
            target_brand = top_brand["brand"]
            logger.info(f"\nTesting competitor comparison for {target_brand}...")
            
            comparison = analyzer.compare_competitor_sentiment(df_brands, target_brand)
            if comparison.count() > 0:
                comparison.show(10, truncate=False)
            else:
                logger.warning("No competitor data found")
                
        # 3. Test share of voice
        logger.info("\n" + "="*50)
        logger.info("Testing share of voice calculation...")
        sov = analyzer.calculate_share_of_voice(df_brands)
        
        logger.info("Share of voice results:")
        sov.select(
            "window_start", "brand", "share_of_voice", 
            "sov_rank", "market_position"
        ).show(20, truncate=False)
        
        # 4. Test sentiment momentum
        logger.info("\n" + "="*50)
        logger.info("Testing sentiment momentum...")
        momentum = analyzer.compute_sentiment_momentum(sov)
        
        logger.info("Momentum results:")
        momentum.select(
            "brand", "window_start", "sentiment_momentum",
            "momentum_trend", "trend_strength"
        ).show(20, truncate=False)
        
        # 5. Test feature analysis
        features = ["price", "quality", "performance", "battery", "camera"]
        logger.info(f"\nTesting feature analysis for: {features}")
        
        feature_sentiment = analyzer.analyze_feature_sentiment(df_brands, features)
        if feature_sentiment.count() > 0:
            feature_sentiment.show(20, truncate=False)
        else:
            logger.warning("No feature mentions found")
            
        # 6. Generate insights
        if top_brand:
            logger.info(f"\nGenerating insights for {target_brand}...")
            insights_path = str(get_path(f"data/analytics/insights_{target_brand}.json"))
            insights = analyzer.generate_competitive_insights(momentum, target_brand, insights_path)
            
            logger.info("Generated insights:")
            logger.info(f"Summary: {insights['strategic_summary']}")
            logger.info(f"Recommendations: {len(insights['recommendations'])}")
            
        # 7. Create visualizations
        logger.info("\nCreating visualizations...")
        analyzer.visualize_competitive_landscape(momentum, target_brand if top_brand else None)
        
        # Save results
        output_base = str(get_path("data/analytics/competitor_analysis"))
        os.makedirs(output_base, exist_ok=True)
        
        brand_sentiment.coalesce(4).write.mode("overwrite").parquet(f"{output_base}/brand_sentiment")
        sov.coalesce(4).write.mode("overwrite").parquet(f"{output_base}/share_of_voice")
        momentum.coalesce(4).write.mode("overwrite").parquet(f"{output_base}/sentiment_momentum")
        
        logger.info(f"\nResults saved to {output_base}")
        logger.info("Enhanced competitor analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()