"""
Temporal Analysis Module for MICAP
Analyzes sentiment trends over time periods
Implements time series analysis and anomaly detection
"""

import os
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.path_utils import get_path

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, avg, count, stddev, sum as spark_sum,
    window, date_format, hour, dayofweek, month,
    year, weekofyear, lag, lead, when, abs as spark_abs,
    percentile_approx, collect_list, udf, pandas_udf
)
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, ArrayType, StructType, StructField
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# safe ratio helper  (Spark ≥3.5)
def safe_ratio(numer, denom):
    return F.try_divide(numer, denom)         # NULL when denom = 0
    # On Spark <3.5 use: F.when(denom != 0, numer / denom)


class TemporalAnalyzer:
    """
    Performs temporal analysis of sentiment data
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize temporal analyzer

        Args:
            spark: Active SparkSession
        """
        self.spark = spark
        self.time_windows = {
            'hourly': '1 hour',
            'daily': '1 day',
            'weekly': '1 week',
            'monthly': '1 month'
        }

    def aggregate_sentiment_by_time(self, df: DataFrame,
                                    time_window: str = 'hourly') -> DataFrame:
        """
        Aggregate sentiment scores by time windows

        Args:
            df: Input DataFrame with timestamp and sentiment
            time_window: Time window for aggregation

        Returns:
            DataFrame with aggregated sentiment metrics
        """
        logger.info(f"Aggregating sentiment by {time_window} windows...")

        window_duration = self.time_windows.get(time_window, '1 hour')

        # Aggregate sentiment metrics
        agg_df = df.groupBy(
            F.window("timestamp", window_duration).alias("time_window")
        ).agg(
            F.count("sentiment").alias("tweet_count"),
            F.avg("sentiment").alias("avg_sentiment"),
            F.stddev("sentiment").alias("sentiment_stddev"),
            spark_sum(when(F.col("sentiment") == 1, 1).otherwise(0)).alias("positive_count"),
            spark_sum(when(F.col("sentiment") == 0, 1).otherwise(0)).alias("negative_count"),
            F.avg("vader_compound").alias("avg_vader_compound"),
            F.avg("emoji_sentiment").alias("avg_emoji_sentiment")
        )

        # Calculate sentiment ratio
        agg_df = agg_df.withColumn(
            "positive_ratio",
            safe_ratio(F.col("positive_count"), F.col("tweet_count"))
        )

        # Extract window start and end times
        agg_df = agg_df.withColumn("window_start", F.col("time_window.start")) \
            .withColumn("window_end", F.col("time_window.end")) \
            .drop("time_window")

        # Add time-based features
        agg_df = self._add_time_features(agg_df)

        return agg_df.orderBy("window_start")

    def _add_time_features(self, df: DataFrame) -> DataFrame:
        """
        Add time-based features for analysis

        Args:
            df: DataFrame with window_start column

        Returns:
            DataFrame with additional time features
        """
        df = df.withColumn("hour", F.hour(col("window_start"))) \
            .withColumn("day_of_week", F.dayofweek(col("window_start"))) \
            .withColumn("month", F.month(col("window_start"))) \
            .withColumn("year", F.year(col("window_start"))) \
            .withColumn("week_of_year", F.weekofyear(col("window_start")))

        # Add is_weekend flag
        df = df.withColumn(
            "is_weekend",
            when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0)
        )

        return df

    def detect_sentiment_anomalies(self, df: DataFrame,
                                   metric: str = "avg_sentiment",
                                   window_size: int = 24,
                                   threshold: float = 2.0) -> DataFrame:
        """
        Detect anomalies in sentiment patterns using statistical methods

        Args:
            df: Aggregated sentiment DataFrame
            metric: Metric to analyze for anomalies
            window_size: Rolling window size
            threshold: Standard deviation threshold for anomaly

        Returns:
            DataFrame with anomaly flags
        """
        logger.info(f"Detecting anomalies in {metric}...")

        # Define window for rolling statistics
        window_spec = Window.orderBy("window_start").rowsBetween(
            -window_size, -1
        )

        # Calculate rolling statistics
        df = df.withColumn(
            f"{metric}_rolling_mean",
            F.avg(col(metric)).over(window_spec)
        )
        df = df.withColumn(
            f"{metric}_rolling_std",
            F.stddev(col(metric)).over(window_spec)
        )

        # Calculate z-score
        # df = df.withColumn(
        #     f"{metric}_zscore",
        #     (F.col(metric) - F.col(f"{metric}_rolling_mean")) / F.col(f"{metric}_rolling_std")
        # )
        df = df.withColumn(
            f"{metric}_zscore",
            safe_ratio((F.col(metric) - F.col(f"{metric}_rolling_mean")), F.col(f"{metric}_rolling_std"))
        )

        # Flag anomalies
        df = df.withColumn(
            "is_anomaly",
            when(spark_abs(F.col(f"{metric}_zscore")) > threshold, 1).otherwise(0)
        )

        # Classify anomaly type
        df = df.withColumn(
            "anomaly_type",
            when(F.col("is_anomaly") == 1,
                 when(F.col(f"{metric}_zscore") > 0, "positive_spike")
                 .otherwise("negative_spike"))
            .otherwise("normal")
        )

        # Count anomalies
        anomaly_count = df.filter(F.col("is_anomaly") == 1).count()
        total_count = df.count()
        logger.info(f"Detected {anomaly_count} anomalies out of {total_count} time windows")

        return df

    def analyze_sentiment_trends(self, df: DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Analyze sentiment trends across different time granularities

        Args:
            df: Input DataFrame with sentiment data

        Returns:
            Dictionary of trend DataFrames
        """
        logger.info("Analyzing sentiment trends...")

        trends = {}

        # Hourly trends
        hourly_df = self.aggregate_sentiment_by_time(df, 'hourly')
        trends['hourly'] = hourly_df.toPandas()

        # Daily trends
        daily_df = self.aggregate_sentiment_by_time(df, 'daily')
        trends['daily'] = daily_df.toPandas()

        # Weekly trends
        weekly_df = self.aggregate_sentiment_by_time(df, 'weekly')
        trends['weekly'] = weekly_df.toPandas()

        # Hour of day patterns
        hour_patterns = df.groupBy("hour").agg(
            avg("sentiment").alias("avg_sentiment"),
            count("sentiment").alias("tweet_count"),
            avg("vader_compound").alias("avg_vader_compound")
        ).orderBy("hour").toPandas()
        trends['hour_of_day'] = hour_patterns

        # Day of week patterns
        dow_patterns = df.withColumn(
            "day_name",
            when(F.col("day_of_week") == 1, "Sunday")
            .when(F.col("day_of_week") == 2, "Monday")
            .when(F.col("day_of_week") == 3, "Tuesday")
            .when(F.col("day_of_week") == 4, "Wednesday")
            .when(F.col("day_of_week") == 5, "Thursday")
            .when(F.col("day_of_week") == 6, "Friday")
            .when(F.col("day_of_week") == 7, "Saturday")
        ).groupBy("day_of_week", "day_name").agg(
            avg("sentiment").alias("avg_sentiment"),
            count("sentiment").alias("tweet_count")
        ).orderBy("day_of_week").toPandas()
        trends['day_of_week'] = dow_patterns

        return trends

    def calculate_sentiment_momentum(self, df: DataFrame,
                                     window_size: int = 7) -> DataFrame:
        """
        Calculate sentiment momentum indicators

        Args:
            df: Aggregated sentiment DataFrame
            window_size: Window size for momentum calculation

        Returns:
            DataFrame with momentum indicators
        """
        logger.info("Calculating sentiment momentum...")

        # Define window specifications
        window_spec = Window.orderBy("window_start").rowsBetween(
            -window_size, Window.currentRow
        )
        lag_window = Window.orderBy("window_start")

        # Calculate moving averages
        df = df.withColumn(
            "sentiment_ma",
            F.avg("avg_sentiment").over(window_spec)
        )

        # Calculate rate of change
        df = df.withColumn(
            "sentiment_lag",
            F.lag("avg_sentiment", window_size).over(lag_window)
        )
        # df = df.withColumn(
        #     "sentiment_roc",
        #     ((F.col("avg_sentiment") - F.col("sentiment_lag")) / F.col("sentiment_lag")) * 100
        # )
        df = df.withColumn(
            "sentiment_roc",
            (safe_ratio((F.col("avg_sentiment") - F.col("sentiment_lag")), F.col("sentiment_lag"))) * 100
        )

        # Calculate momentum
        df = df.withColumn(
            "sentiment_momentum",
            F.col("avg_sentiment") - F.col("sentiment_lag")
        )

        # Calculate acceleration
        df = df.withColumn(
            "momentum_lag",
            F.lag("sentiment_momentum", 1).over(lag_window)
        )
        df = df.withColumn(
            "sentiment_acceleration",
            F.col("sentiment_momentum") - F.col("momentum_lag")
        )

        # Trend classification
        df = df.withColumn(
            "trend",
            when(F.col("sentiment_momentum") > 0.05, "uptrend")
            .when(F.col("sentiment_momentum") < -0.05, "downtrend")
            .otherwise("neutral")
        )

        return df

    def perform_time_series_decomposition(self, df: pd.DataFrame,
                                          period: int = 24) -> Dict:
        """
        Perform time series decomposition on sentiment data

        Args:
            df: Pandas DataFrame with time series data
            period: Period for seasonal decomposition

        Returns:
            Dictionary with decomposition components
        """
        logger.info("Performing time series decomposition...")

        # Ensure datetime index
        df = df.set_index('window_start')

        # Perform decomposition
        decomposition = seasonal_decompose(
            df['avg_sentiment'],
            model='additive',
            period=period
        )

        # Extract components
        components = {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'observed': df['avg_sentiment']
        }

        # Perform stationarity test
        adf_result = adfuller(df['avg_sentiment'].dropna())
        stationarity = {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'is_stationary': adf_result[1] < 0.05
        }

        logger.info(f"Stationarity test p-value: {stationarity['p_value']:.4f}")

        return {
            'components': components,
            'stationarity': stationarity
        }

    def create_temporal_visualizations(self, trends: Dict[str, pd.DataFrame],
                                       output_dir: str = str(get_path("data/visualizations"))):
        """
        Create temporal analysis visualizations

        Args:
            trends: Dictionary of trend DataFrames
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')

        # 1. Hourly sentiment trends
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Sentiment over time
        ax = axes[0]
        hourly_df = trends['hourly']
        ax.plot(hourly_df['window_start'], hourly_df['avg_sentiment'],
                label='Average Sentiment', color='blue', alpha=0.7)
        ax.fill_between(hourly_df['window_start'],
                        hourly_df['avg_sentiment'] - hourly_df['sentiment_stddev'],
                        hourly_df['avg_sentiment'] + hourly_df['sentiment_stddev'],
                        alpha=0.3, color='blue', label='±1 std dev')
        ax.set_xlabel('Time')
        ax.set_ylabel('Average Sentiment')
        ax.set_title('Hourly Sentiment Trends')
        ax.legend()

        # Tweet volume
        ax2 = axes[1]
        ax2.plot(hourly_df['window_start'], hourly_df['tweet_count'],
                 color='green', alpha=0.7)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Tweet Count')
        ax2.set_title('Hourly Tweet Volume')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/hourly_sentiment_trends.png", dpi=300)
        plt.close()

        # 2. Hour of day patterns
        fig, ax = plt.subplots(figsize=(10, 6))
        hour_df = trends['hour_of_day']

        ax.bar(hour_df['hour'], hour_df['avg_sentiment'],
               color='skyblue', alpha=0.8)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Sentiment')
        ax.set_title('Sentiment by Hour of Day')
        ax.set_xticks(range(0, 24))

        # Add tweet count as line
        ax2 = ax.twinx()
        ax2.plot(hour_df['hour'], hour_df['tweet_count'],
                 color='red', marker='o', label='Tweet Count')
        ax2.set_ylabel('Tweet Count', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/hour_of_day_patterns.png", dpi=300)
        plt.close()

        # 3. Day of week patterns
        fig, ax = plt.subplots(figsize=(10, 6))
        dow_df = trends['day_of_week']

        ax.bar(dow_df['day_name'], dow_df['avg_sentiment'],
               color='lightcoral', alpha=0.8)
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Average Sentiment')
        ax.set_title('Sentiment by Day of Week')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/day_of_week_patterns.png", dpi=300)
        plt.close()

        # 4. Daily sentiment heatmap
        if len(trends['daily']) > 7:
            daily_df = trends['daily']
            daily_df['date'] = pd.to_datetime(daily_df['window_start']).dt.date
            daily_df['weekday'] = pd.to_datetime(daily_df['window_start']).dt.day_name()
            daily_df['week'] = pd.to_datetime(daily_df['window_start']).dt.isocalendar().week

            # Create pivot table
            pivot_df = daily_df.pivot_table(
                values='avg_sentiment',
                index='weekday',
                columns='week',
                aggfunc='mean'
            )

            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                         'Friday', 'Saturday', 'Sunday']
            pivot_df = pivot_df.reindex(day_order)

            # Create heatmap
            fig, ax = plt.subplots(figsize=(15, 8))
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdBu_r',
                        center=0.5, ax=ax)
            ax.set_title('Sentiment Heatmap by Day and Week')
            ax.set_xlabel('Week Number')
            ax.set_ylabel('Day of Week')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/sentiment_heatmap.png", dpi=300)
            plt.close()

        logger.info(f"Visualizations saved to {output_dir}")


def main():
    """
    Demonstrate temporal analysis functionality
    """
    from config.spark_config import create_spark_session

    # Create Spark session
    spark = create_spark_session("TemporalAnalysis")

    # Load data
    logger.info("Loading data...")
    df = spark.read.parquet(str(get_path("data/processed/pipeline_features")))

    # Add time features
    df = df.withColumn("hour", F.hour(col("timestamp"))) \
        .withColumn("day_of_week", F.dayofweek(col("timestamp")))

    # Initialize analyzer
    analyzer = TemporalAnalyzer(spark)

    # Analyze trends
    trends = analyzer.analyze_sentiment_trends(df)

    # Aggregate hourly data
    hourly_df = analyzer.aggregate_sentiment_by_time(df, 'hourly')

    # Detect anomalies
    anomaly_df = analyzer.detect_sentiment_anomalies(hourly_df)

    # Calculate momentum
    momentum_df = analyzer.calculate_sentiment_momentum(anomaly_df)

    # Save results
    output_path = str(get_path("data/analytics/temporal"))
    os.makedirs(output_path, exist_ok=True)

    momentum_df.coalesce(1).write.mode("overwrite").parquet(
        f"{output_path}/sentiment_momentum"
    )

    # Create visualizations
    analyzer.create_temporal_visualizations(trends)

    # Show sample results
    logger.info("\nSample hourly sentiment trends:")
    hourly_df.select(
        "window_start", "tweet_count", "avg_sentiment",
        "positive_ratio", "avg_vader_compound"
    ).show(10)

    logger.info("\nDetected anomalies:")
    anomaly_df.filter(F.col("is_anomaly") == 1).select(
        "window_start", "avg_sentiment", "avg_sentiment_zscore", "anomaly_type"
    ).show(10)

    # Perform time series decomposition on daily data
    daily_pandas = trends['daily']
    if len(daily_pandas) > 14:  # Need at least 2 weeks
        decomp_results = analyzer.perform_time_series_decomposition(
            daily_pandas, period=7
        )
        logger.info(f"\nStationarity test: {decomp_results['stationarity']}")

    # Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()