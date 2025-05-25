"""
Data Ingestion Module for MICAP
Handles loading, validation, and partitioning of tweet data
Optimized for local processing on M4 Mac
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, from_unixtime, to_timestamp, date_format,
    when, isnan, isnull, count, trim, length
)
from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, TimestampType
)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.spark_config import create_spark_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Handles data ingestion for sentiment analysis
    Includes validation, cleaning, and partitioning
    """

    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize DataIngestion with Spark session

        Args:
            spark: Optional SparkSession, creates new if not provided
        """
        self.spark = spark or create_spark_session("DataIngestion")

        # Define schema for Sentiment140 dataset
        self.schema = StructType([
            StructField("polarity", IntegerType(), True),  # 0 = negative, 4 = positive
            StructField("tweet_id", StringType(), True),  # Tweet ID
            StructField("date", StringType(), True),  # Tweet date
            StructField("query", StringType(), True),  # Query (NO_QUERY)
            StructField("user", StringType(), True),  # Username
            StructField("text", StringType(), True)  # Tweet text
        ])

    def load_sentiment140_data(self, file_path: str) -> DataFrame:
        """
        Load Sentiment140 dataset with proper schema and initial cleaning

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame: Loaded and initially processed data
        """
        logger.info(f"Loading data from: {file_path}")

        try:
            # Load CSV with schema
            df = self.spark.read.csv(
                file_path,
                schema=self.schema,
                encoding='iso-8859-1',  # Sentiment140 uses latin-1 encoding
                header=False,
                mode='DROPMALFORMED'  # Skip malformed records
            )

            # Initial data processing
            df = df.filter(col("text").isNotNull() & (length(trim(col("text"))) > 0))

            # Convert polarity: 0 (negative) -> 0, 4 (positive) -> 1
            df = df.withColumn(
                "sentiment",
                when(col("polarity") == 0, 0)
                .when(col("polarity") == 4, 1)
                .otherwise(None)
            ).drop("polarity")

            # Parse date string to timestamp
            # Sentiment140 date format: "Mon May 11 03:17:40 UTC 2009"
            self.spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
            df = df.withColumn(
                "timestamp",
                to_timestamp(col("date"), "EEE MMM dd HH:mm:ss zzz yyyy")
            )

            # Add date partitioning columns for efficient querying
            df = df.withColumn("year", date_format(col("timestamp"), "yyyy")) \
                .withColumn("month", date_format(col("timestamp"), "MM")) \
                .withColumn("day", date_format(col("timestamp"), "dd")) \
                .withColumn("hour", date_format(col("timestamp"), "HH"))

            logger.info(f"Successfully loaded {df.count()} records")

            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def validate_data_quality(self, df: DataFrame) -> Tuple[DataFrame, dict]:
        """
        Validate data quality and return cleaned data with quality metrics

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (cleaned DataFrame, quality metrics dict)
        """
        logger.info("Starting data quality validation...")

        initial_count = df.count()

        # Quality checks
        quality_metrics = {
            "initial_count": initial_count,
            "null_text_count": df.filter(col("text").isNull()).count(),
            "empty_text_count": df.filter(length(trim(col("text"))) == 0).count(),
            "null_sentiment_count": df.filter(col("sentiment").isNull()).count(),
            "invalid_date_count": df.filter(col("timestamp").isNull()).count(),
        }

        # Remove records with null or empty text
        df_clean = df.filter(
            col("text").isNotNull() &
            (length(trim(col("text"))) > 0) &
            col("sentiment").isNotNull() &
            col("timestamp").isNotNull()
        )

        quality_metrics["final_count"] = df_clean.count()
        quality_metrics["removed_count"] = initial_count - quality_metrics["final_count"]
        quality_metrics["removal_percentage"] = (
                quality_metrics["removed_count"] / initial_count * 100
        )

        # Log quality metrics
        logger.info("Data Quality Report:")
        for metric, value in quality_metrics.items():
            logger.info(f"  {metric}: {value}")

        return df_clean, quality_metrics

    def partition_by_date(self, df: DataFrame, output_path: str) -> str:
        """
        Partition data by date for efficient processing
        Optimized for local file system on M4 Mac

        Args:
            df: DataFrame to partition
            output_path: Base path for partitioned data

        Returns:
            str: Path to partitioned data
        """
        logger.info(f"Partitioning data to: {output_path}")

        # For local development, we'll use fewer partitions
        # to avoid too many small files
        df_repartitioned = df.repartition(10, "year", "month")

        # Write partitioned data
        df_repartitioned.write \
            .mode("overwrite") \
            .partitionBy("year", "month") \
            .parquet(output_path)

        logger.info("Data partitioning completed")
        return output_path

    def save_to_local_storage(self, df: DataFrame, path: str, format: str = "parquet"):
        """
        Save DataFrame to local storage with optimization for M4 Mac

        Args:
            df: DataFrame to save
            path: Output path
            format: Output format (parquet, csv, json)
        """
        logger.info(f"Saving data to {path} in {format} format")

        # Coalesce to reduce number of files for local storage
        df_coalesced = df.coalesce(4)  # 4 files for 4 efficiency cores

        if format == "parquet":
            df_coalesced.write.mode("overwrite").parquet(path)
        elif format == "csv":
            df_coalesced.write.mode("overwrite").option("header", True).csv(path)
        elif format == "json":
            df_coalesced.write.mode("overwrite").json(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Data saved successfully")

    def create_sample_dataset(self, df: DataFrame, sample_size: float = 0.01) -> DataFrame:
        """
        Create a sample dataset for development and testing
        Useful for quick iterations on M4 Mac

        Args:
            df: Full DataFrame
            sample_size: Fraction of data to sample (default 1%)

        Returns:
            DataFrame: Sampled data
        """
        logger.info(f"Creating sample dataset with {sample_size * 100}% of data")

        # Stratified sampling to maintain sentiment distribution
        df_sample = df.sampleBy("sentiment",
                                fractions={0: sample_size, 1: sample_size},
                                seed=42)

        sample_count = df_sample.count()
        logger.info(f"Sample dataset created with {sample_count} records")

        return df_sample


def main():
    """
    Main function to demonstrate data ingestion
    """
    # Initialize ingestion
    ingestion = DataIngestion()

    # Load data
    data_path = "/Users/ali/Documents/Projects/micap/data/raw/testdata.manual.2009.06.14.csv"
    df = ingestion.load_sentiment140_data(data_path)

    # Validate data quality
    df_clean, quality_metrics = ingestion.validate_data_quality(df)

    # Create sample for development
    df_sample = ingestion.create_sample_dataset(df_clean, sample_size=0.01)

    # Save sample data
    ingestion.save_to_local_storage(
        df_sample,
        "/Users/ali/Documents/Projects/micap/data/processed/sentiment140_sample",
        format="parquet"
    )

    # Partition full data
    ingestion.partition_by_date(
        df_clean,
        "/Users/ali/Documents/Projects/micap/data/processed/sentiment140_partitioned"
    )

    # Show sample records
    logger.info("Sample records:")
    df_sample.select("tweet_id", "text", "sentiment", "timestamp").show(5, truncate=True)

    # Stop Spark session
    ingestion.spark.stop()


if __name__ == "__main__":
    main()