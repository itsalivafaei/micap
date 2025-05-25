"""
Spark configuration optimized for MacBook Air M4 24GB RAM
This module handles Spark session creation with optimal settings for local development
"""

import os
from pyspark.sql import SparkSession
from pyspark import SparkConf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session(app_name="MICAP", local_mode=True):
    """
    Create and configure Spark session optimized for M4 Mac with 24GB RAM

    Args:
        app_name (str): Name of the Spark application
        local_mode (bool): Whether to run in local mode (True for development)

    Returns:
        SparkSession: Configured Spark session
    """

    # Calculate memory allocation (leaving 4GB for system)
    total_memory = 20  # GB available for Spark
    driver_memory = f"{int(total_memory * 0.6)}g"  # 60% for driver
    executor_memory = f"{int(total_memory * 0.4)}g"  # 40% for executors

    # Create Spark configuration
    conf = SparkConf().setAppName(app_name)

    if local_mode:
        # Local mode configuration for M4 Mac
        conf.setMaster("local[*]")  # Use all available cores

        # Memory settings
        conf.set("spark.driver.memory", driver_memory)
        conf.set("spark.executor.memory", executor_memory)

        # Optimization for M4 architecture
        conf.set("spark.sql.adaptive.enabled", "true")
        conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")

        # Tune for local SSD performance
        conf.set("spark.local.dir", "/tmp/spark-temp")
        conf.set("spark.sql.shuffle.partitions", "100")  # Reduced for local mode

        # UI settings
        conf.set("spark.ui.port", "4040")
        conf.set("spark.ui.showConsoleProgress", "true")

    logger.info(f"Creating Spark session with driver memory: {driver_memory}")

    # Create session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # Set log level
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Spark session created successfully")
    logger.info(f"Spark version: {spark.version}")
    logger.info(f"Driver memory: {driver_memory}")
    logger.info(f"Executor memory: {executor_memory}")

    return spark