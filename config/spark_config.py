"""
Spark configuration optimized for MacBook Air M4 24GB RAM
This module handles Spark session creation with optimal settings for local development
FIXED: Added proper environment variable handling to prevent numpy import issues
"""

import os
import sys
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

    # FIX: Set Python executable paths to ensure consistent environment
    python_exec = sys.executable  # Gets the current Python interpreter path

    # Set environment variables to ensure all Spark processes use the same Python
    os.environ["PYSPARK_PYTHON"] = python_exec
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec
    use_gpu = os.environ.get("ENABLE_GPU_LIBS") == "1"

    logger.info(f"Using Python executable: {python_exec}")

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

        # FIX: Explicitly set Python executable in Spark config
        conf.set("spark.pyspark.python", python_exec)
        conf.set("spark.pyspark.driver.python", python_exec)
        conf.set("spark.executorEnv.PYSPARK_PYTHON", python_exec)

        # Optimization for M4 architecture
        conf.set("spark.sql.adaptive.enabled", "true")
        conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")

        #### New
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "8000")
        conf.set("spark.sql.shuffle.partitions", "24")
        conf.set("spark.driver.maxResultSize", "4g")
        conf.set("spark.network.timeout", "600s")
        conf.set("spark.storage.memoryFraction", "0.4")
        conf.set("spark.executor.memoryOverhead", "2048")


        # Tune for local SSD performance
        conf.set("spark.local.dir", "/tmp/spark-temp")
        conf.set("spark.sql.shuffle.partitions", "100")  # Reduced for local mode

        # UI settings
        conf.set("spark.ui.port", "4040")
        conf.set("spark.ui.showConsoleProgress", "true")

        # FIX: Add serialization settings that might help with numpy issues
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")  # Disable Arrow for now

        # ---------------------------------
        # production / fork-safe profile
        # ---------------------------------
        # these three lines fix the executor crashes
        # conf.set("spark.python.use.daemon", "false")
        # conf.set("spark.python.worker.reuse", "false")
        conf.set("spark.python.use.daemon", str(not use_gpu).lower())
        conf.set("spark.python.worker.reuse", str(not use_gpu).lower())
        conf.set("spark.executorEnv.PYTORCH_ENABLE_MPS_FALLBACK", "1")

        # still propagate the Apple flag as defence in depth
        conf.set("spark.driverEnv.OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
        conf.set("spark.executorEnv.OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

        conf.set("spark.python.worker.faulthandler.enabled", "true")

        # conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        # conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 8000)


    # for production fork-safe vs fast profiling of python fork()
    # if local_mode:
    #     conf.set("spark.driverEnv.OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    #     conf.set("spark.executorEnv.OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    # else:
    #     # 1) Never reuse the Python daemon → no fork() after GPU libs load
    #     conf.set("spark.python.use.daemon", "false")
    #     conf.set("spark.python.worker.reuse", "false")
    #     # 2) ALSO propagate Apple’s flag as extra belt-and-braces
    #     conf.set("spark.driverEnv.OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    #     conf.set("spark.executorEnv.OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

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