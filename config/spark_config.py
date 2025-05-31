"""
Spark Configuration Module for MICAP
Handles Spark session creation and optimization for different environments
Optimized for M4 Mac development and production deployments
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_optimal_memory_config():
    """
    Get optimal memory configuration based on system resources
    
    Returns:
        Tuple of (driver_memory, executor_memory)
    """
    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if total_memory_gb >= 16:
            # High-memory system (16GB+)
            driver_memory = "8g"
            executor_memory = "6g"
        elif total_memory_gb >= 8:
            # Medium-memory system (8-16GB)
            driver_memory = "4g" 
            executor_memory = "3g"
        else:
            # Low-memory system (<8GB)
            driver_memory = "2g"
            executor_memory = "1g"
            
        logger.info(f"Detected {total_memory_gb:.1f}GB total memory")
        logger.info(f"Configured driver: {driver_memory}, executor: {executor_memory}")
        
        return driver_memory, executor_memory
        
    except ImportError:
        # Fallback if psutil not available
        logger.warning("psutil not available, using default memory settings")
        return "4g", "3g"


def create_spark_session(app_name: str, 
                        environment: str = "development",
                        enable_adaptive: bool = True,
                        enable_arrow: bool = False) -> SparkSession:
    """
    Create optimized Spark session for MICAP
    
    Args:
        app_name: Application name for Spark UI
        environment: deployment environment (development, testing, production)
        enable_adaptive: Enable adaptive query execution
        enable_arrow: Enable Arrow-based columnar data transfers
        
    Returns:
        Configured SparkSession
    """
    # Guarantee all workers use the same venv Python
    python_exec = sys.executable
    logger.info(f"Using Python executable: {python_exec}")
    
    # Set environment variables
    os.environ["PYSPARK_PYTHON"] = python_exec
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec
    
    # Get optimal memory configuration
    driver_memory, executor_memory = get_optimal_memory_config()
    logger.info(f"Creating Spark session with driver memory: {driver_memory}")
    
    # Base configuration
    conf = SparkConf().setAppName(app_name)
    
    # Environment-specific configurations
    if environment == "testing":
        # Minimal resources for testing
        conf.setMaster("local[1]")
        conf.set("spark.driver.memory", "1g")
        conf.set("spark.executor.memory", "1g") 
        conf.set("spark.sql.adaptive.enabled", "false")
        conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
        logger.info("Configured for testing environment")
        
    elif environment == "production":
        # Production optimizations
        conf.setMaster(os.getenv("SPARK_MASTER_URL", "local[*]"))
        conf.set("spark.driver.memory", driver_memory)
        conf.set("spark.executor.memory", executor_memory)
        conf.set("spark.sql.adaptive.enabled", str(enable_adaptive).lower())
        conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        conf.set("spark.sql.execution.arrow.pyspark.enabled", str(enable_arrow).lower())
        logger.info("Configured for production environment")
        
    else:
        # Development configuration (default)
        conf.setMaster("local[*]")
        conf.set("spark.driver.memory", driver_memory) 
        conf.set("spark.executor.memory", executor_memory)
        conf.set("spark.sql.adaptive.enabled", str(enable_adaptive).lower())
        conf.set("spark.sql.adaptive.coalescePartitions.enabled", str(enable_adaptive).lower())
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        conf.set("spark.sql.execution.arrow.pyspark.enabled", str(enable_arrow).lower())
        logger.info("Configured for development environment")
    
    # Common optimizations
    conf.set("spark.pyspark.python", python_exec)
    conf.set("spark.pyspark.driver.python", python_exec)  
    conf.set("spark.executorEnv.PYSPARK_PYTHON", python_exec)
    
    # M4 Mac specific optimizations
    conf.set("spark.local.dir", "/tmp/spark-temp")
    conf.set("spark.sql.shuffle.partitions", "200")
    conf.set("spark.default.parallelism", "8")
    
    # Improve stability and performance
    conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
    conf.set("spark.sql.execution.arrow.fallback.enabled", "true")
    conf.set("spark.driver.maxResultSize", "2g")
    conf.set("spark.network.timeout", "300s")
    conf.set("spark.executor.heartbeatInterval", "20s")
    
    # Word2Vec specific optimizations
    conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")  # Reduce batch size
    conf.set("spark.sql.shuffle.partitions", "100")  # Reduce from 200
    conf.set("spark.driver.maxResultSize", "4g")     # Increase result size
    conf.set("spark.executor.memory", "6g")          # Increase executor memory
    conf.set("spark.executor.memoryFraction", "0.8") # More memory for execution
    conf.set("spark.storage.memoryFraction", "0.6")  # More storage memory
    conf.set("spark.rdd.compress", "true")           # Enable RDD compression
    conf.set("spark.network.timeout", "600s")        # Increase network timeout
    conf.set("spark.executor.heartbeatInterval", "60s") # Increase heartbeat
    
    # Create session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    
    # Set log level
    spark.sparkContext.setLogLevel("WARN")
    
    # Log configuration
    logger.info("Spark session created successfully")
    logger.info(f"Spark version: {spark.version}")
    logger.info(f"Driver memory: {driver_memory}")
    logger.info(f"Executor memory: {executor_memory}")
    
    return spark


def create_minimal_spark_session(app_name: str) -> SparkSession:
    """
    Create minimal Spark session for testing and debugging
    
    Args:
        app_name: Application name
        
    Returns:
        Minimal SparkSession
    """
    return create_spark_session(
        app_name=app_name,
        environment="testing",
        enable_adaptive=False,
        enable_arrow=False
    )


def stop_spark_session(spark: SparkSession) -> None:
    """
    Safely stop Spark session with cleanup
    
    Args:
        spark: SparkSession to stop
    """
    try:
        if spark:
            spark.stop()
            logger.info("Spark session stopped successfully")
    except Exception as e:
        logger.warning(f"Error stopping Spark session: {e}")


# Environment detection
def get_environment() -> str:
    """
    Detect current environment from environment variables
    
    Returns:
        Environment string (development, testing, production)
    """
    env = os.getenv("ENV", "development").lower()
    if env in ["test", "testing"]:
        return "testing"
    elif env in ["prod", "production"]:
        return "production"
    else:
        return "development"


if __name__ == "__main__":
    # Test configuration
    env = get_environment()
    logger.info(f"Detected environment: {env}")
    
    spark = create_spark_session("ConfigTest", environment=env)
    
    # Test basic functionality
    df = spark.range(10).toDF("number")
    count = df.count()
    logger.info(f"Test DataFrame count: {count}")
    
    stop_spark_session(spark)
