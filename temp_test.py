#!/usr/bin/env python
"""Test script to verify Spark and NumPy work together on ARM64"""

import os
import sys
import platform

print("=== System Information ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Machine: {platform.machine()}")
print(f"JAVA_HOME: {os.environ.get('JAVA_HOME', 'Not set')}")

# Test Java
import subprocess

try:
    java_version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT)
    print(f"\nJava version output:\n{java_version.decode()}")
except Exception as e:
    print(f"Error checking Java: {e}")

# Test NumPy locally
print("\n=== Testing NumPy locally ===")
try:
    import numpy as np

    print(f"NumPy version: {np.__version__}")
    print(f"NumPy location: {np.__file__}")
    arr = np.array([1, 2, 3])
    print(f"NumPy test: {arr} - Success!")
except Exception as e:
    print(f"NumPy error: {e}")

# Test PySpark
print("\n=== Testing PySpark ===")
try:
    from pyspark.sql import SparkSession

    # Create session
    spark = SparkSession.builder \
        .appName("TestNumPy") \
        .master("local[2]") \
        .config("spark.pyspark.python", sys.executable) \
        .config("spark.pyspark.driver.python", sys.executable) \
        .getOrCreate()

    print("Spark session created successfully")

    # Test NumPy in Spark
    print("\n=== Testing NumPy in Spark workers ===")


    def test_numpy(_):
        import numpy as np
        return [(np.__version__, np.array([1, 2, 3]).sum())]


    rdd = spark.sparkContext.parallelize([1])
    result = rdd.mapPartitions(test_numpy).collect()

    print(f"NumPy in Spark worker - Version: {result[0][0]}, Sum test: {result[0][1]}")
    print("✅ Success! NumPy works in Spark workers")

    spark.stop()

except Exception as e:
    print(f"❌ PySpark error: {e}")
    import traceback

    traceback.print_exc()

print("\n=== Test Complete ===")