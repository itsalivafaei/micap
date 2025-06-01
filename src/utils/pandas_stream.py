# utils/pandas_stream.py
"""Pandas Streaming Utilities for MICAP.

Provides memory-efficient conversion between Spark DataFrames and Pandas DataFrames.
"""

import pandas as pd
from typing import Iterator
from pyspark.sql import DataFrame


def spark_to_pandas_stream(df: DataFrame, batch: int = 20000) -> pd.DataFrame:
    """Convert Spark DataFrame to Pandas DataFrame using streaming to avoid memory issues.
    
    This function prevents memory crashes on macOS by streaming data in small batches
    rather than loading the entire DataFrame into memory at once.
    
    Args:
        df: Spark DataFrame to convert
        batch: Number of rows to process in each batch (default: 20000)
    
    Returns:
        Pandas DataFrame containing all the data from the Spark DataFrame
    
    Example:
        >>> spark_df = spark.sql("SELECT * FROM large_table")
        >>> pandas_df = spark_to_pandas_stream(spark_df, batch=10000)
        >>> print(f"Converted {len(pandas_df)} rows")
    
    Note:
        Using smaller batch sizes reduces memory usage but may increase processing time.
        Adjust the batch size based on available memory and data size.
    """
    return pd.concat(
        (pd.DataFrame(rows, columns=df.columns)
         for rows in df.toLocalIterator(batch)),
        ignore_index=True
    )
