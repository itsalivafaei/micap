# utils/pandas_stream.py
def spark_to_pandas_stream(df, batch=20000):
    import pandas as pd, itertools as it
    return pd.concat(
        (pd.DataFrame(rows, columns=df.columns)
         for rows in df.toLocalIterator(batch)),
        ignore_index=True)
