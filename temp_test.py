from pyspark.sql import SparkSession, functions as F, types as T

spark = SparkSession.builder.getOrCreate()

# Inspect architecture **inside a worker**
arch = (
    spark.range(1)
         .mapInPandas(lambda it: [{"arch": __import__("platform").machine()}], schema="arch string")
         .collect()[0]["arch"]
)
print("worker arch =", arch)   # should print arm64 now
