# Databricks notebook source
# MAGIC %sql
# MAGIC create catalog if not exists lakehouse;
# MAGIC create schema if not exists  lakehouse.myschema;
# MAGIC create volume if not exists lakehouse.myschema.myvolume;

# COMMAND ----------

# MAGIC %md
# MAGIC # Beginner Medallion Demo (Duplicates Only)
# MAGIC **Goal:** Ingest clean CSVs to Bronze, remove duplicates in Silver, build a simple Gold table.
# MAGIC
# MAGIC **Duplicate rules:**
# MAGIC - Customers: keep the newest row by `updated_at` per `customer_id`.
# MAGIC - Orders: keep the newest row by `order_date` per `order_id`.
# MAGIC
# MAGIC Upload CSVs to `/Volumes/lakehouse/myschema/myvolume/raw/`:
# MAGIC - customers.csv
# MAGIC - orders.csv

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0) Config

# COMMAND ----------

dbutils.widgets.text("catalog", "lakehouse")
dbutils.widgets.text("schema", "myschema")
dbutils.widgets.text("raw_path", "/Volumes/lakehouse/myschema/myvolume/raw/", "Raw CSV Path")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
raw_path = dbutils.widgets.get("raw_path").rstrip("/")

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
spark.sql(f"USE SCHEMA {schema}")

print(f"Using {catalog}.{schema}")
print(f"Raw path: {raw_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Bronze — Load CSVs as-is (Delta)

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists lakehouse.myschema.bronze_customers;
# MAGIC drop table if exists lakehouse.myschema.bronze_orders

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.window import Window

bronze_customers = (spark.read.format("csv")
    .option("header", True)
    .option("inferSchema", True)
    .load(f"{raw_path}/customers.csv"))

bronze_orders = (spark.read.format("csv")
    .option("header", True)
    .option("inferSchema", True)
    .load(f"{raw_path}/orders.csv"))

bronze_customers.write.mode("overwrite").format("delta").saveAsTable("lakehouse.myschema.bronze_customers")
bronze_orders.write.mode("overwrite").format("delta").saveAsTable("lakehouse.myschema.bronze_orders")

display(spark.table("lakehouse.myschema.bronze_customers"))
display(spark.table("lakehouse.myschema.bronze_orders").orderBy("order_id", "order_date"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Silver — Deduplicate

# COMMAND ----------

# Customers: keep newest by updated_at per customer_id
c = spark.table("lakehouse.myschema.bronze_customers").withColumn("updated_ts", to_timestamp("updated_at"))

w_c = Window.partitionBy("customer_id").orderBy(col("updated_ts").desc_nulls_last())
silver_customers = (c
    .withColumn("rn", row_number().over(w_c))
    .filter(col("rn") == 1)
    .drop("rn", "updated_ts")
)

silver_customers.write.mode("overwrite").format("delta").saveAsTable("lakehouse.myschema.silver_customers")

# Orders: keep newest by order_date per order_id
o = spark.table("lakehouse.myschema.bronze_orders").withColumn("order_ts", to_timestamp("order_date"))

w_o = Window.partitionBy("order_id").orderBy(col("order_ts").desc_nulls_last())
silver_orders = (o
    .withColumn("rn", row_number().over(w_o))
    .filter(col("rn") == 1)
    .drop("rn", "order_ts")
)

silver_orders.write.mode("overwrite").format("delta").saveAsTable("lakehouse.myschema.silver_orders")

display(spark.table("lakehouse.myschema.silver_customers").orderBy("customer_id"))
display(spark.table("lakehouse.myschema.silver_orders").orderBy("order_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Gold — Daily sales summary

# COMMAND ----------

gold_sales_by_day = (spark.table("lakehouse.myschema.silver_orders")
    .withColumn("order_ts", to_timestamp("order_date"))
    .withColumn("sales", col("qty") * col("unit_price"))
    .groupBy(to_date(col("order_ts")).alias("order_date"))
    .agg(sum("sales").alias("total_sales"), count("*").alias("order_count"))
)

gold_sales_by_day.write.mode("overwrite").format("delta").saveAsTable("lakehouse.myschema.gold_sales_by_day")

display(spark.table("lakehouse.myschema.gold_sales_by_day").orderBy("order_date"))