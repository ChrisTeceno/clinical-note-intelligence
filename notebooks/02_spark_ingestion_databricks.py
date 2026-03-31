# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # Clinical Note Intelligence -- PySpark Ingestion Pipeline
# MAGIC Pipeline that ingests MTSamples clinical transcriptions, cleans text,
# MAGIC deduplicates, runs quality checks, and writes to partitioned Delta tables.

# COMMAND ----------

# Configuration
RAW_DATA_PATH = "/Volumes/workspace/default/raw-data/mtsamples.csv"
OUTPUT_PATH = "/Volumes/workspace/default/raw-data/mtsamples_clean"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schema Definition
# MAGIC Enforce types on read to catch malformed rows early.

# COMMAND ----------

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

MTSAMPLES_SCHEMA = StructType(
    [
        StructField("id", IntegerType(), nullable=False),
        StructField("description", StringType(), nullable=True),
        StructField("medical_specialty", StringType(), nullable=True),
        StructField("sample_name", StringType(), nullable=True),
        StructField("transcription", StringType(), nullable=True),
        StructField("keywords", StringType(), nullable=True),
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read CSV
# MAGIC Load the raw MTSamples CSV with the enforced schema.

# COMMAND ----------

df_raw = (
    spark.read.option("header", "true")
    .option("multiLine", "true")
    .option("escape", '"')
    .schema(MTSAMPLES_SCHEMA)
    .csv(RAW_DATA_PATH)
)

raw_count = df_raw.count()
print(f"Raw row count: {raw_count}")
display(df_raw.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text Cleaning Functions
# MAGIC Self-contained cleaning pipeline: strip HTML, fix encoding artifacts,
# MAGIC and normalize whitespace. Registered as PySpark UDFs.

# COMMAND ----------

import html
import re
import unicodedata

from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def strip_html_tags(text):
    """Remove HTML tags and decode HTML entities."""
    if text is None:
        return None
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return text


def normalize_whitespace(text):
    """Collapse runs of whitespace into single spaces and strip edges."""
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_encoding(text):
    """Normalize Unicode and replace common smart-quote / dash artifacts."""
    if text is None:
        return None
    text = unicodedata.normalize("NFKD", text)
    replacements = {
        "\u2019": "'",   # right single quote
        "\u2018": "'",   # left single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2026": "...", # ellipsis
        "\u00a0": " ",   # non-breaking space
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def clean_text(text):
    """Full cleaning pipeline: HTML removal, encoding fix, whitespace normalization."""
    text = strip_html_tags(text)
    text = fix_encoding(text)
    text = normalize_whitespace(text)
    return text


clean_text_udf = F.udf(clean_text, StringType())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Cleaning
# MAGIC Run `clean_text` over all free-text columns.

# COMMAND ----------

TEXT_COLUMNS = ["description", "transcription", "keywords", "sample_name"]

df_cleaned = df_raw
for col_name in TEXT_COLUMNS:
    df_cleaned = df_cleaned.withColumn(col_name, clean_text_udf(F.col(col_name)))

# Show before/after for a single row
print("=== BEFORE cleaning (raw) ===")
display(df_raw.select("id", "transcription").limit(1))
print("=== AFTER cleaning ===")
display(df_cleaned.select("id", "transcription").limit(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deduplication
# MAGIC Drop exact duplicates on the transcription column.

# COMMAND ----------

count_before = df_cleaned.count()
df_dedup = df_cleaned.dropDuplicates(["transcription"])
count_after = df_dedup.count()

print(f"Before dedup: {count_before}")
print(f"After dedup:  {count_after}")
print(f"Removed:      {count_before - count_after} duplicate rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Report
# MAGIC Null rates, specialty distribution, and transcription length statistics.

# COMMAND ----------

# --- Total row count ---
total = df_dedup.count()
print(f"Total rows: {total}\n")

# --- Null rates per column ---
print("Null rates:")
for col_name in df_dedup.columns:
    null_count = df_dedup.where(F.col(col_name).isNull()).count()
    rate = null_count / total if total > 0 else 0.0
    print(f"  {col_name}: {rate:.2%} ({null_count} nulls)")

# --- Top 15 specialty distribution (bar chart) ---
print("\nSpecialty distribution (top 15):")
df_specialty = (
    df_dedup.groupBy("medical_specialty")
    .count()
    .orderBy(F.desc("count"))
    .limit(15)
)
display(df_specialty)

# --- Transcription length statistics ---
df_with_len = df_dedup.withColumn("transcription_length", F.length(F.col("transcription")))
length_stats = df_with_len.agg(
    F.min("transcription_length").alias("min_length"),
    F.max("transcription_length").alias("max_length"),
    F.mean("transcription_length").alias("mean_length"),
    F.expr("percentile_approx(transcription_length, 0.5)").alias("median_length"),
).first()

print("\nTranscription length statistics:")
print(f"  Min:    {length_stats['min_length']}")
print(f"  Max:    {length_stats['max_length']}")
print(f"  Mean:   {length_stats['mean_length']:.0f}")
print(f"  Median: {length_stats['median_length']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Delta
# MAGIC Persist the cleaned, deduplicated data as a Delta table partitioned by specialty.

# COMMAND ----------

(
    df_dedup.repartition("medical_specialty")
    .write.format("delta")
    .mode("overwrite")
    .partitionBy("medical_specialty")
    .save(OUTPUT_PATH)
)

print(f"Delta table written to {OUTPUT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Output
# MAGIC Read the Delta table back and confirm the data is intact.

# COMMAND ----------

# Verify file output
dbutils.fs.ls(OUTPUT_PATH)

# COMMAND ----------

df_verify = spark.read.format("delta").load(OUTPUT_PATH)
print(f"Verified row count: {df_verify.count()}")
df_verify.printSchema()

# Query a sample by specialty
print("\nSample rows from 'Surgery':")
display(
    df_verify.where(F.col("medical_specialty") == " Surgery")
    .select("id", "sample_name", "medical_specialty")
    .limit(5)
)

# COMMAND ----------

import urllib.request                                                                
try:                                                                                 
      resp = urllib.request.urlopen("https://api.anthropic.com/v1/messages", timeout=5)
except urllib.error.HTTPError as e:                                                  
      print(f"API reachable! (got HTTP {e.code} — expected, no auth header)")          
except urllib.error.URLError as e:                                                   
      print(f"BLOCKED: {e.reason}")                                                    
except Exception as e:                                                               
      print(f"Error: {e}")  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Raw rows loaded | *see output above* |
# MAGIC | Rows after dedup | *see output above* |
# MAGIC | Output format | Delta (partitioned by `medical_specialty`) |
# MAGIC | Output path | `/FileStore/tables/clinical_pipeline/mtsamples_clean` |
# MAGIC
# MAGIC ### Next Steps
# MAGIC 1. **Extraction pipeline** -- run locally with Claude API (Haiku 4.5 via Batch API)
# MAGIC    to extract structured clinical variables (diagnoses, procedures, medications, ICD-10 codes).
# MAGIC 2. **ICD-10 mapping** -- cross-reference extracted codes against CMS code tables.
# MAGIC 3. **HITL validation** -- review extracted data through the Streamlit dashboard.