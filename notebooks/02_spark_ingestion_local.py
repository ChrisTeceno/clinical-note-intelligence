"""
Clinical Note Intelligence -- PySpark Ingestion Pipeline (local version)

Same logic as the Databricks notebook but runs locally with spark-submit or plain Python.
Writes Parquet instead of Delta and uses print() for output.

Usage:
    PYTHONPATH=src conda run -n clinical-pipeline python notebooks/02_spark_ingestion_local.py
"""

from __future__ import annotations

import html
import re
import sys
import unicodedata
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mtsamples.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "mtsamples_clean.parquet"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
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

TEXT_COLUMNS = ["description", "transcription", "keywords", "sample_name"]

# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------


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
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder.master("local[*]")
        .appName("ClinicalNoteIntelligence-Local")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def main() -> None:
    if not RAW_DATA_PATH.exists():
        print(f"ERROR: CSV not found at {RAW_DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    spark = create_spark_session()
    clean_text_udf = F.udf(clean_text, StringType())

    try:
        # ---- Read ----
        print(f"Reading CSV from {RAW_DATA_PATH}")
        df_raw = (
            spark.read.option("header", "true")
            .option("multiLine", "true")
            .option("escape", '"')
            .schema(MTSAMPLES_SCHEMA)
            .csv(str(RAW_DATA_PATH))
        )
        raw_count = df_raw.count()
        print(f"Raw row count: {raw_count}")
        df_raw.show(5, truncate=80)

        # ---- Clean ----
        print("Cleaning text columns...")
        df_cleaned = df_raw
        for col_name in TEXT_COLUMNS:
            df_cleaned = df_cleaned.withColumn(col_name, clean_text_udf(F.col(col_name)))

        # ---- Deduplicate ----
        df_dedup = df_cleaned.dropDuplicates(["transcription"])
        dedup_count = df_dedup.count()
        print(f"After dedup: {dedup_count} (removed {raw_count - dedup_count} duplicates)")

        # ---- Quality report ----
        print("\n--- Data Quality Report ---")
        total = dedup_count
        print(f"Total rows: {total}\n")

        print("Null rates:")
        for col_name in df_dedup.columns:
            null_count = df_dedup.where(F.col(col_name).isNull()).count()
            rate = null_count / total if total > 0 else 0.0
            print(f"  {col_name}: {rate:.2%} ({null_count} nulls)")

        print("\nSpecialty distribution (top 15):")
        specialty_rows = (
            df_dedup.groupBy("medical_specialty")
            .count()
            .orderBy(F.desc("count"))
            .limit(15)
            .collect()
        )
        for row in specialty_rows:
            specialty = row["medical_specialty"] or "NULL"
            print(f"  {specialty}: {row['count']}")

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

        # ---- Write ----
        print(f"\nWriting partitioned Parquet to {OUTPUT_PATH}")
        (
            df_dedup.repartition("medical_specialty")
            .write.mode("overwrite")
            .partitionBy("medical_specialty")
            .parquet(str(OUTPUT_PATH))
        )

        # ---- Verify ----
        df_verify = spark.read.parquet(str(OUTPUT_PATH))
        print(f"\nVerified row count: {df_verify.count()}")
        df_verify.printSchema()

        print("Sample rows from 'Surgery':")
        (
            df_verify.where(F.col("medical_specialty") == " Surgery")
            .select("id", "sample_name", "medical_specialty")
            .show(5, truncate=80)
        )

        print("Ingestion complete.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
