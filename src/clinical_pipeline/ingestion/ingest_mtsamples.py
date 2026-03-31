"""Main ingestion entry point: read MTSamples CSV, clean, deduplicate, write to parquet."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from clinical_pipeline.config import Settings, get_settings
from clinical_pipeline.ingestion.cleaners import register_udfs
from clinical_pipeline.ingestion.data_quality import run_quality_checks
from clinical_pipeline.ingestion.schema import MTSAMPLES_SCHEMA
from clinical_pipeline.ingestion.spark_session import create_spark_session

logger = logging.getLogger(__name__)

TEXT_COLUMNS = ["description", "transcription", "keywords", "sample_name"]


def read_mtsamples(spark: SparkSession, csv_path: Path) -> DataFrame:
    """Read MTSamples CSV with enforced schema."""
    return (
        spark.read.option("header", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .schema(MTSAMPLES_SCHEMA)
        .csv(str(csv_path))
    )


def clean_dataframe(df: DataFrame) -> DataFrame:
    """Apply text cleaning UDF to all text columns."""
    udfs = register_udfs()
    clean_text_udf = udfs["clean_text"]
    for col_name in TEXT_COLUMNS:
        df = df.withColumn(col_name, clean_text_udf(F.col(col_name)))
    return df


def deduplicate(df: DataFrame) -> DataFrame:
    """Drop exact duplicates on transcription text and keep the first id."""
    return df.dropDuplicates(["transcription"])


def ingest(settings: Settings | None = None) -> None:
    """Run the full ingestion pipeline."""
    settings = settings or get_settings()
    csv_path = settings.raw_dir / "mtsamples.csv"
    output_path = settings.processed_dir / "mtsamples.parquet"

    if not csv_path.exists():
        logger.error("MTSamples CSV not found at %s", csv_path)
        sys.exit(1)

    spark = create_spark_session(master=settings.spark_master)

    try:
        logger.info("Reading CSV from %s", csv_path)
        df = read_mtsamples(spark, csv_path)
        raw_count = df.count()
        logger.info("Raw rows: %d", raw_count)

        logger.info("Cleaning text columns")
        df = clean_dataframe(df)

        logger.info("Deduplicating")
        df = deduplicate(df)
        dedup_count = df.count()
        logger.info("Rows after dedup: %d (removed %d)", dedup_count, raw_count - dedup_count)

        logger.info("Running data quality checks")
        report = run_quality_checks(df)
        logger.info("Quality report:\n%s", report)

        logger.info("Writing partitioned parquet to %s", output_path)
        (
            df.repartition("medical_specialty")
            .write.mode("overwrite")
            .partitionBy("medical_specialty")
            .parquet(str(output_path))
        )

        logger.info("Ingestion complete")
    finally:
        spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ingest()
