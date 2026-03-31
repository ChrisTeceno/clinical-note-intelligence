from __future__ import annotations

import os

from pyspark.sql import SparkSession


def _is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def create_spark_session(
    app_name: str = "ClinicalNoteIntelligence",
    master: str = "local[*]",
) -> SparkSession:
    """Create a SparkSession for local or Databricks execution.

    On Databricks the runtime provides a pre-configured session, so we
    retrieve it via ``SparkSession.builder.getOrCreate()``.  Locally we
    build a fresh session with sensible defaults for single-machine work.
    """
    if _is_databricks():
        return SparkSession.builder.getOrCreate()

    return (
        SparkSession.builder.master(master)
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
