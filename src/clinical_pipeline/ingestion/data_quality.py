"""Data quality checks for ingested clinical notes."""

from __future__ import annotations

from dataclasses import dataclass, field

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass
class QualityReport:
    row_count: int = 0
    null_rates: dict[str, float] = field(default_factory=dict)
    specialty_distribution: dict[str, int] = field(default_factory=dict)
    note_length_stats: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Row count: {self.row_count}",
            "",
            "Null rates:",
        ]
        for col, rate in sorted(self.null_rates.items()):
            lines.append(f"  {col}: {rate:.2%}")

        lines.append("")
        lines.append("Specialty distribution (top 15):")
        for specialty, count in sorted(
            self.specialty_distribution.items(), key=lambda x: -x[1]
        )[:15]:
            lines.append(f"  {specialty}: {count}")

        lines.append("")
        lines.append("Transcription length stats:")
        for stat, value in self.note_length_stats.items():
            lines.append(f"  {stat}: {value:.0f}")

        return "\n".join(lines)


def compute_null_rates(df: DataFrame) -> dict[str, float]:
    """Compute the fraction of nulls for each column."""
    total = df.count()
    if total == 0:
        return {c: 0.0 for c in df.columns}
    rates: dict[str, float] = {}
    for col_name in df.columns:
        null_count = df.where(F.col(col_name).isNull()).count()
        rates[col_name] = null_count / total
    return rates


def compute_specialty_distribution(df: DataFrame) -> dict[str, int]:
    """Count notes per medical specialty."""
    rows = (
        df.groupBy("medical_specialty")
        .count()
        .orderBy(F.desc("count"))
        .collect()
    )
    return {row["medical_specialty"] or "NULL": row["count"] for row in rows}


def compute_note_length_stats(df: DataFrame) -> dict[str, float]:
    """Compute min/max/mean/median length of transcription text."""
    df_with_len = df.withColumn("_len", F.length(F.col("transcription")))
    stats_row = df_with_len.agg(
        F.min("_len").alias("min"),
        F.max("_len").alias("max"),
        F.mean("_len").alias("mean"),
        F.expr("percentile_approx(_len, 0.5)").alias("median"),
    ).first()

    if stats_row is None:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}

    return {
        "min": float(stats_row["min"] or 0),
        "max": float(stats_row["max"] or 0),
        "mean": float(stats_row["mean"] or 0),
        "median": float(stats_row["median"] or 0),
    }


def run_quality_checks(df: DataFrame) -> QualityReport:
    """Run all quality checks and return a summary report."""
    return QualityReport(
        row_count=df.count(),
        null_rates=compute_null_rates(df),
        specialty_distribution=compute_specialty_distribution(df),
        note_length_stats=compute_note_length_stats(df),
    )
