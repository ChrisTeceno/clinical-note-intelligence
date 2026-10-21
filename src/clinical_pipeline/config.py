from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude access (required for extraction)",
    )
    openai_compat_api_key: str = Field(
        default="",
        description="Bearer token for an OpenAI-compatible extraction server (omlx needs one)",
    )
    database_url: str = Field(
        default="sqlite:///data/clinical_notes.db",
        description="Database connection string (SQLite default, Postgres for production)",
    )
    data_dir: Path = Field(
        default=Path("./data"),
        description="Root directory for data files",
    )
    spark_master: str = Field(
        default="local[*]",
        description="Spark master URL. Use 'local[*]' for local mode.",
    )
    mimic_source: Literal["demo", "full"] = Field(
        default="demo",
        description="Which MIMIC-IV dataset to evaluate against: 'demo' or 'full'",
    )
    mimic_demo_root: Path = Field(
        default=Path.home() / "Downloads" / "mimic-iv-clinical-database-demo-2.2",
        description="Root of the MIMIC-IV clinical database demo (contains hosp/)",
    )
    mimic_full_root: Path = Field(
        default=Path.home() / "physionet.org" / "files" / "mimiciv" / "3.1",
        description="Root of the full MIMIC-IV 3.1 download (contains hosp/ and icu/)",
    )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reference_dir(self) -> Path:
        return self.data_dir / "reference"

    @property
    def imaging_dir(self) -> Path:
        return self.data_dir / "imaging"

    @property
    def mimic_root(self) -> Path:
        return self.mimic_full_root if self.mimic_source == "full" else self.mimic_demo_root

    @property
    def mimic_hosp_dir(self) -> Path:
        return self.mimic_root / "hosp"

    @property
    def evaluation_dir(self) -> Path:
        # Full-data output lives under data/full/ so it never lands in the tracked demo caches
        if self.mimic_source == "full":
            return self.data_dir / "full" / "evaluation"
        return self.data_dir / "evaluation"

    @property
    def optimization_dir(self) -> Path:
        if self.mimic_source == "full":
            return self.data_dir / "full" / "optimization"
        return self.data_dir / "optimization"


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
