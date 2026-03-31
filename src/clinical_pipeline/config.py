from pathlib import Path

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


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
