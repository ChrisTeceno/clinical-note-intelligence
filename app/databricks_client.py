"""Databricks SQL API client for querying Delta tables."""

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Load from .env
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
_env_vars: dict[str, str] = {}
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            _env_vars[key.strip()] = val.strip()

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", _env_vars.get("DATABRICKS_HOST", ""))
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", _env_vars.get("DATABRICKS_TOKEN", ""))
DATABRICKS_WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", _env_vars.get("DATABRICKS_WAREHOUSE_ID", ""))

DELTA_TABLE = "delta.`/Volumes/workspace/default/raw-data/mtsamples_clean`"


def is_configured() -> bool:
    return bool(DATABRICKS_HOST and DATABRICKS_TOKEN and DATABRICKS_WAREHOUSE_ID)


def _execute_sql(sql: str) -> dict:
    """Execute a SQL statement via the Databricks SQL Statement API."""
    url = f"{DATABRICKS_HOST}/api/2.0/sql/statements"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "warehouse_id": DATABRICKS_WAREHOUSE_ID,
            "statement": sql,
            "wait_timeout": "30s",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60)
def query_databricks(sql: str) -> pd.DataFrame:
    """Execute SQL and return a DataFrame."""
    result = _execute_sql(sql)

    if result.get("status", {}).get("state") != "SUCCEEDED":
        error = result.get("status", {}).get("error", {}).get("message", "Unknown error")
        raise RuntimeError(f"Databricks query failed: {error}")

    manifest = result.get("manifest", {})
    columns = [c["name"] for c in manifest.get("schema", {}).get("columns", [])]
    rows = result.get("result", {}).get("data_array", [])

    return pd.DataFrame(rows, columns=columns)
