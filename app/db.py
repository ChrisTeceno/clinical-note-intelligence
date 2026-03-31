"""Shared database helpers for the Streamlit dashboard."""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "clinical_notes.db"


def get_connection() -> sqlite3.Connection:
    """Return a sqlite3 connection with row factory enabled."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=30)
def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute *sql* and return the result as a DataFrame. Cached for 30s."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    return df


def execute(sql: str, params: tuple = ()) -> None:
    """Execute a write statement (UPDATE / INSERT)."""
    conn = get_connection()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()
