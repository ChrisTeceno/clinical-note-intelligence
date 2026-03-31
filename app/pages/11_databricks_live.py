"""Databricks Live — query Delta tables directly from Databricks."""

import altair as alt
import streamlit as st

st.title("Databricks Live Data")
st.caption("Real-time queries against the Delta table on Databricks serverless compute.")

# Import the client (lazy — only fails if credentials missing)
try:
    from databricks_client import DELTA_TABLE, is_configured, query_databricks
except ImportError:
    st.error("Databricks client not available.")
    st.stop()

if not is_configured():
    st.warning(
        "Databricks credentials not configured. Add DATABRICKS_HOST, "
        "DATABRICKS_TOKEN, and DATABRICKS_WAREHOUSE_ID to your .env file."
    )
    st.stop()

st.info(
    "**Live connection** to Databricks serverless SQL warehouse. "
    "Data is queried in real-time from the Delta table written by the PySpark ingestion pipeline.",
    icon=":material/cloud:",
)

# ---------------------------------------------------------------------------
# KPIs from Delta table
# ---------------------------------------------------------------------------
try:
    count_df = query_databricks(f"SELECT COUNT(*) as total FROM {DELTA_TABLE}")
    total_notes = int(count_df.iloc[0]["total"])
except Exception as e:
    st.error(f"Failed to connect to Databricks: {e}")
    st.stop()

specialty_df = query_databricks(f"""
    SELECT medical_specialty, COUNT(*) as count
    FROM {DELTA_TABLE}
    WHERE medical_specialty IS NOT NULL
    GROUP BY medical_specialty
    ORDER BY count DESC
""")

note_length_df = query_databricks(f"""
    SELECT
        medical_specialty,
        COUNT(*) as notes,
        ROUND(AVG(LENGTH(transcription))) as avg_length,
        MIN(LENGTH(transcription)) as min_length,
        MAX(LENGTH(transcription)) as max_length
    FROM {DELTA_TABLE}
    WHERE transcription IS NOT NULL
    GROUP BY medical_specialty
    ORDER BY notes DESC
    LIMIT 15
""")

null_df = query_databricks(f"""
    SELECT
        SUM(CASE WHEN description IS NULL THEN 1 ELSE 0 END) as desc_nulls,
        SUM(CASE WHEN transcription IS NULL THEN 1 ELSE 0 END) as trans_nulls,
        SUM(CASE WHEN keywords IS NULL THEN 1 ELSE 0 END) as kw_nulls,
        SUM(CASE WHEN medical_specialty IS NULL THEN 1 ELSE 0 END) as spec_nulls,
        COUNT(*) as total
    FROM {DELTA_TABLE}
""")

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
st.subheader("Ingestion Summary")

k1, k2, k3 = st.columns(3)
k1.metric("Total Notes in Delta", f"{total_notes:,}")
k2.metric("Specialties", f"{len(specialty_df):,}")

nulls = null_df.iloc[0]
total = int(nulls["total"])
trans_null_pct = int(nulls["trans_nulls"]) / total * 100 if total > 0 else 0
k3.metric("Transcription Null Rate", f"{trans_null_pct:.1f}%")

st.divider()

# Specialty distribution
st.subheader("Specialty Distribution")

if not specialty_df.empty:
    # Convert count to int for charting
    specialty_df["count"] = specialty_df["count"].astype(int)
    top_20 = specialty_df.head(20)

    chart = (
        alt.Chart(top_20)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#3498db")
        .encode(
            x=alt.X("count:Q", title="Notes"),
            y=alt.Y("medical_specialty:N", title=None, sort="-x"),
            tooltip=["medical_specialty", "count"],
        )
        .properties(height=max(300, len(top_20) * 25))
    )
    st.altair_chart(chart, use_container_width=True)

st.divider()

# Note length by specialty
st.subheader("Note Length by Specialty")

if not note_length_df.empty:
    note_length_df["avg_length"] = note_length_df["avg_length"].astype(float)
    note_length_df["notes"] = note_length_df["notes"].astype(int)
    st.dataframe(
        note_length_df,
        column_config={
            "medical_specialty": "Specialty",
            "notes": st.column_config.NumberColumn("Notes"),
            "avg_length": st.column_config.NumberColumn("Avg Length (chars)", format="%d"),
            "min_length": st.column_config.NumberColumn("Min", format="%d"),
            "max_length": st.column_config.NumberColumn("Max", format="%d"),
        },
        hide_index=True,
        use_container_width=True,
    )

st.divider()

# Data quality
st.subheader("Data Quality")

null_data = {
    "Field": ["description", "transcription", "keywords", "medical_specialty"],
    "Null Count": [
        int(nulls["desc_nulls"]),
        int(nulls["trans_nulls"]),
        int(nulls["kw_nulls"]),
        int(nulls["spec_nulls"]),
    ],
}
null_data["Null Rate"] = [f"{n / total * 100:.1f}%" for n in null_data["Null Count"]]
st.dataframe(null_data, hide_index=True, use_container_width=True)

st.divider()

# Sample notes
st.subheader("Sample Notes")
selected_spec = st.selectbox("Filter by specialty", ["All"] + specialty_df["medical_specialty"].tolist())

if selected_spec == "All":
    sample_sql = f"SELECT id, medical_specialty, sample_name, LEFT(transcription, 200) as preview FROM {DELTA_TABLE} LIMIT 10"
else:
    sample_sql = f"SELECT id, medical_specialty, sample_name, LEFT(transcription, 200) as preview FROM {DELTA_TABLE} WHERE medical_specialty = '{selected_spec}' LIMIT 10"

sample_df = query_databricks(sample_sql)
if not sample_df.empty:
    st.dataframe(sample_df, hide_index=True, use_container_width=True)
else:
    st.info("No notes found for this specialty.")

st.caption("Data served live from Databricks Delta table via SQL Statement API.")
