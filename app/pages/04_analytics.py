"""Analytics — extraction quality metrics and cost analysis."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from db import query_df
from tooltips import TOOLTIP_CSS, fmt_pct_ci, tt, tt_heading

st.title("Analytics")
st.caption("Extraction quality, ICD-10 accuracy, and cost metrics across the pipeline.")

# Inject tooltip CSS once
st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ICD-10 match rates by specialty
# ---------------------------------------------------------------------------
st.subheader("ICD-10 Match Rates by Specialty")

match_by_spec = query_df("""
    SELECT
        cn.medical_specialty AS specialty,
        d.match_type,
        COUNT(*) AS count
    FROM diagnoses d
    JOIN extractions e ON e.id = d.extraction_id
    JOIN clinical_notes cn ON cn.id = e.note_id
    GROUP BY cn.medical_specialty, d.match_type
    ORDER BY cn.medical_specialty
""")

if not match_by_spec.empty:
    chart = (
        alt.Chart(match_by_spec)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Diagnoses", stack="zero"),
            y=alt.Y("specialty:N", title=None, sort="-x"),
            color=alt.Color(
                "match_type:N",
                scale=alt.Scale(
                    domain=["exact", "partial", "none"],
                    range=["#2ecc71", "#f39c12", "#e74c3c"],
                ),
                title="Match Type",
            ),
            tooltip=["specialty", "match_type", "count"],
        )
        .properties(height=max(300, len(match_by_spec["specialty"].unique()) * 35))
    )
    st.altair_chart(chart, use_container_width=True)

    # Per-specialty exact-match rates with Wilson CIs
    st.markdown(
        tt_heading(
            "Exact Match Rates by Specialty (95% Wilson CI)",
            "Wilson score confidence intervals for the proportion"
            " of diagnoses with an exact ICD-10 code match,"
            " per specialty. Wider intervals indicate fewer"
            " observations.",
            style="font-size:1.05em;font-weight:600",
        ),
        unsafe_allow_html=True,
    )
    pivot = match_by_spec.pivot_table(
        index="specialty", columns="match_type",
        values="count", fill_value=0,
    ).reset_index()
    ci_rows = []
    for _, r in pivot.iterrows():
        spec = r["specialty"]
        exact_n = int(r.get("exact", 0))
        total_n = int(r.get("exact", 0)) + int(r.get("partial", 0)) + int(r.get("none", 0))
        ci_rows.append({
            "Specialty": spec,
            "Exact": exact_n,
            "Total Dx": total_n,
            "Exact Match Rate": fmt_pct_ci(exact_n, total_n),
        })
    if ci_rows:
        ci_display = pd.DataFrame(ci_rows).sort_values("Total Dx", ascending=False)
        st.dataframe(ci_display, hide_index=True, use_container_width=True)
else:
    st.info("No diagnosis data available.")

# ---------------------------------------------------------------------------
# Confidence distribution by specialty
# ---------------------------------------------------------------------------
st.subheader("Confidence Distribution by Specialty")

conf_by_spec = query_df("""
    SELECT
        cn.medical_specialty AS specialty,
        d.confidence,
        COUNT(*) AS count
    FROM diagnoses d
    JOIN extractions e ON e.id = d.extraction_id
    JOIN clinical_notes cn ON cn.id = e.note_id
    GROUP BY cn.medical_specialty, d.confidence
    ORDER BY cn.medical_specialty
""")

if not conf_by_spec.empty:
    chart = (
        alt.Chart(conf_by_spec)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Diagnoses", stack="zero"),
            y=alt.Y("specialty:N", title=None, sort="-x"),
            color=alt.Color(
                "confidence:N",
                scale=alt.Scale(
                    domain=["high", "medium", "low"],
                    range=["#2ecc71", "#f39c12", "#e74c3c"],
                ),
                title="Confidence",
            ),
            tooltip=["specialty", "confidence", "count"],
        )
        .properties(height=max(300, len(conf_by_spec["specialty"].unique()) * 35))
    )
    st.altair_chart(chart, use_container_width=True)

    # Per-specialty high-confidence rates with Wilson CIs
    st.markdown(
        tt_heading(
            "High-Confidence Rates by Specialty (95% Wilson CI)",
            "Wilson score confidence intervals for the proportion"
            " of diagnoses classified as high confidence,"
            " per specialty.",
            style="font-size:1.05em;font-weight:600",
        ),
        unsafe_allow_html=True,
    )
    conf_pivot = conf_by_spec.pivot_table(
        index="specialty", columns="confidence",
        values="count", fill_value=0,
    ).reset_index()
    conf_ci_rows = []
    for _, r in conf_pivot.iterrows():
        spec = r["specialty"]
        high_n = int(r.get("high", 0))
        total_n = int(r.get("high", 0)) + int(r.get("medium", 0)) + int(r.get("low", 0))
        conf_ci_rows.append({
            "Specialty": spec,
            "High Conf.": high_n,
            "Total Dx": total_n,
            "High-Confidence Rate": fmt_pct_ci(high_n, total_n),
        })
    if conf_ci_rows:
        conf_ci_display = pd.DataFrame(conf_ci_rows).sort_values("Total Dx", ascending=False)
        st.dataframe(conf_ci_display, hide_index=True, use_container_width=True)
else:
    st.info("No diagnosis data available.")

# ---------------------------------------------------------------------------
# Extraction counts per specialty
# ---------------------------------------------------------------------------
st.subheader("Extraction Counts per Specialty")

counts_df = query_df("""
    SELECT
        cn.medical_specialty AS specialty,
        COUNT(DISTINCT d.id) AS diagnoses,
        COUNT(DISTINCT p.id) AS procedures,
        COUNT(DISTINCT m.id) AS medications
    FROM extractions e
    JOIN clinical_notes cn ON cn.id = e.note_id
    LEFT JOIN diagnoses d ON d.extraction_id = e.id
    LEFT JOIN procedures p ON p.extraction_id = e.id
    LEFT JOIN medications m ON m.extraction_id = e.id
    GROUP BY cn.medical_specialty
    ORDER BY diagnoses DESC
""")

if not counts_df.empty:
    melted = counts_df.melt(
        id_vars=["specialty"],
        value_vars=["diagnoses", "procedures", "medications"],
        var_name="entity_type",
        value_name="count",
    )
    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Entities Extracted", stack="zero"),
            y=alt.Y("specialty:N", title=None, sort="-x"),
            color=alt.Color(
                "entity_type:N",
                scale=alt.Scale(
                    domain=["diagnoses", "procedures", "medications"],
                    range=["#3498db", "#9b59b6", "#1abc9c"],
                ),
                title="Entity Type",
            ),
            tooltip=["specialty", "entity_type", "count"],
        )
        .properties(height=max(300, len(counts_df) * 35))
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No extraction data available.")

# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
errors_path = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "processed"
    / "extraction_results"
    / "errors.json"
)

st.subheader("Extraction Errors")

if errors_path.exists():
    with open(errors_path) as f:
        errors = json.load(f)

    total_notes = query_df("SELECT COUNT(*) AS n FROM clinical_notes").iloc[0]["n"]
    error_count = len(errors)
    success_count = total_notes - error_count
    error_rate = (error_count / total_notes * 100) if total_notes > 0 else 0

    e1, e2, e3 = st.columns(3)
    e1.metric("Successful Extractions", f"{success_count:,}")
    e2.metric("Failed Extractions", f"{error_count:,}")
    e3.metric("Error Rate", f"{error_rate:.1f}%")
    if total_notes > 0:
        e3.caption(f"95% CI: {fmt_pct_ci(error_count, total_notes)}")

    if errors:
        error_df = pd.DataFrame(errors)
        st.dataframe(
            error_df,
            column_config={
                "note_id": "Note ID",
                "error": st.column_config.TextColumn("Error Message", width="large"),
            },
            hide_index=True,
            use_container_width=True,
        )
else:
    st.info("No error log found.")

# ---------------------------------------------------------------------------
# Cost per note analysis
# ---------------------------------------------------------------------------
cost_path = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "processed"
    / "extraction_results"
    / "cost_report.json"
)

st.subheader("Cost Analysis")

if cost_path.exists():
    with open(cost_path) as f:
        cost_data = json.load(f)

    summary = cost_data.get("summary", {})
    records = cost_data.get("records", [])

    c1, c2, c3 = st.columns(3)
    total_cost = summary.get("estimated_cost_usd", 0)
    total_reqs = summary.get("total_requests", 0)
    cost_per_note = total_cost / max(total_reqs, 1)

    c1.metric("Total Cost", f"${total_cost:.2f}")
    c1.markdown(
        tt(
            '<span style="font-size:0.75em;color:#888">Pricing</span>',
            "Based on Anthropic API pricing: Haiku 4.5 at "
            "$0.80/M input, $4.00/M output tokens",
        ),
        unsafe_allow_html=True,
    )
    c2.metric("Cost / Note", f"${cost_per_note:.4f}")
    c3.metric("Model", records[0]["model"] if records else "N/A")

    if records:
        records_df = pd.DataFrame(records)
        records_df["timestamp"] = pd.to_datetime(records_df["timestamp"])
        records_df["total_tokens"] = records_df["input_tokens"] + records_df["output_tokens"]
        records_df["note_index"] = range(1, len(records_df) + 1)

        # Token usage over time
        st.markdown("**Token Usage per Request**")
        token_melted = records_df[["note_index", "input_tokens", "output_tokens"]].melt(
            id_vars=["note_index"],
            value_vars=["input_tokens", "output_tokens"],
            var_name="token_type",
            value_name="tokens",
        )
        chart = (
            alt.Chart(token_melted)
            .mark_bar()
            .encode(
                x=alt.X("note_index:O", title="Request #"),
                y=alt.Y("tokens:Q", title="Tokens", stack="zero"),
                color=alt.Color(
                    "token_type:N",
                    scale=alt.Scale(
                        domain=["input_tokens", "output_tokens"],
                        range=["#3498db", "#e67e22"],
                    ),
                    title="Token Type",
                ),
                tooltip=["note_index", "token_type", "tokens"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
else:
    st.info("No cost report found.")
