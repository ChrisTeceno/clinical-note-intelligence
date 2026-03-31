"""Dashboard — overview metrics and KPIs."""

import json
from pathlib import Path

import altair as alt
import streamlit as st
from db import query_df
from tooltips import TOOLTIP_CSS, fmt_pct_ci, tt, tt_heading

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Pipeline Dashboard")
st.caption("High-level overview of the Clinical Note Intelligence extraction pipeline.")

# Inject tooltip CSS once
st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI row (with tooltip explanations)
# ---------------------------------------------------------------------------
total_notes = query_df("SELECT COUNT(*) AS n FROM clinical_notes").iloc[0]["n"]
total_diagnoses = query_df("SELECT COUNT(*) AS n FROM diagnoses").iloc[0]["n"]
total_procedures = query_df("SELECT COUNT(*) AS n FROM procedures").iloc[0]["n"]
total_medications = query_df("SELECT COUNT(*) AS n FROM medications").iloc[0]["n"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Notes Processed", f"{total_notes:,}")
k1.markdown(
    tt(
        '<span style="font-size:0.75em;color:#888">What is this?</span>',
        "Total unique clinical notes ingested from MTSamples "
        "and processed through the PySpark pipeline",
    ),
    unsafe_allow_html=True,
)
k2.metric("Diagnoses", f"{total_diagnoses:,}")
k2.markdown(
    tt(
        '<span style="font-size:0.75em;color:#888">What is this?</span>',
        "Total diagnosis entities extracted by Claude from all processed clinical notes",
    ),
    unsafe_allow_html=True,
)
k3.metric("Procedures", f"{total_procedures:,}")
k3.markdown(
    tt(
        '<span style="font-size:0.75em;color:#888">What is this?</span>',
        "Total procedure entities extracted, each with a suggested CPT code",
    ),
    unsafe_allow_html=True,
)
k4.metric("Medications", f"{total_medications:,}")
k4.markdown(
    tt(
        '<span style="font-size:0.75em;color:#888">What is this?</span>',
        "Total medication entities extracted, including dosage, frequency, and route",
    ),
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Charts row 1 — ICD-10 match rate & confidence distribution
# ---------------------------------------------------------------------------
col_match, col_conf = st.columns(2)

with col_match:
    st.markdown(
        tt_heading(
            "ICD-10 Match Rate",
            "How extracted diagnosis codes compare to the CMS 2025 "
            "ICD-10-CM code table. Exact = code matches verbatim. "
            "Partial = matched by description similarity (rapidfuzz). "
            "None = no match found, requires manual coding.",
            style="font-size:1.25em;font-weight:600",
        ),
        unsafe_allow_html=True,
    )
    match_df = query_df(
        "SELECT match_type, COUNT(*) AS count FROM diagnoses GROUP BY match_type"
    )
    if not match_df.empty:
        # Wilson CI annotations for each match type
        total_dx = int(match_df["count"].sum())
        ci_parts = []
        for mt in ["exact", "partial", "none"]:
            row = match_df[match_df["match_type"] == mt]
            n = int(row["count"].iloc[0]) if not row.empty else 0
            ci_parts.append(f"**{mt}:** {fmt_pct_ci(n, total_dx)}")
        st.caption(" &nbsp;|&nbsp; ".join(ci_parts))

        color_scale = alt.Scale(
            domain=["exact", "partial", "none"],
            range=["#2ecc71", "#f39c12", "#e74c3c"],
        )
        chart = (
            alt.Chart(match_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("match_type:N", title="Match Type", sort=["exact", "partial", "none"]),
                y=alt.Y("count:Q", title="Diagnoses"),
                color=alt.Color("match_type:N", scale=color_scale, legend=None),
                tooltip=["match_type", "count"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No diagnoses found.")

with col_conf:
    st.markdown(
        tt_heading(
            "Confidence Distribution",
            "Claude's self-assessed confidence for each extracted diagnosis. "
            "High = explicitly stated. Medium = strongly implied or clinical shorthand. "
            "Low = differential diagnosis, rule-out, or uncertainty language.",
            style="font-size:1.25em;font-weight:600",
        ),
        unsafe_allow_html=True,
    )
    conf_df = query_df(
        "SELECT confidence, COUNT(*) AS count FROM diagnoses GROUP BY confidence"
    )
    if not conf_df.empty:
        # Wilson CI annotations for each confidence level
        total_conf = int(conf_df["count"].sum())
        ci_parts = []
        for cl in ["high", "medium", "low"]:
            row = conf_df[conf_df["confidence"] == cl]
            n = int(row["count"].iloc[0]) if not row.empty else 0
            ci_parts.append(f"**{cl}:** {fmt_pct_ci(n, total_conf)}")
        st.caption(" &nbsp;|&nbsp; ".join(ci_parts))

        pie = (
            alt.Chart(conf_df)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color(
                    "confidence:N",
                    scale=alt.Scale(
                        domain=["high", "medium", "low"],
                        range=["#2ecc71", "#f39c12", "#e74c3c"],
                    ),
                ),
                tooltip=["confidence", "count"],
            )
            .properties(height=300)
        )
        st.altair_chart(pie, use_container_width=True)
    else:
        st.info("No diagnoses found.")

# ---------------------------------------------------------------------------
# Charts row 2 — Top specialties
# ---------------------------------------------------------------------------
st.subheader("Top 10 Medical Specialties")
specialty_df = query_df("""
    SELECT cn.medical_specialty AS specialty, COUNT(DISTINCT e.id) AS extractions
    FROM extractions e
    JOIN clinical_notes cn ON cn.id = e.note_id
    GROUP BY cn.medical_specialty
    ORDER BY extractions DESC
    LIMIT 10
""")

if not specialty_df.empty:
    bar = (
        alt.Chart(specialty_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#3498db")
        .encode(
            x=alt.X("extractions:Q", title="Extractions"),
            y=alt.Y("specialty:N", title=None, sort="-x"),
            tooltip=["specialty", "extractions"],
        )
        .properties(height=max(250, len(specialty_df) * 30))
    )
    st.altair_chart(bar, use_container_width=True)
else:
    st.info("No extraction data available.")

# ---------------------------------------------------------------------------
# Recent extractions table
# ---------------------------------------------------------------------------
st.subheader("Recent Extractions")
recent_df = query_df("""
    SELECT
        e.id,
        cn.medical_specialty AS specialty,
        e.chief_complaint,
        e.status,
        e.created_at,
        (SELECT COUNT(*) FROM diagnoses d WHERE d.extraction_id = e.id) AS dx_count,
        (SELECT COUNT(*) FROM procedures p WHERE p.extraction_id = e.id) AS px_count,
        (SELECT COUNT(*) FROM medications m WHERE m.extraction_id = e.id) AS rx_count
    FROM extractions e
    JOIN clinical_notes cn ON cn.id = e.note_id
    ORDER BY e.created_at DESC
    LIMIT 15
""")

if not recent_df.empty:
    st.dataframe(
        recent_df,
        column_config={
            "id": st.column_config.TextColumn("ID", width="small"),
            "specialty": "Specialty",
            "chief_complaint": "Chief Complaint",
            "status": "Status",
            "created_at": st.column_config.DatetimeColumn("Created", format="YYYY-MM-DD HH:mm"),
            "dx_count": st.column_config.NumberColumn("Dx", help="Diagnoses"),
            "px_count": st.column_config.NumberColumn("Px", help="Procedures"),
            "rx_count": st.column_config.NumberColumn("Rx", help="Medications"),
        },
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("No extractions yet.")

# ---------------------------------------------------------------------------
# Cost summary
# ---------------------------------------------------------------------------
cost_path = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "extraction_results" / "cost_report.json"
)
if cost_path.exists():
    st.subheader("Cost Summary")
    with open(cost_path) as f:
        cost_data = json.load(f)
    summary = cost_data.get("summary", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Requests", f"{summary.get('total_requests', 0):,}")
    c2.metric("Input Tokens", f"{summary.get('total_input_tokens', 0):,}")
    c3.metric("Output Tokens", f"{summary.get('total_output_tokens', 0):,}")
    c4.metric("Estimated Cost", f"${summary.get('estimated_cost_usd', 0):.2f}")
    c4.markdown(
        tt(
            '<span style="font-size:0.75em;color:#888">Pricing</span>',
            "Based on Anthropic API pricing: Haiku 4.5 at "
            "$0.80/M input, $4.00/M output tokens",
        ),
        unsafe_allow_html=True,
    )

    cost_per_note = summary.get("estimated_cost_usd", 0) / max(summary.get("total_requests", 1), 1)
    st.caption(f"Average cost per note: **${cost_per_note:.4f}**")
