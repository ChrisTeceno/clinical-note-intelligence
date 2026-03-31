"""Experiments -- run history and performance trends."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from tooltips import TOOLTIP_CSS, tt_heading

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Experiment History")

st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)

st.info(
    "**About this page:** Each evaluation run is logged with its metrics, model, "
    "cost, and a description of what changed. Use this to track improvement over "
    "time and compare experiments side by side.",
    icon=":material/labs:",
)

# ---------------------------------------------------------------------------
# Load run history
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation"
HISTORY_PATH = EVAL_DIR / "run_history.json"
RESULTS_PATH = EVAL_DIR / "evaluation_results.json"
COST_PATH = EVAL_DIR / "eval_cost_report.json"


def _load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            return []
    return []


def _seed_baseline() -> list[dict]:
    """Create a baseline entry from existing cached evaluation results."""
    if not RESULTS_PATH.exists():
        return []

    eval_data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    overall = eval_data.get("overall", {})
    n_admissions = eval_data.get("n_admissions", 0)

    # Read cost info
    total_cost = 0.0
    model = "claude-haiku-4-5-20251001"
    if COST_PATH.exists():
        cost_data = json.loads(COST_PATH.read_text(encoding="utf-8"))
        total_cost = cost_data.get("summary", {}).get("estimated_cost_usd", 0.0)
        records = cost_data.get("records", [])
        if records:
            model = records[0].get("model", model)

    # Use the earliest cost record timestamp, or fall back to file mtime
    timestamp = ""
    if COST_PATH.exists():
        cost_data = json.loads(COST_PATH.read_text(encoding="utf-8"))
        records = cost_data.get("records", [])
        if records:
            timestamp = records[0].get("timestamp", "")
    if not timestamp:
        import datetime
        mtime = RESULTS_PATH.stat().st_mtime
        timestamp = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc).isoformat()

    baseline = {
        "run_id": "baseline",
        "timestamp": timestamp,
        "model": model,
        "n_admissions": n_admissions,
        "description": "baseline (cached results)",
        "diagnoses": overall.get("diagnoses", {}),
        "procedures": overall.get("procedures", {}),
        "medications": overall.get("medications", {}),
        "total_cost_usd": total_cost,
        "duration_seconds": 0.0,
    }

    # Persist so it only seeds once
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps([baseline], indent=2), encoding="utf-8")
    return [baseline]


runs = _load_history()
if not runs:
    runs = _seed_baseline()

if not runs:
    st.warning(
        "No experiment history found. Run the evaluation pipeline first:\n\n"
        "```\n"
        "PYTHONPATH=src python -m clinical_pipeline.evaluation.run_eval "
        "--mimic-path /path/to/mimic-iv-demo/hosp --description 'baseline'\n"
        "```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Build DataFrame
# ---------------------------------------------------------------------------
rows = []
for r in runs:
    rows.append({
        "Run ID": r.get("run_id", ""),
        "Timestamp": r.get("timestamp", ""),
        "Description": r.get("description", ""),
        "Model": r.get("model", ""),
        "N": r.get("n_admissions", 0),
        "Dx F1": r.get("diagnoses", {}).get("f1", 0.0),
        "Dx P": r.get("diagnoses", {}).get("precision", 0.0),
        "Dx R": r.get("diagnoses", {}).get("recall", 0.0),
        "Proc F1": r.get("procedures", {}).get("f1", 0.0),
        "Proc P": r.get("procedures", {}).get("precision", 0.0),
        "Proc R": r.get("procedures", {}).get("recall", 0.0),
        "Rx F1": r.get("medications", {}).get("f1", 0.0),
        "Rx P": r.get("medications", {}).get("precision", 0.0),
        "Rx R": r.get("medications", {}).get("recall", 0.0),
        "Cost ($)": r.get("total_cost_usd", 0.0),
        "Duration (s)": r.get("duration_seconds", 0.0),
    })

df = pd.DataFrame(rows).sort_values("Timestamp", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Best run highlight
# ---------------------------------------------------------------------------
best_idx = df["Dx F1"].idxmax()
best_run = df.iloc[best_idx]

st.markdown(
    tt_heading(
        "Best Run",
        "The run with the highest Diagnosis F1 score.",
        style="font-size:1.15em;font-weight:600",
    ),
    unsafe_allow_html=True,
)

b1, b2, b3, b4 = st.columns(4)
b1.metric("Diagnosis F1", f"{best_run['Dx F1']:.1%}")
b2.metric("Procedures F1", f"{best_run['Proc F1']:.1%}")
b3.metric("Medications F1", f"{best_run['Rx F1']:.1%}")
b4.metric("Run", best_run["Description"] or best_run["Run ID"])

st.divider()

# ---------------------------------------------------------------------------
# Run history table
# ---------------------------------------------------------------------------
st.subheader("Run History")

st.dataframe(
    df,
    column_config={
        "Run ID": st.column_config.TextColumn("Run ID", width="small"),
        "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
        "Description": st.column_config.TextColumn("Description", width="medium"),
        "Model": st.column_config.TextColumn("Model", width="medium"),
        "N": st.column_config.NumberColumn("Admissions", format="%d"),
        "Dx F1": st.column_config.ProgressColumn("Dx F1", min_value=0, max_value=1, format="%.3f"),
        "Proc F1": st.column_config.ProgressColumn(
            "Proc F1", min_value=0, max_value=1, format="%.3f",
        ),
        "Rx F1": st.column_config.ProgressColumn("Rx F1", min_value=0, max_value=1, format="%.3f"),
        "Cost ($)": st.column_config.NumberColumn("Cost ($)", format="$%.4f"),
        "Duration (s)": st.column_config.NumberColumn("Duration (s)", format="%.1f"),
    },
    hide_index=True,
    use_container_width=True,
)

st.divider()

# ---------------------------------------------------------------------------
# F1 Trend Chart
# ---------------------------------------------------------------------------
st.subheader("F1 Trend Over Time")
st.caption("Track how extraction quality evolves across experiment runs.")

# Need at least 1 run for a chart (line appears with 2+)
trend_rows = []
for _, row in df.iterrows():
    ts = row["Timestamp"]
    label = row["Description"] or row["Run ID"]
    entity_cols = [("Diagnoses", "Dx F1"), ("Procedures", "Proc F1"), ("Medications", "Rx F1")]
    for entity, col in entity_cols:
        trend_rows.append({
            "Timestamp": ts, "Label": label, "Entity": entity, "F1": row[col],
        })

trend_df = pd.DataFrame(trend_rows)

if len(df) >= 2:
    trend_chart = (
        alt.Chart(trend_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Timestamp:T", title="Run Time"),
            y=alt.Y("F1:Q", title="F1 Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "Entity:N",
                scale=alt.Scale(
                    domain=["Diagnoses", "Procedures", "Medications"],
                    range=["#3498db", "#9b59b6", "#1abc9c"],
                ),
            ),
            tooltip=["Label", "Entity", alt.Tooltip("F1:Q", format=".3f"), "Timestamp:T"],
        )
        .properties(height=350)
    )
    st.altair_chart(trend_chart, use_container_width=True)
else:
    st.info("Run at least two evaluations to see the trend chart.")
    # Show a bar chart for the single run
    bar_chart = (
        alt.Chart(trend_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Entity:N", title=None),
            y=alt.Y("F1:Q", title="F1 Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "Entity:N",
                scale=alt.Scale(
                    domain=["Diagnoses", "Procedures", "Medications"],
                    range=["#3498db", "#9b59b6", "#1abc9c"],
                ),
            ),
            tooltip=["Entity", alt.Tooltip("F1:Q", format=".3f")],
        )
        .properties(height=300)
    )
    st.altair_chart(bar_chart, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Run Comparison
# ---------------------------------------------------------------------------
st.subheader("Run Comparison")
st.caption("Select two runs to compare side by side.")

if len(df) < 2:
    st.info("Need at least two runs to compare.")
else:
    run_options = {
        f"{row['Run ID']} - {row['Description'] or row['Timestamp']}": i
        for i, row in df.iterrows()
    }

    c1, c2 = st.columns(2)
    with c1:
        label_a = st.selectbox("Run A", options=list(run_options.keys()), index=0)
    with c2:
        default_b = min(1, len(run_options) - 1)
        label_b = st.selectbox("Run B", options=list(run_options.keys()), index=default_b)

    run_a = df.iloc[run_options[label_a]]
    run_b = df.iloc[run_options[label_b]]

    metrics_to_compare = [
        ("Diagnosis F1", "Dx F1"),
        ("Diagnosis Precision", "Dx P"),
        ("Diagnosis Recall", "Dx R"),
        ("Procedures F1", "Proc F1"),
        ("Procedures Precision", "Proc P"),
        ("Procedures Recall", "Proc R"),
        ("Medications F1", "Rx F1"),
        ("Medications Precision", "Rx P"),
        ("Medications Recall", "Rx R"),
        ("Cost ($)", "Cost ($)"),
    ]

    st.markdown("---")

    # Header row
    h1, h2, h3 = st.columns([2, 2, 2])
    h1.markdown("**Metric**")
    h2.markdown(f"**Run A:** {run_a['Description'] or run_a['Run ID']}")
    h3.markdown(f"**Run B:** {run_b['Description'] or run_b['Run ID']}")

    for label, col_name in metrics_to_compare:
        val_a = run_a[col_name]
        val_b = run_b[col_name]
        delta = val_b - val_a

        m1, m2, m3 = st.columns([2, 2, 2])
        m1.markdown(f"**{label}**")

        if col_name == "Cost ($)":
            m2.markdown(f"${val_a:.4f}")
            # For cost, lower is better -- flip the arrow logic
            if abs(delta) < 1e-6:
                m3.markdown(f"${val_b:.4f}")
            elif delta < 0:
                m3.markdown(f"${val_b:.4f} :green[{delta:+.4f}]")
            else:
                m3.markdown(f"${val_b:.4f} :red[{delta:+.4f}]")
        else:
            m2.markdown(f"{val_a:.1%}")
            if abs(delta) < 1e-6:
                m3.markdown(f"{val_b:.1%}")
            elif delta > 0:
                m3.markdown(f"{val_b:.1%} :green[{delta:+.1%}]")
            else:
                m3.markdown(f"{val_b:.1%} :red[{delta:+.1%}]")
