"""Prompt Optimization Dashboard — view and run the autoresearch optimization loop."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from clinical_pipeline.extraction.prompts import (
    SYSTEM_PROMPT as BASELINE_PROMPT,
)

st.title("Prompt Optimization")
st.caption(
    "Karpathy-style autoresearch loop: systematically mutate the extraction "
    "prompt and keep changes that improve F1."
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OPT_DIR = DATA_DIR / "optimization"
HISTORY_PATH = OPT_DIR / "prompt_history.json"
BEST_PROMPT_PATH = OPT_DIR / "best_prompt.txt"

# ---------------------------------------------------------------------------
# Current best prompt vs baseline
# ---------------------------------------------------------------------------
st.subheader("Current Prompts")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Baseline Prompt**")
    st.code(BASELINE_PROMPT, language="text")

with col2:
    st.markdown("**Best Optimized Prompt**")
    if BEST_PROMPT_PATH.exists():
        best_prompt = BEST_PROMPT_PATH.read_text(encoding="utf-8")
        st.code(best_prompt, language="text")
        if best_prompt == BASELINE_PROMPT:
            st.info("Best prompt is identical to baseline (no improvements found yet).")
    else:
        st.info("No optimization has been run yet.")

st.divider()

# ---------------------------------------------------------------------------
# Optimization history
# ---------------------------------------------------------------------------
st.subheader("Optimization History")

if HISTORY_PATH.exists():
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    iterations = history.get("iterations", [])

    if iterations:
        # Summary metrics
        n_kept = sum(1 for it in iterations if it.get("kept"))
        n_total = len(iterations)
        baseline = history.get("baseline_metrics", {})
        best = history.get("best_metrics", {})

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Iterations", n_total)
        mcol2.metric("Mutations Kept", n_kept)
        mcol3.metric(
            "Total Duration",
            f"{history.get('total_duration_seconds', 0):.1f}s",
        )

        # Baseline vs best metrics
        st.markdown("**Metrics Comparison**")
        metric_rows = []
        for entity in ("diagnoses", "procedures", "medications"):
            b = baseline.get(entity, {})
            be = best.get(entity, {})
            metric_rows.append({
                "Entity": entity.title(),
                "Baseline F1": f"{b.get('f1', 0):.4f}",
                "Best F1": f"{be.get('f1', 0):.4f}",
                "Baseline P": f"{b.get('precision', 0):.4f}",
                "Best P": f"{be.get('precision', 0):.4f}",
                "Baseline R": f"{b.get('recall', 0):.4f}",
                "Best R": f"{be.get('recall', 0):.4f}",
            })
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

        # Iteration details
        st.markdown("**Iteration Details**")
        iter_rows = []
        for it in iterations:
            iter_rows.append({
                "Iteration": it["iteration"],
                "Mutation": it["mutation_name"],
                "Description": it["mutation_description"],
                "Target Value": f"{it['target_value']:.4f}",
                "Best Value": f"{it['best_value']:.4f}",
                "Kept": "Yes" if it["kept"] else "No",
                "Duration (s)": f"{it.get('duration_seconds', 0):.1f}",
            })
        st.dataframe(pd.DataFrame(iter_rows), use_container_width=True, hide_index=True)

        # Chart: target metric over iterations
        st.markdown("**Target Metric Over Iterations**")
        chart_data = pd.DataFrame({
            "Iteration": [it["iteration"] for it in iterations],
            "Target Value": [it["target_value"] for it in iterations],
            "Best Value": [it["best_value"] for it in iterations],
        }).set_index("Iteration")
        st.line_chart(chart_data)

        # Which mutations helped vs didn't
        st.markdown("**Mutation Effectiveness**")
        mutation_stats: dict[str, dict] = {}
        for it in iterations:
            name = it["mutation_name"]
            if name not in mutation_stats:
                mutation_stats[name] = {"tried": 0, "kept": 0}
            mutation_stats[name]["tried"] += 1
            if it["kept"]:
                mutation_stats[name]["kept"] += 1
        mut_rows = [
            {
                "Mutation": name,
                "Times Tried": stats["tried"],
                "Times Kept": stats["kept"],
                "Success Rate": f"{stats['kept'] / stats['tried'] * 100:.0f}%"
                if stats["tried"] > 0
                else "N/A",
            }
            for name, stats in mutation_stats.items()
        ]
        st.dataframe(pd.DataFrame(mut_rows), use_container_width=True, hide_index=True)

    else:
        st.info("No iterations recorded in history.")

else:
    st.info("No optimization history found. Run an optimization first.")

st.divider()

# ---------------------------------------------------------------------------
# Run optimization
# ---------------------------------------------------------------------------
st.subheader("Run Optimization")

st.warning(
    "Running optimization will call the Claude API multiple times. "
    "Each iteration extracts from the eval set, so costs scale with "
    "iterations x eval set size."
)

# Check for cached evaluation data (previous runs)
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
has_cached_data = (EVAL_DIR / "selected_hadm_ids.json").exists() and (EVAL_DIR / "synthetic_notes").is_dir()

# Default MIMIC path — check common locations
DEFAULT_MIMIC_PATHS = [
    "/home/deploy/data/mimic-iv-demo/hosp",
    "/Users/teceno/Downloads/mimic-iv-clinical-database-demo-2.2/hosp",
    str(Path.home() / "Downloads/mimic-iv-clinical-database-demo-2.2/hosp"),
]
default_mimic = ""
for p in DEFAULT_MIMIC_PATHS:
    if Path(p).exists():
        default_mimic = p
        break

mimic_path_str = st.text_input(
    "MIMIC-IV hosp/ directory path",
    value=default_mimic,
    help="Path to the MIMIC-IV demo hosp/ directory. Cached data from previous evaluation runs will be reused.",
)

if has_cached_data:
    st.caption("Previous evaluation data found — synthetic notes and ground truth will be reused.")

with st.form("opt_form"):
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        n_iterations = st.number_input(
            "Number of iterations", min_value=1, max_value=20, value=5
        )
        n_admissions = st.number_input(
            "Eval set size (admissions)", min_value=1, max_value=30, value=10
        )
    with opt_col2:
        target = st.selectbox(
            "Target metric",
            ["diagnoses_f1", "procedures_f1", "medications_f1"],
        )
        use_rag = st.checkbox("Enable ICD-10 RAG")

    # Cost estimate
    est_calls = (n_iterations + 1) * n_admissions  # +1 for baseline
    st.info(
        f"Estimated API calls: ~{est_calls} extractions "
        f"(plus {n_admissions} note generations if not cached). "
        f"Approximate cost with Haiku: ~${est_calls * 0.003:.2f}"
    )

    submitted = st.form_submit_button("Run Optimization")

if submitted:
    if not mimic_path_str or not Path(mimic_path_str).exists():
        st.error("MIMIC-IV path not found. Please check the path above.")
    else:
        from clinical_pipeline.optimization.run_optimize import (
            run_optimization,
        )

        with st.spinner("Running optimization loop — this takes a few minutes..."):
            result = run_optimization(
                mimic_path=Path(mimic_path_str),
                n_iterations=n_iterations,
                target_metric=target,
                n_admissions=n_admissions,
                use_rag=use_rag,
            )
        st.success(
            f"Optimization complete. Kept {result['n_kept']}/{n_iterations} mutations "
            f"in {result['total_duration_seconds']:.1f}s."
        )
        st.rerun()
