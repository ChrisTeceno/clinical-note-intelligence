"""Evaluation — synthetic data extraction quality metrics."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from tooltips import TOOLTIP_CSS, tt, tt_heading

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Synthetic Data Evaluation")

st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)

st.info(
    "**About this evaluation:** Synthetic discharge summaries are generated from "
    "structured MIMIC-IV demo data (ICD codes, procedures, medications) using Claude, "
    "then run through the extraction pipeline. Results are scored against the known "
    "ground truth. This validates the pipeline design. Real MIMIC clinical notes will "
    "be used once credentialing is complete.",
    icon=":material/science:",
)

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation"
RESULTS_PATH = EVAL_DIR / "evaluation_results.json"

if not RESULTS_PATH.exists():
    st.warning(
        "No evaluation results found. Run the evaluation pipeline first:\n\n"
        "```\n"
        "PYTHONPATH=src python -m clinical_pipeline.evaluation.run_eval "
        "--mimic-path /path/to/mimic-iv-demo/hosp\n"
        "```"
    )
    st.stop()

with open(RESULTS_PATH) as f:
    eval_data = json.load(f)

overall = eval_data.get("overall", {})
per_admission = eval_data.get("per_admission", {})
n_admissions = eval_data.get("n_admissions", 0)
failed = eval_data.get("failed_extractions", [])

# ---------------------------------------------------------------------------
# Overall KPIs
# ---------------------------------------------------------------------------
st.subheader("Overall Performance")

entity_types = ["diagnoses", "procedures", "medications"]
cols = st.columns(len(entity_types))

for col, etype in zip(cols, entity_types):
    metrics = overall.get(etype, {})
    p = metrics.get("precision", 0)
    r = metrics.get("recall", 0)
    f1 = metrics.get("f1", 0)
    tp = metrics.get("tp", 0)
    fp = metrics.get("fp", 0)
    fn = metrics.get("fn", 0)

    with col:
        st.markdown(
            tt_heading(
                etype.title(),
                f"Precision: fraction of extracted {etype} that match ground truth. "
                f"Recall: fraction of ground truth {etype} that were extracted. "
                f"F1: harmonic mean of precision and recall.",
                style="font-size:1.15em;font-weight:600",
            ),
            unsafe_allow_html=True,
        )
        st.metric("F1 Score", f"{f1:.1%}")
        st.metric("Precision", f"{p:.1%}")
        st.metric("Recall", f"{r:.1%}")
        st.caption(f"TP={tp}  FP={fp}  FN={fn}")

st.caption(f"Evaluated on **{n_admissions}** synthetic admissions.")
if failed:
    st.warning(f"{len(failed)} admission(s) failed extraction: {failed}")

st.divider()

# ---------------------------------------------------------------------------
# F1 by entity type chart
# ---------------------------------------------------------------------------
st.subheader("Precision / Recall / F1 by Entity Type")

prf_rows = []
for etype in entity_types:
    metrics = overall.get(etype, {})
    for metric_name in ["precision", "recall", "f1"]:
        prf_rows.append({
            "Entity Type": etype.title(),
            "Metric": metric_name.title(),
            "Value": metrics.get(metric_name, 0),
        })

if prf_rows:
    prf_df = pd.DataFrame(prf_rows)
    chart = (
        alt.Chart(prf_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Metric:N", title=None, sort=["Precision", "Recall", "F1"]),
            y=alt.Y("Value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "Entity Type:N",
                scale=alt.Scale(
                    domain=["Diagnoses", "Procedures", "Medications"],
                    range=["#3498db", "#9b59b6", "#1abc9c"],
                ),
            ),
            column=alt.Column("Entity Type:N", title=None),
            tooltip=["Entity Type", "Metric", alt.Tooltip("Value:Q", format=".1%")],
        )
        .properties(width=150, height=300)
    )
    st.altair_chart(chart)

st.divider()

# ---------------------------------------------------------------------------
# Per-admission breakdown
# ---------------------------------------------------------------------------
st.subheader("Per-Admission Breakdown")

admission_rows = []
for hadm_id, results in per_admission.items():
    row = {"Admission ID": int(hadm_id)}
    for r in results:
        etype = r["entity_type"]
        row[f"{etype.title()} F1"] = r["f1"]
        row[f"{etype.title()} P"] = r["precision"]
        row[f"{etype.title()} R"] = r["recall"]
        row[f"{etype.title()} TP"] = r["true_positives"]
        row[f"{etype.title()} FP"] = r["false_positives"]
        row[f"{etype.title()} FN"] = r["false_negatives"]
    admission_rows.append(row)

if admission_rows:
    adm_df = pd.DataFrame(admission_rows).sort_values("Admission ID")
    st.dataframe(
        adm_df,
        column_config={
            "Admission ID": st.column_config.NumberColumn(format="%d"),
            "Diagnoses F1": st.column_config.ProgressColumn(
                "Dx F1", min_value=0, max_value=1, format="%.2f"
            ),
            "Procedures F1": st.column_config.ProgressColumn(
                "Proc F1", min_value=0, max_value=1, format="%.2f"
            ),
            "Medications F1": st.column_config.ProgressColumn(
                "Rx F1", min_value=0, max_value=1, format="%.2f"
            ),
        },
        hide_index=True,
        use_container_width=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Synthetic Note Viewer
# ---------------------------------------------------------------------------
st.subheader("Synthetic Note Viewer")
st.caption("Select an admission to view the synthetic discharge summary, ground truth, and what the pipeline extracted.")

NOTES_DIR = EVAL_DIR / "synthetic_notes"
EXTRACTIONS_DIR = EVAL_DIR / "extractions"
GT_DIR = EVAL_DIR  # ground truth loaded from MIMIC

hadm_ids_available = sorted(int(h) for h in per_admission.keys())
selected_hadm = st.selectbox(
    "Select Admission",
    options=hadm_ids_available,
    format_func=lambda x: f"Admission #{x}",
)

if selected_hadm:
    # Load synthetic note
    note_path = NOTES_DIR / f"{selected_hadm}.txt"
    note_text = note_path.read_text(encoding="utf-8") if note_path.exists() else "Note not found."

    # Load extraction
    ext_path = EXTRACTIONS_DIR / f"{selected_hadm}.json"
    extraction_data = json.loads(ext_path.read_text()) if ext_path.exists() else {}

    # Get scores for this admission
    adm_results = per_admission.get(str(selected_hadm), [])
    score_cols = st.columns(3)
    for col, etype in zip(score_cols, entity_types):
        r = next((r for r in adm_results if r["entity_type"] == etype), None)
        if r:
            col.metric(f"{etype.title()} F1", f"{r['f1']:.0%}",
                       help=f"P={r['precision']:.0%} R={r['recall']:.0%} | TP={r['true_positives']} FP={r['false_positives']} FN={r['false_negatives']}")

    note_tab, gt_tab, ext_tab = st.tabs(["Synthetic Note", "Ground Truth", "Extracted"])

    with note_tab:
        st.markdown(
            f"<div style='max-height:500px;overflow-y:auto;padding:12px;"
            f"border:1px solid #ddd;border-radius:8px;font-size:0.88em;"
            f"line-height:1.6;white-space:pre-wrap'>{note_text}</div>",
            unsafe_allow_html=True,
        )

    with gt_tab:
        # Load ground truth from MIMIC data
        gt_hadm_path = EVAL_DIR / "ground_truth" / f"{selected_hadm}.json"
        if gt_hadm_path.exists():
            gt_data = json.loads(gt_hadm_path.read_text())
        else:
            # Reconstruct from evaluation details
            gt_data = None

        if gt_data:
            gt_dx = gt_data.get("diagnoses", [])
            gt_px = gt_data.get("procedures", [])
            gt_rx = gt_data.get("medications", [])

            if gt_dx:
                st.markdown(f"**Diagnoses ({len(gt_dx)})**")
                for d in gt_dx:
                    st.markdown(f"- `{d.get('icd_code', '')}` — {d.get('description', '')}")

            if gt_px:
                st.markdown(f"**Procedures ({len(gt_px)})**")
                for p in gt_px:
                    st.markdown(f"- `{p.get('icd_code', '')}` — {p.get('description', '')}")

            if gt_rx:
                st.markdown(f"**Medications ({len(gt_rx)})**")
                for m in gt_rx:
                    drug = m.get("drug", "")
                    dose = f" {m.get('dose_val', '')} {m.get('dose_unit', '')}".strip()
                    route = m.get("route", "")
                    st.markdown(f"- {drug}{dose} ({route})" if route else f"- {drug}{dose}")
        else:
            st.info("Ground truth file not cached. Run evaluation again to generate.")

    with ext_tab:
        ext_dx = extraction_data.get("diagnoses", [])
        ext_px = extraction_data.get("procedures", [])
        ext_rx = extraction_data.get("medications", [])

        if ext_dx:
            st.markdown(f"**Extracted Diagnoses ({len(ext_dx)})**")
            for d in ext_dx:
                icd = d.get("icd10_suggestion", "N/A")
                conf = d.get("confidence", "?")
                st.markdown(f"- `{icd}` **{d['name']}** ({conf})")

        if ext_px:
            st.markdown(f"**Extracted Procedures ({len(ext_px)})**")
            for p in ext_px:
                cpt = p.get("cpt_suggestion", "N/A")
                st.markdown(f"- `{cpt}` **{p['name']}** ({p.get('confidence', '?')})")

        if ext_rx:
            st.markdown(f"**Extracted Medications ({len(ext_rx)})**")
            for m in ext_rx:
                details = " | ".join(filter(None, [m.get("dosage"), m.get("frequency"), m.get("route")]))
                st.markdown(f"- **{m['name']}** {f'({details})' if details else ''} ({m.get('confidence', '?')})")

st.divider()

# ---------------------------------------------------------------------------
# Confusion examples: FP and FN details
# ---------------------------------------------------------------------------
st.subheader("Error Analysis")

selected_type = st.selectbox("Entity type", entity_types, format_func=str.title)

fp_rows = []
fn_rows = []

for hadm_id, results in per_admission.items():
    for r in results:
        if r["entity_type"] != selected_type:
            continue
        for d in r.get("details", []):
            if d["matched"]:
                continue
            if d.get("extracted_value") and not d.get("ground_truth_value"):
                fp_rows.append({
                    "Admission ID": int(hadm_id),
                    "Extracted (Hallucinated)": d["extracted_value"],
                    "Match Method": d["match_method"],
                })

        # Collect FNs: ground truth items not matched by any extraction
        matched_gt = {
            d["ground_truth_value"]
            for d in r.get("details", [])
            if d["matched"] and d.get("ground_truth_value")
        }
        # We need the GT details — reconstruct from TP + FN counts
        # FNs are GT items minus matched ones; details only have matched + FP extractions
        # For FN display, note the fn count; detailed GT listing requires the ground truth file
        fn_count = r.get("false_negatives", 0)
        if fn_count > 0:
            fn_rows.append({
                "Admission ID": int(hadm_id),
                "Missed Count": fn_count,
                "Matched Count": r.get("true_positives", 0),
                "Total GT": r.get("true_positives", 0) + fn_count,
            })

col_fp, col_fn = st.columns(2)

with col_fp:
    st.markdown(
        tt_heading(
            "False Positives (Hallucinated)",
            f"Extracted {selected_type} that do not match any ground truth entity. "
            "These may be hallucinated by the note generator or the extraction pipeline.",
            style="font-size:1.05em;font-weight:600",
        ),
        unsafe_allow_html=True,
    )
    if fp_rows:
        st.dataframe(pd.DataFrame(fp_rows), hide_index=True, use_container_width=True)
    else:
        st.success(f"No false positives for {selected_type}.")

with col_fn:
    st.markdown(
        tt_heading(
            "False Negatives (Missed)",
            f"Ground truth {selected_type} that the extraction pipeline failed to find. "
            "Higher counts indicate the pipeline is missing entities.",
            style="font-size:1.05em;font-weight:600",
        ),
        unsafe_allow_html=True,
    )
    if fn_rows:
        st.dataframe(pd.DataFrame(fn_rows), hide_index=True, use_container_width=True)
    else:
        st.success(f"No false negatives for {selected_type}.")

st.divider()

# ---------------------------------------------------------------------------
# ICD-10 chapter-level analysis (diagnoses only)
# ---------------------------------------------------------------------------
st.subheader("Diagnosis F1 by ICD-10 Chapter")
st.caption(
    "Categories based on the first character of the ICD-10 code. "
    "Shows how extraction quality varies across clinical domains."
)

ICD10_CHAPTERS = {
    "A": "Infectious diseases",
    "B": "Infectious diseases",
    "C": "Neoplasms",
    "D": "Blood / neoplasms",
    "E": "Endocrine / metabolic",
    "F": "Mental / behavioral",
    "G": "Nervous system",
    "H": "Eye / ear",
    "I": "Circulatory",
    "J": "Respiratory",
    "K": "Digestive",
    "L": "Skin",
    "M": "Musculoskeletal",
    "N": "Genitourinary",
    "O": "Pregnancy",
    "P": "Perinatal",
    "Q": "Congenital",
    "R": "Signs / symptoms",
    "S": "Injury",
    "T": "Injury / poisoning",
    "V": "External causes",
    "W": "External causes",
    "X": "External causes",
    "Y": "External causes",
    "Z": "Health status / encounters",
}

# Collect per-chapter TP/FP/FN from match details
chapter_stats: dict[str, dict[str, int]] = {}

for hadm_id, results in per_admission.items():
    for r in results:
        if r["entity_type"] != "diagnoses":
            continue
        for d in r.get("details", []):
            gt_val = d.get("ground_truth_value", "")
            ext_val = d.get("extracted_value", "")
            code = gt_val or ext_val
            if not code:
                continue
            chapter = code[0].upper() if code else "?"
            if chapter not in chapter_stats:
                chapter_stats[chapter] = {"tp": 0, "fp": 0}
            if d["matched"]:
                chapter_stats[chapter]["tp"] += 1
            else:
                chapter_stats[chapter]["fp"] += 1

if chapter_stats:
    chapter_rows = []
    for ch, stats in sorted(chapter_stats.items()):
        tp = stats["tp"]
        fp = stats["fp"]
        total = tp + fp
        label = ICD10_CHAPTERS.get(ch, "Other")
        chapter_rows.append({
            "Chapter": f"{ch} - {label}",
            "TP": tp,
            "FP": fp,
            "Total Extracted": total,
            "Accuracy": tp / total if total > 0 else 0,
        })

    ch_df = pd.DataFrame(chapter_rows)
    chart = (
        alt.Chart(ch_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Accuracy:Q", title="Match Accuracy", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Chapter:N", title=None, sort="-x"),
            color=alt.Color(
                "Accuracy:Q",
                scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
                legend=None,
            ),
            tooltip=["Chapter", "TP", "FP", "Total Extracted",
                      alt.Tooltip("Accuracy:Q", format=".1%")],
        )
        .properties(height=max(300, len(ch_df) * 30))
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No chapter-level data available.")

# ---------------------------------------------------------------------------
# Cost summary
# ---------------------------------------------------------------------------
cost_path = EVAL_DIR / "eval_cost_report.json"
if cost_path.exists():
    st.divider()
    st.subheader("Evaluation Cost")
    with open(cost_path) as f:
        cost_data = json.load(f)
    summary = cost_data.get("summary", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Requests", f"{summary.get('total_requests', 0):,}")
    total_tokens = summary.get("total_input_tokens", 0) + summary.get("total_output_tokens", 0)
    c2.metric("Total Tokens", f"{total_tokens:,}")
    c3.metric("Estimated Cost", f"${summary.get('estimated_cost_usd', 0):.2f}")
    c3.markdown(
        tt(
            '<span style="font-size:0.75em;color:#888">Pricing</span>',
            "Extraction cost only (Haiku 4.5). Synthetic note generation cost "
            "is not tracked separately.",
        ),
        unsafe_allow_html=True,
    )
