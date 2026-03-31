"""Medical Imaging — chest X-ray classification with TorchXRayVision."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Medical Imaging Analysis")

st.info(
    "**About this module:** Chest X-ray images from the NIH ChestX-ray14 sample dataset "
    "are classified using TorchXRayVision models pretrained on various clinical datasets. "
    "Predictions for up to 18 pathology classes are evaluated against ground-truth labels. "
    "Use the model selector to compare performance across architectures and training data.",
    icon=":material/image:",
)

# ---------------------------------------------------------------------------
# Discover available model results
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "imaging"
RESULTS_DIR = DATA_DIR / "results"
IMAGES_DIR = DATA_DIR / "sample" / "images"

if not IMAGES_DIR.exists():
    IMAGES_DIR = DATA_DIR / "images"

# Find all model result directories
available_models: dict[str, Path] = {}
if RESULTS_DIR.exists():
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and (d / "evaluation_results.json").exists():
            available_models[d.name] = d / "evaluation_results.json"

# Also check the root results file
root_results = RESULTS_DIR / "evaluation_results.json"
if root_results.exists() and not available_models:
    available_models["default"] = root_results

if not available_models:
    st.warning(
        "No imaging results found. Run the imaging pipeline first:\n\n"
        "```\nPYTHONPATH=src python -m clinical_pipeline.imaging.run_imaging\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------
MODEL_LABELS = {
    "densenet121-res224-all": "DenseNet-121 All Datasets (18 pathologies)",
    "densenet121-res224-nih": "DenseNet-121 NIH (14 pathologies)",
    "densenet121-res224-chex": "DenseNet-121 CheXpert (11 pathologies)",
    "densenet121-res224-mimic_nb": "DenseNet-121 MIMIC NoBrainer (11 pathologies)",
    "densenet121-res224-mimic_ch": "DenseNet-121 MIMIC CheXpert (11 pathologies)",
    "densenet121-res224-pc": "DenseNet-121 PadChest (15 pathologies)",
    "densenet121-res224-rsna": "DenseNet-121 RSNA (2 pathologies)",
    "resnet50-res512-all": "ResNet-50 All Datasets (18 pathologies)",
}

model_keys = list(available_models.keys())
selected_model = st.selectbox(
    "Select Model",
    options=model_keys,
    format_func=lambda k: MODEL_LABELS.get(k, k),
)

# ---------------------------------------------------------------------------
# Load selected model results
# ---------------------------------------------------------------------------
with open(available_models[selected_model]) as f:
    results = json.load(f)

per_class = results.get("per_class", {})
macro = results.get("macro", {})
micro = results.get("micro", {})
n_images = results.get("n_images", 0)
model_name = results.get("model", selected_model)
per_image = results.get("per_image", [])
threshold = results.get("threshold", 0.5)

# ---------------------------------------------------------------------------
# Model comparison summary (always visible)
# ---------------------------------------------------------------------------
st.subheader("Model Comparison")

comparison_rows = []
for mk, mp in available_models.items():
    with open(mp) as f:
        mr = json.load(f)
    mc_per_class = mr.get("per_class", {})
    aucs = [v["auc"] for v in mc_per_class.values() if v.get("auc") is not None]
    mean_auc = sum(aucs) / len(aucs) if aucs else 0
    comparison_rows.append({
        "Model": MODEL_LABELS.get(mk, mk),
        "Images": mr.get("n_images", 0),
        "Pathologies": len(mc_per_class),
        "Mean AUC": mean_auc,
        "Macro F1": mr.get("macro", {}).get("f1", 0),
        "Macro Precision": mr.get("macro", {}).get("precision", 0),
        "Macro Recall": mr.get("macro", {}).get("recall", 0),
    })

comp_df = pd.DataFrame(comparison_rows).sort_values("Mean AUC", ascending=False)
st.dataframe(
    comp_df,
    column_config={
        "Mean AUC": st.column_config.ProgressColumn("Mean AUC", min_value=0, max_value=1, format="%.3f"),
        "Macro F1": st.column_config.ProgressColumn("Macro F1", min_value=0, max_value=1, format="%.3f"),
        "Macro Precision": st.column_config.NumberColumn(format="%.3f"),
        "Macro Recall": st.column_config.NumberColumn(format="%.3f"),
    },
    hide_index=True,
    use_container_width=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Overview KPIs for selected model
# ---------------------------------------------------------------------------
st.subheader(f"Selected: {MODEL_LABELS.get(selected_model, selected_model)}")

aucs = [v["auc"] for v in per_class.values() if v.get("auc") is not None]
mean_auc = sum(aucs) / len(aucs) if aucs else 0

kpi_cols = st.columns(5)
kpi_cols[0].metric("Images", f"{n_images:,}")
kpi_cols[1].metric("Pathologies", len(per_class))
kpi_cols[2].metric("Mean AUC", f"{mean_auc:.3f}")
kpi_cols[3].metric("Macro F1", f"{macro.get('f1', 0):.1%}")
kpi_cols[4].metric("Macro Precision", f"{macro.get('precision', 0):.1%}")

st.divider()

# ---------------------------------------------------------------------------
# Per-pathology AUC / F1 bar chart
# ---------------------------------------------------------------------------
st.subheader("Per-Pathology Performance")

chart_rows = []
for pathology, metrics in per_class.items():
    for metric_name in ["f1", "auc"]:
        val = metrics.get(metric_name)
        if val is not None:
            chart_rows.append({
                "Pathology": pathology,
                "Metric": metric_name.upper(),
                "Value": val,
            })

if chart_rows:
    chart_df = pd.DataFrame(chart_rows)
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Pathology:N", title=None, sort="-x"),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(
                    domain=["F1", "AUC"],
                    range=["#3498db", "#e67e22"],
                ),
            ),
            row=alt.Row("Metric:N", title=None),
            tooltip=["Pathology", "Metric", alt.Tooltip("Value:Q", format=".3f")],
        )
        .properties(width=500, height=alt.Step(20))
    )
    st.altair_chart(chart)

# Detailed table
st.markdown("**Detailed Metrics**")
table_rows = []
for pathology, metrics in per_class.items():
    table_rows.append({
        "Pathology": pathology,
        "Precision": metrics.get("precision", 0),
        "Recall": metrics.get("recall", 0),
        "F1": metrics.get("f1", 0),
        "AUC": metrics.get("auc"),
        "Support": metrics.get("support", 0),
    })

if table_rows:
    table_df = pd.DataFrame(table_rows).sort_values("AUC", ascending=False)
    st.dataframe(
        table_df,
        column_config={
            "F1": st.column_config.ProgressColumn("F1", min_value=0, max_value=1, format="%.3f"),
            "AUC": st.column_config.ProgressColumn("AUC", min_value=0, max_value=1, format="%.3f"),
            "Precision": st.column_config.NumberColumn(format="%.3f"),
            "Recall": st.column_config.NumberColumn(format="%.3f"),
        },
        hide_index=True,
        use_container_width=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Sample predictions viewer
# ---------------------------------------------------------------------------
st.subheader("Sample Predictions")
st.caption(
    "Browse individual chest X-ray predictions. "
    "Colours: **green** = true positive, **red** = false positive, **orange** = false negative."
)

if not per_image:
    st.info("No per-image data available for this model.")
    st.stop()

image_options = [item["image"] for item in per_image]
selected_idx = st.selectbox(
    "Select image",
    range(len(image_options)),
    format_func=lambda i: image_options[i],
)

item = per_image[selected_idx]
image_name = item["image"]
gt_labels: list[str] = item["ground_truth"]
preds: dict[str, float] = item["predictions"]

img_col, pred_col = st.columns([1, 1])

with img_col:
    img_path = IMAGES_DIR / image_name
    if img_path.exists():
        st.image(str(img_path), caption=image_name, use_container_width=True)
    else:
        st.warning(f"Image file not found: {img_path}")

with pred_col:
    st.markdown("**Ground Truth**")
    if gt_labels:
        st.markdown(", ".join(f"`{l}`" for l in gt_labels))
    else:
        st.markdown("`No Finding`")

    st.markdown("**Predicted Pathologies**")

    sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
    gt_set = set(gt_labels)

    for name, prob in sorted_preds:
        is_predicted = prob >= threshold
        is_gt = name in gt_set

        if is_predicted and is_gt:
            color = "green"
            tag = "TP"
        elif is_predicted and not is_gt:
            color = "red"
            tag = "FP"
        elif not is_predicted and is_gt:
            color = "orange"
            tag = "FN"
        else:
            continue

        bar_width = int(prob * 100)
        st.markdown(
            f":{color}[**{tag}**] **{name}** — {prob:.3f} "
            f"<span style='display:inline-block;width:{bar_width}px;height:8px;"
            f"background:{color};border-radius:4px'></span>",
            unsafe_allow_html=True,
        )

    high_tn = [
        (name, prob) for name, prob in sorted_preds
        if prob < threshold and name not in gt_set and prob > 0.3
    ]
    if high_tn:
        with st.expander("Near-threshold true negatives"):
            for name, prob in high_tn:
                st.markdown(f"**{name}** — {prob:.3f}")

st.divider()

# ---------------------------------------------------------------------------
# Confusion examples
# ---------------------------------------------------------------------------
st.subheader("Confusion Examples")
st.caption("Highest-confidence false positives and false negatives across all evaluated images.")

fp_examples: list[dict] = []
fn_examples: list[dict] = []

for item in per_image:
    gt_set = set(item["ground_truth"])
    for pathology, prob in item["predictions"].items():
        if prob >= threshold and pathology not in gt_set:
            fp_examples.append({
                "Image": item["image"],
                "Pathology": pathology,
                "Confidence": prob,
            })
        elif prob < threshold and pathology in gt_set:
            fn_examples.append({
                "Image": item["image"],
                "Pathology": pathology,
                "Confidence": prob,
            })

fp_col, fn_col = st.columns(2)

with fp_col:
    st.markdown("**False Positives** (predicted but not in ground truth)")
    if fp_examples:
        fp_df = pd.DataFrame(fp_examples).sort_values("Confidence", ascending=False).head(20)
        st.dataframe(
            fp_df,
            column_config={"Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")},
            hide_index=True, use_container_width=True,
        )
    else:
        st.success("No false positives found.")

with fn_col:
    st.markdown("**False Negatives** (missed ground-truth labels)")
    if fn_examples:
        fn_df = pd.DataFrame(fn_examples).sort_values("Confidence", ascending=False).head(20)
        st.dataframe(
            fn_df,
            column_config={"Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")},
            hide_index=True, use_container_width=True,
        )
    else:
        st.success("No false negatives found.")
