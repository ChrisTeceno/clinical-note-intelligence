"""Evaluate chest X-ray predictions against NIH ground-truth labels."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from clinical_pipeline.imaging.nih_loader import NIH_PATHOLOGIES

logger = logging.getLogger(__name__)


def evaluate_predictions(
    predictions: list[dict[str, float]],
    ground_truth: pd.DataFrame,
    threshold: float = 0.5,
) -> dict:
    """Compute per-pathology and aggregate classification metrics.

    Parameters
    ----------
    predictions : list[dict[str, float]]
        One dict per image mapping pathology names to predicted probabilities.
        Must be aligned with the rows of *ground_truth*.
    ground_truth : pd.DataFrame
        DataFrame that contains a ``labels`` column (list of label strings)
        aligned 1-to-1 with *predictions*.
    threshold : float
        Probability cutoff for converting predictions to binary labels.

    Returns
    -------
    dict
        ``per_class`` : dict mapping pathology -> {precision, recall, f1, support, auc}
        ``macro``     : macro-averaged precision/recall/f1
        ``micro``     : micro-averaged precision/recall/f1
        ``n_images``  : number of images evaluated
    """
    n_images = len(predictions)
    n_classes = len(NIH_PATHOLOGIES)

    # Build binary matrices: (n_images, n_classes)
    y_true = np.zeros((n_images, n_classes), dtype=int)
    y_prob = np.zeros((n_images, n_classes), dtype=np.float32)
    y_pred = np.zeros((n_images, n_classes), dtype=int)

    gt_labels_list = ground_truth["labels"].tolist()

    for i, (preds, gt_labels) in enumerate(zip(predictions, gt_labels_list)):
        for j, pathology in enumerate(NIH_PATHOLOGIES):
            if pathology in gt_labels:
                y_true[i, j] = 1
            prob = preds.get(pathology, 0.0)
            y_prob[i, j] = prob
            y_pred[i, j] = int(prob >= threshold)

    # Per-class metrics from sklearn classification_report
    report = classification_report(
        y_true, y_pred, target_names=NIH_PATHOLOGIES, output_dict=True, zero_division=0
    )

    # Per-class AUC (only computable when both classes are present)
    per_class: dict[str, dict] = {}
    for j, pathology in enumerate(NIH_PATHOLOGIES):
        cls_metrics = report.get(pathology, {})
        entry: dict = {
            "precision": round(cls_metrics.get("precision", 0.0), 4),
            "recall": round(cls_metrics.get("recall", 0.0), 4),
            "f1": round(cls_metrics.get("f1-score", 0.0), 4),
            "support": int(cls_metrics.get("support", 0)),
        }
        # AUC requires at least one positive and one negative sample
        if y_true[:, j].sum() > 0 and y_true[:, j].sum() < n_images:
            try:
                entry["auc"] = round(float(roc_auc_score(y_true[:, j], y_prob[:, j])), 4)
            except ValueError:
                entry["auc"] = None
        else:
            entry["auc"] = None
        per_class[pathology] = entry

    # Macro / micro averages
    macro = report.get("macro avg", {})
    micro = report.get("micro avg", {})

    return {
        "per_class": per_class,
        "macro": {
            "precision": round(macro.get("precision", 0.0), 4),
            "recall": round(macro.get("recall", 0.0), 4),
            "f1": round(macro.get("f1-score", 0.0), 4),
        },
        "micro": {
            "precision": round(micro.get("precision", 0.0), 4),
            "recall": round(micro.get("recall", 0.0), 4),
            "f1": round(micro.get("f1-score", 0.0), 4),
        },
        "n_images": n_images,
        "threshold": threshold,
    }
