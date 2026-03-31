"""Run Claude Vision chest X-ray classification pipeline."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_pipeline.config import get_settings
from clinical_pipeline.imaging.claude_classifier import ClaudeVisionClassifier, NIH_PATHOLOGIES
from clinical_pipeline.imaging.evaluator import evaluate_predictions
from clinical_pipeline.imaging.nih_loader import load_nih_metadata

logger = logging.getLogger(__name__)


def _stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Select up to *n* images with stratified sampling across pathologies."""
    rng = np.random.RandomState(seed)
    selected_indices: set[int] = set()

    for pathology in NIH_PATHOLOGIES:
        exploded = df.explode("labels")
        subset = exploded[exploded["labels"] == pathology]
        if subset.empty:
            continue
        candidates = subset.index.difference(pd.Index(list(selected_indices)))
        take = max(1, n // (len(NIH_PATHOLOGIES) + 1))
        if len(candidates) > take:
            chosen = rng.choice(candidates.tolist(), size=take, replace=False)
        else:
            chosen = candidates.tolist()
        selected_indices.update(chosen)

    # Fill with no-finding images
    no_finding = df[df["labels"].apply(len) == 0]
    if not no_finding.empty:
        nf_candidates = no_finding.index.difference(pd.Index(list(selected_indices)))
        take_nf = max(1, n // (len(NIH_PATHOLOGIES) + 1))
        if len(nf_candidates) > take_nf:
            chosen_nf = rng.choice(nf_candidates.tolist(), size=take_nf, replace=False)
        else:
            chosen_nf = nf_candidates.tolist()
        selected_indices.update(chosen_nf)

    # Fill randomly if needed
    remaining = n - len(selected_indices)
    if remaining > 0:
        pool = df.index.difference(pd.Index(list(selected_indices)))
        if len(pool) > remaining:
            extra = rng.choice(pool.tolist(), size=remaining, replace=False)
        else:
            extra = pool.tolist()
        selected_indices.update(extra)

    selected = sorted(selected_indices)[:n]
    return df.loc[selected].reset_index(drop=True)


def run_claude_vision_pipeline(
    data_dir: Path,
    n_images: int = 100,
    api_key: str | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """Run Claude Vision classification on NIH chest X-rays.

    Results are saved in the same format as TorchXRayVision results
    so the imaging dashboard auto-discovers them.
    """
    start = time.monotonic()
    settings = get_settings()
    api_key = api_key or settings.anthropic_api_key

    model_label = "claude-haiku-vision"
    results_dir = data_dir / "results" / model_label
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_cache = results_dir / "predictions.json"
    results_path = results_dir / "evaluation_results.json"

    # Load metadata
    logger.info("Loading NIH metadata from %s", data_dir)
    df = load_nih_metadata(data_dir)
    sample_df = _stratified_sample(df, n_images)
    logger.info("Selected %d images", len(sample_df))

    # Resolve image directory
    images_dir = data_dir / "sample" / "images"
    if not images_dir.exists():
        images_dir = data_dir / "images"

    # Check cache for per-image predictions
    cached_preds: dict[str, dict[str, float]] = {}
    per_image_cache = results_dir / "per_image_cache"
    per_image_cache.mkdir(parents=True, exist_ok=True)
    for f in per_image_cache.glob("*.json"):
        cached_preds[f.stem] = json.loads(f.read_text())

    # Run classification
    classifier = ClaudeVisionClassifier(api_key=api_key, model=model)
    predictions: list[dict[str, float]] = []
    valid_indices: list[int] = []
    total_cost = 0.0

    for idx, row in sample_df.iterrows():
        image_name = row["Image Index"]
        img_path = images_dir / image_name
        stem = img_path.stem

        if stem in cached_preds:
            predictions.append(cached_preds[stem])
            valid_indices.append(idx)
            logger.info("[%d/%d] Cached: %s", len(valid_indices), len(sample_df), image_name)
            continue

        if not img_path.exists():
            logger.warning("Image not found: %s", img_path)
            continue

        pred = classifier.predict(img_path)
        predictions.append(pred)
        valid_indices.append(idx)

        # Cache per-image
        (per_image_cache / f"{stem}.json").write_text(json.dumps(pred))

        # Rough cost estimate: ~1500 input tokens per image for Haiku vision
        total_cost += 0.0012  # ~$0.80/M tokens * ~1500 tokens

    sample_df = sample_df.loc[valid_indices].reset_index(drop=True)
    logger.info("Classified %d images", len(predictions))

    # Save bulk predictions cache
    cache_data = {
        "image_names": sample_df["Image Index"].tolist(),
        "predictions": predictions,
        "model": model_label,
        "n_images": len(predictions),
    }
    predictions_cache.write_text(json.dumps(cache_data, indent=2))

    # Evaluate
    logger.info("Evaluating predictions...")
    results = evaluate_predictions(predictions, sample_df, threshold=0.5)
    results["model"] = model_label
    results["duration_seconds"] = round(time.monotonic() - start, 2)
    results["image_names"] = sample_df["Image Index"].tolist()
    results["estimated_cost_usd"] = round(total_cost, 4)

    # Per-image details for the viewer
    per_image: list[dict] = []
    for i, row in sample_df.iterrows():
        per_image.append({
            "image": row["Image Index"],
            "ground_truth": row["labels"],
            "predictions": predictions[i],
        })
    results["per_image"] = per_image

    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s (cost: $%.4f)", results_path, total_cost)

    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run Claude Vision CXR classification")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--n-images", type=int, default=100)
    args = parser.parse_args()

    data_dir = args.data_dir or get_settings().imaging_dir
    run_claude_vision_pipeline(data_dir=data_dir, n_images=args.n_images)
