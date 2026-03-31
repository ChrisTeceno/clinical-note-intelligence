"""Main entry point for the medical imaging pipeline."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_pipeline.config import get_settings
from clinical_pipeline.imaging.classifier import ChestXRayClassifier
from clinical_pipeline.imaging.evaluator import evaluate_predictions
from clinical_pipeline.imaging.nih_loader import NIH_PATHOLOGIES, load_image, load_nih_metadata

logger = logging.getLogger(__name__)


def _stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Select up to *n* images with stratified sampling across pathologies.

    Ensures representation of each pathology label.  Images with multiple
    labels count toward each of their labels.
    """
    rng = np.random.RandomState(seed)

    # Explode labels so each row is (image_index, single_label)
    exploded = df.explode("labels")
    # Include "No Finding" images (empty label list => NaN after explode)
    no_finding = df[df["labels"].apply(len) == 0]

    selected_indices: set[int] = set()

    # Round-robin across pathologies
    for pathology in NIH_PATHOLOGIES:
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

    # Add some "No Finding" images
    if not no_finding.empty:
        nf_candidates = no_finding.index.difference(pd.Index(list(selected_indices)))
        take_nf = max(1, n // (len(NIH_PATHOLOGIES) + 1))
        if len(nf_candidates) > take_nf:
            chosen_nf = rng.choice(nf_candidates.tolist(), size=take_nf, replace=False)
        else:
            chosen_nf = nf_candidates.tolist()
        selected_indices.update(chosen_nf)

    # If we still need more, fill randomly
    remaining = n - len(selected_indices)
    if remaining > 0:
        pool = df.index.difference(pd.Index(list(selected_indices)))
        if len(pool) > remaining:
            extra = rng.choice(pool.tolist(), size=remaining, replace=False)
        else:
            extra = pool.tolist()
        selected_indices.update(extra)

    # Trim if we overshot
    selected = sorted(selected_indices)[:n]
    return df.loc[selected].reset_index(drop=True)


def run_imaging_pipeline(
    data_dir: Path,
    n_images: int = 100,
    model_name: str = "densenet121-res224-nih",
    threshold: float = 0.5,
    batch_size: int = 16,
) -> dict:
    """Run the full imaging pipeline.

    Steps
    -----
    1. Load NIH metadata
    2. Select *n_images* (stratified by pathology)
    3. Load and preprocess images
    4. Run TorchXRayVision inference
    5. Evaluate against ground truth
    6. Save results to ``data/imaging/results/``

    Parameters
    ----------
    data_dir : Path
        Root of the NIH sample data directory.
    n_images : int
        Number of images to process.
    model_name : str
        TorchXRayVision weight identifier.
    threshold : float
        Probability threshold for binary classification.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    dict
        Evaluation results including per-class and aggregate metrics.
    """
    start = time.monotonic()

    results_dir = data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_cache = results_dir / "predictions.json"
    results_path = results_dir / "evaluation_results.json"

    # Step 1: Load metadata
    logger.info("Loading NIH metadata from %s", data_dir)
    df = load_nih_metadata(data_dir)

    # Step 2: Stratified sample
    sample_df = _stratified_sample(df, n_images)
    logger.info("Selected %d images for evaluation", len(sample_df))

    # Resolve image directory
    images_dir = data_dir / "sample" / "images"
    if not images_dir.exists():
        # Try flat structure
        images_dir = data_dir / "images"

    # Step 3 & 4: Load images and run inference (with caching)
    if predictions_cache.exists():
        logger.info("Loading cached predictions from %s", predictions_cache)
        cached = json.loads(predictions_cache.read_text(encoding="utf-8"))
        cached_preds = cached["predictions"]
        cached_images = cached["image_names"]
        # Rebuild sample_df to match cached order
        sample_df = df[df["Image Index"].isin(cached_images)].reset_index(drop=True)
        # Maintain alignment
        name_to_idx = {name: i for i, name in enumerate(cached_images)}
        order = [name_to_idx[name] for name in sample_df["Image Index"]]
        predictions = [cached_preds[i] for i in order]
        logger.info("Loaded %d cached predictions", len(predictions))
    else:
        classifier = ChestXRayClassifier(model_name=model_name)

        # Load images
        images: list[np.ndarray] = []
        valid_indices: list[int] = []
        for idx, row in sample_df.iterrows():
            img_path = images_dir / row["Image Index"]
            if not img_path.exists():
                logger.warning("Image not found: %s", img_path)
                continue
            try:
                img = load_image(img_path)
                images.append(img)
                valid_indices.append(idx)
            except Exception:
                logger.exception("Failed to load image: %s", img_path)

        if not images:
            raise FileNotFoundError(
                f"No valid images found in {images_dir}. "
                "Ensure the NIH sample dataset has been extracted."
            )

        sample_df = sample_df.loc[valid_indices].reset_index(drop=True)
        logger.info("Loaded %d images, running inference...", len(images))

        predictions = classifier.predict_batch(images, batch_size=batch_size)

        # Cache predictions
        cache_data = {
            "image_names": sample_df["Image Index"].tolist(),
            "predictions": predictions,
            "model": model_name,
            "n_images": len(predictions),
        }
        predictions_cache.write_text(
            json.dumps(cache_data, indent=2), encoding="utf-8"
        )
        logger.info("Cached %d predictions to %s", len(predictions), predictions_cache)

    # Step 5: Evaluate
    logger.info("Evaluating predictions...")
    results = evaluate_predictions(predictions, sample_df, threshold=threshold)
    results["model"] = model_name
    results["duration_seconds"] = round(time.monotonic() - start, 2)
    results["image_names"] = sample_df["Image Index"].tolist()

    # Save per-image predictions alongside results for the Streamlit viewer
    per_image: list[dict] = []
    for i, row in sample_df.iterrows():
        per_image.append({
            "image": row["Image Index"],
            "ground_truth": row["labels"],
            "predictions": predictions[i],
        })
    results["per_image"] = per_image

    # Step 6: Save
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", results_path)

    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run chest X-ray imaging pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to NIH sample data directory (default: data/imaging/)",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=100,
        help="Number of images to process (default: 100)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="densenet121-res224-nih",
        help="TorchXRayVision model weights",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = get_settings().imaging_dir

    run_imaging_pipeline(
        data_dir=data_dir,
        n_images=args.n_images,
        model_name=args.model,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
