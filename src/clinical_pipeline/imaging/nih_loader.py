"""Load and preprocess the NIH ChestX-ray14 sample dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

# The 14 NIH pathology labels used across the pipeline.
NIH_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


def load_nih_metadata(data_dir: Path) -> pd.DataFrame:
    """Load sample_labels.csv and parse Finding Labels into lists.

    Parameters
    ----------
    data_dir : Path
        Root of the NIH sample data (contains ``sample_labels.csv`` and
        ``sample/images/``).

    Returns
    -------
    pd.DataFrame
        DataFrame with an added ``labels`` column containing a list of
        individual findings per image.  ``"No Finding"`` is represented as
        an empty list.
    """
    csv_path = data_dir / "sample_labels.csv"
    df = pd.read_csv(csv_path)

    def _parse_labels(raw: str) -> list[str]:
        if raw.strip() == "No Finding":
            return []
        return [l.strip() for l in raw.split("|") if l.strip()]

    df["labels"] = df["Finding Labels"].apply(_parse_labels)
    logger.info("Loaded %d image records from %s", len(df), csv_path)
    return df


def load_image(image_path: Path, target_size: int = 224) -> np.ndarray:
    """Load a chest X-ray image and prepare it for TorchXRayVision.

    The image is converted to single-channel grayscale, resized to
    ``target_size x target_size``, and normalised to the [-1024, 1024]
    range expected by ``torchxrayvision``.

    Parameters
    ----------
    image_path : Path
        Path to a PNG chest X-ray image.
    target_size : int
        Spatial dimension for the square output (default 224).

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(1, target_size, target_size)`` in
        [-1024, 1024].
    """
    img = Image.open(image_path).convert("L")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)  # [0, 255]
    # Normalise to [-1024, 1024] as required by torchxrayvision
    arr = arr / 255.0 * 2048.0 - 1024.0
    # Add channel dimension: (1, H, W)
    return arr[np.newaxis, :, :]
