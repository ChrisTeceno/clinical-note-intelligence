"""Chest X-ray pathology classification using TorchXRayVision."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torchxrayvision as xrv

logger = logging.getLogger(__name__)

# The 14 clinically meaningful pathology labels output by the NIH model.
# The model's ``pathologies`` list has 18 slots; the last four are empty.
_NIH_PATHOLOGY_COUNT = 14


class ChestXRayClassifier:
    """Wrapper around TorchXRayVision DenseNet model.

    Parameters
    ----------
    model_name : str
        Weight identifier accepted by ``xrv.models.DenseNet``.
        Default is the NIH-pretrained 224-px DenseNet-121.
    """

    def __init__(self, model_name: str = "densenet121-res224-nih") -> None:
        logger.info("Loading TorchXRayVision model: %s", model_name)
        self.model = xrv.models.DenseNet(weights=model_name)
        self.model.eval()
        self.pathologies: list[str] = list(
            self.model.pathologies[:_NIH_PATHOLOGY_COUNT]
        )
        self.model_name = model_name
        logger.info("Model ready — %d pathologies", len(self.pathologies))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> dict[str, float]:
        """Run inference on a single preprocessed image.

        Parameters
        ----------
        image : np.ndarray
            Float32 array of shape ``(1, 224, 224)`` in [-1024, 1024].

        Returns
        -------
        dict[str, float]
            ``{pathology_name: probability}`` for each of the 14 classes.
        """
        tensor = torch.from_numpy(image).unsqueeze(0)  # (1, 1, 224, 224)
        with torch.no_grad():
            outputs = self.model(tensor)
        probs = outputs[0].cpu().numpy()
        return {
            name: float(probs[i])
            for i, name in enumerate(self.pathologies)
        }

    def predict_batch(
        self, images: Sequence[np.ndarray], batch_size: int = 16
    ) -> list[dict[str, float]]:
        """Run inference on a batch of preprocessed images.

        Parameters
        ----------
        images : Sequence[np.ndarray]
            Each element is a float32 array of shape ``(1, 224, 224)``.
        batch_size : int
            Number of images per forward pass.

        Returns
        -------
        list[dict[str, float]]
            One prediction dict per input image.
        """
        results: list[dict[str, float]] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensor = torch.from_numpy(np.stack(batch))  # (B, 1, 224, 224)
            with torch.no_grad():
                outputs = self.model(tensor)
            probs = outputs.cpu().numpy()
            for row in probs:
                results.append({
                    name: float(row[i])
                    for i, name in enumerate(self.pathologies)
                })
        return results
