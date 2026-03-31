"""Claude Vision classifier for chest X-ray pathology detection."""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

NIH_PATHOLOGIES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
]

SYSTEM_PROMPT = """\
You are an expert radiologist analyzing a chest X-ray image. \
For each of the following 14 pathologies, rate your confidence that it is \
present in this image on a scale of 0.0 to 1.0:

- 0.0 = definitely absent
- 0.3 = unlikely but cannot exclude
- 0.5 = equivocal
- 0.7 = likely present
- 1.0 = definitely present

Use the classify_chest_xray tool to report your findings. \
Include a brief evidence description for any finding with confidence above 0.3.\
"""

CLASSIFY_TOOL = {
    "name": "classify_chest_xray",
    "description": "Report pathology findings from a chest X-ray with confidence scores.",
    "input_schema": {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "description": "Confidence assessment for each pathology.",
                "items": {
                    "type": "object",
                    "properties": {
                        "pathology": {
                            "type": "string",
                            "enum": NIH_PATHOLOGIES,
                            "description": "Name of the pathology.",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence score (0.0 = absent, 1.0 = present).",
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Brief description of radiographic evidence, if any.",
                        },
                    },
                    "required": ["pathology", "confidence"],
                },
            },
        },
        "required": ["findings"],
    },
}


class ClaudeVisionClassifier:
    """Chest X-ray classifier using Claude's vision capabilities."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        max_retries: int = 3,
        delay: float = 0.5,
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.delay = delay

    def predict(self, image_path: Path) -> dict[str, float]:
        """Classify a single chest X-ray image.

        Returns a dict mapping each NIH pathology to a confidence score.
        """
        image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
        media_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analyze this chest X-ray. Assess the confidence "
                            "for each of the 14 standard pathologies."
                        ),
                    },
                ],
            }
        ]

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    tools=[CLASSIFY_TOOL],
                    tool_choice={"type": "tool", "name": "classify_chest_xray"},
                    messages=messages,
                )

                # Extract tool use result
                for block in response.content:
                    if block.type == "tool_use":
                        findings = block.input.get("findings", [])
                        result = {p: 0.0 for p in NIH_PATHOLOGIES}
                        for f in findings:
                            name = f.get("pathology", "")
                            if name in result:
                                result[name] = float(f.get("confidence", 0.0))
                        return result

                logger.warning("No tool_use block in response for %s", image_path.name)
                return {p: 0.0 for p in NIH_PATHOLOGIES}

            except anthropic.RateLimitError:
                logger.warning("Rate limited on attempt %d/%d", attempt, self.max_retries)
                time.sleep(2 ** attempt)
                last_error = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    logger.warning("Server error %d, retrying...", e.status_code)
                    time.sleep(2 ** attempt)
                    last_error = e
                else:
                    raise

        logger.error("Failed to classify %s after %d attempts: %s", image_path.name, self.max_retries, last_error)
        return {p: 0.0 for p in NIH_PATHOLOGIES}

    def predict_batch(
        self, image_paths: list[Path], delay: float | None = None
    ) -> list[dict[str, float]]:
        """Classify a batch of images with rate limiting."""
        delay = delay or self.delay
        results = []
        for i, path in enumerate(image_paths):
            logger.info("[%d/%d] Classifying %s", i + 1, len(image_paths), path.name)
            result = self.predict(path)
            results.append(result)
            if i < len(image_paths) - 1:
                time.sleep(delay)
        return results
