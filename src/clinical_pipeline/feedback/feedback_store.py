"""Append-only store for HITL reviewer corrections."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ENTITY_TYPES = ("diagnosis", "procedure", "medication")
ACTIONS = ("corrected", "added", "removed", "confirmed")


@dataclass(slots=True)
class FeedbackItem:
    """A single reviewer correction or confirmation."""

    note_id: str
    entity_type: str  # "diagnosis", "procedure", "medication"
    action: str  # "corrected", "added", "removed", "confirmed"
    original_value: dict  # what the model extracted
    corrected_value: dict | None  # what the reviewer changed it to
    note_snippet: str  # relevant portion of the clinical note
    timestamp: str
    reviewer: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> FeedbackItem:
        return cls(**{k: v for k, v in data.items() if k in cls.__slots__})


class FeedbackStore:
    """Append-only store for HITL corrections. Saved to data/feedback/corrections.json."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.items: list[FeedbackItem] = self._load()

    def add(self, item: FeedbackItem) -> None:
        """Add a correction."""
        self.items.append(item)
        self._save()
        logger.info(
            "Recorded %s feedback for note %s (%s)",
            item.action,
            item.note_id,
            item.entity_type,
        )

    def get_corrections(
        self, entity_type: str | None = None, limit: int = 20
    ) -> list[FeedbackItem]:
        """Get recent corrections, optionally filtered by type."""
        filtered = self.items
        if entity_type is not None:
            filtered = [f for f in filtered if f.entity_type == entity_type]
        # Return most recent first
        return list(reversed(filtered))[:limit]

    def get_few_shot_examples(self, n: int = 5) -> list[dict]:
        """Convert recent corrections into few-shot prompt examples.

        Returns examples in the format:
        [{"note_snippet": "...", "correct_extraction": {...}, "common_mistake": "..."}]
        """
        corrections = [f for f in self.items if f.action in ("corrected", "added")]
        # Most recent first
        corrections = list(reversed(corrections))[:n]
        examples = []
        for item in corrections:
            example: dict = {"note_snippet": item.note_snippet}
            if item.action == "corrected":
                example["correct_extraction"] = item.corrected_value or {}
                original_name = item.original_value.get("name", "")
                corrected_name = (item.corrected_value or {}).get("name", "")
                if original_name and corrected_name and original_name != corrected_name:
                    example["common_mistake"] = (
                        f"Model extracted '{original_name}' but the correct "
                        f"extraction is '{corrected_name}'"
                    )
                else:
                    original_code = item.original_value.get("icd10_suggestion", "")
                    corrected_code = (item.corrected_value or {}).get(
                        "icd10_suggestion", ""
                    )
                    if original_code != corrected_code:
                        example["common_mistake"] = (
                            f"Model suggested code '{original_code}' but the "
                            f"correct code is '{corrected_code}'"
                        )
                    else:
                        example["common_mistake"] = (
                            f"Model output was corrected for '{original_name}'"
                        )
            elif item.action == "added":
                example["correct_extraction"] = item.corrected_value or {}
                example["common_mistake"] = (
                    "Model missed this entity entirely — it should have been extracted"
                )
            examples.append(example)
        return examples

    def summary(self) -> dict:
        """Return counts by entity_type, action, and common error patterns."""
        by_type: Counter[str] = Counter()
        by_action: Counter[str] = Counter()
        error_names: Counter[str] = Counter()

        for item in self.items:
            by_type[item.entity_type] += 1
            by_action[item.action] += 1
            if item.action == "corrected":
                name = item.original_value.get("name", "unknown")
                error_names[name] += 1
            elif item.action == "removed":
                name = item.original_value.get("name", "unknown")
                error_names[f"hallucinated: {name}"] += 1
            elif item.action == "added":
                name = (item.corrected_value or {}).get("name", "unknown")
                error_names[f"missed: {name}"] += 1

        return {
            "total": len(self.items),
            "by_entity_type": dict(by_type),
            "by_action": dict(by_action),
            "common_errors": dict(error_names.most_common(20)),
        }

    def _load(self) -> list[FeedbackItem]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return [FeedbackItem.from_dict(d) for d in data]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(
                "Could not parse feedback at %s, starting fresh", self.path
            )
            return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([item.to_dict() for item in self.items], indent=2),
            encoding="utf-8",
        )
