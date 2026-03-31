"""Track evaluation runs over time for experiment comparison."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunRecord:
    """A single evaluation run's summary metrics."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model: str = ""
    n_admissions: int = 0
    description: str = ""
    diagnoses: dict = field(default_factory=dict)
    procedures: dict = field(default_factory=dict)
    medications: dict = field(default_factory=dict)
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RunRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__slots__})


class RunTracker:
    """Append-only run history stored as a JSON array."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.runs: list[RunRecord] = self._load()

    def record(self, run: RunRecord) -> None:
        """Append a new run to history and persist."""
        self.runs.append(run)
        self._save()
        logger.info("Recorded run %s: %s", run.run_id, run.description)

    def get_history(self) -> list[RunRecord]:
        """Return all runs sorted by timestamp (oldest first)."""
        return sorted(self.runs, key=lambda r: r.timestamp)

    def _load(self) -> list[RunRecord]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return [RunRecord.from_dict(d) for d in data]
        except (json.JSONDecodeError, KeyError):
            logger.warning("Could not parse run history at %s, starting fresh", self.path)
            return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([r.to_dict() for r in self.runs], indent=2),
            encoding="utf-8",
        )
