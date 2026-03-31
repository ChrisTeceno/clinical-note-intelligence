"""Track token usage and estimated cost per extraction run."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Pricing per million tokens (as of early 2025 for Haiku 4.5)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-5-20241022": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-6-20250514": {"input": 3.00, "output": 15.00},
}

# Batch API gives 50% discount
BATCH_DISCOUNT = 0.5


@dataclass
class UsageRecord:
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CostTracker:
    """Accumulates token usage across a run and computes estimated cost."""

    def __init__(self, is_batch: bool = False) -> None:
        self.records: list[UsageRecord] = []
        self.is_batch = is_batch

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.records.append(UsageRecord(model=model, input_tokens=input_tokens, output_tokens=output_tokens))

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_requests(self) -> int:
        return len(self.records)

    def estimated_cost(self, model: str | None = None) -> float:
        """Compute estimated USD cost for all recorded usage.

        If *model* is None, uses each record's own model for pricing.
        """
        total = 0.0
        for rec in self.records:
            m = model or rec.model
            pricing = MODEL_PRICING.get(m)
            if pricing is None:
                logger.warning("No pricing data for model %s, skipping cost estimate", m)
                continue
            input_cost = (rec.input_tokens / 1_000_000) * pricing["input"]
            output_cost = (rec.output_tokens / 1_000_000) * pricing["output"]
            total += input_cost + output_cost

        if self.is_batch:
            total *= BATCH_DISCOUNT

        return total

    def summary(self) -> dict[str, object]:
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost(), 4),
            "is_batch": self.is_batch,
        }

    def save(self, path: Path) -> None:
        """Save usage records and summary to a JSON file."""
        data = {
            "summary": self.summary(),
            "records": [
                {
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "timestamp": r.timestamp,
                }
                for r in self.records
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        logger.info("Cost report saved to %s", path)
