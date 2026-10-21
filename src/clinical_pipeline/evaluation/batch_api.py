"""Message Batches API helpers: submit requests, poll to completion, collect results."""

from __future__ import annotations

import logging
import time

import anthropic

from clinical_pipeline.extraction.cost_tracker import BATCH_DISCOUNT, MODEL_PRICING

logger = logging.getLogger(__name__)

POLL_INITIAL_SECONDS = 15.0
POLL_MAX_SECONDS = 120.0
POLL_BACKOFF = 1.5
DEFAULT_TIMEOUT_SECONDS = 4 * 3600.0

# Rough per-request token counts used only for the pre-submit cost estimate
ESTIMATED_NOTE_TOKENS = (900, 1000)
ESTIMATED_EXTRACTION_TOKENS = (3000, 1500)


class BatchError(Exception):
    """Raised when a batch cannot be submitted or does not finish in time."""


def estimate_batch_cost(model: str, n_requests: int, tokens_per_request: tuple[int, int]) -> float:
    """Rough USD cost of a batch: (input, output) tokens per request x count x batch price."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        logger.warning("No pricing data for model %s, estimating as zero", model)
        return 0.0
    input_tokens, output_tokens = tokens_per_request
    per_request = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    return per_request * n_requests * BATCH_DISCOUNT


def run_batch(
    client: anthropic.Anthropic,
    requests: dict[str, dict],
    label: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    poll_initial: float = POLL_INITIAL_SECONDS,
) -> dict[str, anthropic.types.Message]:
    """Submit ``{custom_id: params}`` as one batch and wait for it to end.

    Returns custom_id -> Message for succeeded items only. Errored, canceled and
    expired items are logged and left out.
    """
    if not requests:
        return {}

    batch = client.messages.batches.create(
        requests=[{"custom_id": cid, "params": params} for cid, params in requests.items()]
    )
    logger.info("Submitted %s batch %s with %d requests", label, batch.id, len(requests))

    deadline = time.monotonic() + timeout_seconds
    delay = poll_initial
    while batch.processing_status != "ended":
        if time.monotonic() + delay > deadline:
            raise BatchError(
                f"{label} batch {batch.id} not finished after {timeout_seconds:.0f}s; "
                "it keeps running server side"
            )
        time.sleep(delay)
        delay = min(delay * POLL_BACKOFF, POLL_MAX_SECONDS)
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        logger.info(
            "%s batch %s: %s (processing=%d succeeded=%d errored=%d)",
            label,
            batch.id,
            batch.processing_status,
            counts.processing,
            counts.succeeded,
            counts.errored,
        )

    messages: dict[str, anthropic.types.Message] = {}
    for item in client.messages.batches.results(batch.id):
        if item.result.type == "succeeded":
            messages[item.custom_id] = item.result.message
        else:
            logger.warning(
                "%s batch item %s ended as %s: %s",
                label,
                item.custom_id,
                item.result.type,
                getattr(item.result, "error", ""),
            )
    logger.info("%s batch %s: %d/%d succeeded", label, batch.id, len(messages), len(requests))
    return messages
