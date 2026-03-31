"""ClinicalExtractor: calls Claude API to extract structured clinical data from a note."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import anthropic

from clinical_pipeline.extraction.models import ClinicalExtraction
from clinical_pipeline.extraction.prompts import (
    EXTRACTION_TOOL,
    SYSTEM_PROMPT,
    build_extraction_prompt,
    build_rag_extraction_prompt,
)

if TYPE_CHECKING:
    from clinical_pipeline.extraction.cost_tracker import CostTracker
    from clinical_pipeline.extraction.icd10_rag import ICD10Retriever

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4096


class ExtractionError(Exception):
    """Raised when extraction fails after retries."""


class ClinicalExtractor:
    """Extracts structured clinical data from a single note using Claude."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        cost_tracker: CostTracker | None = None,
        max_retries: int = 3,
        retriever: ICD10Retriever | None = None,
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.cost_tracker = cost_tracker
        self.max_retries = max_retries
        self.retriever = retriever

    def extract(self, note_id: str, transcription: str) -> ClinicalExtraction:
        """Extract clinical data from a single note.

        Uses Claude's tool_use to guarantee structured JSON output.
        Retries on transient API errors.
        """
        if self.retriever is not None:
            relevant_codes = self.retriever.retrieve(transcription, top_k=50)
            user_message = build_rag_extraction_prompt(transcription, relevant_codes)
        else:
            user_message = build_extraction_prompt(transcription)
        messages = [
            {"role": "user", "content": user_message},
        ]

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._call_api(messages)
                extraction = self._parse_response(response, note_id)

                if self.cost_tracker is not None:
                    self.cost_tracker.record(
                        model=self.model,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                    )

                return extraction

            except anthropic.RateLimitError:
                logger.warning("Rate limited on attempt %d/%d", attempt, self.max_retries)
                last_error = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    logger.warning(
                        "Server error %d on attempt %d/%d", e.status_code, attempt, self.max_retries
                    )
                    last_error = e
                else:
                    raise ExtractionError(f"API error for note {note_id}: {e}") from e
            except Exception as e:
                raise ExtractionError(f"Unexpected error for note {note_id}: {e}") from e

        raise ExtractionError(
            f"Failed to extract note {note_id} after {self.max_retries} attempts: {last_error}"
        )

    def _call_api(self, messages: list[dict[str, str]]) -> anthropic.types.Message:
        """Make the API call to Claude with tool_use for structured output."""
        return self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "extract_clinical_data"},
        )

    def _parse_response(
        self, response: anthropic.types.Message, note_id: str
    ) -> ClinicalExtraction:
        """Extract the tool_use result and validate with Pydantic."""
        for block in response.content:
            if block.type == "tool_use":
                data = block.input
                data["note_id"] = note_id
                return ClinicalExtraction.model_validate(data)

        raise ExtractionError("No tool_use block found in response")
