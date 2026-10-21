"""OpenAICompatExtractor: extraction against any OpenAI-compatible /v1/chat/completions server.

Targets local servers (Ollama, omlx) so credentialed note text never leaves the machine.
Uses the same system prompt and EXTRACTION_TOOL schema as the Anthropic path.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import TYPE_CHECKING

import httpx
from pydantic import ValidationError

from clinical_pipeline.extraction.extractor import (
    MAX_TOKENS,
    ClinicalExtractor,
    ExtractionError,
)
from clinical_pipeline.extraction.models import ClinicalExtraction
from clinical_pipeline.extraction.prompts import EXTRACTION_TOOL, SYSTEM_PROMPT

if TYPE_CHECKING:
    from clinical_pipeline.extraction.cost_tracker import CostTracker
    from clinical_pipeline.extraction.icd10_rag import ICD10Retriever

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 900.0
FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class OpenAICompatExtractor(ClinicalExtractor):
    """Extracts structured clinical data through an OpenAI-compatible chat server.

    Tries function calling first. If the server rejects ``tools`` (HTTP 400) or the model
    answers in plain text, falls back to JSON-schema structured output and then to parsing
    the message content as JSON. Failures are counted by kind in ``failure_counts``.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        cost_tracker: CostTracker | None = None,
        max_retries: int = 3,
        retriever: ICD10Retriever | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cost_tracker = cost_tracker
        self.max_retries = max_retries
        self.retriever = retriever
        self.system_prompt = system_prompt
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(headers=headers, timeout=REQUEST_TIMEOUT)
        self.use_tools = True
        self.failure_counts: Counter[str] = Counter()

    def extract(self, note_id: str, transcription: str) -> ClinicalExtraction:
        """Extract clinical data from a single note, retrying invalid or failed responses."""
        messages = self.build_messages(transcription)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._call_api(messages)
                extraction = self._parse_response(response, note_id)
            except ExtractionError as e:
                logger.warning("Bad response on attempt %d/%d: %s", attempt, self.max_retries, e)
                last_error = e
                continue
            except httpx.HTTPError as e:
                self.failure_counts["http_error"] += 1
                logger.warning("HTTP error on attempt %d/%d: %s", attempt, self.max_retries, e)
                last_error = e
                continue

            if self.cost_tracker is not None:
                usage = response.get("usage") or {}
                self.cost_tracker.record(
                    model=self.model,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                )
            return extraction

        raise ExtractionError(
            f"Failed to extract note {note_id} after {self.max_retries} attempts: {last_error}"
        )

    def _request_params(self, messages: list[dict[str, str]]) -> dict:
        """Chat-completions body with the extraction tool forced (or JSON-schema output)."""
        params: dict = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": self.system_prompt}, *messages],
            "stream": False,
        }
        if self.use_tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": EXTRACTION_TOOL["name"],
                        "description": EXTRACTION_TOOL["description"],
                        "parameters": EXTRACTION_TOOL["input_schema"],
                    },
                }
            ]
            params["tool_choice"] = {
                "type": "function",
                "function": {"name": EXTRACTION_TOOL["name"]},
            }
        else:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": EXTRACTION_TOOL["name"],
                    "schema": EXTRACTION_TOOL["input_schema"],
                },
            }
        return params

    def _call_api(self, messages: list[dict[str, str]]) -> dict:
        """POST to /chat/completions, switching to JSON-schema output if tools are rejected."""
        url = f"{self.base_url}/chat/completions"
        response = self.client.post(url, json=self._request_params(messages))
        if response.status_code == 400 and self.use_tools:
            logger.warning("Server rejected tools for %s, using JSON-schema output", self.model)
            self.use_tools = False
            response = self.client.post(url, json=self._request_params(messages))
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: dict, note_id: str) -> ClinicalExtraction:
        """Pull the tool-call arguments (or JSON content) and validate with Pydantic."""
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            self.failure_counts["bad_response"] += 1
            raise ExtractionError(f"Malformed chat completion: {e!r}") from e

        raw = None
        for call in message.get("tool_calls") or []:
            if call.get("function", {}).get("name") == EXTRACTION_TOOL["name"]:
                raw = call["function"].get("arguments")
                break
        if raw is None:
            raw = message.get("content")
        if not raw:
            self.failure_counts["no_output"] += 1
            raise ExtractionError("No tool call or content in response")

        try:
            data = raw if isinstance(raw, dict) else json.loads(_strip_fence(raw))
        except json.JSONDecodeError as e:
            self.failure_counts["invalid_json"] += 1
            raise ExtractionError(f"Invalid JSON: {e}") from e
        if not isinstance(data, dict):
            self.failure_counts["invalid_json"] += 1
            raise ExtractionError("JSON output is not an object")

        data["note_id"] = note_id
        try:
            return ClinicalExtraction.model_validate(data)
        except ValidationError as e:
            self.failure_counts["schema_error"] += 1
            raise ExtractionError(f"Schema validation failed: {e.error_count()} errors") from e


def _strip_fence(text: str) -> str:
    """Remove a Markdown code fence some models wrap around JSON output."""
    text = text.strip()
    match = FENCE_RE.match(text)
    return match.group(1) if match else text
