"""Tests for the OpenAI-compatible extraction client, with a mocked HTTP client (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from clinical_pipeline.evaluation import run_eval, synthetic_notes
from clinical_pipeline.evaluation.ground_truth import GroundTruth, GroundTruthDiagnosis
from clinical_pipeline.extraction.cost_tracker import CostTracker
from clinical_pipeline.extraction.extractor import DEFAULT_MODEL, ExtractionError
from clinical_pipeline.extraction.openai_compat import OpenAICompatExtractor
from clinical_pipeline.extraction.prompts import EXTRACTION_TOOL, SYSTEM_PROMPT

BASE_URL = "http://127.0.0.1:11434/v1"
OTHER_URL = "http://127.0.0.1:8000/v1"
PAYLOAD = {
    "diagnoses": [
        {"name": "Hypertension", "confidence": "high", "evidence_span": "h/o HTN"},
    ],
    "procedures": [],
    "medications": [],
}


def _response(message: dict, status: int = 200) -> httpx.Response:
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    body = {"choices": [{"message": message}], "usage": usage}
    return httpx.Response(status, json=body, request=httpx.Request("POST", BASE_URL))


def _tool_response(arguments: str) -> httpx.Response:
    call = {"function": {"name": EXTRACTION_TOOL["name"], "arguments": arguments}}
    return _response({"content": None, "tool_calls": [call]})


def _extractor(*responses: httpx.Response, **kwargs) -> tuple[OpenAICompatExtractor, MagicMock]:
    client = MagicMock()
    client.post.side_effect = list(responses)
    extractor = OpenAICompatExtractor(
        base_url=BASE_URL + "/", model="local", client=client, **kwargs
    )
    return extractor, client


def test_request_forces_the_extraction_tool_with_the_shared_prompt_and_schema():
    extractor, client = _extractor(_tool_response(json.dumps(PAYLOAD)))
    extractor.extract("1", "note text")

    url = client.post.call_args.args[0]
    body = client.post.call_args.kwargs["json"]
    assert url == f"{BASE_URL}/chat/completions"
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert body["messages"][1]["role"] == "user" and "note text" in body["messages"][1]["content"]
    assert body["tools"][0]["function"]["parameters"] == EXTRACTION_TOOL["input_schema"]
    assert body["tool_choice"]["function"]["name"] == EXTRACTION_TOOL["name"]


def test_tool_call_is_validated_into_clinical_extraction_and_cost_recorded():
    tracker = CostTracker()
    extractor, _ = _extractor(_tool_response(json.dumps(PAYLOAD)), cost_tracker=tracker)

    extraction = extractor.extract("7", "note")

    assert extraction.note_id == "7"
    assert extraction.diagnoses[0].name == "Hypertension"
    assert (tracker.total_input_tokens, tracker.total_output_tokens) == (10, 5)
    assert not extractor.failure_counts


def test_fenced_json_content_is_accepted_when_no_tool_call():
    content = "```json\n" + json.dumps(PAYLOAD) + "\n```"
    extractor, _ = _extractor(_response({"content": content}))

    assert extractor.extract("1", "note").diagnoses[0].name == "Hypertension"


def test_server_rejecting_tools_falls_back_to_json_schema():
    rejected = httpx.Response(
        400, json={"error": "tools unsupported"}, request=httpx.Request("POST", BASE_URL)
    )
    extractor, client = _extractor(rejected, _response({"content": json.dumps(PAYLOAD)}))

    extraction = extractor.extract("1", "note")

    assert extraction.diagnoses[0].name == "Hypertension"
    first, second = (c.kwargs["json"] for c in client.post.call_args_list)
    assert "tools" in first and "tools" not in second
    assert second["response_format"]["json_schema"]["schema"] == EXTRACTION_TOOL["input_schema"]
    assert extractor.use_tools is False


def test_failures_are_counted_by_kind_and_retried():
    bad_schema = json.dumps({"diagnoses": [{"name": "x"}]})
    extractor, client = _extractor(
        _tool_response("{not json"),
        _tool_response(bad_schema),
        _tool_response(json.dumps(PAYLOAD)),
    )

    extraction = extractor.extract("1", "note")

    assert extraction.diagnoses[0].name == "Hypertension"
    assert client.post.call_count == 3
    assert dict(extractor.failure_counts) == {"invalid_json": 1, "schema_error": 1}


def test_extraction_error_after_max_retries():
    extractor, _ = _extractor(*[_tool_response("nope")] * 2, max_retries=2)

    with pytest.raises(ExtractionError):
        extractor.extract("1", "note")
    assert extractor.failure_counts["invalid_json"] == 2


def test_empty_message_counts_as_no_output():
    extractor, _ = _extractor(_response({"content": ""}), max_retries=1)

    with pytest.raises(ExtractionError):
        extractor.extract("1", "note")
    assert extractor.failure_counts["no_output"] == 1


def test_http_errors_are_counted_and_retried():
    server_error = httpx.Response(500, request=httpx.Request("POST", BASE_URL))
    extractor, _ = _extractor(server_error, _tool_response(json.dumps(PAYLOAD)))

    assert extractor.extract("1", "note").note_id == "1"
    assert extractor.failure_counts["http_error"] == 1


def test_provider_and_base_url_change_the_cache_key():
    base = run_eval._config_key("local", SYSTEM_PROMPT, False, "openai-compat", BASE_URL)
    assert run_eval._config_key("local", SYSTEM_PROMPT, False, "openai-compat", OTHER_URL) != base
    slash = run_eval._config_key("local", SYSTEM_PROMPT, False, "openai-compat", BASE_URL + "/")
    assert slash == base
    assert run_eval._config_key("local", SYSTEM_PROMPT, False) != base


def test_anthropic_key_is_unchanged_by_the_provider_arguments():
    key = run_eval._config_key(DEFAULT_MODEL, SYSTEM_PROMPT, False)
    assert key == run_eval._config_key(DEFAULT_MODEL, SYSTEM_PROMPT, False, "anthropic", "")


def _setup(tmp_path: Path) -> SimpleNamespace:
    eval_dir = tmp_path / "eval"
    (eval_dir / "synthetic_notes").mkdir(parents=True)
    (eval_dir / "synthetic_notes" / "1.txt").write_text("note 1", encoding="utf-8")
    (eval_dir / "selected_hadm_ids.json").write_text(json.dumps([1]))
    return SimpleNamespace(
        mimic_hosp_dir=tmp_path,
        evaluation_dir=eval_dir,
        anthropic_api_key="test",
        openai_compat_api_key="local-key",
        reference_dir=tmp_path,
    )


def _run_local(settings, http_client: MagicMock, **kwargs) -> None:
    gt = GroundTruth(
        hadm_id=1,
        diagnoses=[GroundTruthDiagnosis(icd_code="I10", description="Hypertension", seq_num=1)],
        procedures=[],
        medications=[],
    )
    with (
        patch.object(run_eval, "load_ground_truth", return_value={1: gt}),
        patch.object(synthetic_notes.anthropic, "Anthropic", return_value=MagicMock()),
        patch("clinical_pipeline.extraction.openai_compat.httpx.Client", return_value=http_client),
    ):
        run_eval.run_evaluation(
            settings=settings, provider="openai-compat", base_url=BASE_URL, model="local", **kwargs
        )


def test_local_results_get_their_own_cache_dir_and_config(tmp_path: Path):
    settings = _setup(tmp_path)
    http_client = MagicMock()
    http_client.post.return_value = _tool_response(json.dumps(PAYLOAD))

    _run_local(settings, http_client)
    _run_local(settings, http_client)

    assert http_client.post.call_count == 1
    extractions = settings.evaluation_dir / "extractions"
    assert not (extractions / "1.json").exists()
    key_dir = next(p for p in extractions.iterdir() if p.is_dir())
    config = json.loads((key_dir / "config.json").read_text())
    assert config["provider"] == "openai-compat" and config["base_url"] == BASE_URL
    assert (key_dir / "1.json").exists()


def test_openai_compat_requires_base_url_and_model_and_no_batch(tmp_path: Path):
    settings = _setup(tmp_path)
    with pytest.raises(ValueError):
        run_eval.run_evaluation(settings=settings, provider="openai-compat", model="local")
    with pytest.raises(ValueError):
        run_eval.run_evaluation(
            settings=settings,
            provider="openai-compat",
            base_url=BASE_URL,
            model="local",
            batch=True,
        )
