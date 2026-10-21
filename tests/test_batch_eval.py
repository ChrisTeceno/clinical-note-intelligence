"""Tests for the Batch API eval path, with a mocked Anthropic client (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from clinical_pipeline.evaluation import batch_api, run_eval, synthetic_notes
from clinical_pipeline.evaluation.batch_api import BatchError, estimate_batch_cost, run_batch
from clinical_pipeline.evaluation.ground_truth import GroundTruth, GroundTruthDiagnosis
from clinical_pipeline.extraction.cost_tracker import MODEL_PRICING, CostTracker
from clinical_pipeline.extraction.extractor import ClinicalExtractor

HAIKU = "claude-haiku-4-5-20251001"


def _gt(hadm_id: int) -> GroundTruth:
    return GroundTruth(
        hadm_id=hadm_id,
        diagnoses=[GroundTruthDiagnosis(icd_code="I10", description="Hypertension", seq_num=1)],
        procedures=[],
        medications=[],
    )


def _message(content, input_tokens=100, output_tokens=50):
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _text_message(text: str):
    return _message([SimpleNamespace(type="text", text=text)])


def _tool_message(note_id_ignored="x"):
    block = SimpleNamespace(
        type="tool_use",
        input={"diagnoses": [], "procedures": [], "medications": []},
    )
    return _message([block])


def _succeeded(custom_id: str, message):
    return SimpleNamespace(
        custom_id=custom_id, result=SimpleNamespace(type="succeeded", message=message)
    )


def _errored(custom_id: str):
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="errored", error={"type": "error"}),
    )


def _client(results):
    """Mock client whose batch is already ended and yields ``results``."""
    client = MagicMock()
    counts = SimpleNamespace(processing=0, succeeded=len(results), errored=0)
    client.messages.batches.create.return_value = SimpleNamespace(
        id="msgbatch_1", processing_status="ended", request_counts=counts
    )
    client.messages.batches.results.return_value = iter(results)
    return client


def test_run_batch_builds_requests_and_maps_results_by_custom_id():
    client = _client([_succeeded("2", _text_message("b")), _succeeded("1", _text_message("a"))])
    requests = {"1": {"model": HAIKU}, "2": {"model": HAIKU}}

    messages = run_batch(client, requests, label="test")

    sent = client.messages.batches.create.call_args.kwargs["requests"]
    assert sent == [
        {"custom_id": "1", "params": {"model": HAIKU}},
        {"custom_id": "2", "params": {"model": HAIKU}},
    ]
    assert messages["1"].content[0].text == "a"
    assert messages["2"].content[0].text == "b"


def test_run_batch_skips_errored_items():
    client = _client([_succeeded("1", _text_message("a")), _errored("2")])

    messages = run_batch(client, {"1": {}, "2": {}}, label="test")

    assert list(messages) == ["1"]


def test_run_batch_polls_until_ended():
    client = _client([_succeeded("1", _text_message("a"))])
    counts = SimpleNamespace(processing=1, succeeded=0, errored=0)
    client.messages.batches.create.return_value = SimpleNamespace(
        id="msgbatch_1", processing_status="in_progress", request_counts=counts
    )
    client.messages.batches.retrieve.return_value = SimpleNamespace(
        id="msgbatch_1", processing_status="ended", request_counts=counts
    )

    with patch.object(batch_api.time, "sleep") as sleep:
        messages = run_batch(client, {"1": {}}, label="test", poll_initial=1.0)

    sleep.assert_called_once_with(1.0)
    assert "1" in messages


def test_run_batch_times_out():
    client = _client([])
    counts = SimpleNamespace(processing=1, succeeded=0, errored=0)
    client.messages.batches.create.return_value = SimpleNamespace(
        id="msgbatch_1", processing_status="in_progress", request_counts=counts
    )

    with pytest.raises(BatchError):
        run_batch(client, {"1": {}}, label="test", timeout_seconds=0, poll_initial=1.0)


def test_batch_cost_applies_discount_and_haiku_price():
    assert MODEL_PRICING[HAIKU] == {"input": 1.00, "output": 5.00}
    tracker = CostTracker(is_batch=True)
    tracker.record(HAIKU, input_tokens=1_000_000, output_tokens=1_000_000)
    assert tracker.estimated_cost() == pytest.approx(3.00)  # (1 + 5) * 0.5

    assert estimate_batch_cost(HAIKU, 10, (1_000_000, 0)) == pytest.approx(5.00)


def test_prompt_override_reaches_request_params():
    extractor = ClinicalExtractor(api_key="test", system_prompt="custom prompt")
    params = extractor.build_request_params("patient note")
    assert params["system"] == "custom prompt"
    assert params["tool_choice"]["name"] == "extract_clinical_data"
    assert "patient note" in params["messages"][0]["content"]


def test_note_generation_batch_skips_cached_and_writes_cache(tmp_path: Path):
    notes_dir = tmp_path / "synthetic_notes"
    notes_dir.mkdir()
    (notes_dir / "1.txt").write_text("cached note", encoding="utf-8")
    client = _client([_succeeded("2", _text_message("new note"))])
    tracker = CostTracker(is_batch=True)

    with patch.object(synthetic_notes.anthropic, "Anthropic", return_value=client):
        notes = synthetic_notes.generate_batch_api(
            {1: _gt(1), 2: _gt(2)}, api_key="test", cache_dir=tmp_path, cost_tracker=tracker
        )

    sent = client.messages.batches.create.call_args.kwargs["requests"]
    assert [r["custom_id"] for r in sent] == ["2"]
    assert notes == {1: "cached note", 2: "new note"}
    assert (notes_dir / "2.txt").read_text(encoding="utf-8") == "new note"
    assert tracker.total_requests == 1


def test_budget_check_aborts_over_max_cost(tmp_path: Path):
    hadm_ids = list(range(1000))
    with pytest.raises(BatchError):
        run_eval._check_batch_budget(tmp_path, hadm_ids, HAIKU, max_cost=0.01)
    run_eval._check_batch_budget(tmp_path, hadm_ids[:30], HAIKU, max_cost=3.00)


def test_run_evaluation_batch_end_to_end(tmp_path: Path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("optimized prompt", encoding="utf-8")
    settings = SimpleNamespace(
        mimic_hosp_dir=tmp_path,
        evaluation_dir=tmp_path / "eval",
        anthropic_api_key="test",
        reference_dir=tmp_path,
    )
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "selected_hadm_ids.json").write_text(json.dumps([1, 2]))

    note_client = _client(
        [_succeeded("1", _text_message("note 1")), _succeeded("2", _text_message("note 2"))]
    )
    extraction_client = _client([_succeeded("1", _tool_message()), _errored("2")])
    created = []

    def fake_anthropic(api_key):
        client = note_client if not created else extraction_client
        created.append(client)
        return client

    with (
        patch.object(run_eval, "load_ground_truth", return_value={1: _gt(1), 2: _gt(2)}),
        patch.object(synthetic_notes.anthropic, "Anthropic", side_effect=fake_anthropic),
        patch(
            "clinical_pipeline.extraction.extractor.anthropic.Anthropic",
            side_effect=fake_anthropic,
        ),
    ):
        summary = run_eval.run_evaluation(
            settings=settings, batch=True, prompt_file=prompt_file, description="poc"
        )

    sent = extraction_client.messages.batches.create.call_args.kwargs["requests"]
    assert {r["custom_id"] for r in sent} == {"1", "2"}
    assert all(r["params"]["system"] == "optimized prompt" for r in sent)
    assert summary.failed_extractions == [2]
    assert summary.n_admissions == 1
    history = json.loads((tmp_path / "eval" / "run_history.json").read_text())
    assert history[-1]["description"] == "poc [batch] [prompt: prompt.txt]"
    report = json.loads((tmp_path / "eval" / "eval_cost_report.json").read_text())
    assert report["summary"]["is_batch"] is True
    assert report["summary"]["total_requests"] == 3
