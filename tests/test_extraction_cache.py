"""Tests for the config-keyed extraction cache, with a mocked Anthropic client (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from clinical_pipeline.evaluation import run_eval, synthetic_notes
from clinical_pipeline.evaluation.ground_truth import GroundTruth, GroundTruthDiagnosis
from clinical_pipeline.extraction.extractor import DEFAULT_MODEL
from clinical_pipeline.extraction.prompts import SYSTEM_PROMPT

OTHER_MODEL = "claude-sonnet-4-5-20250929"


def _gt(hadm_id: int) -> GroundTruth:
    return GroundTruth(
        hadm_id=hadm_id,
        diagnoses=[GroundTruthDiagnosis(icd_code="I10", description="Hypertension", seq_num=1)],
        procedures=[],
        medications=[],
    )


def _tool_message():
    block = SimpleNamespace(
        type="tool_use",
        input={"diagnoses": [], "procedures": [], "medications": []},
    )
    return SimpleNamespace(
        content=[block], usage=SimpleNamespace(input_tokens=100, output_tokens=50)
    )


def _setup(tmp_path: Path) -> SimpleNamespace:
    eval_dir = tmp_path / "eval"
    (eval_dir / "synthetic_notes").mkdir(parents=True)
    (eval_dir / "synthetic_notes" / "1.txt").write_text("note 1", encoding="utf-8")
    (eval_dir / "selected_hadm_ids.json").write_text(json.dumps([1]))
    return SimpleNamespace(
        mimic_hosp_dir=tmp_path,
        evaluation_dir=eval_dir,
        anthropic_api_key="test",
        reference_dir=tmp_path,
    )


def _run(settings, **kwargs) -> MagicMock:
    """Run an evaluation of admission 1 and return the mock client that served extraction."""
    client = MagicMock()
    client.messages.create.return_value = _tool_message()
    with (
        patch.object(run_eval, "load_ground_truth", return_value={1: _gt(1)}),
        patch.object(synthetic_notes.anthropic, "Anthropic", return_value=MagicMock()),
        patch("clinical_pipeline.extraction.extractor.anthropic.Anthropic", return_value=client),
    ):
        run_eval.run_evaluation(settings=settings, **kwargs)
    return client


def _prompt_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "prompt.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_config_uses_legacy_path(tmp_path: Path):
    settings = _setup(tmp_path)
    client = _run(settings)

    assert client.messages.create.call_count == 1
    extractions = settings.evaluation_dir / "extractions"
    assert (extractions / "1.json").exists()
    assert [p for p in extractions.iterdir() if p.is_dir()] == []


def test_different_prompt_or_model_makes_new_requests_and_dir(tmp_path: Path):
    settings = _setup(tmp_path)
    _run(settings)
    extractions = settings.evaluation_dir / "extractions"

    prompt_client = _run(settings, prompt_file=_prompt_file(tmp_path, "optimized prompt"))
    model_client = _run(settings, model=OTHER_MODEL)

    assert prompt_client.messages.create.call_count == 1
    assert model_client.messages.create.call_count == 1
    keyed = [p for p in extractions.iterdir() if p.is_dir()]
    assert len(keyed) == 2
    assert all(len(p.name) == 12 and (p / "1.json").exists() for p in keyed)


def test_same_non_default_config_hits_its_cache(tmp_path: Path):
    settings = _setup(tmp_path)
    prompt_file = _prompt_file(tmp_path, "optimized prompt")

    first = _run(settings, prompt_file=prompt_file)
    second = _run(settings, prompt_file=prompt_file)

    assert first.messages.create.call_count == 1
    assert second.messages.create.call_count == 0
    key_dir = next(p for p in (settings.evaluation_dir / "extractions").iterdir() if p.is_dir())
    config = json.loads((key_dir / "config.json").read_text())
    assert config["model"] == DEFAULT_MODEL
    assert config["use_rag"] is False
    assert config["prompt_sha"] == run_eval._sha256("optimized prompt")[:12]


def test_use_rag_changes_the_key():
    base = run_eval._config_key(DEFAULT_MODEL, SYSTEM_PROMPT, False)
    assert run_eval._config_key(DEFAULT_MODEL, SYSTEM_PROMPT, True) != base
    assert run_eval._config_key(OTHER_MODEL, SYSTEM_PROMPT, False) != base
    assert run_eval._config_key(DEFAULT_MODEL, SYSTEM_PROMPT + " ", False) != base


def test_budget_check_counts_per_config(tmp_path: Path, caplog):
    legacy = tmp_path / "extractions"
    legacy.mkdir()
    for hadm_id in (1, 2):
        (legacy / f"{hadm_id}.json").write_text("{}", encoding="utf-8")

    def uncached(**kwargs) -> int:
        caplog.clear()
        with caplog.at_level("INFO", logger=run_eval.logger.name):
            run_eval._check_batch_budget(tmp_path, [1, 2], DEFAULT_MODEL, 100.0, **kwargs)
        return int(caplog.text.split("extractions $")[0].rsplit(", ", 1)[1].split()[0])

    assert uncached() == 0
    assert uncached(system_prompt="optimized prompt") == 2
    assert uncached(use_rag=True) == 2
