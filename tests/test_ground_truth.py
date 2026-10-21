"""Tests for the MIMIC ground-truth loader and the MIMIC source settings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from clinical_pipeline.config import Settings
from clinical_pipeline.evaluation import ground_truth
from clinical_pipeline.evaluation.ground_truth import load_ground_truth, select_admissions


def _write(hosp: Path, name: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(hosp / f"{name}.csv.gz", index=False)


@pytest.fixture
def hosp(tmp_path: Path) -> Path:
    hosp = tmp_path / "hosp"
    hosp.mkdir()

    dx = []
    # hadm 100: 6 ICD-10 dx (simple tier); hadm 200: 12 (moderate); hadm 300: 22 (complex)
    for hadm_id, n in ((100, 6), (200, 12), (300, 22)):
        for seq in range(1, n + 1):
            dx.append(
                {"subject_id": 1, "hadm_id": hadm_id, "seq_num": seq,
                 "icd_code": f"A{seq:02d}", "icd_version": 10}
            )
    # ICD-9 rows must be ignored everywhere
    dx.append({"subject_id": 1, "hadm_id": 100, "seq_num": 7, "icd_code": "4019", "icd_version": 9})
    dx.append({"subject_id": 2, "hadm_id": 400, "seq_num": 1, "icd_code": "4019", "icd_version": 9})
    _write(hosp, "diagnoses_icd", dx)

    d_dx = [
        {"icd_code": f"A{seq:02d}", "icd_version": 10, "long_title": f"Dx title {seq}"}
        for seq in range(1, 23)
    ]
    d_dx.append({"icd_code": "4019", "icd_version": 9, "long_title": "Hypertension"})
    _write(hosp, "d_icd_diagnoses", d_dx)

    _write(hosp, "procedures_icd", [
        {"subject_id": 1, "hadm_id": 100, "seq_num": 1, "icd_code": "0BH17EZ", "icd_version": 10},
        {"subject_id": 1, "hadm_id": 100, "seq_num": 2, "icd_code": "3893", "icd_version": 9},
        {"subject_id": 1, "hadm_id": 200, "seq_num": 1, "icd_code": "NOPE000", "icd_version": 10},
    ])
    _write(hosp, "d_icd_procedures", [
        {"icd_code": "0BH17EZ", "icd_version": 10, "long_title": "Insertion of airway"},
        {"icd_code": "3893", "icd_version": 9, "long_title": "Venous catheterization"},
    ])

    _write(hosp, "prescriptions", [
        {"subject_id": 1, "hadm_id": 100, "drug": "Aspirin", "dose_val_rx": "81",
         "dose_unit_rx": "mg", "route": "PO"},
        # duplicate of the row above (same hadm_id, drug, dose_val_rx, route)
        {"subject_id": 1, "hadm_id": 100, "drug": "Aspirin", "dose_val_rx": "81",
         "dose_unit_rx": "mg", "route": "PO"},
        {"subject_id": 1, "hadm_id": 100, "drug": "Heparin", "dose_val_rx": None,
         "dose_unit_rx": None, "route": None},
        # admission outside the selected set
        {"subject_id": 1, "hadm_id": 300, "drug": "Insulin", "dose_val_rx": "5",
         "dose_unit_rx": "units", "route": "SC"},
    ])
    return hosp


def test_load_ground_truth(hosp: Path) -> None:
    result = load_ground_truth(hosp, [100, 200])

    assert set(result) == {100, 200}
    gt = result[100]
    assert [d.icd_code for d in gt.diagnoses] == [f"A{i:02d}" for i in range(1, 7)]
    assert gt.diagnoses[0].description == "Dx title 1"
    assert [p.icd_code for p in gt.procedures] == ["0BH17EZ"]
    assert gt.procedures[0].description == "Insertion of airway"
    # unknown code keeps an empty description
    assert result[200].procedures[0].description == ""


def test_icd9_rows_are_filtered(hosp: Path) -> None:
    result = load_ground_truth(hosp, [100, 400])

    assert all(d.icd_code != "4019" for d in result[100].diagnoses)
    assert result[400].diagnoses == []


def test_prescriptions_dedup_and_filter(hosp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A tiny chunk size forces the duplicate pair across a chunk boundary
    monkeypatch.setattr(ground_truth, "RX_CHUNK_ROWS", 1)
    result = load_ground_truth(hosp, [100])

    meds = result[100].medications
    assert [(m.drug, m.dose, m.route) for m in meds] == [
        ("Aspirin", "81.0 mg", "PO"),
        ("Heparin", None, None),
    ]


def test_select_admissions_is_deterministic(hosp: Path) -> None:
    first = select_admissions(hosp, n=3, seed=42)
    second = select_admissions(hosp, n=3, seed=42)

    assert first == second == [100, 200, 300]
    # hadm 400 has only ICD-9 rows and never qualifies
    assert 400 not in select_admissions(hosp, n=10, seed=42)


def _settings(monkeypatch: pytest.MonkeyPatch, source: str | None) -> Settings:
    monkeypatch.chdir(Path(__file__).parent)  # no .env picked up
    monkeypatch.setenv("DATA_DIR", "data")
    monkeypatch.setenv("MIMIC_DEMO_ROOT", "/demo")
    monkeypatch.setenv("MIMIC_FULL_ROOT", "/full")
    if source is None:
        monkeypatch.delenv("MIMIC_SOURCE", raising=False)
    else:
        monkeypatch.setenv("MIMIC_SOURCE", source)
    return Settings()


def test_settings_demo_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, None)

    assert settings.mimic_source == "demo"
    assert settings.mimic_root == Path("/demo")
    assert settings.mimic_hosp_dir == Path("/demo/hosp")
    assert settings.evaluation_dir == Path("data/evaluation")
    assert settings.optimization_dir == Path("data/optimization")


def test_settings_full_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, "full")

    assert settings.mimic_root == Path("/full")
    assert settings.mimic_hosp_dir == Path("/full/hosp")
    assert settings.evaluation_dir == Path("data/full/evaluation")
    assert settings.optimization_dir == Path("data/full/optimization")
