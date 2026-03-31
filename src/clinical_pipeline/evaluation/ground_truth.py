"""Load and structure MIMIC-IV demo ground truth for evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GroundTruthDiagnosis:
    icd_code: str
    description: str
    seq_num: int


@dataclass(frozen=True, slots=True)
class GroundTruthProcedure:
    icd_code: str
    description: str


@dataclass(frozen=True, slots=True)
class GroundTruthMedication:
    drug: str
    dose: str | None
    route: str | None


@dataclass(slots=True)
class GroundTruth:
    hadm_id: int
    diagnoses: list[GroundTruthDiagnosis] = field(default_factory=list)
    procedures: list[GroundTruthProcedure] = field(default_factory=list)
    medications: list[GroundTruthMedication] = field(default_factory=list)


def load_ground_truth(mimic_path: Path, hadm_ids: list[int]) -> dict[int, GroundTruth]:
    """Load ground truth from MIMIC demo for selected admissions.

    Parameters
    ----------
    mimic_path:
        Path to the ``hosp/`` directory of the MIMIC-IV demo.
    hadm_ids:
        Admission IDs to load (must be ICD-10 admissions).

    Returns
    -------
    dict mapping hadm_id -> GroundTruth
    """
    hadm_set = set(hadm_ids)

    # --- Diagnoses ---
    dx_df = pd.read_csv(mimic_path / "diagnoses_icd.csv.gz")
    dx_df = dx_df[(dx_df["icd_version"] == 10) & (dx_df["hadm_id"].isin(hadm_set))]

    d_dx = pd.read_csv(mimic_path / "d_icd_diagnoses.csv.gz")
    d_dx = d_dx[d_dx["icd_version"] == 10].rename(columns={"long_title": "description"})
    dx_df = dx_df.merge(d_dx[["icd_code", "description"]], on="icd_code", how="left")
    dx_df["description"] = dx_df["description"].fillna("")

    # --- Procedures ---
    proc_df = pd.read_csv(mimic_path / "procedures_icd.csv.gz")
    proc_df = proc_df[(proc_df["icd_version"] == 10) & (proc_df["hadm_id"].isin(hadm_set))]

    d_proc = pd.read_csv(mimic_path / "d_icd_procedures.csv.gz")
    d_proc = d_proc[d_proc["icd_version"] == 10].rename(columns={"long_title": "description"})
    proc_df = proc_df.merge(d_proc[["icd_code", "description"]], on="icd_code", how="left")
    proc_df["description"] = proc_df["description"].fillna("")

    # --- Prescriptions ---
    rx_df = pd.read_csv(mimic_path / "prescriptions.csv.gz")
    rx_df = rx_df[rx_df["hadm_id"].isin(hadm_set)]
    # Deduplicate: keep unique drug/dose/route per admission
    rx_df = rx_df.drop_duplicates(subset=["hadm_id", "drug", "dose_val_rx", "route"])

    # --- Assemble GroundTruth per admission ---
    result: dict[int, GroundTruth] = {}
    for hadm_id in hadm_ids:
        gt = GroundTruth(hadm_id=hadm_id)

        hadm_dx = dx_df[dx_df["hadm_id"] == hadm_id].sort_values("seq_num")
        for _, row in hadm_dx.iterrows():
            gt.diagnoses.append(
                GroundTruthDiagnosis(
                    icd_code=str(row["icd_code"]).strip(),
                    description=row["description"],
                    seq_num=int(row["seq_num"]),
                )
            )

        hadm_proc = proc_df[proc_df["hadm_id"] == hadm_id]
        for _, row in hadm_proc.iterrows():
            gt.procedures.append(
                GroundTruthProcedure(
                    icd_code=str(row["icd_code"]).strip(),
                    description=row["description"],
                )
            )

        hadm_rx = rx_df[rx_df["hadm_id"] == hadm_id]
        for _, row in hadm_rx.iterrows():
            dose_val = row.get("dose_val_rx")
            dose_unit = row.get("dose_unit_rx")
            dose = None
            if pd.notna(dose_val) and str(dose_val).strip():
                dose = str(dose_val).strip()
                if pd.notna(dose_unit) and str(dose_unit).strip():
                    dose = f"{dose} {str(dose_unit).strip()}"
            route = str(row["route"]).strip() if pd.notna(row.get("route")) else None

            gt.medications.append(
                GroundTruthMedication(
                    drug=str(row["drug"]).strip(),
                    dose=dose,
                    route=route,
                )
            )

        result[hadm_id] = gt

    logger.info(
        "Loaded ground truth for %d admissions (%d dx, %d proc, %d rx)",
        len(result),
        sum(len(gt.diagnoses) for gt in result.values()),
        sum(len(gt.procedures) for gt in result.values()),
        sum(len(gt.medications) for gt in result.values()),
    )
    return result


def select_admissions(mimic_path: Path, n: int = 30, seed: int = 42) -> list[int]:
    """Select a diverse set of ICD-10 admissions from the MIMIC demo.

    Picks admissions with a mix of diagnosis counts: some simple (5-10 dx),
    some moderate (11-20), and some complex (20+).
    """
    dx_df = pd.read_csv(mimic_path / "diagnoses_icd.csv.gz")
    dx10 = dx_df[dx_df["icd_version"] == 10]
    counts = dx10.groupby("hadm_id").size().reset_index(name="n_dx")

    # Bin into complexity tiers
    simple = counts[(counts["n_dx"] >= 5) & (counts["n_dx"] <= 10)]
    moderate = counts[(counts["n_dx"] >= 11) & (counts["n_dx"] <= 20)]
    complex_ = counts[counts["n_dx"] > 20]

    # Sample proportionally: ~30% simple, ~40% moderate, ~30% complex
    n_simple = max(1, round(n * 0.3))
    n_complex = max(1, round(n * 0.3))
    n_moderate = n - n_simple - n_complex

    selected = pd.concat([
        simple.sample(n=min(n_simple, len(simple)), random_state=seed),
        moderate.sample(n=min(n_moderate, len(moderate)), random_state=seed),
        complex_.sample(n=min(n_complex, len(complex_)), random_state=seed),
    ])

    hadm_ids = sorted(selected["hadm_id"].tolist())
    logger.info(
        "Selected %d admissions (simple=%d, moderate=%d, complex=%d)",
        len(hadm_ids),
        min(n_simple, len(simple)),
        min(n_moderate, len(moderate)),
        min(n_complex, len(complex_)),
    )
    return hadm_ids
