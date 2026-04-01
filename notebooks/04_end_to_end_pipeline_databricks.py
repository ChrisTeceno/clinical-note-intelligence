# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # Clinical Note Intelligence -- End-to-End Pipeline
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC **Full lifecycle**: Raw CSV ingestion &rarr; PySpark cleaning &rarr; Claude API extraction &rarr; ICD-10 validation &rarr; Delta table persistence &rarr; result visualization
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC This notebook is self-contained and runs on **Databricks** (recommended) or locally with `delta-spark`.
# MAGIC %md
# MAGIC Each stage produces artifacts consumed by the next, demonstrating a production-grade clinical data automation pipeline.
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC | Stage | What | Output |
# MAGIC %md
# MAGIC |-------|------|--------|
# MAGIC %md
# MAGIC | 1 | PySpark Ingestion & Cleaning | Delta table of 2,358 deduplicated clinical notes |
# MAGIC %md
# MAGIC | 2 | Claude API Extraction | Structured diagnoses, procedures, medications via `tool_use` |
# MAGIC %md
# MAGIC | 3 | ICD-10 Code Validation | Exact + fuzzy match against CMS 2025 (74K codes) |
# MAGIC %md
# MAGIC | 4 | Delta Table Persistence | Star-schema tables in Databricks (notes, extractions, entities) |
# MAGIC %md
# MAGIC | 5 | Results & Visualization | Interactive charts, quality metrics, sample deep-dives |

# COMMAND ----------

# Install dependencies (Databricks: runs once per cluster restart)
# Locally: pip install anthropic rapidfuzz pydantic delta-spark matplotlib
%pip install anthropic rapidfuzz pydantic>=2.0 matplotlib -q

# COMMAND ----------

import os
import json
import time
import html
import re
import unicodedata
import tempfile
import urllib.request
import zipfile
import uuid
import logging
from datetime import datetime, timezone
from typing import Literal, Optional
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pydantic import BaseModel, Field

# Detect environment
try:
    spark  # noqa: F821 — available in Databricks
    ON_DATABRICKS = True
except NameError:
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName("ClinicalNoteIntelligence")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalogExtension")
        .getOrCreate()
    )
    ON_DATABRICKS = False

print(f"Environment: {'Databricks' if ON_DATABRICKS else 'Local PySpark'}")
print(f"Spark version: {spark.version}")

# COMMAND ----------

# --- Configuration ---
# Databricks: use Volumes paths and widgets for secrets
# Local: use relative paths and environment variables

if ON_DATABRICKS:
    RAW_DATA_PATH = "/Volumes/workspace/default/raw-data/mtsamples.csv"
    DELTA_BASE = "/Volumes/workspace/default/clinical-pipeline"
    dbutils.widgets.text("api_key", "", "Anthropic API Key")
    ANTHROPIC_API_KEY = dbutils.widgets.get("api_key")
else:
    RAW_DATA_PATH = str(Path("../data/raw/mtsamples.csv").resolve())
    DELTA_BASE = str(Path("../data/processed/delta").resolve())
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

NOTES_DELTA = f"{DELTA_BASE}/notes_clean"
EXTRACTIONS_DELTA = f"{DELTA_BASE}/extractions"
DIAGNOSES_DELTA = f"{DELTA_BASE}/diagnoses"
PROCEDURES_DELTA = f"{DELTA_BASE}/procedures"
MEDICATIONS_DELTA = f"{DELTA_BASE}/medications"

N_SAMPLE_NOTES = 10  # Number of notes to extract (increase for full run)

if not ANTHROPIC_API_KEY:
    print("WARNING: No API key. Stage 1 will run; Stages 2-5 require ANTHROPIC_API_KEY.")
else:
    print(f"API key configured (ends ...{ANTHROPIC_API_KEY[-4:]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC %md
# MAGIC ## Stage 1: PySpark Ingestion & Cleaning
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC Load the MTSamples clinical note dataset (5,028 rows), enforce a schema, clean text artifacts,
# MAGIC %md
# MAGIC deduplicate on transcription content, and persist as a partitioned Delta table.
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC **Why PySpark?** The pipeline is designed to scale to millions of notes (e.g., full MIMIC-IV).
# MAGIC %md
# MAGIC PySpark gives us distributed text cleaning, schema enforcement, and native Delta Lake support.

# COMMAND ----------

from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from pyspark.sql import functions as F

# --- Schema Definition ---
MTSAMPLES_SCHEMA = StructType([
    StructField("id", IntegerType(), nullable=False),
    StructField("description", StringType(), nullable=True),
    StructField("medical_specialty", StringType(), nullable=True),
    StructField("sample_name", StringType(), nullable=True),
    StructField("transcription", StringType(), nullable=True),
    StructField("keywords", StringType(), nullable=True),
])

# --- Read CSV with schema enforcement ---
df_raw = (
    spark.read
    .option("header", "true")
    .option("multiLine", "true")
    .option("escape", '"')
    .schema(MTSAMPLES_SCHEMA)
    .csv(RAW_DATA_PATH)
)

raw_count = df_raw.count()
print(f"Raw row count: {raw_count:,}")
df_raw.limit(3).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Text Cleaning
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC Clinical notes from MTSamples contain HTML tags, Unicode artifacts (smart quotes, em-dashes),
# MAGIC %md
# MAGIC and irregular whitespace. We apply three cleaning passes as PySpark UDFs:
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC 1. **HTML removal** -- strip tags, decode entities
# MAGIC %md
# MAGIC 2. **Encoding normalization** -- NFKD Unicode, replace smart quotes/dashes
# MAGIC %md
# MAGIC 3. **Whitespace collapse** -- single spaces, trimmed edges

# COMMAND ----------

def clean_text(text):
    """Full cleaning pipeline: HTML removal, encoding fix, whitespace normalization."""
    if text is None:
        return None
    # Strip HTML tags and decode entities
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    # Normalize Unicode
    text = unicodedata.normalize("NFKD", text)
    for src, dst in {
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
    }.items():
        text = text.replace(src, dst)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


clean_text_udf = F.udf(clean_text, StringType())

df_cleaned = df_raw
for col_name in ["description", "transcription", "keywords", "sample_name"]:
    df_cleaned = df_cleaned.withColumn(col_name, clean_text_udf(F.col(col_name)))

# Show before/after on one row
sample_id = df_raw.select("id").first()[0]
raw_text = df_raw.filter(F.col("id") == sample_id).select("transcription").first()[0]
clean_text_val = df_cleaned.filter(F.col("id") == sample_id).select("transcription").first()[0]
print(f"RAW  (first 200 chars): {(raw_text or '')[:200]}")
print(f"CLEAN (first 200 chars): {(clean_text_val or '')[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deduplication & Data Quality Report

# COMMAND ----------

count_before = df_cleaned.count()
df_dedup = df_cleaned.dropDuplicates(["transcription"]).filter("transcription IS NOT NULL")
count_after = df_dedup.count()

print(f"Before dedup: {count_before:,}")
print(f"After dedup:  {count_after:,}")
print(f"Removed:      {count_before - count_after:,} rows (duplicates + nulls)")

# Null rates
print("\nNull rates:")
for col_name in df_dedup.columns:
    null_count = df_dedup.where(F.col(col_name).isNull()).count()
    rate = null_count / count_after if count_after > 0 else 0.0
    print(f"  {col_name}: {rate:.1%} ({null_count:,} nulls)")

# Transcription length statistics
df_with_len = df_dedup.withColumn("transcription_length", F.length(F.col("transcription")))
stats = df_with_len.agg(
    F.min("transcription_length").alias("min"),
    F.max("transcription_length").alias("max"),
    F.mean("transcription_length").alias("mean"),
    F.expr("percentile_approx(transcription_length, 0.5)").alias("median"),
).first()

print(f"\nTranscription length: min={stats['min']}, median={stats['median']}, "
      f"mean={stats['mean']:.0f}, max={stats['max']}")

# COMMAND ----------

# Specialty distribution — top 15
specialty_pd = (
    df_dedup.groupBy("medical_specialty").count()
    .orderBy(F.desc("count"))
    .limit(15)
    .toPandas()
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(specialty_pd["medical_specialty"][::-1], specialty_pd["count"][::-1], color="#4A90D9")
ax.set_xlabel("Number of Notes")
ax.set_title("Top 15 Medical Specialties in MTSamples")
for i, v in enumerate(specialty_pd["count"][::-1]):
    ax.text(v + 5, i, str(v), va="center", fontsize=9)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Cleaned Notes to Delta

# COMMAND ----------

(
    df_dedup
    .repartition("medical_specialty")
    .write.format("delta")
    .mode("overwrite")
    .partitionBy("medical_specialty")
    .save(NOTES_DELTA)
)

# Verify
df_notes = spark.read.format("delta").load(NOTES_DELTA)
verified_count = df_notes.count()
n_specialties = df_notes.select("medical_specialty").distinct().count()
print(f"Delta table written to {NOTES_DELTA}")
print(f"Verified: {verified_count:,} notes across {n_specialties} specialties")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC %md
# MAGIC ## Stage 2: LLM-Based Clinical Extraction
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC Extract diagnoses, procedures, and medications from clinical notes using **Claude Haiku** with
# MAGIC %md
# MAGIC structured `tool_use` output. Tool use guarantees valid JSON matching our Pydantic schema --
# MAGIC %md
# MAGIC no regex parsing, no output format errors.
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC **Key design decisions:**
# MAGIC %md
# MAGIC - **`tool_choice: {"type": "tool"}`** forces Claude to respond only via the extraction tool (guaranteed structured output)
# MAGIC %md
# MAGIC - **Pydantic validation** catches any schema violations before downstream processing
# MAGIC %md
# MAGIC - **Evidence spans** anchor every extraction to verbatim text from the note
# MAGIC %md
# MAGIC - **Confidence levels** (high/medium/low) enable downstream filtering and prioritized review

# COMMAND ----------

# --- Pydantic Models ---
# These define the exact schema Claude must produce via tool_use.

class Diagnosis(BaseModel):
    name: str = Field(description="Name of the diagnosis")
    icd10_suggestion: Optional[str] = Field(default=None, description="Suggested ICD-10-CM code")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence in this extraction")
    evidence_span: str = Field(description="Verbatim text from the note supporting this diagnosis")

class Procedure(BaseModel):
    name: str = Field(description="Name of the procedure performed or planned")
    cpt_suggestion: Optional[str] = Field(default=None, description="Suggested CPT code")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence in this extraction")
    evidence_span: str = Field(description="Verbatim text from the note supporting this procedure")

class Medication(BaseModel):
    name: str = Field(description="Medication name")
    dosage: Optional[str] = Field(default=None, description="Dosage if mentioned")
    frequency: Optional[str] = Field(default=None, description="Frequency if mentioned")
    route: Optional[str] = Field(default=None, description="Route of administration")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence in this extraction")
    evidence_span: str = Field(description="Verbatim text from the note supporting this medication")

class ClinicalExtraction(BaseModel):
    note_id: str = Field(description="Identifier of the source clinical note")
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    chief_complaint: Optional[str] = Field(default=None, description="Chief complaint or reason for visit")
    medical_specialty: Optional[str] = Field(default=None, description="Medical specialty of this note")

print(f"Schema fields: {list(ClinicalExtraction.model_fields.keys())}")
print(f"Diagnosis fields: {list(Diagnosis.model_fields.keys())}")

# COMMAND ----------

import anthropic

# --- System Prompt & Tool Schema ---
SYSTEM_PROMPT = """\
You are a clinical data extraction specialist. Your task is to extract structured \
medical information from clinical notes with high accuracy.

Use the extract_clinical_data tool to return your results.

Rules:
1. Only extract information explicitly stated in the note. Do not infer diagnoses \
that are not mentioned.
2. For each extraction, provide the exact verbatim text span from the note as evidence.
3. Assign confidence levels:
   - "high": explicitly stated and unambiguous
   - "medium": strongly implied or uses clinical shorthand
   - "low": mentioned in differential or uncertain context
4. For ICD-10 codes, suggest the most specific code you can. Use the format "X00.00". \
If unsure, provide the broader category code.
5. If a field has no relevant information in the note, return an empty list or null.\
"""

CONFIDENCE_ENUM = ["high", "medium", "low"]

EXTRACTION_TOOL = {
    "name": "extract_clinical_data",
    "description": (
        "Extract structured clinical data from a clinical note. "
        "Call this tool with all diagnoses, procedures, medications, "
        "chief complaint, and medical specialty found in the note."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "diagnoses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "icd10_suggestion": {"type": ["string", "null"]},
                        "confidence": {"type": "string", "enum": CONFIDENCE_ENUM},
                        "evidence_span": {"type": "string"},
                    },
                    "required": ["name", "confidence", "evidence_span"],
                },
            },
            "procedures": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "cpt_suggestion": {"type": ["string", "null"]},
                        "confidence": {"type": "string", "enum": CONFIDENCE_ENUM},
                        "evidence_span": {"type": "string"},
                    },
                    "required": ["name", "confidence", "evidence_span"],
                },
            },
            "medications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "dosage": {"type": ["string", "null"]},
                        "frequency": {"type": ["string", "null"]},
                        "route": {"type": ["string", "null"]},
                        "confidence": {"type": "string", "enum": CONFIDENCE_ENUM},
                        "evidence_span": {"type": "string"},
                    },
                    "required": ["name", "confidence", "evidence_span"],
                },
            },
            "chief_complaint": {"type": ["string", "null"]},
            "medical_specialty": {"type": ["string", "null"]},
        },
        "required": ["diagnoses", "procedures", "medications", "chief_complaint", "medical_specialty"],
    },
}

print(f"Tool schema: {EXTRACTION_TOOL['name']} with {len(EXTRACTION_TOOL['input_schema']['properties'])} top-level fields")

# COMMAND ----------

logger = logging.getLogger("clinical_extractor")
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4096


class ExtractionError(Exception):
    """Raised when extraction fails after retries."""


class ClinicalExtractor:
    """Extracts structured clinical data from a single note using Claude tool_use."""

    def __init__(self, api_key, model=DEFAULT_MODEL, max_retries=3):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def extract(self, note_id, transcription):
        """Extract clinical data from a single note. Returns a validated ClinicalExtraction."""
        messages = [{"role": "user", "content": f"Extract structured clinical data from this note:\n\n{transcription}"}]

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=[EXTRACTION_TOOL],
                    tool_choice={"type": "tool", "name": "extract_clinical_data"},
                )
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens

                for block in response.content:
                    if block.type == "tool_use":
                        data = block.input
                        data["note_id"] = str(note_id)
                        return ClinicalExtraction.model_validate(data)

                raise ExtractionError("No tool_use block in response")

            except anthropic.RateLimitError:
                last_error = Exception("Rate limit exceeded")
                time.sleep(2 ** attempt)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last_error = e
                    time.sleep(2 ** attempt)
                else:
                    raise ExtractionError(f"API error for note {note_id}: {e}") from e
            except ExtractionError:
                raise
            except Exception as e:
                raise ExtractionError(f"Unexpected error for note {note_id}: {e}") from e

        raise ExtractionError(f"Failed after {self.max_retries} attempts: {last_error}")

    @property
    def estimated_cost(self):
        """Estimate USD cost based on Haiku pricing ($0.25/1M input, $1.25/1M output)."""
        return (self.total_input_tokens * 0.25 + self.total_output_tokens * 1.25) / 1_000_000

print("ClinicalExtractor ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Extraction on Sample Notes

# COMMAND ----------

assert ANTHROPIC_API_KEY, "Set the api_key widget (Databricks) or ANTHROPIC_API_KEY env var before running."

df_notes = spark.read.format("delta").load(NOTES_DELTA)
sample_notes = df_notes.orderBy("id").limit(N_SAMPLE_NOTES).toPandas()

extractor = ClinicalExtractor(api_key=ANTHROPIC_API_KEY)
extractions = []   # full ClinicalExtraction objects
results = []       # summary rows for display
errors = []

for idx, row in sample_notes.iterrows():
    label = f"{idx+1}/{N_SAMPLE_NOTES}: {row['sample_name'] or 'Untitled'}"
    try:
        ext = extractor.extract(str(row["id"]), row["transcription"])
        extractions.append((row, ext))
        results.append({
            "note_id": row["id"],
            "specialty": row["medical_specialty"],
            "sample_name": row["sample_name"],
            "n_dx": len(ext.diagnoses),
            "n_px": len(ext.procedures),
            "n_rx": len(ext.medications),
            "chief_complaint": ext.chief_complaint,
        })
        print(f"  {label} -> {len(ext.diagnoses)} dx, {len(ext.procedures)} px, {len(ext.medications)} rx")
    except Exception as e:
        errors.append({"note_id": row["id"], "error": str(e)})
        print(f"  {label} -> ERROR: {e}")
    time.sleep(0.3)

results_df = pd.DataFrame(results)
print(f"\nExtracted {len(results)}/{N_SAMPLE_NOTES} notes | "
      f"Tokens: {extractor.total_input_tokens:,} in + {extractor.total_output_tokens:,} out | "
      f"Cost: ${extractor.estimated_cost:.4f}")
results_df

# COMMAND ----------

# Show one full extraction as JSON for inspection
if extractions:
    sample_row, sample_ext = extractions[0]
    print(f"--- Full extraction for note {sample_row['id']}: {sample_row['sample_name']} ---\n")
    print(sample_ext.model_dump_json(indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC %md
# MAGIC ## Stage 3: ICD-10 Code Validation
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC Cross-reference Claude's ICD-10 suggestions against the **CMS 2025 ICD-10-CM code table** (74,260 codes).
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC Two-tier matching strategy:
# MAGIC %md
# MAGIC 1. **Exact match** -- normalize the code (strip dots, uppercase) and look up directly
# MAGIC %md
# MAGIC 2. **Fuzzy match** -- use `rapidfuzz.fuzz.WRatio` on the diagnosis name against all CMS code descriptions (threshold: 80%)
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC Codes that match neither tier are flagged for manual review in the HITL dashboard.

# COMMAND ----------

# Download and parse CMS ICD-10-CM 2025 code table
icd10_url = "https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip"
tmp_dir = tempfile.mkdtemp()
zip_path = os.path.join(tmp_dir, "icd10cm_2025.zip")

print("Downloading CMS ICD-10-CM 2025 code table...")
urllib.request.urlretrieve(icd10_url, zip_path)

with zipfile.ZipFile(zip_path, "r") as zf:
    txt_files = [f for f in zf.namelist() if f.endswith(".txt")]
    zf.extractall(tmp_dir)

txt_path = os.path.join(tmp_dir, txt_files[0])

# Parse fixed-width format: first 7 chars = code, rest = description
icd10_codes = {}
with open(txt_path, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if len(line) < 8:
            continue
        code = line[:7].strip().upper().replace(".", "")
        description = line[7:].strip()
        if code and description:
            icd10_codes[code] = description

print(f"Loaded {len(icd10_codes):,} ICD-10-CM codes")
# Show examples
for code, desc in list(icd10_codes.items())[:3]:
    print(f"  {code}: {desc}")

# COMMAND ----------

from rapidfuzz import fuzz, process

PARTIAL_MATCH_THRESHOLD = 80

match_results = []

for row, extraction in extractions:
    for dx in extraction.diagnoses:
        suggested = (dx.icd10_suggestion or "").strip().upper().replace(".", "")
        match_type = "none"
        matched_code = None
        matched_desc = None
        score = 0.0

        # Tier 1: Exact code match
        if suggested and suggested in icd10_codes:
            match_type = "exact"
            matched_code = suggested
            matched_desc = icd10_codes[suggested]
            score = 100.0
        # Tier 2: Fuzzy description match
        elif dx.name:
            result = process.extractOne(
                dx.name, icd10_codes,
                scorer=fuzz.WRatio,
                score_cutoff=PARTIAL_MATCH_THRESHOLD,
            )
            if result is not None:
                matched_desc, score, matched_code = result
                match_type = "partial"

        match_results.append({
            "note_id": extraction.note_id,
            "diagnosis": dx.name,
            "confidence": dx.confidence,
            "suggested_code": dx.icd10_suggestion,
            "match_type": match_type,
            "matched_code": matched_code,
            "matched_description": matched_desc,
            "score": round(score, 1),
        })

match_df = pd.DataFrame(match_results)
print(f"Validated {len(match_df)} diagnosis codes")
match_df

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC %md
# MAGIC ## Stage 4: Persist Results to Delta Tables
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC Write all extraction results to structured Delta tables following a **star schema**:
# MAGIC %md
# MAGIC - `extractions` (fact) -- one row per note, links to entity dimension tables
# MAGIC %md
# MAGIC - `diagnoses`, `procedures`, `medications` (dimensions) -- one row per extracted entity
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC This enables SQL analytics in Databricks, version-controlled data with Delta's time travel,
# MAGIC %md
# MAGIC and serves as the source of truth for the HITL review dashboard.

# COMMAND ----------

# Build DataFrames from extraction results
from pyspark.sql.types import (
    ArrayType, FloatType, IntegerType, StringType, StructField, StructType, TimestampType,
)

now = datetime.now(timezone.utc).isoformat()

# --- Extractions fact table ---
extraction_rows = []
dx_rows = []
px_rows = []
rx_rows = []

# Build a lookup from match_df for ICD-10 validation results
match_lookup = {}
for _, m in match_df.iterrows():
    key = (str(m["note_id"]), m["diagnosis"])
    match_lookup[key] = m

for row, ext in extractions:
    ext_id = str(uuid.uuid4())
    note_id = str(row["id"])

    extraction_rows.append({
        "extraction_id": ext_id,
        "note_id": note_id,
        "medical_specialty": ext.medical_specialty or row["medical_specialty"],
        "chief_complaint": ext.chief_complaint,
        "n_diagnoses": len(ext.diagnoses),
        "n_procedures": len(ext.procedures),
        "n_medications": len(ext.medications),
        "model": DEFAULT_MODEL,
        "status": "pending",
        "extracted_at": now,
    })

    for dx in ext.diagnoses:
        m = match_lookup.get((note_id, dx.name), {})
        dx_rows.append({
            "diagnosis_id": str(uuid.uuid4()),
            "extraction_id": ext_id,
            "note_id": note_id,
            "name": dx.name,
            "icd10_suggested": dx.icd10_suggestion,
            "icd10_matched": m.get("matched_code") if isinstance(m, dict) else getattr(m, "matched_code", None),
            "match_type": m.get("match_type", "none") if isinstance(m, dict) else getattr(m, "match_type", "none"),
            "match_score": float(m.get("score", 0)) if isinstance(m, dict) else float(getattr(m, "score", 0)),
            "confidence": dx.confidence,
            "evidence_span": dx.evidence_span,
        })

    for px in ext.procedures:
        px_rows.append({
            "procedure_id": str(uuid.uuid4()),
            "extraction_id": ext_id,
            "note_id": note_id,
            "name": px.name,
            "cpt_suggestion": px.cpt_suggestion,
            "confidence": px.confidence,
            "evidence_span": px.evidence_span,
        })

    for rx in ext.medications:
        rx_rows.append({
            "medication_id": str(uuid.uuid4()),
            "extraction_id": ext_id,
            "note_id": note_id,
            "name": rx.name,
            "dosage": rx.dosage,
            "frequency": rx.frequency,
            "route": rx.route,
            "confidence": rx.confidence,
            "evidence_span": rx.evidence_span,
        })

print(f"Prepared: {len(extraction_rows)} extractions, "
      f"{len(dx_rows)} diagnoses, {len(px_rows)} procedures, {len(rx_rows)} medications")

# COMMAND ----------

# Write to Delta tables
spark.createDataFrame(extraction_rows).write.format("delta").mode("overwrite").save(EXTRACTIONS_DELTA)
spark.createDataFrame(dx_rows).write.format("delta").mode("overwrite").save(DIAGNOSES_DELTA)
spark.createDataFrame(px_rows).write.format("delta").mode("overwrite").save(PROCEDURES_DELTA)
spark.createDataFrame(rx_rows).write.format("delta").mode("overwrite").save(MEDICATIONS_DELTA)

# Verify all tables
for name, path in [
    ("extractions", EXTRACTIONS_DELTA),
    ("diagnoses", DIAGNOSES_DELTA),
    ("procedures", PROCEDURES_DELTA),
    ("medications", MEDICATIONS_DELTA),
]:
    count = spark.read.format("delta").load(path).count()
    print(f"  {name}: {count:,} rows -> {path}")

# COMMAND ----------

# Register as temporary SQL views for cross-table queries
spark.read.format("delta").load(NOTES_DELTA).createOrReplaceTempView("notes")
spark.read.format("delta").load(EXTRACTIONS_DELTA).createOrReplaceTempView("extractions")
spark.read.format("delta").load(DIAGNOSES_DELTA).createOrReplaceTempView("diagnoses")
spark.read.format("delta").load(PROCEDURES_DELTA).createOrReplaceTempView("procedures")
spark.read.format("delta").load(MEDICATIONS_DELTA).createOrReplaceTempView("medications")

# Verify with a join query
spark.sql("""
    SELECT e.note_id, e.medical_specialty, e.n_diagnoses, e.n_procedures, e.n_medications,
           e.chief_complaint, e.status
    FROM extractions e
    ORDER BY e.n_diagnoses DESC
    LIMIT 5
""").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC %md
# MAGIC ## Stage 5: Results & Visualization
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC Query the Delta tables to produce quality metrics and visual summaries of the extraction pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5a. Pipeline KPIs

# COMMAND ----------

total_notes = len(results)
total_dx = sum(r["n_dx"] for r in results)
total_px = sum(r["n_px"] for r in results)
total_rx = sum(r["n_rx"] for r in results)

# ICD-10 match rates
if not match_df.empty:
    n_codes = len(match_df)
    n_exact = (match_df["match_type"] == "exact").sum()
    n_partial = (match_df["match_type"] == "partial").sum()
    n_none = (match_df["match_type"] == "none").sum()
else:
    n_codes = n_exact = n_partial = n_none = 0

kpi_data = {
    "Notes processed": total_notes,
    "Total diagnoses": total_dx,
    "Total procedures": total_px,
    "Total medications": total_rx,
    "Avg entities/note": f"{(total_dx + total_px + total_rx) / max(total_notes, 1):.1f}",
    "ICD-10 exact match": f"{n_exact}/{n_codes} ({n_exact/max(n_codes,1):.0%})",
    "ICD-10 partial match": f"{n_partial}/{n_codes} ({n_partial/max(n_codes,1):.0%})",
    "ICD-10 no match": f"{n_none}/{n_codes} ({n_none/max(n_codes,1):.0%})",
    "API cost": f"${extractor.estimated_cost:.4f}",
    "Model": DEFAULT_MODEL,
}

print("=" * 55)
print("  PIPELINE QUALITY SUMMARY")
print("=" * 55)
for k, v in kpi_data.items():
    print(f"  {k:<25} {v}")
print("=" * 55)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5b. Entity Distribution by Specialty

# COMMAND ----------

# Entity counts by specialty from Delta
entity_by_spec = spark.sql("""
    SELECT e.medical_specialty,
           SUM(e.n_diagnoses) as diagnoses,
           SUM(e.n_procedures) as procedures,
           SUM(e.n_medications) as medications
    FROM extractions e
    GROUP BY e.medical_specialty
    ORDER BY diagnoses + procedures + medications DESC
""").toPandas()

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(entity_by_spec))
w = 0.25
ax.bar([i - w for i in x], entity_by_spec["diagnoses"], w, label="Diagnoses", color="#E74C3C")
ax.bar(x, entity_by_spec["procedures"], w, label="Procedures", color="#3498DB")
ax.bar([i + w for i in x], entity_by_spec["medications"], w, label="Medications", color="#2ECC71")
ax.set_xticks(x)
ax.set_xticklabels(entity_by_spec["medical_specialty"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Count")
ax.set_title("Extracted Entities by Medical Specialty")
ax.legend()
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5c. Confidence Distribution & ICD-10 Match Rates

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: Confidence distribution across all entity types
all_conf = []
for _, ext in extractions:
    for dx in ext.diagnoses:
        all_conf.append({"type": "Diagnosis", "confidence": dx.confidence})
    for px in ext.procedures:
        all_conf.append({"type": "Procedure", "confidence": px.confidence})
    for rx in ext.medications:
        all_conf.append({"type": "Medication", "confidence": rx.confidence})

conf_df = pd.DataFrame(all_conf)
if not conf_df.empty:
    pivot = conf_df.groupby(["type", "confidence"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=["high", "medium", "low"], fill_value=0)
    pivot.plot.bar(ax=axes[0], color={"high": "#2ECC71", "medium": "#F39C12", "low": "#E74C3C"})
    axes[0].set_title("Confidence Distribution by Entity Type")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Confidence")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Right: ICD-10 match type breakdown
if not match_df.empty:
    match_counts = match_df["match_type"].value_counts()
    colors = {"exact": "#2ECC71", "partial": "#F39C12", "none": "#E74C3C"}
    match_counts.plot.pie(
        ax=axes[1],
        autopct="%1.0f%%",
        colors=[colors.get(t, "#95a5a6") for t in match_counts.index],
        startangle=90,
    )
    axes[1].set_ylabel("")
    axes[1].set_title("ICD-10 Code Match Types")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5d. Sample Deep-Dive: One Note End-to-End
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC Select one note and show the full pipeline output: raw text, extracted entities with evidence spans, and ICD-10 validation results.

# COMMAND ----------

# Pick the note with the most diagnoses for the richest example
best_idx = max(range(len(extractions)), key=lambda i: len(extractions[i][1].diagnoses))
demo_row, demo_ext = extractions[best_idx]

print(f"Note ID: {demo_row['id']}  |  Specialty: {demo_row['medical_specialty']}  |  {demo_row['sample_name']}")
print(f"Chief Complaint: {demo_ext.chief_complaint or 'N/A'}")
print("-" * 80)
print("CLINICAL NOTE (first 800 chars):")
print(demo_row["transcription"][:800])
print("..." if len(demo_row["transcription"]) > 800 else "")
print()

print(f"DIAGNOSES ({len(demo_ext.diagnoses)}):")
for dx in demo_ext.diagnoses:
    code = dx.icd10_suggestion or "N/A"
    # Look up matched code
    m = match_lookup.get((str(demo_row["id"]), dx.name), {})
    matched = m.get("matched_code", "") if isinstance(m, dict) else getattr(m, "matched_code", "")
    mtype = m.get("match_type", "") if isinstance(m, dict) else getattr(m, "match_type", "")
    print(f"  [{dx.confidence.upper():6s}] {dx.name}")
    print(f"           ICD-10: {code} -> {matched or 'no match'} ({mtype})")
    print(f"           Evidence: \"{dx.evidence_span[:120]}\"")

print(f"\nPROCEDURES ({len(demo_ext.procedures)}):")
for px in demo_ext.procedures:
    print(f"  [{px.confidence.upper():6s}] {px.name} (CPT: {px.cpt_suggestion or 'N/A'})")
    print(f"           Evidence: \"{px.evidence_span[:120]}\"")

print(f"\nMEDICATIONS ({len(demo_ext.medications)}):")
for rx in demo_ext.medications:
    details = " | ".join(filter(None, [rx.dosage, rx.frequency, rx.route]))
    print(f"  [{rx.confidence.upper():6s}] {rx.name} {f'({details})' if details else ''}")
    print(f"           Evidence: \"{rx.evidence_span[:120]}\"")


# COMMAND ----------

# MAGIC %md
# MAGIC ### 5e. Query Delta Tables with SQL
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC Demonstrate that all results are now queryable as standard SQL tables -- the same queries
# MAGIC %md
# MAGIC power the live Streamlit HITL dashboard at [clinical.christeceno.com](https://clinical.christeceno.com).

# COMMAND ----------

# Join extractions with diagnoses, showing ICD-10 validation results
dx_detail = spark.sql("""
    SELECT
        e.medical_specialty,
        d.name AS diagnosis,
        d.icd10_suggested,
        d.icd10_matched,
        d.match_type,
        d.match_score,
        d.confidence,
        SUBSTR(d.evidence_span, 1, 80) AS evidence_preview
    FROM diagnoses d
    JOIN extractions e ON d.extraction_id = e.extraction_id
    ORDER BY e.medical_specialty, d.confidence
""")

dx_detail.toPandas()

# COMMAND ----------

# Medications with dosage information from Delta
rx_detail = spark.sql("""
    SELECT
        e.medical_specialty,
        m.name AS medication,
        m.dosage,
        m.frequency,
        m.route,
        m.confidence,
        SUBSTR(m.evidence_span, 1, 80) AS evidence_preview
    FROM medications m
    JOIN extractions e ON m.extraction_id = e.extraction_id
    ORDER BY m.name
""")

rx_detail.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC %md
# MAGIC ## Pipeline Summary
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC | Stage | Status | Key Output |
# MAGIC %md
# MAGIC |-------|--------|------------|
# MAGIC %md
# MAGIC | **1. PySpark Ingestion** | Complete | Delta table: cleaned, deduplicated clinical notes |
# MAGIC %md
# MAGIC | **2. Claude Extraction** | Complete | Structured entities via `tool_use` (Pydantic-validated) |
# MAGIC %md
# MAGIC | **3. ICD-10 Validation** | Complete | Exact + fuzzy match against CMS 2025 (74K codes) |
# MAGIC %md
# MAGIC | **4. Delta Persistence** | Complete | Star-schema tables: extractions, diagnoses, procedures, medications |
# MAGIC %md
# MAGIC | **5. Visualization** | Complete | KPIs, charts, SQL queries, sample deep-dive |
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC ### Architecture
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC ```
# MAGIC %md
# MAGIC CSV -> PySpark (clean/dedup) -> Delta Table
# MAGIC %md
# MAGIC          |
# MAGIC %md
# MAGIC          v
# MAGIC %md
# MAGIC     Claude Haiku (tool_use) -> Pydantic validation -> ICD-10 matching
# MAGIC %md
# MAGIC          |
# MAGIC %md
# MAGIC          v
# MAGIC %md
# MAGIC     Delta Tables (star schema) -> SQL analytics
# MAGIC %md
# MAGIC          |
# MAGIC %md
# MAGIC          v
# MAGIC %md
# MAGIC     Streamlit HITL Dashboard (approve/reject/edit)
# MAGIC %md
# MAGIC ```
# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC ### Next Steps
# MAGIC %md
# MAGIC - **Scale**: Process all 2,358 notes using Anthropic Batch API (~$4 estimated cost)
# MAGIC %md
# MAGIC - **Evaluate**: Run against MIMIC-IV ground truth (Dx F1=83.8% with prompt optimization)
# MAGIC %md
# MAGIC - **HITL Review**: Live dashboard at [clinical.christeceno.com](https://clinical.christeceno.com)
# MAGIC %md
# MAGIC - **Production**: Replace SQLite with Databricks-managed Delta tables, add Unity Catalog governance
