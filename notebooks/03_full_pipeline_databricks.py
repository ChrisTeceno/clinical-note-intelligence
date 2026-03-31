# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Clinical Note Intelligence -- Full Pipeline
# MAGIC End-to-end pipeline: PySpark ingestion -> Claude API extraction -> ICD-10 matching -> quality analysis

# COMMAND ----------

# MAGIC %pip install anthropic rapidfuzz "pydantic>=2.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

RAW_DATA_PATH = "/Volumes/workspace/default/raw-data/mtsamples.csv"
OUTPUT_PATH = "/Volumes/workspace/default/raw-data/mtsamples_clean"

dbutils.widgets.text("api_key", "", "Anthropic API Key")
ANTHROPIC_API_KEY = dbutils.widgets.get("api_key")

if not ANTHROPIC_API_KEY:
    print("WARNING: No API key provided. Set the 'api_key' widget at the top of the notebook.")
    print("Stages 1 (ingestion) will run; Stage 2 (extraction) and 3 (ICD-10) require a key.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Stage 1: PySpark Ingestion & Cleaning
# MAGIC Load MTSamples CSV, clean text, deduplicate, and write to Delta.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Schema Definition & CSV Read

# COMMAND ----------

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

MTSAMPLES_SCHEMA = StructType(
    [
        StructField("id", IntegerType(), nullable=False),
        StructField("description", StringType(), nullable=True),
        StructField("medical_specialty", StringType(), nullable=True),
        StructField("sample_name", StringType(), nullable=True),
        StructField("transcription", StringType(), nullable=True),
        StructField("keywords", StringType(), nullable=True),
    ]
)

df_raw = (
    spark.read.option("header", "true")
    .option("multiLine", "true")
    .option("escape", '"')
    .schema(MTSAMPLES_SCHEMA)
    .csv(RAW_DATA_PATH)
)

raw_count = df_raw.count()
print(f"Raw row count: {raw_count}")
display(df_raw.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Text Cleaning
# MAGIC Strip HTML, fix encoding artifacts, normalize whitespace. Applied as PySpark UDFs.

# COMMAND ----------

import html
import re
import unicodedata

from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def strip_html_tags(text):
    """Remove HTML tags and decode HTML entities."""
    if text is None:
        return None
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return text


def normalize_whitespace(text):
    """Collapse runs of whitespace into single spaces and strip edges."""
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_encoding(text):
    """Normalize Unicode and replace common smart-quote / dash artifacts."""
    if text is None:
        return None
    text = unicodedata.normalize("NFKD", text)
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def clean_text(text):
    """Full cleaning pipeline: HTML removal, encoding fix, whitespace normalization."""
    text = strip_html_tags(text)
    text = fix_encoding(text)
    text = normalize_whitespace(text)
    return text


clean_text_udf = F.udf(clean_text, StringType())

TEXT_COLUMNS = ["description", "transcription", "keywords", "sample_name"]

df_cleaned = df_raw
for col_name in TEXT_COLUMNS:
    df_cleaned = df_cleaned.withColumn(col_name, clean_text_udf(F.col(col_name)))

print("=== BEFORE cleaning (raw) ===")
display(df_raw.select("id", "transcription").limit(1))
print("=== AFTER cleaning ===")
display(df_cleaned.select("id", "transcription").limit(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deduplication & Quality Report

# COMMAND ----------

count_before = df_cleaned.count()
df_dedup = df_cleaned.dropDuplicates(["transcription"])
count_after = df_dedup.count()

print(f"Before dedup: {count_before}")
print(f"After dedup:  {count_after}")
print(f"Removed:      {count_before - count_after} duplicate rows")

# --- Null rates per column ---
print("\nNull rates:")
for col_name in df_dedup.columns:
    null_count = df_dedup.where(F.col(col_name).isNull()).count()
    rate = null_count / count_after if count_after > 0 else 0.0
    print(f"  {col_name}: {rate:.2%} ({null_count} nulls)")

# --- Top 15 specialty distribution ---
print("\nSpecialty distribution (top 15):")
df_specialty = (
    df_dedup.groupBy("medical_specialty")
    .count()
    .orderBy(F.desc("count"))
    .limit(15)
)
display(df_specialty)

# --- Transcription length statistics ---
df_with_len = df_dedup.withColumn("transcription_length", F.length(F.col("transcription")))
length_stats = df_with_len.agg(
    F.min("transcription_length").alias("min_length"),
    F.max("transcription_length").alias("max_length"),
    F.mean("transcription_length").alias("mean_length"),
    F.expr("percentile_approx(transcription_length, 0.5)").alias("median_length"),
).first()

print("\nTranscription length statistics:")
print(f"  Min:    {length_stats['min_length']}")
print(f"  Max:    {length_stats['max_length']}")
print(f"  Mean:   {length_stats['mean_length']:.0f}")
print(f"  Median: {length_stats['median_length']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Cleaned Data as Delta

# COMMAND ----------

(
    df_dedup.repartition("medical_specialty")
    .write.format("delta")
    .mode("overwrite")
    .partitionBy("medical_specialty")
    .save(OUTPUT_PATH)
)

print(f"Delta table written to {OUTPUT_PATH}")

# Verify
df_verify = spark.read.format("delta").load(OUTPUT_PATH)
print(f"Verified row count: {df_verify.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Stage 2: LLM-Based Clinical Extraction
# MAGIC Extract diagnoses, procedures, and medications from clinical notes using Claude Haiku
# MAGIC with structured `tool_use` output.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pydantic Models

# COMMAND ----------

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Diagnosis(BaseModel):
    name: str = Field(description="Name of the diagnosis")
    icd10_suggestion: Optional[str] = Field(
        default=None,
        description="Suggested ICD-10-CM code (e.g. 'I25.10')",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in this extraction"
    )
    evidence_span: str = Field(
        description="Verbatim text from the note supporting this diagnosis"
    )


class Procedure(BaseModel):
    name: str = Field(description="Name of the procedure performed or planned")
    cpt_suggestion: Optional[str] = Field(
        default=None, description="Suggested CPT code if identifiable"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in this extraction"
    )
    evidence_span: str = Field(
        description="Verbatim text from the note supporting this procedure"
    )


class Medication(BaseModel):
    name: str = Field(description="Medication name")
    dosage: Optional[str] = Field(default=None, description="Dosage if mentioned")
    frequency: Optional[str] = Field(default=None, description="Frequency if mentioned")
    route: Optional[str] = Field(
        default=None, description="Route of administration if mentioned"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in this extraction"
    )
    evidence_span: str = Field(
        description="Verbatim text from the note supporting this medication"
    )


class ClinicalExtraction(BaseModel):
    note_id: str = Field(description="Identifier of the source clinical note")
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    chief_complaint: Optional[str] = Field(
        default=None, description="Chief complaint or reason for visit"
    )
    medical_specialty: Optional[str] = Field(
        default=None, description="Medical specialty of this note"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extraction Tool Schema & Prompt

# COMMAND ----------

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
5. If a field has no relevant information in the note, return an empty list or null.

Example: For a note mentioning "Type 2 diabetes mellitus" in the assessment, extract \
a diagnosis with name="Type 2 diabetes mellitus", icd10_suggestion="E11.9", \
confidence="high", and evidence_span set to the verbatim text.\
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
                "description": "Diagnoses found in the note",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the diagnosis"},
                        "icd10_suggestion": {
                            "type": ["string", "null"],
                            "description": "Suggested ICD-10-CM code (e.g. 'I25.10')",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": CONFIDENCE_ENUM,
                            "description": "Confidence in this extraction",
                        },
                        "evidence_span": {
                            "type": "string",
                            "description": "Verbatim text from the note supporting this diagnosis",
                        },
                    },
                    "required": ["name", "confidence", "evidence_span"],
                },
            },
            "procedures": {
                "type": "array",
                "description": "Procedures performed or planned in the note",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the procedure performed or planned",
                        },
                        "cpt_suggestion": {
                            "type": ["string", "null"],
                            "description": "Suggested CPT code if identifiable",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": CONFIDENCE_ENUM,
                            "description": "Confidence in this extraction",
                        },
                        "evidence_span": {
                            "type": "string",
                            "description": "Verbatim text from the note supporting this procedure",
                        },
                    },
                    "required": ["name", "confidence", "evidence_span"],
                },
            },
            "medications": {
                "type": "array",
                "description": "Medications mentioned in the note",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Medication name"},
                        "dosage": {"type": ["string", "null"], "description": "Dosage if mentioned"},
                        "frequency": {
                            "type": ["string", "null"],
                            "description": "Frequency if mentioned",
                        },
                        "route": {
                            "type": ["string", "null"],
                            "description": "Route of administration if mentioned",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": CONFIDENCE_ENUM,
                            "description": "Confidence in this extraction",
                        },
                        "evidence_span": {
                            "type": "string",
                            "description": "Verbatim text from the note supporting this medication",
                        },
                    },
                    "required": ["name", "confidence", "evidence_span"],
                },
            },
            "chief_complaint": {
                "type": ["string", "null"],
                "description": "Chief complaint or reason for visit, null if not stated",
            },
            "medical_specialty": {
                "type": ["string", "null"],
                "description": "Medical specialty of this note, null if not identifiable",
            },
        },
        "required": [
            "diagnoses",
            "procedures",
            "medications",
            "chief_complaint",
            "medical_specialty",
        ],
    },
}


def build_extraction_prompt(transcription):
    """Build the user message for a single note extraction."""
    return f"Extract structured clinical data from this note:\n\n{transcription}"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clinical Extractor

# COMMAND ----------

import anthropic
import time
import logging

logger = logging.getLogger("clinical_extractor")

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4096


class ExtractionError(Exception):
    """Raised when extraction fails after retries."""


class ClinicalExtractor:
    """Extracts structured clinical data from a single note using Claude."""

    def __init__(self, api_key, model=DEFAULT_MODEL, max_retries=3):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def extract(self, note_id, transcription):
        """Extract clinical data from a single note via tool_use."""
        user_message = build_extraction_prompt(transcription)
        messages = [{"role": "user", "content": user_message}]

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

                # Parse the tool_use block
                for block in response.content:
                    if block.type == "tool_use":
                        data = block.input
                        data["note_id"] = str(note_id)
                        return ClinicalExtraction.model_validate(data)

                raise ExtractionError("No tool_use block found in response")

            except anthropic.RateLimitError:
                logger.warning("Rate limited on attempt %d/%d", attempt, self.max_retries)
                last_error = Exception("Rate limit exceeded")
                time.sleep(2 ** attempt)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    logger.warning("Server error %d on attempt %d/%d", e.status_code, attempt, self.max_retries)
                    last_error = e
                    time.sleep(2 ** attempt)
                else:
                    raise ExtractionError(f"API error for note {note_id}: {e}") from e
            except ExtractionError:
                raise
            except Exception as e:
                raise ExtractionError(f"Unexpected error for note {note_id}: {e}") from e

        raise ExtractionError(
            f"Failed to extract note {note_id} after {self.max_retries} attempts: {last_error}"
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Extraction on Sample Notes (10 notes)

# COMMAND ----------

import pandas as pd

assert ANTHROPIC_API_KEY, "Set the api_key widget before running extraction."

df_clean = spark.read.format("delta").load(OUTPUT_PATH)
sample_notes = df_clean.filter("transcription IS NOT NULL").limit(10).toPandas()

extractor = ClinicalExtractor(api_key=ANTHROPIC_API_KEY)
results = []
extractions = []  # keep full extraction objects for ICD-10 validation

for idx, row in sample_notes.iterrows():
    print(f"Extracting note {idx+1}/10: {row['sample_name']}")
    try:
        extraction = extractor.extract(str(row["id"]), row["transcription"])
        extractions.append(extraction)
        results.append(
            {
                "note_id": row["id"],
                "specialty": row["medical_specialty"],
                "sample_name": row["sample_name"],
                "n_diagnoses": len(extraction.diagnoses),
                "n_procedures": len(extraction.procedures),
                "n_medications": len(extraction.medications),
                "chief_complaint": extraction.chief_complaint,
                "diagnoses": [d.name for d in extraction.diagnoses],
                "icd10_codes": [d.icd10_suggestion for d in extraction.diagnoses],
            }
        )
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(0.5)

results_df = pd.DataFrame(results)
print(f"\nSuccessfully extracted {len(results)}/10 notes")
display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Stage 3: ICD-10 Code Validation
# MAGIC Cross-reference Claude's ICD-10 suggestions against the CMS 2025 code table
# MAGIC using exact lookup and rapidfuzz description matching.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download & Load CMS ICD-10-CM 2025 Codes

# COMMAND ----------

import urllib.request
import zipfile
import os
import tempfile

icd10_url = "https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip"
tmp_dir = tempfile.mkdtemp()
zip_path = os.path.join(tmp_dir, "icd10cm_2025.zip")

print("Downloading CMS ICD-10-CM 2025 code table...")
urllib.request.urlretrieve(icd10_url, zip_path)
print(f"Downloaded to {zip_path}")

# Extract the txt file
with zipfile.ZipFile(zip_path, "r") as zf:
    txt_files = [f for f in zf.namelist() if f.endswith(".txt")]
    print(f"Files in zip: {txt_files}")
    zf.extractall(tmp_dir)

# Find the code descriptions file
txt_path = None
for f in txt_files:
    full = os.path.join(tmp_dir, f)
    if os.path.exists(full):
        txt_path = full
        break

assert txt_path is not None, f"Could not find ICD-10 txt file in {txt_files}"
print(f"Using: {txt_path}")

# Parse the CMS fixed-width format into a dict: code -> description
icd10_codes = {}
with open(txt_path, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if len(line) < 8:
            continue
        code = line[:7].strip()
        description = line[7:].strip()
        if code and description:
            icd10_codes[code.upper().replace(".", "")] = description

print(f"Loaded {len(icd10_codes)} ICD-10-CM codes")

# Show a few examples
for i, (code, desc) in enumerate(list(icd10_codes.items())[:5]):
    print(f"  {code}: {desc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Match Extracted Codes Against CMS Table

# COMMAND ----------

from rapidfuzz import fuzz, process

PARTIAL_MATCH_THRESHOLD = 80

match_results = []

for extraction in extractions:
    for dx in extraction.diagnoses:
        suggested = (dx.icd10_suggestion or "").strip().upper().replace(".", "")
        match_type = "none"
        matched_code = None
        matched_desc = None
        score = 0.0

        # Try exact code match
        if suggested and suggested in icd10_codes:
            match_type = "exact"
            matched_code = suggested
            matched_desc = icd10_codes[suggested]
            score = 100.0
        # Try fuzzy description matching
        elif dx.name and icd10_codes:
            result = process.extractOne(
                dx.name,
                icd10_codes,
                scorer=fuzz.WRatio,
                score_cutoff=PARTIAL_MATCH_THRESHOLD,
            )
            if result is not None:
                matched_desc, score, matched_code = result
                match_type = "partial"

        match_results.append(
            {
                "note_id": extraction.note_id,
                "diagnosis": dx.name,
                "suggested_code": dx.icd10_suggestion,
                "match_type": match_type,
                "matched_code": matched_code,
                "matched_description": matched_desc,
                "score": round(score, 1),
            }
        )

match_df = pd.DataFrame(match_results)
if not match_df.empty:
    display(match_df)
else:
    print("No diagnoses with ICD-10 codes to validate.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extraction Quality Summary

# COMMAND ----------

total_notes = len(results)
total_diagnoses = sum(r["n_diagnoses"] for r in results)
total_procedures = sum(r["n_procedures"] for r in results)
total_medications = sum(r["n_medications"] for r in results)

print("=" * 60)
print("EXTRACTION QUALITY SUMMARY")
print("=" * 60)
print(f"Notes processed:      {total_notes}")
print(f"Total diagnoses:      {total_diagnoses}")
print(f"Total procedures:     {total_procedures}")
print(f"Total medications:    {total_medications}")
print()

if not match_df.empty:
    total_codes = len(match_df)
    exact = len(match_df[match_df["match_type"] == "exact"])
    partial = len(match_df[match_df["match_type"] == "partial"])
    none_ = len(match_df[match_df["match_type"] == "none"])

    print("ICD-10 Code Validation:")
    print(f"  Total codes checked: {total_codes}")
    print(f"  Exact matches:       {exact} ({exact/total_codes:.1%})")
    print(f"  Partial matches:     {partial} ({partial/total_codes:.1%})")
    print(f"  No match:            {none_} ({none_/total_codes:.1%})")
    print(f"  Any match rate:      {(exact+partial)/total_codes:.1%}")

    exact_pct = f"{exact/total_codes:.0%}" if total_codes > 0 else "N/A"
else:
    exact_pct = "N/A"
    print("ICD-10 Code Validation: No codes to validate.")

# Summary table as DataFrame
summary_data = [
    {"Metric": "Notes processed", "Value": str(total_notes)},
    {"Metric": "Total diagnoses", "Value": str(total_diagnoses)},
    {"Metric": "Total procedures", "Value": str(total_procedures)},
    {"Metric": "Total medications", "Value": str(total_medications)},
    {"Metric": "Avg diagnoses/note", "Value": f"{total_diagnoses/total_notes:.1f}" if total_notes else "0"},
    {"Metric": "Avg procedures/note", "Value": f"{total_procedures/total_notes:.1f}" if total_notes else "0"},
    {"Metric": "Avg medications/note", "Value": f"{total_medications/total_notes:.1f}" if total_notes else "0"},
]
display(pd.DataFrame(summary_data))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Pipeline Summary
# MAGIC
# MAGIC | Stage | Status | Output |
# MAGIC |-------|--------|--------|
# MAGIC | PySpark Ingestion | Complete | Delta table with cleaned & deduplicated notes |
# MAGIC | Claude Extraction | Complete | 10 notes extracted (diagnoses, procedures, medications) |
# MAGIC | ICD-10 Validation | Complete | Code match rates shown above |
# MAGIC | HITL Review | Streamlit dashboard (local) | Approve/reject interface |
# MAGIC | Evaluation | MIMIC-IV ground truth (local) | F1: Dx=81.5%, Proc=55.8%, Rx=47.5% |
# MAGIC
# MAGIC ### Next Steps
# MAGIC - **Scale extraction**: Use the Anthropic Batch API to process all notes at lower cost.
# MAGIC - **HITL review**: Launch `streamlit run streamlit_app/app.py` locally to approve/reject extractions.
# MAGIC - **Evaluation**: Run `python -m clinical_pipeline.evaluation.run_eval` against MIMIC-IV ground truth.
