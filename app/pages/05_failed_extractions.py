"""Failed Extractions — view failed notes and rerun extraction."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from db import query_df

st.title("Failed Extractions")
st.caption("View notes that failed extraction and rerun them.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ERRORS_PATH = PROJECT_ROOT / "data" / "processed" / "extraction_results" / "errors.json"
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "mtsamples.csv"

# ---------------------------------------------------------------------------
# Load errors
# ---------------------------------------------------------------------------
if not ERRORS_PATH.exists():
    st.success("No extraction errors found. All notes processed successfully.")
    st.stop()

with open(ERRORS_PATH) as f:
    errors = json.load(f)

if not errors:
    st.success("No extraction errors found.")
    st.stop()

st.metric("Failed Notes", len(errors))

# ---------------------------------------------------------------------------
# Load the raw CSV to show the actual note content
# ---------------------------------------------------------------------------
raw_df = pd.read_csv(RAW_CSV) if RAW_CSV.exists() else pd.DataFrame()

# ---------------------------------------------------------------------------
# Display each failed note
# ---------------------------------------------------------------------------
for i, err in enumerate(errors):
    note_id = err["note_id"]
    error_msg = err["error"]

    # Parse error type
    if "Expecting value" in error_msg:
        error_type = "Empty / non-JSON response"
    elif "Extra data" in error_msg:
        error_type = "Extra text outside JSON"
    elif "Expecting ',' delimiter" in error_msg:
        error_type = "Malformed JSON structure"
    else:
        error_type = "Unknown"

    with st.expander(f"Note #{note_id} — {error_type}", expanded=(i == 0)):
        st.markdown(f"**Error:** `{error_msg}`")
        st.markdown(f"**Error Type:** {error_type}")

        # Show the actual note content
        if not raw_df.empty:
            try:
                idx = int(note_id)
                if idx in raw_df.index:
                    row = raw_df.loc[idx]
                else:
                    row = raw_df[raw_df.iloc[:, 0] == idx]
                    row = row.iloc[0] if not row.empty else None

                if row is not None:
                    st.markdown(f"**Specialty:** {row.get('medical_specialty', 'N/A')}")
                    st.markdown(f"**Sample Name:** {row.get('sample_name', 'N/A')}")
                    st.markdown(f"**Description:** {row.get('description', 'N/A')}")

                    transcription = row.get("transcription", "")
                    if transcription and isinstance(transcription, str):
                        char_count = len(transcription)
                        st.markdown(f"**Note Length:** {char_count:,} characters")
                        st.markdown("**Clinical Note:**")
                        st.markdown(
                            f"<div style='max-height:300px;overflow-y:auto;padding:12px;"
                            f"border:1px solid #ddd;border-radius:8px;font-size:0.85em;"
                            f"line-height:1.5;white-space:pre-wrap'>{transcription}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning("No transcription text found for this note.")
                else:
                    st.warning(f"Note #{note_id} not found in raw CSV.")
            except (ValueError, IndexError, KeyError):
                st.warning(f"Could not load note #{note_id} from raw data.")

st.divider()

# ---------------------------------------------------------------------------
# Rerun controls
# ---------------------------------------------------------------------------
st.subheader("Rerun Failed Extractions")

st.markdown(
    "Failed extractions were caused by Claude returning malformed JSON. "
    "Rerunning sends the same notes through the extraction pipeline again."
)

# Select which to rerun
failed_ids = [e["note_id"] for e in errors]
selected = st.multiselect(
    "Select notes to rerun",
    options=failed_ids,
    default=failed_ids,
    format_func=lambda x: f"Note #{x}",
)

if st.button("Rerun Selected", type="primary", disabled=not selected):
    with st.spinner(f"Rerunning extraction for {len(selected)} notes..."):
        # Write a small script that reruns just the selected notes
        rerun_script = PROJECT_ROOT / "data" / "processed" / "extraction_results" / "_rerun.py"
        rerun_script.write_text(f"""
import sys, json, logging
sys.path.insert(0, "{PROJECT_ROOT / 'src'}")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from clinical_pipeline.config import get_settings
from clinical_pipeline.extraction.extractor import ClinicalExtractor, ExtractionError
from clinical_pipeline.extraction.cost_tracker import CostTracker
import pandas as pd

settings = get_settings()
df = pd.read_csv(settings.raw_dir / "mtsamples.csv")
tracker = CostTracker()
extractor = ClinicalExtractor(api_key=settings.anthropic_api_key, cost_tracker=tracker)

note_ids = {json.dumps(selected)}
results = []
remaining_errors = []

for nid in note_ids:
    try:
        idx = int(nid)
        row = df.loc[idx] if idx in df.index else None
        if row is None:
            remaining_errors.append({{"note_id": nid, "error": "Note not found in CSV"}})
            continue
        transcription = row.get("transcription", "")
        if not transcription or not isinstance(transcription, str):
            remaining_errors.append({{"note_id": nid, "error": "No transcription text"}})
            continue

        extraction = extractor.extract(nid, transcription)
        # Save individual result
        out_path = settings.processed_dir / "extraction_results" / "extractions" / f"{{nid}}.json"
        out_path.write_text(extraction.model_dump_json(indent=2))
        results.append(nid)
        print(f"SUCCESS: Note #{{nid}}")
    except ExtractionError as e:
        remaining_errors.append({{"note_id": nid, "error": str(e)}})
        print(f"FAILED: Note #{{nid}}: {{e}}")

# Update errors.json with only remaining errors
errors_path = settings.processed_dir / "extraction_results" / "errors.json"
errors_path.write_text(json.dumps(remaining_errors, indent=2))

print(f"\\nDone: {{len(results)}} succeeded, {{len(remaining_errors)}} still failed")
print(f"Cost: {{tracker.summary()}}")
""")

        result = subprocess.run(
            ["conda", "run", "-n", "clinical-pipeline", "python", str(rerun_script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode == 0:
            st.success("Rerun complete!")
            st.code(result.stdout, language="text")
            st.info(
                "Newly extracted notes need to be loaded into the database. "
                "Run `make load-db` or rerun the load_to_db script."
            )
        else:
            st.error("Rerun failed.")
            st.code(result.stderr, language="text")

        # Clean up
        rerun_script.unlink(missing_ok=True)
        st.cache_data.clear()
