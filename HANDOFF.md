# Agent Handoff — Clinical Note Intelligence Pipeline

## Project State (2026-03-31)

### What This Project Is
End-to-end clinical data automation system for a job application. Extracts diagnoses, procedures, medications from clinical notes using Claude API, validates ICD-10 codes, and presents results through a HITL Streamlit dashboard.

### Live URLs
- **Dashboard**: https://clinical.christeceno.com
- **Portfolio**: https://christeceno.com
- **GitHub**: https://github.com/ChrisTeceno/clinical-note-intelligence

### Infrastructure
- **VPS**: Hetzner CPX21, IP 5.161.193.61, SSH: `deploy@5.161.193.61`
- **Domain**: christeceno.com on Cloudflare (API token in global memory)
- **Streamlit**: systemd service `clinical-pipeline`, nginx reverse proxy port 8501
- **Databricks**: API access configured (token in .env)

### Key Files
- `app/main.py` — Streamlit navigation (13 pages)
- `app/pages/03_note_detail.py` — HITL review page with edit/approve/reject
- `app/pages/00_overview.py` — Landing page
- `src/clinical_pipeline/extraction/extractor.py` — Claude API extraction (tool_use)
- `src/clinical_pipeline/extraction/prompts.py` — System prompt + tool schema
- `src/clinical_pipeline/extraction/icd10_rag.py` — RAG retriever
- `src/clinical_pipeline/evaluation/run_eval.py` — Evaluation pipeline
- `src/clinical_pipeline/optimization/` — Karpathy autoresearch prompt optimizer
- `src/clinical_pipeline/feedback/feedback_store.py` — HITL corrections store
- `src/clinical_pipeline/imaging/` — Chest X-ray classification (TorchXRayVision + Claude Vision)
- `scripts/sync_databricks_notebooks.py` — Pull notebooks from Databricks API
- `scripts/convert_databricks_notebook.py` — Convert to dark-themed HTML
- `data/clinical_notes.db` — SQLite with 50 extracted notes
- `data/evaluation/`: 30 MIMIC-IV demo evaluation results, run history (`MIMIC_SOURCE=full` writes to gitignored `data/full/` instead)
- `data/feedback/` — HITL corrections (JSON)
- `data/optimization/` — Prompt optimization history + best prompt
- `data/imaging/results/` — 7 model results (6 TorchXRayVision + Claude Vision)

### Experiment Results (run_history.json)
1. **Baseline Haiku**: Dx F1=81.5%, Px F1=55.8%, Rx F1=47.5%
2. **Sonnet 4**: Dx F1=82.1% (best precision 93.5%)
3. **Haiku + RAG**: Rx F1=51.1% (best medication extraction)
4. **Prompt optimization (unvalidated)**: the run did not test its candidates. The prompt patch never reached the extractor (fixed in PR #2), so every candidate ran with the default prompt and the Dx F1=83.8% figure is not a real gain. On 30 full MIMIC-IV admissions the saved best prompt lowered medication F1 from 0.560 to 0.443. The default SYSTEM_PROMPT remains the production prompt.

### Deployment Command
```bash
# Sync code to VPS
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' --exclude 'data/raw/mtsamples.csv' --exclude 'data/imaging/sample' --exclude 'data/imaging/sample.zip' --exclude 'Certs' -e ssh . deploy@5.161.193.61:/home/deploy/clinical-note-intelligence/

# Restart
ssh root@5.161.193.61 "systemctl restart clinical-pipeline"

# Validate
conda run -n clinical-pipeline python -c "from playwright.sync_api import sync_playwright; ..."
```

### Conda Environment
`clinical-pipeline` (Python 3.12) — all deps installed including streamlit, anthropic, pyspark, torchxrayvision, playwright, sklearn, rapidfuzz

### IMMEDIATE TODO (where we left off)

1. **After approve/reject, auto-advance to next pending note and scroll to top**
   - File: `app/pages/03_note_detail.py`, lines ~550-590
   - Currently: `st.rerun()` after approve/reject reloads the same note
   - Needed: Query next pending note, set it via `st.query_params`, scroll to top
   - The selectbox index needs to advance to the next pending note

2. **Commit and push latest changes** — there are uncommitted edits to note_detail.py

### Other Pending Items
- PhysioNet credentialing is complete (MIMIC-IV 3.1 access granted)
- Full 2,358 note extraction not yet run (~$8-10 API cost)
- Real MIMIC-IV note evaluation needs access to MIMIC-IV-Note, a separate PhysioNet project (follow-up)
- Full MIMIC-IV 3.1 support is in: set `MIMIC_SOURCE=full` and `MIMIC_FULL_ROOT`; no full-data run has been done yet. Public deploy stays on the demo (`MIMIC_DEMO_ROOT=/home/deploy/data/mimic-iv-demo`)
- Portfolio headshot on christeceno.com uses inline styles (CSS class wasn't applying)

### User Preferences (from memory)
- Always validate web UIs with Playwright screenshots before presenting
- No emojis in code/docs unless asked
- Prefers concise responses
- Clinical background: RN, USAF medic — understands medical terminology
- christeceno.com is the personal domain
