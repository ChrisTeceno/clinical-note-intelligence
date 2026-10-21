# Prompt optimization artifacts

- `best_prompt.txt`: the prompt the optimizer saved as its best candidate.
- `prompt_history.json`: the per-iteration record of candidates and scores.
- `opt_hadm_ids.json`, `synthetic_notes/`: the admissions and notes the run used.

These files come from the optimizer before PR #2 and are unvalidated. In that version the
prompt patch never reached the extractor, so every candidate ran with the default prompt.
The reported gain (Dx F1 83.2 to 83.8) is noise.

Full-data check (2026-10-21, 30 full MIMIC-IV admissions): `best_prompt.txt` lowered
medication F1 from 0.560 to 0.443 and matched the default prompt on the other metrics.
The default SYSTEM_PROMPT remains the production prompt.

The optimizer now evaluates candidates with the patched prompt, so a fresh run is needed
for a real result. The files are kept as history only.
