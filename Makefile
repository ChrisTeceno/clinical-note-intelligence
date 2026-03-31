.PHONY: setup test lint ingest extract run-app clean

setup:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	@echo "Activate with: source .venv/bin/activate"

test:
	.venv/bin/pytest tests/ -v

lint:
	.venv/bin/ruff check src/ tests/
	.venv/bin/ruff format --check src/ tests/

ingest:
	.venv/bin/python -m clinical_pipeline.ingestion.ingest_mtsamples

extract:
	.venv/bin/python -m clinical_pipeline.extraction.batch_runner

run-app:
	.venv/bin/streamlit run app/main.py

clean:
	rm -rf .venv build dist *.egg-info
	rm -rf spark-warehouse metastore_db derby.log
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
