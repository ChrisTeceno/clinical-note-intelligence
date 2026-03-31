#!/usr/bin/env python3
"""Sync notebooks from Databricks, convert to dark-themed HTML, and deploy.

Usage:
    python scripts/sync_databricks_notebooks.py
    python scripts/sync_databricks_notebooks.py --deploy  # also rsync to VPS
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "https://dbc-cfaa4306-63f1.cloud.databricks.com")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
WORKSPACE_PATH = os.environ.get("DATABRICKS_WORKSPACE_PATH", "/Users/chris.teceno@gmail.com")

if not DATABRICKS_TOKEN:
    # Try loading from .env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DATABRICKS_TOKEN="):
                DATABRICKS_TOKEN = line.split("=", 1)[1].strip()
    if not DATABRICKS_TOKEN:
        print("ERROR: DATABRICKS_TOKEN not set. Add it to .env or set as environment variable.")
        sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
STATIC_DIR = PROJECT_ROOT / "static" / "notebooks"
CONVERTER = PROJECT_ROOT / "scripts" / "convert_databricks_notebook.py"

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}


# ---------------------------------------------------------------------------
# Databricks API helpers
# ---------------------------------------------------------------------------

def list_notebooks() -> list[dict]:
    """List all notebooks in the user's workspace."""
    resp = requests.get(
        f"{DATABRICKS_HOST}/api/2.0/workspace/list",
        headers=HEADERS,
        json={"path": WORKSPACE_PATH},
    )
    resp.raise_for_status()
    return [
        obj for obj in resp.json().get("objects", [])
        if obj.get("object_type") == "NOTEBOOK"
    ]


def export_notebook_source(path: str) -> str:
    """Export a notebook as raw Python source."""
    resp = requests.get(
        f"{DATABRICKS_HOST}/api/2.0/workspace/export",
        headers=HEADERS,
        params={"path": path, "format": "SOURCE"},
    )
    resp.raise_for_status()
    content_b64 = resp.json()["content"]
    return base64.b64decode(content_b64).decode("utf-8")


def export_notebook_html(path: str) -> str:
    """Export a notebook as Databricks HTML (with outputs)."""
    resp = requests.get(
        f"{DATABRICKS_HOST}/api/2.0/workspace/export",
        headers=HEADERS,
        params={"path": path, "format": "HTML"},
    )
    resp.raise_for_status()
    content_b64 = resp.json()["content"]
    return base64.b64decode(content_b64).decode("utf-8")


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_to_clean_html(notebook_name: str, html_content: str) -> str:
    """Convert Databricks HTML export to clean dark-themed HTML using our converter."""
    # Save temp HTML
    temp_input = STATIC_DIR / f"_temp_{notebook_name}.html"
    temp_output = STATIC_DIR / f"{notebook_name}.html"
    temp_input.write_text(html_content, encoding="utf-8")

    # Run converter
    result = subprocess.run(
        [sys.executable, str(CONVERTER), str(temp_input), str(temp_output)],
        capture_output=True,
        text=True,
    )

    # Cleanup temp
    temp_input.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  Converter failed: {result.stderr}")
        return ""

    print(f"  {result.stdout.strip()}")
    return str(temp_output)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def sync(deploy: bool = False):
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching notebook list from Databricks...")
    notebooks = list_notebooks()
    print(f"Found {len(notebooks)} notebooks\n")

    manifest = []

    for nb in notebooks:
        name = nb["path"].split("/")[-1]
        print(f"Processing: {name}")

        # 1. Save raw source locally
        source = export_notebook_source(nb["path"])
        source_path = NOTEBOOKS_DIR / f"{name}.py"
        source_path.write_text(source, encoding="utf-8")
        print(f"  Source saved: {source_path.name} ({len(source.splitlines())} lines)")

        # 2. Export HTML (with outputs) and convert
        html_content = export_notebook_html(nb["path"])
        output_path = convert_to_clean_html(name, html_content)

        manifest.append({
            "name": name,
            "path": nb["path"],
            "source_file": f"{name}.py",
            "html_file": f"{name}.html" if output_path else None,
            "language": nb.get("language", "PYTHON"),
        })
        print()

    # Save manifest for the Streamlit page
    manifest_path = STATIC_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest saved: {manifest_path}")

    # Deploy to VPS
    if deploy:
        print("\nDeploying to VPS...")
        result = subprocess.run(
            [
                "rsync", "-avz", "-e", "ssh",
                str(STATIC_DIR) + "/",
                "deploy@5.161.193.61:/home/deploy/clinical-note-intelligence/static/notebooks/",
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout.split("\n")[-3] if result.stdout else result.stderr)
        print("Deployed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Databricks notebooks")
    parser.add_argument("--deploy", action="store_true", help="Deploy to VPS after sync")
    args = parser.parse_args()
    sync(deploy=args.deploy)
