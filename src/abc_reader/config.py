"""
Global configuration for ABC Reading Evaluator.

All settings can be overridden via environment variables for CI/deployment.
"""

import os
from pathlib import Path

# ---------- Paths ----------
# Project root (two levels up from src/abc_reader/)
ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = Path(os.getenv("ABC_DATA_DIR", ROOT / "data"))
DOWNLOADS_DIR = DATA_DIR / "downloads"
STUDENT_DIR = DOWNLOADS_DIR / "student"
REFERENCE_DIR = DOWNLOADS_DIR / "reference"
REPORT_DIR = DATA_DIR / "reports"

# Ensure dirs exist
for d in (STUDENT_DIR, REFERENCE_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def cleanup_data_dir():
    """Remove all downloaded audio after report generation."""
    import shutil
    if DOWNLOADS_DIR.exists():
        shutil.rmtree(DOWNLOADS_DIR)
        DOWNLOADS_DIR.mkdir(parents=True)


# ---------- ASR ----------
ASR_MODEL = os.getenv("ABC_ASR_MODEL", "tiny")
# tiny | base | small | medium | large-v3


# ---------- Browser ----------
CDP_URL = os.getenv("ABC_CDP_URL", "http://localhost:9222")


# ---------- GitHub Pages ----------
GITHUB_REPO = os.getenv("ABC_GITHUB_REPO", "kojie2008/abc-reading-reports")
GITHUB_TOKEN = os.getenv("ABC_GITHUB_TOKEN", "")
# Empty = publishing disabled unless env var provided


# ---------- Output ----------
class OutputPaths:
    """Convenience accessor for output directories."""
    downloads = DOWNLOADS_DIR
    student = STUDENT_DIR
    reference = REFERENCE_DIR
    reports = REPORT_DIR


class ReportConfig:
    dir = REPORT_DIR

# ---------- HF mirror ----------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
