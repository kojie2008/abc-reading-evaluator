"""
Audio downloading — student recordings + reference audio.
"""

import os
import re
from pathlib import Path

import requests

from .config import STUDENT_DIR, REFERENCE_DIR

_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _safe_filename(work_id: str, page_num: int, kind: str, url: str) -> str:
    """Generate a safe filename from work_id, page and URL extension."""
    ext = (os.path.splitext(url.split("?")[0])[1]) or ".wav"
    clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", work_id).strip("_")
    return f"{clean_id}_p{page_num}_{kind}{ext}"


def download_file(url: str, save_path: str) -> bool:
    """
    Download a single audio file. Returns True on success.
    Skips if the file already exists with content.
    """
    if not url:
        return False
    path = Path(save_path)
    if path.exists() and path.stat().st_size > 0:
        return True

    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": _USER_AGENT})
        if resp.status_code != 200 or len(resp.content) < 1000:
            print(f"[下载] ⚠ {path.name}: HTTP {resp.status_code}, {len(resp.content)} bytes")
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        print(f"[下载] ✓ {path.name} ({len(resp.content) / 1024:.0f} KB)")
        return True
    except requests.RequestException as e:
        print(f"[下载] ✗ {path.name}: {e}")
        return False


def download_all(pages: list, work_id: str) -> dict:
    """
    Download all audio for a book work.

    Returns: {page_num: {"student": "path", "reference": "path"}}
    """
    results: dict = {}
    for page in pages:
        pn = page["page_num"]
        entry = {"student": "", "reference": ""}

        if page.get("student_audio_url"):
            fname = _safe_filename(work_id, pn, "student", page["student_audio_url"])
            dest = STUDENT_DIR / fname
            if download_file(page["student_audio_url"], str(dest)):
                entry["student"] = str(dest)

        if page.get("reference_audio_url"):
            fname = _safe_filename(work_id, pn, "reference", page["reference_audio_url"])
            dest = REFERENCE_DIR / fname
            if download_file(page["reference_audio_url"], str(dest)):
                entry["reference"] = str(dest)

        results[pn] = entry
    return results
