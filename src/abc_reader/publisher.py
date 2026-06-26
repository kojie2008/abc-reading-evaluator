"""
GitHub Pages publisher — push HTML reports to a public repo.

Uses the GitHub Contents API to create/update files.
"""

import base64
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from .config import GITHUB_REPO, GITHUB_TOKEN


def _safe_basename(local_path: str) -> str:
    """Strip non-ASCII characters from filename for GitHub API safety."""
    base = Path(local_path).name
    safe = re.sub(r"[^\x20-\x7E]", "", base).strip().replace(" ", "_")
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or f"report_{datetime.now():%Y%m%d_%H%M%S}.html"


def _api_put(path_in_repo: str, content_b64: str, token: str) -> dict:
    """PUT a file on GitHub. Handles 422 (exists) automatically."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path_in_repo}"
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}
    data = {"message": f"add report: {path_in_repo}", "content": content_b64}

    # Try update first (need sha)
    get_req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(get_req, timeout=15) as resp:
            existing = json.loads(resp.read().decode())
            if "sha" in existing:
                data["sha"] = existing["sha"]
    except Exception:
        pass  # File doesn't exist yet — good

    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="PUT")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def publish(local_path: str) -> dict:
    """
    Push a local HTML report to GitHub Pages.

    Returns: {"ok": bool, "url": str | None, "error": str | None}
    """
    path = Path(local_path)
    if not path.exists():
        return {"ok": False, "url": None, "error": "文件不存在"}

    token = os.environ.get("ABC_GITHUB_TOKEN") or GITHUB_TOKEN
    if not token:
        print("[发布] ⏭️ 无 GITHUB_TOKEN，跳过发布")
        return {"ok": False, "url": None, "error": "未配置 GITHUB_TOKEN"}

    safe_name = _safe_basename(local_path)
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")

    # Build pages URL
    username, reponame = GITHUB_REPO.split("/", 1)
    base_url = f"https://{username}.github.io/{reponame}"

    try:
        _api_put(safe_name, b64, token)
        url = f"{base_url}/{safe_name}"
        print(f"[发布] ✅ 已上传: {safe_name}")
        if url:
            print(f"[发布] 📎 {url}")
        return {"ok": True, "url": url, "error": None}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:200] if e.fp else str(e)
        print(f"[发布] ❌ HTTP {e.code}: {detail}")
        return {"ok": False, "url": None, "error": f"HTTP {e.code}"}
    except Exception as e:
        print(f"[发布] ❌ {e}")
        return {"ok": False, "url": None, "error": str(e)}
