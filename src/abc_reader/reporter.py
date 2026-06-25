"""Report generation for ABC Reading evaluation results."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ReportConfig


def _safe_name(text: str) -> str:
    """Convert text to safe ASCII filename."""
    safe = re.sub(r"[^\x20-\x7E]", "", text).strip().replace(" ", "_")
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "report"


def _build_percent_bar(percent: float, width: int = 120) -> str:
    """Build an inline SVG percent bar."""
    color = "#22c55e" if percent >= 80 else "#eab308" if percent >= 60 else "#ef4444"
    return f"""<svg width="{width}" height="16" style="vertical-align:middle">
  <rect width="{width}" height="16" rx="8" fill="#e5e7eb"/>
  <rect width="{int(width * percent / 100)}" height="16" rx="8" fill="{color}"/>
</svg>"""


def _error_summary(all_errors: dict) -> str:
    """Build consolidated error word cards."""
    subs = list(set(e["expected"] for e in all_errors.get("substitutions", [])))
    dels = list(set(all_errors.get("deletions", [])))
    parts = []

    if subs:
        cards = []
        for w in sorted(subs)[:40]:
            readings = list(set(
                e["read_as"] for e in all_errors["substitutions"]
                if e["expected"] == w
            ))
            cards.append(
                f'<span class="word-card error-card"><strong>{w}</strong> → 读成 "{", ".join(readings[:2])}"</span>'
            )
        parts.append('<div class="word-section"><h3>🔴 读错的单词 ({})</h3>{}</div>'.format(
            len(subs), " ".join(cards)
        ))

    if dels:
        cards = [f'<span class="word-card del-card"><strong>{w}</strong></span>' for w in sorted(dels)[:30]]
        parts.append('<div class="word-section"><h3>⏭️ 漏读的单词 ({})</h3>{}</div>'.format(
            len(dels), " ".join(cards)
        ))

    return "".join(parts)


def _page_block(pr: dict) -> str:
    """Build a single page analysis block."""
    pn = pr["page_num"]
    text = (pr.get("text") or "")[:200]
    score = pr.get("score", {})
    errors = pr.get("errors", {})
    marked = (pr.get("marked_text") or "")[:300]
    acc = score.get("accuracy", 0)
    subs = errors.get("substitutions", [])
    dels = errors.get("deletions", [])
    inss = errors.get("insertions", [])

    badges = ""
    if subs:
        badges += f'<span class="badge badge-sub">替换 {len(subs)}</span> '
    if dels:
        badges += f'<span class="badge badge-del">漏读 {len(dels)}</span> '
    if inss:
        badges += f'<span class="badge badge-ins">多读 {len(inss)}</span> '

    detail = ""
    if subs:
        items = [
            f'<span class="word-error">{e["expected"]}</span> → <span class="word-wrong">{e["read_as"]}</span>'
            for e in subs[:8]
        ]
        detail += f'<div class="error-section">🔊 读错: {", ".join(items)}{" ...等{}处".format(len(subs)) if len(subs) > 8 else ""}</div>'
    if dels:
        items = [f'<span class="word-deleted">{w}</span>' for w in dels[:8]]
        detail += f'<div class="error-section">⏭️ 漏读: {", ".join(items)}</div>'
    if inss:
        items = [f'<span class="word-inserted">{w}</span>' for w in inss[:6]]
        detail += f'<div class="error-section">➕ 多读: {", ".join(items)}</div>'

    return f"""<div class="page-block">
  <div class="page-header">
    <span class="page-num">📄 第 {pn} 页</span>
    {badges}
    <span class="page-accuracy">准确率: {acc:.1f}%</span>
  </div>
  <div class="page-original"><strong>原文:</strong> {text}</div>
  <div class="page-marked"><strong>朗读分析:</strong> {marked or "（无音频或未识别）"}</div>
  {detail}
</div>"""


def generate_html(
    opus_info: dict,
    page_results: list,
    overall_score: dict,
    all_errors: dict,
    output_path: str | None = None,
) -> str:
    """Generate an HTML evaluation report."""
    name = opus_info.get("name", "未知")
    book = opus_info.get("book_info", {}).get("pictureBookName", "未知")
    level = opus_info.get("book_info", {}).get("lowerLevelDesc", "")
    order = opus_info.get("opus_order", 0)
    abctime_score = opus_info.get("score", 0)

    ts = datetime.now()
    if not output_path:
        safe = _safe_name(f"{name}_{book}")
        output_path = str(ReportConfig.dir / f"{safe}_{ts:%Y%m%d_%H%M%S}.html")

    acc = overall_score.get("accuracy", 0)
    total = overall_score.get("total_words", 0)
    correct = overall_score.get("correct_count", 0)
    n_sub = len(all_errors.get("substitutions", []))
    n_del = len(all_errors.get("deletions", []))
    n_ins = len(all_errors.get("insertions", []))

    pages_html = "\n".join(_page_block(pr) for pr in page_results)
    errors_html = _error_summary(all_errors)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>朗读评测报告 - {name}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif; background:#f5f7fa; color:#333; padding:16px; max-width:920px; margin:0 auto; }}
.header {{ background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; border-radius:16px; padding:24px; margin-bottom:16px; }}
.header h1 {{ font-size:22px; margin-bottom:6px; }}
.header .meta {{ font-size:13px; opacity:.85; }}
.score-overview {{ background:#fff; border-radius:16px; padding:20px; margin-bottom:16px; box-shadow:0 2px 12px rgba(0,0,0,.06); }}
.score-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:12px; margin-top:14px; }}
.score-item {{ text-align:center; padding:14px 8px; background:#f8f9ff; border-radius:12px; }}
.score-item .num {{ font-size:28px; font-weight:bold; color:#667eea; }}
.score-item .label {{ font-size:12px; color:#999; margin-top:3px; }}
.score-item .num.good {{ color:#22c55e; }}
.score-item .num.ok {{ color:#eab308; }}
.score-item .num.bad {{ color:#ef4444; }}
.page-block {{ background:#fff; border-radius:12px; padding:14px 18px; margin-bottom:10px; box-shadow:0 1px 4px rgba(0,0,0,.05); border-left:4px solid #667eea; }}
.page-header {{ display:flex; align-items:center; gap:8px; margin-bottom:8px; flex-wrap:wrap; }}
.page-num {{ font-weight:bold; font-size:14px; }}
.page-accuracy {{ margin-left:auto; font-size:12px; color:#666; white-space:nowrap; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:bold; color:#fff; }}
.badge-sub {{ background:#f59e0b; }}
.badge-del {{ background:#ef4444; }}
.badge-ins {{ background:#8b5cf6; }}
.page-original,.page-marked {{ font-size:13px; line-height:1.6; margin-bottom:6px; color:#444; }}
.page-marked {{ color:#555; }}
.error-section {{ margin-top:6px; font-size:12px; color:#555; }}
.word-error {{ color:#f59e0b; font-weight:bold; }}
.word-wrong {{ color:#aaa; text-decoration:line-through; }}
.word-deleted {{ color:#ef4444; text-decoration:line-through; }}
.word-inserted {{ color:#8b5cf6; font-style:italic; }}
.word-section {{ margin-top:18px; }}
.word-section h3 {{ font-size:15px; margin-bottom:10px; }}
.word-card {{ display:inline-block; padding:4px 10px; margin:3px; border-radius:8px; font-size:12px; background:#fef3c7; border:1px solid #f59e0b; }}
.word-card.del-card {{ background:#fee2e2; border-color:#ef4444; }}
.footer {{ text-align:center; color:#bbb; font-size:11px; margin-top:24px; }}
</style>
</head>
<body>
<div class="header">
  <h1>📖 {name} 的朗读评测报告</h1>
  <div class="meta">绘本: 《{book}》 | Level: {level} | 第 {order} 个作品</div>
  <div class="meta" style="margin-top:4px">ABC Reading 评分: {abctime_score} 分 | 评测时间: {ts:%Y-%m-%d %H:%M}</div>
</div>

<div class="score-overview">
  <h2>📊 整体评分</h2>
  <div class="score-grid">
    <div class="score-item"><div class="num {'good' if acc>=80 else 'ok' if acc>=60 else 'bad'}">{acc:.1f}%</div><div class="label">单词准确率</div></div>
    <div class="score-item"><div class="num">{correct}/{total}</div><div class="label">正确单词数</div></div>
    <div class="score-item"><div class="num ok">{n_sub}</div><div class="label">读错(替换)</div></div>
    <div class="score-item"><div class="num bad">{n_del}</div><div class="label">漏读</div></div>
    <div class="score-item"><div class="num">{n_ins}</div><div class="label">多读</div></div>
  </div>
</div>

{errors_html}

<h2>📄 逐页分析</h2>
{pages_html}

<div class="footer">Generated by ABC Reading Evaluator v2.0 | {ts:%Y-%m-%d %H:%M:%S}</div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def generate_json(
    opus_info: dict,
    page_results: list,
    overall_score: dict,
    all_errors: dict,
    output_path: str | None = None,
) -> str:
    """Generate a JSON evaluation report."""
    if not output_path:
        name = opus_info.get("name", "unknown")
        book = opus_info.get("book_info", {}).get("pictureBookName", "unknown")
        safe = _safe_name(f"{name}_{book}")
        output_path = str(ReportConfig.dir / f"{safe}_{datetime.now():%Y%m%d_%H%M%S}.json")

    report = {
        "meta": {
            "student_name": opus_info.get("name", ""),
            "book_name": book,
            "book_level": opus_info.get("book_info", {}).get("lowerLevelDesc", ""),
            "opus_order": opus_info.get("opus_order", 0),
            "abctime_score": opus_info.get("score", 0),
            "total_pages": len(page_results),
            "generated_at": datetime.now().isoformat(),
        },
        "overall_score": overall_score,
        "overall_errors": {
            "substitutions": sorted(set(e["expected"] for e in all_errors.get("substitutions", []))),
            "deletions": sorted(set(all_errors.get("deletions", []))),
            "insertions": sorted(set(all_errors.get("insertions", []))),
        },
        "page_results": [
            {
                "page_num": pr["page_num"],
                "text": pr.get("text", ""),
                "score": pr.get("score", {}),
                "errors": pr.get("errors", {}),
            }
            for pr in page_results
        ],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return output_path
