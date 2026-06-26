"""
Report generation for ABC Reading Evaluator v3.0.
"""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ReportConfig


def _safe(text: str) -> str:
    s = re.sub(r"[^\x20-\x7E]", "", text).strip().replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "report"


def _banner_html(name: str, book: str, level: str, order: int, abctime_score: int,
                 accuracy: float, total_words: int, ts: datetime) -> str:
    return f"""<div class="banner">
  <div class="banner-stars">⭐ 🌟 ✨</div>
  <div class="banner-label">✦ 朗 读 评 测 报 告 ✦</div>
  <h1 class="banner-name">{name}</h1>
  <div class="banner-sub">《{book}》 · Level {level} · 第 {order} 个作品</div>
  <div class="banner-flex">
    {_ring_svg(accuracy, 110, 7)}
    <div class="banner-side">
      <div class="banner-big">{total_words}</div>
      <div class="banner-sm">原文总词数</div>
      <div class="banner-big" style="font-size:18px;margin-top:8px">{abctime_score} 分</div>
      <div class="banner-sm">ABC Reading 评分</div>
    </div>
  </div>
  <div class="banner-ts">{ts:%Y-%m-%d %H:%M}</div>
</div>"""


def _ring_svg(percent: float, size: int = 120, stroke: int = 8) -> str:
    r = (size - stroke) / 2
    cx = cy = size / 2
    circ = 2 * 3.14159 * r
    offset = circ * (1 - percent / 100)
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(255,255,255,0.15)" stroke-width="{stroke}"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#fff" stroke-width="{stroke}"
    stroke-linecap="round" stroke-dasharray="{circ}" stroke-dashoffset="{offset}"
    transform="rotate(-90, {cx}, {cy})"/>
  <text x="{cx}" y="{cy - 2}" text-anchor="middle" fill="#fff" font-size="{size * 0.22}" font-weight="bold">{percent:.0f}%</text>
  <text x="{cx}" y="{cy + 16}" text-anchor="middle" fill="rgba(255,255,255,0.7)" font-size="{size * 0.1}">准确率</text>
</svg>"""


def _bar(w: int, h: int, pct: float, color: str) -> str:
    return f"""<svg width="{w}" height="{h}" style="vertical-align:middle;border-radius:{h/2}px">
  <rect width="{w}" height="{h}" rx="{h/2}" fill="#e5e7eb"/>
  <rect width="{int(w * pct / 100)}" height="{h}" rx="{h/2}" fill="{color}"/>
</svg>"""


_DIM_COMMENTS = {
    "pronunciation": {
        "good": "发音清晰准确，单词读得很到位",
        "ok": "大部分发音正确，个别单词需要再听一下原音",
        "poor": "发音需要加强，建议多听原音跟读",
        "label": "发音准确率",
    },
    "final_sound": {
        "good": "尾音保留得很好，-ed、-s 词尾都能正确发音",
        "ok": "部分尾音有遗漏，注意句子末尾单词的尾部发音",
        "poor": "尾音遗漏较多，重点练习 -ed/-s/-ing 词尾",
        "label": "尾音保留",
    },
    "pausing": {
        "good": "朗读节奏自然流畅",
        "ok": "注意标点符号处的适当停顿",
        "poor": "需要多注意句号和逗号处的停顿",
        "label": "流畅性",
    },
    "clarity": {
        "good": "声音洪亮，每个单词都读得很清楚",
        "ok": "整体清晰度不错，继续加油",
        "poor": "鼓励大声朗读，字正腔圆",
        "label": "音量清晰度",
    },
    "completeness": {
        "good": "朗读完整，没有漏页漏行",
        "ok": "大部分页面都读了",
        "poor": "建议逐页完成朗读",
        "label": "朗读完整性",
    },
}


def _dimension_comment(dim_name: str, pct: float) -> str:
    cmt = _DIM_COMMENTS.get(dim_name, {})
    if pct >= 80:
        return cmt.get("good", "")
    elif pct >= 50:
        return cmt.get("ok", "")
    else:
        return cmt.get("poor", "")


def _training_suggestions(six: dict) -> str:
    dims = six.get("dimensions", {})
    suggestions = []
    dim_order = [
        ("pronunciation", "发音", "多听原音跟读，用自然拼读法记单词"),
        ("final_sound", "尾音", "放慢语速把尾音读清楚"),
        ("pausing", "停顿", "朗读时注意句号和逗号的停顿"),
        ("clarity", "清晰度", "鼓励大声朗读，避免含糊吞音"),
        ("completeness", "完整性", "逐页完成朗读，不跳页不漏行"),
    ]
    scored = []
    for key, label, tip in dim_order:
        d = dims.get(key)
        if d and d["max"] > 0:
            pct = d["score"] / d["max"] * 100
            scored.append((pct, key, label, tip))
    scored.sort(key=lambda x: x[0])
    total = six.get("total", 0)
    if total < 60:
        suggestions.append(
            "<div class='tip-row'><span class='tip-badge'>💪</span><span class='tip-text'><strong>整体建议</strong>：建议每天朗读15-20分钟，先听原音再跟读</span></div>"
        )
    for i, (pct, key, label, tip) in enumerate(scored[:3]):
        if pct < 75:
            emoji = "①②③④⑤"[i]
            suggestions.append(
                f"<div class='tip-row'><span class='tip-badge tip-{i+1}'>{emoji}</span><span class='tip-text'><strong>{label}</strong>：{tip}</span></div>"
            )
    if not suggestions:
        suggestions.append(
            "<div class='tip-row'><span class='tip-badge'>🎉</span><span class='tip-text'>表现很棒！继续加油</span></div>"
        )
    return f"""<div class="card"><h3>📈 综合提升建议</h3>
  {"".join(suggestions)}
</div>"""


def _score_card_html(dims: dict) -> str:
    rows = ""
    for k in ["pronunciation", "final_sound", "pausing", "clarity", "completeness"]:
        d = dims[k]
        pct = d["score"] / d["max"] * 100
        color = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 50 else "#ef4444"
        icon = d.get("icon", "📊")
        comment = _dimension_comment(k, pct)
        rows += f"""<div class="s-row">
  <div class="s-head">
    <span class="s-icon">{icon}</span>
    <span class="s-label">{d['label']}</span>
  </div>
  <div class="s-body">
    <div class="s-barline">
      <span class="s-val" style="color:{color}">{d['score']:.1f}/{d['max']}</span>
      {_bar(70, 6, pct, color)}
    </div>
    <div class="s-comment" style="color:{color}">{comment}</div>
  </div>
</div>"""
    return f"""<div class="card"><h3>📊 五维评分</h3>
  <div class="s-grid">{rows}</div>
</div>"""


def _pass_fail_html(six: dict) -> str:
    cls = "pass" if six["passed"] else "fail"
    label = "🎉 达到推荐标准" if six["passed"] else "💪 继续加油，下次更好"
    return f"""<div class="card result-{cls}">
  <div class="result-score">{six['total']:.0f}<span style="font-size:14px;color:rgba(255,255,255,0.7)">/100</span></div>
  <div class="result-label">{label}</div>
</div>"""


def _classified_errors_html(classified: dict) -> str:
    sections = ""
    cat_info = {
        "grammatical_-ed": ("🔊", "尾音 -ed", "#f59e0b"),
        "grammatical_-ing": ("🔊", "尾音 -ing", "#f59e0b"),
        "grammatical_-s/es": ("🔊", "尾音 -s/es", "#f59e0b"),
        "function_word": ("📖", "高频词", "#3b82f6"),
        "polysyllabic": ("📈", "多音节词", "#8b5cf6"),
        "vowel": ("🔤", "元音替换", "#ec4899"),
        "consonant": ("🔤", "辅音替换", "#14b8a6"),
        "other": ("❓", "其他错误", "#6b7280"),
    }
    for cat, info in sorted(classified.items()):
        data = classified[cat]
        if not data:
            continue
        icon, label, color = cat_info.get(cat, ("❓", cat, "#999"))
        words = data.get("word_list", [])[:12]
        word_tags = " ".join(f'<span class="wtag" style="border-color:{color}30;background:{color}10;color:{color}">{w}</span>'
                            for w in words)
        sections += f"""<div class="err-cat">
  <div class="err-cat-head">
    <span>{icon}</span>
    <span class="err-cat-label">{label}</span>
    <span class="err-cat-count">{data['count']}次</span>
  </div>
  <div class="err-cat-tags">{word_tags}</div>
</div>"""
    return f"""<div class="card"><h3>🔍 错误分类统计</h3>
  <div class="err-grid">{sections}</div>
</div>"""


def _audio_html(url: str, label: str, icon: str, color: str) -> str:
    if not url:
        return ""
    return f"""<div class="ap" style="border-color:{color}">
  <span class="ap-icon">{icon}</span>
  <span class="ap-label">{label}</span>
  <audio controls preload="metadata" style="width:100%;height:36px">
    <source src="{url}" type="audio/wav">
    <source src="{url}" type="audio/mpeg">
  </audio>
</div>"""


def _marked_text_html(errors: dict, text: str) -> str:
    words = text.strip().split()
    if not words:
        return text
    sub_words = {e["expected"] for e in errors.get("substitutions", [])}
    del_words = set(errors.get("deletions", []))
    result = []
    for w in words:
        w_clean = w.strip(".,!?;:'\"").lower()
        if w_clean in del_words:
            result.append(f'<span class="w-del">{w}</span>')
        elif w_clean in sub_words:
            read_as = ""
            for e in errors.get("substitutions", []):
                if e["expected"].lower() == w_clean:
                    read_as = e["read_as"]
                    break
            if read_as:
                result.append(f'<span class="w-err">{w}<span class="w-read">→{read_as}</span></span>')
            else:
                result.append(f'<span class="w-err">{w}</span>')
        else:
            result.append(f'<span class="w-correct">{w}</span>')
    return " ".join(result)


def _page_block_html(pr: dict) -> str:
    pn = pr["page_num"]
    text = pr.get("text", "")
    acc = pr.get("score", {}).get("accuracy", 0)
    errors = pr.get("errors", {})
    ref_url = pr.get("reference_audio_url", "")
    stu_url = pr.get("student_audio_url", "")

    n_sub = len(errors.get("substitutions", []))
    n_del = len(errors.get("deletions", []))
    n_ins = len(errors.get("insertions", []))

    badges = ""
    if n_sub:
        badges += f'<span class="badge sub">替换 {n_sub}</span> '
    if n_del:
        badges += f'<span class="badge del">漏读 {n_del}</span> '
    if n_ins:
        badges += f'<span class="badge ins">多读 {n_ins}</span> '

    acc_color = "#22c55e" if acc >= 80 else "#f59e0b" if acc >= 50 else "#ef4444"
    marked = _marked_text_html(errors, text)
    audio_block = _audio_html(ref_url, "原音", "🎧", "#667eea")
    audio_block += _audio_html(stu_url, "跟读", "🎙️", "#f093fb")

    return f"""<div class="page">
  <div class="page-hd">
    <div class="page-num">{pn}</div>
    <div class="page-info">
      <div class="page-acc" style="color:{acc_color}">{acc:.1f}%</div>
      {badges}
    </div>
  </div>
  <div class="page-body">
    <div class="page-text">{marked}</div>
    {audio_block}
  </div>
</div>"""


def _training_html(training: list) -> str:
    if not training:
        return ""
    blocks = ""
    for t in training:
        ttype = t.get("type", "")
        title = t.get("title", "")
        desc = t.get("description", "")
        body = ""

        if ttype == "final_sound":
            words = t.get("words", [])
            sentence = t.get("drill_sentence", "")
            body = f"""<div class="train-desc">{desc}</div>
            <div class="train-words">{"".join(f'<span class="tw">{w}</span>' for w in words)}</div>
            <div class="train-sentence">📝 {sentence}</div>"""

        elif ttype == "function_word":
            words = t.get("words", [])
            sentences = t.get("sentences", [])
            body = f"""<div class="train-desc">{desc}</div>
            <div class="train-words">{"".join(f'<span class="tw fw">{w["word"]}</span>' for w in words)}</div>"""
            if sentences:
                body += f"""<div class="train-sentences">{"".join(f'<div class="ts">🔊 {s}</div>' for s in sentences)}</div>"""

        elif ttype == "tricky_words":
            words = t.get("words", [])
            body = f"""<div class="train-desc">{desc}</div>
            <div class="train-grid">{"".join(
                f'<div class="tw-card"><strong>{w["word"]}</strong><span class="tw-err">读成: {", ".join(w["errors"][:2])}</span><span class="tw-n">×{w["count"]}</span></div>'
                for w in words
            )}</div>"""

        elif ttype == "minimal_pairs":
            pairs = t.get("pairs", [])
            body = f"""<div class="train-desc">{desc}</div>
            <div class="train-grid">{"".join(
                f'<div class="tp-card"><span class="tp-correct">✅ {p["correct"]}</span><span class="tp-wrong">❌ {p["wrong"]}</span><span class="tp-drill">{" · ".join(p["drills"][:6])}</span></div>'
                for p in pairs
            )}</div>"""

        elif ttype == "deletion_alert":
            words = t.get("words", [])
            tip = t.get("tip", "")
            body = f"""<div class="train-desc">{desc}</div>
            <div class="train-words">{"".join(f'<span class="tw da">{w["word"]}</span>' for w in words)}</div>
            <div class="train-tip">💡 {tip}</div>"""

        blocks += f"""<div class="train-block">
  <div class="train-title">{title}</div>
  {body}
</div>"""

    return f"""<div class="card"><h3>🏋️ 专项训练</h3>{blocks}</div>"""


def _disclaimer_html(disclaimer: str) -> str:
    return f"""<div class="card disclaimer">
  <h4>⚠️ 评分说明</h4>
  <p>{disclaimer}</p>
</div>"""


def generate_html(
    opus_info: dict,
    page_results: list,
    overall_score: dict,
    all_errors: dict,
    six_dimension: dict,
    classified_errors: dict,
    training: list[dict],
    output_path: str | None = None,
) -> str:
    name = opus_info.get("name", "未知")
    book = opus_info.get("book_info", {}).get("pictureBookName", "未知")
    level = opus_info.get("book_info", {}).get("lowerLevelDesc", "")
    order = opus_info.get("opus_order", 0)
    abctime_score = opus_info.get("score", 0)
    ts = datetime.now()

    if not output_path:
        safe = _safe(f"{name}_{book}")
        output_path = str(ReportConfig.dir / f"{safe}_{ts:%Y%m%d_%H%M%S}.html")

    acc = overall_score.get("accuracy", 0)
    total = overall_score.get("total_words", 0)

    # Build sections
    banner = _banner_html(name, book, level, order, abctime_score, acc, total, ts)
    score_card = _score_card_html(six_dimension["dimensions"])
    result_badge = _pass_fail_html(six_dimension)
    training_suggestions = _training_suggestions(six_dimension)
    err_classified = _classified_errors_html(classified_errors)
    pages = "\n".join(_page_block_html(pr) for pr in page_results)
    training_sec = _training_html(training)
    disclaimer = _disclaimer_html(six_dimension.get("disclaimer", ""))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>朗读评测报告 - {name}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;background:#f0f2f5;color:#333;padding:0;max-width:420px;margin:0 auto;min-height:100vh}}
.banner{{background:linear-gradient(135deg,#667eea 0%,#f093fb 50%,#f5576c 100%);padding:28px 20px 24px;position:relative;overflow:hidden}}
.banner-stars{{position:absolute;top:12px;right:16px;font-size:20px;opacity:.4}}
.banner-label{{font-size:11px;color:rgba(255,255,255,.6);letter-spacing:3px;margin-bottom:2px}}
.banner-name{{font-size:26px;color:#fff;font-weight:700;line-height:1.2}}
.banner-sub{{font-size:13px;color:rgba(255,255,255,.8);margin-top:2px}}
.banner-flex{{display:flex;align-items:center;gap:16px;margin-top:14px}}
.banner-side{{color:#fff}}
.banner-big{{font-size:32px;font-weight:700;line-height:1}}
.banner-sm{{font-size:11px;opacity:.7;margin-top:2px}}
.banner-ts{{margin-top:10px;font-size:11px;color:rgba(255,255,255,.5)}}
.content{{padding:10px 12px}}
.card{{background:#fff;border-radius:14px;padding:14px;margin-bottom:10px;box-shadow:0 1px 6px rgba(0,0,0,.06)}}
.card h3{{font-size:14px;color:#555;margin-bottom:10px;display:flex;align-items:center;gap:6px}}
.card h4{{font-size:13px;color:#666;margin-bottom:6px}}
.s-grid{{display:flex;flex-direction:column;gap:8px}}
.s-row{{display:flex;flex-direction:column;gap:2px;padding:6px 0;border-bottom:1px solid #f5f5f5}}
.s-row:last-child{{border-bottom:none}}
.s-head{{display:flex;align-items:center;gap:6px}}
.s-icon{{width:22px;text-align:center;font-size:13px}}
.s-label{{font-size:12px;color:#666;font-weight:600}}
.s-body{{display:flex;flex-direction:column;gap:1px}}
.s-barline{{display:flex;align-items:center;gap:6px}}
.s-val{{font-size:13px;font-weight:600;white-space:nowrap;width:52px;text-align:right;flex-shrink:0}}
.s-comment{{font-size:11px;line-height:1.4;padding-left:28px}}
.result-pass,.result-fail{{text-align:center;padding:16px}}
.result-pass{{background:linear-gradient(135deg,#22c55e,#16a34a);color:#fff}}
.result-fail{{background:linear-gradient(135deg,#f59e0b,#ea580c);color:#fff}}
.result-score{{font-size:40px;font-weight:700}}
.result-label{{font-size:14px;margin-top:4px}}
.tip-row{{display:flex;align-items:flex-start;gap:8px;padding:6px 0;border-bottom:1px solid #f5f5f5}}
.tip-row:last-child{{border-bottom:none}}
.tip-badge{{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:50%;background:#f0f2f5;font-size:11px;font-weight:700;flex-shrink:0}}
.tip-1{{background:#fee2e2;color:#ef4444}}
.tip-2{{background:#fef3c7;color:#f59e0b}}
.tip-3{{background:#dbeafe;color:#3b82f6}}
.tip-text{{font-size:12px;line-height:1.5;color:#555}}
.tip-text strong{{color:#333}}
.err-grid{{display:flex;flex-direction:column;gap:8px}}
.err-cat-head{{display:flex;align-items:center;gap:6px;font-size:13px;margin-bottom:4px}}
.err-cat-label{{font-weight:600;flex:1;font-size:12px;color:#444}}
.err-cat-count{{font-size:11px;color:#999;background:#f5f5f5;padding:1px 8px;border-radius:8px}}
.err-cat-tags{{display:flex;flex-wrap:wrap;gap:4px}}
.wtag{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;border:1px solid}}
.page{{background:#fff;border-radius:12px;margin-bottom:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.04)}}
.page-hd{{display:flex;align-items:center;padding:10px 12px;gap:8px;border-bottom:1px solid #f0f0f0}}
.page-num{{width:32px;height:32px;border-radius:50%;background:#eff6ff;color:#3b82f6;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;flex-shrink:0}}
.page-info{{flex:1;display:flex;flex-wrap:wrap;align-items:center;gap:4px}}
.page-acc{{font-size:13px;font-weight:600;margin-right:4px}}
.badge{{display:inline-block;padding:1px 6px;border-radius:8px;font-size:10px;font-weight:600;color:#fff}}
.badge.sub{{background:#f59e0b}}
.badge.del{{background:#ef4444}}
.badge.ins{{background:#8b5cf6}}
.page-body{{padding:10px 12px}}
.page-text{{font-size:13px;line-height:1.7;margin-bottom:8px;word-break:break-word}}
.w-correct{{color:#333}}
.w-err{{color:#f59e0b;font-weight:600}}
.w-read{{color:#bbb;font-weight:400;font-size:11px;text-decoration:line-through;margin-left:2px}}
.w-del{{color:#ef4444;text-decoration:line-through}}
.ap{{border:1.5px solid;border-radius:10px;padding:6px 10px;margin-bottom:6px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}}
.ap-icon{{font-size:14px}}
.ap-label{{font-size:11px;color:#666;flex-shrink:0}}
.ap audio{{flex:1;min-width:120px;height:32px;border-radius:6px}}
.train-block{{margin-top:10px;padding-top:10px;border-top:1px solid #f0f0f0}}
.train-block:first-child{{margin-top:0;padding-top:0;border-top:none}}
.train-title{{font-size:13px;font-weight:600;color:#444;margin-bottom:6px}}
.train-desc{{font-size:12px;color:#888;margin-bottom:6px}}
.train-words{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}}
.tw{{display:inline-block;padding:4px 10px;border-radius:14px;font-size:12px;background:#fef3c7;border:1px solid #f59e0b;color:#92400e}}
.tw.fw{{background:#dbeafe;border-color:#3b82f6;color:#1e40af}}
.tw.da{{background:#fee2e2;border-color:#ef4444;color:#991b1b}}
.train-sentence{{font-size:12px;color:#666;background:#f9fafb;padding:8px;border-radius:8px;margin-top:4px;line-height:1.6}}
.train-sentences{{margin-top:4px}}
.ts{{font-size:12px;color:#666;padding:4px 0;line-height:1.5}}
.train-grid{{display:flex;flex-direction:column;gap:6px}}
.tw-card{{display:flex;align-items:center;gap:6px;padding:6px 10px;background:#fef3c7;border-radius:10px;font-size:12px}}
.tw-card strong{{color:#92400e}}
.tw-err{{color:#bbb;font-size:11px;text-decoration:line-through;flex:1}}
.tw-n{{color:#f59e0b;font-weight:700;font-size:11px}}
.tp-card{{display:flex;align-items:center;gap:6px;padding:6px 10px;background:#f0fdf4;border-radius:10px;font-size:12px}}
.tp-correct{{color:#22c55e;font-weight:600}}
.tp-wrong{{color:#ef4444;font-weight:600}}
.tp-drill{{color:#888;font-size:11px;flex:1}}
.train-tip{{font-size:12px;color:#3b82f6;background:#eff6ff;padding:8px;border-radius:8px;margin-top:4px}}
.disclaimer{{background:#fefce8;border:1px solid #fde68a}}
.disclaimer p{{font-size:11px;color:#92400e;line-height:1.6}}
.footer{{text-align:center;color:#bbb;font-size:10px;padding:16px 0}}
</style>
</head>
<body>
{banner}
<div class="content">
{score_card}
{result_badge}
{training_suggestions}
{err_classified}
<h3 style="font-size:14px;color:#555;margin:10px 12px 6px;display:flex;align-items:center;gap:6px">📄 逐页分析</h3>
{pages}
{training_sec}
{disclaimer}
<div class="footer">ABC Reading Evaluator v3.0 · {ts:%Y-%m-%d %H:%M}</div>
</div>
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
    six_dimension: dict,
    classified_errors: dict,
    training: list[dict],
    output_path: str | None = None,
) -> str:
    name = opus_info.get("name", "unknown")
    book = opus_info.get("book_info", {}).get("pictureBookName", "unknown")
    ts = datetime.now()

    if not output_path:
        safe = _safe(f"{name}_{book}")
        output_path = str(ReportConfig.dir / f"{safe}_{ts:%Y%m%d_%H%M%S}.json")

    classified_summary = {}
    for cat, info in classified_errors.items():
        if info:
            classified_summary[cat] = {
                "count": info["count"],
                "unique_words": info["unique_words"],
                "word_list": info["word_list"],
            }

    report = {
        "meta": {
            "student_name": name,
            "book_name": book,
            "book_level": opus_info.get("book_info", {}).get("lowerLevelDesc", ""),
            "opus_order": opus_info.get("opus_order", 0),
            "abctime_score": opus_info.get("score", 0),
            "total_pages": len(page_results),
            "generated_at": ts.isoformat(),
            "evaluator_version": "3.0",
        },
        "overall_score": overall_score,
        "six_dimension": six_dimension,
        "classified_errors": classified_summary,
        "training": [
            {
                "type": str(t.get("type", "")),
                "title": str(t.get("title", "")),
                "description": str(t.get("description", "")),
                "data": t.get("words") or t.get("pairs") or t.get("sentences") or [],
            }
            for t in training
        ],
        "page_results": [
            {
                "page_num": int(pr["page_num"]),
                "text": str(pr.get("text", "")),
                "asr_text": str(pr.get("asr_text", "")),
                "score": {k: float(v) if isinstance(v, (int, float)) else v for k, v in (pr.get("score") or {}).items()},
                "errors": pr.get("errors", {}),
                "acoustic": {k: float(v) if hasattr(v, "__float__") else v for k, v in (pr.get("acoustic") or {}).items()} if pr.get("acoustic") else None,
                "fluency": {k: float(v) if hasattr(v, "__float__") else v for k, v in (pr.get("fluency") or {}).items()} if pr.get("fluency") else None,
            }
            for pr in page_results
        ],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return output_path


def generate_friendly_summary(
    opus_info: dict,
    overall_score: dict,
    six_dimension: dict,
    report_url: str = "",
) -> str:
    name = opus_info.get("name", "宝贝")
    book = opus_info.get("book_info", {}).get("pictureBookName", "")
    level = opus_info.get("book_info", {}).get("lowerLevelDesc", "")
    ts = datetime.now()
    total = overall_score.get("total_words", 0)
    correct = overall_score.get("correct_count", 0)
    total_score = six_dimension.get("total", 0)

    if total_score >= 80:
        compliment = "读得太棒了！发音标准，流利自然，继续加油！🎉"
    elif total_score >= 60:
        compliment = "读得很不错！大部分单词都读对了，继续坚持练习会越来越好！👍"
    elif total_score >= 40:
        compliment = "读得真棒！每次都看到你的进步，坚持朗读的你超厉害！🌟"
    else:
        compliment = "敢于开口就是最大的进步！多听原音慢慢跟读🌟"

    dims = six_dimension.get("dimensions", {})
    strengths = []
    for key, label in [("pronunciation","发音"), ("pausing","流利度"), ("clarity","清晰度"), ("completeness","完整性")]:
        d = dims.get(key)
        if d and d["max"] > 0 and d["score"] / d["max"] * 100 >= 60:
            strengths.append(label)
    strength_text = "、".join(strengths) if strengths else "敢于开口"

    lines = [
        f"📚 {name} 的英语朗读打卡",
        "",
        f"今天读了《{book}》（Level {level}）",
        f"✅ {total}个单词中读对了{correct}个",
        f"五维综合评分: {total_score:.0f}/100",
        f"可圈可点：{strength_text}",
        "",
        compliment,
        "",
        "每天一点点，进步看得见 💕",
    ]
    if report_url:
        lines.extend(["", f"🔗 完整报告: {report_url}"])
    lines.append(f"{ts:%m/%d %H:%M}")
    return "\n".join(lines)
