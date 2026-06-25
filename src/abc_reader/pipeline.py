"""
Core pipeline orchestrator.

Wires together: fetcher → downloader → ASR → comparator → reporter → publisher
"""

import os
import sys
from pathlib import Path

from .config import CDP_URL, ASR_MODEL, cleanup_data_dir
from .fetcher import fetch_opus_data, extract_content_list
from .downloader import download_all
from .asr import convert_to_wav, transcribe
from .comparator import compare, format_marked
from .reporter import generate_html, generate_json
from .publisher import publish


def _work_id(opus_info: dict) -> str:
    """Generate a unique work identifier."""
    name = opus_info.get("name", "unknown")
    book = opus_info.get("book_info", {}).get("pictureBookName", "unknown")
    order = opus_info.get("opus_order", 0)
    return f"{name}_{book}_{order}"


def _aggregate(page_results: list) -> tuple[dict, dict]:
    """
    Accumulate scores and errors across all pages.

    Returns (overall_score, all_errors).
    """
    total = {
        "correct_count": 0,
        "total_words": 0,
        "substitution_count": 0,
        "deletion_count": 0,
        "insertion_count": 0,
    }
    all_errors = {"substitutions": [], "deletions": [], "insertions": []}

    for pr in page_results:
        s = pr.get("score", {})
        for k in total:
            total[k] += s.get(k, 0)
        errs = pr.get("errors", {})
        all_errors["substitutions"].extend(errs.get("substitutions", []))
        all_errors["deletions"].extend(errs.get("deletions", []))
        all_errors["insertions"].extend(errs.get("insertions", []))

    total_words = total["total_words"]
    if total_words:
        accuracy = round((total["correct_count"] / total_words) * 100, 2)
        wer = round(
            (
                total["substitution_count"]
                + total["deletion_count"]
                + total["insertion_count"]
            )
            / total_words
            * 100,
            2,
        )
    else:
        accuracy = 0.0
        wer = 0.0

    overall = {
        "accuracy": accuracy,
        "wer": wer,
        **total,
        "student_word_count": sum(
            pr.get("score", {}).get("student_word_count", 0) for pr in page_results
        ),
    }
    return overall, all_errors


async def run(share_url: str, keep_audio: bool = True, skip_publish: bool = False) -> dict:
    """
    Execute the full evaluation pipeline.

    Args:
        share_url: ABC Reading share link.
        keep_audio: If False, delete downloaded audio after run.
        skip_publish: If True, skip GitHub Pages upload.

    Returns:
        dict with keys: opus_info, page_results, overall_score, all_errors, reports
    """
    print("=" * 60)
    print("  ABC Reading 学生朗读评测系统 v2")
    print("=" * 60)

    # ── 1. Fetch ──
    print("\n[1/6] 📡 抓取作品数据…")
    opus_info = await fetch_opus_data(CDP_URL, share_url)
    pages = extract_content_list(opus_info)
    wid = _work_id(opus_info)
    print(f"  👤 {opus_info.get('name')}")
    print(f"  📖 {opus_info['book_info']['pictureBookName']} "
          f"(Level {opus_info['book_info']['lowerLevelDesc']})")
    print(f"  📄 有效朗读页: {len(pages)}")

    # ── 2. Download ──
    print(f"\n[2/6] ⬇️ 下载音频…")
    audio_paths = download_all(pages, wid)

    # ── 3. Load ASR ──
    print(f"\n[3/6] 🎤 加载 ASR 模型 ({ASR_MODEL})…")
    from .asr import get_model
    get_model()

    # ── 4 & 5. ASR + Compare ──
    print(f"\n[4/6] 🎯 语音识别 + [5/6] 文本对比…")
    page_results = []
    total_pages = len(pages)

    for i, page in enumerate(pages, 1):
        pn = page["page_num"]
        text = page["text"]
        stu = audio_paths.get(pn, {}).get("student", "")

        if not stu or not os.path.exists(stu):
            print(f"\n  [{i}/{total_pages}] 第{pn}页: ⚠️ 无学生音频，跳过")
            page_results.append({
                "page_num": pn,
                "text": text,
                "asr_text": "",
                "score": {
                    "accuracy": 0, "wer": 0,
                    "correct_count": 0, "substitution_count": 0,
                    "deletion_count": 0, "insertion_count": 0,
                    "total_words": 0, "student_word_count": 0,
                },
                "errors": {"substitutions": [], "deletions": [], "insertions": []},
                "marked_text": "",
            })
            continue

        print(f"\n  [{i}/{total_pages}] 第{pn}页: 🔍 识别中…", end=" ", flush=True)
        wav = convert_to_wav(stu)
        asr_result = transcribe(wav)
        asr_text = asr_result.get("text", "")
        comp = compare(text, asr_text)
        marked = format_marked(comp["aligned"])

        sc = comp["score"]
        print(f"词数: {sc.get('student_word_count', 0)}, "
              f"准确率: {sc.get('accuracy', 0):.1f}%")

        page_results.append({
            "page_num": pn,
            "text": text,
            "asr_text": asr_text,
            "score": sc,
            "errors": comp["errors"],
            "marked_text": marked,
        })

    # ── Aggregate ──
    overall_score, all_errors = _aggregate(page_results)

    # ── 6. Report ──
    print(f"\n[6/6] 📝 生成评测报告…")
    html_path = generate_html(opus_info, page_results, overall_score, all_errors)
    json_path = generate_json(opus_info, page_results, overall_score, all_errors)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  ✅ 评测完成!")
    print(f"  📄 HTML: {html_path}")
    print(f"  📋 JSON: {json_path}")
    print(f"  总词数: {overall_score['total_words']} | "
          f"正确: {overall_score['correct_count']}")
    print(f"  替换: {overall_score['substitution_count']} | "
          f"漏读: {overall_score['deletion_count']} | "
          f"多读: {overall_score['insertion_count']}")
    print(f"  🎯 单词准确率: {overall_score['accuracy']:.1f}%")
    print(f"{'=' * 60}")

    result = {
        "opus_info": opus_info,
        "page_results": page_results,
        "overall_score": overall_score,
        "all_errors": all_errors,
        "reports": {"html": html_path, "json": json_path},
    }

    # ── Publish ──
    if not skip_publish:
        print(f"\n📤 自动发布到 GitHub Pages…")
        pub_result = publish(html_path)
        if pub_result.get("url"):
            print(f"  🔗 永久链接: {pub_result['url']}")
        result["public_url"] = pub_result.get("url")

    # ── Cleanup ──
    if not keep_audio:
        cleanup_data_dir()

    return result
