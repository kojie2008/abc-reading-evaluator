"""
Core pipeline orchestrator.

Wires together: fetcher → downloader → ASR → comparator → acoustic → reporter → publisher
"""

import os
import sys
from pathlib import Path

from .config import CDP_URL, ASR_MODEL, cleanup_data_dir
from .fetcher import fetch_opus_data, extract_content_list
from .downloader import download_all
from .asr import convert_to_wav, transcribe
from .comparator import compare, format_marked, classify_errors, compute_six_dimensions, generate_training
from .acoustic import acoustic_similarity, fluency_analysis
from .reporter import generate_html, generate_json, generate_friendly_summary, generate_group_message
from .publisher import publish


def _work_id(opus_info: dict) -> str:
    name = opus_info.get("name", "unknown")
    book = opus_info.get("book_info", {}).get("pictureBookName", "unknown")
    order = opus_info.get("opus_order", 0)
    return f"{name}_{book}_{order}"


def _aggregate(page_results: list) -> tuple[dict, dict]:
    total = {
        "correct_count": 0, "total_words": 0,
        "substitution_count": 0, "deletion_count": 0, "insertion_count": 0,
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

    tw = total["total_words"]
    if tw:
        accuracy = round((total["correct_count"] / tw) * 100, 2)
        wer = round((total["substitution_count"] + total["deletion_count"] + total["insertion_count"]) / tw * 100, 2)
    else:
        accuracy, wer = 0.0, 0.0

    overall = {
        "accuracy": accuracy, "wer": wer, **total,
        "student_word_count": sum(pr.get("score", {}).get("student_word_count", 0) for pr in page_results),
    }
    return overall, all_errors


async def run(share_url: str, keep_audio: bool = True, skip_publish: bool = False) -> dict:
    """
    Execute the full evaluation pipeline.
    """
    print("=" * 60)
    print("  ABC Reading 学生朗读评测系统 v3")
    print("=" * 60)

    # ── 1. Fetch ──
    print("\n[1/7] 📡 抓取作品数据…")
    opus_info = await fetch_opus_data(CDP_URL, share_url)
    pages = extract_content_list(opus_info)
    wid = _work_id(opus_info)
    student_name = opus_info.get("name", "未知")
    book_name = opus_info["book_info"]["pictureBookName"]
    book_level = opus_info["book_info"]["lowerLevelDesc"]
    print(f"  👤 {student_name}")
    print(f"  📖 {book_name} (Level {book_level})")
    print(f"  📄 有效朗读页: {len(pages)}")

    # ── 2. Download ──
    print(f"\n[2/7] ⬇️ 下载音频…")
    audio_paths = download_all(pages, wid)

    # ── 3. Load ASR ──
    print(f"\n[3/7] 🎤 加载 ASR 模型 ({ASR_MODEL})…")
    from .asr import get_model
    get_model()

    # ── 4 & 5. ASR + Compare ──
    print(f"\n[4/7] 🎯 语音识别 + [5/7] 文本对比…")
    page_results = []
    total_pages = len(pages)
    skipped_pages = 0
    all_texts = []

    for i, page in enumerate(pages, 1):
        pn = page["page_num"]
        text = page["text"]
        all_texts.append(text)
        stu = audio_paths.get(pn, {}).get("student", "")
        ref = audio_paths.get(pn, {}).get("reference", "")

        if not stu or not os.path.exists(stu):
            print(f"\n  [{i}/{total_pages}] 第{pn}页: ⚠️ 无学生音频，跳过")
            skipped_pages += 1
            page_results.append({
                "page_num": pn, "text": text, "asr_text": "",
                "student_audio_path": stu, "reference_audio_path": ref,
                "student_audio_url": page.get("student_audio_url", ""),
                "reference_audio_url": page.get("reference_audio_url", ""),
                "score": {"accuracy": 0, "wer": 0, "correct_count": 0,
                          "substitution_count": 0, "deletion_count": 0,
                          "insertion_count": 0, "total_words": 0, "student_word_count": 0},
                "errors": {"substitutions": [], "deletions": [], "insertions": []},
                "marked_text": "", "acoustic": None, "fluency": None,
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

        # ── Acoustic analysis ──
        acoustic = None
        fluency = None
        if os.path.exists(ref):
            try:
                print(f"  [{i}/{total_pages}] 第{pn}页: 🎵 声学分析…", end=" ", flush=True)
                acoustic = acoustic_similarity(wav, ref)
                fluency = fluency_analysis(wav, ref, text)
                print(f"相似度 {acoustic.get('similarity', 0)}%, "
                      f"流利度 {fluency.get('flow_score', 0)}%")
            except Exception as e:
                print(f"⚠️ 声学分析失败: {e}")

        page_results.append({
            "page_num": pn, "text": text, "asr_text": asr_text,
            "student_audio_path": stu, "reference_audio_path": ref,
            "student_audio_url": page.get("student_audio_url", ""),
            "reference_audio_url": page.get("reference_audio_url", ""),
            "score": sc,
            "errors": comp["errors"],
            "marked_text": marked,
            "acoustic": acoustic,
            "fluency": fluency,
        })

    # ── Aggregate ──
    overall_score, all_errors = _aggregate(page_results)

    # ── Error classification ──
    print(f"\n[6/7] 📊 错误分类 & 评分…")
    classified = classify_errors(all_errors["substitutions"])

    # Compute six-dimension score
    six_dim = compute_six_dimensions(
        overall_score, classified,
        all_errors["substitutions"],
        all_errors["deletions"],
        all_errors["insertions"],
        total_pages, skipped_pages,
    )

    # Override pronunciation with acoustic similarity if available
    acoustic_sims = [pr["acoustic"]["similarity"] for pr in page_results
                     if pr.get("acoustic")]
    fluency_scores = [pr["fluency"]["flow_score"] for pr in page_results
                      if pr.get("fluency")]

    if acoustic_sims:
        avg_acoustic = sum(acoustic_sims) / len(acoustic_sims)
        # Blend: 60% ASR text accuracy + 40% acoustic similarity
        blended_pron = overall_score["accuracy"] * 0.6 + avg_acoustic * 0.4
        six_dim["dimensions"]["pronunciation"]["score"] = round(blended_pron / 100 * 30, 1)
        six_dim["dimensions"]["pronunciation"]["acoustic_avg"] = round(avg_acoustic, 1)

    if fluency_scores:
        avg_fluency = sum(fluency_scores) / len(fluency_scores)
        # Use fluency score for the pausing dimension
        six_dim["dimensions"]["pausing"]["score"] = round(avg_fluency / 100 * 15, 1)
        six_dim["dimensions"]["pausing"]["fluency_avg"] = round(avg_fluency, 1)

    # Recalculate total
    six_dim["total"] = round(sum(
        d["score"] for d in six_dim["dimensions"].values()
    ), 1)
    six_dim["passed"] = six_dim["total"] >= 60 and (
        six_dim["dimensions"]["pronunciation"]["score"] +
        six_dim["dimensions"]["final_sound"]["score"]
    ) >= 18

    # Generate training material
    training = generate_training(
        classified, all_errors["substitutions"],
        all_errors["deletions"], all_errors["insertions"],
        all_texts,
    )

    # ── 7. Report ──
    print(f"\n[7/7] 📝 生成评测报告…")
    html_path = generate_html(
        opus_info, page_results, overall_score, all_errors,
        six_dim, classified, training,
    )
    json_path = generate_json(
        opus_info, page_results, overall_score, all_errors,
        six_dim, classified, training,
    )

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
    print(f"  📊 五维评分: {six_dim['total']}/{six_dim['max_total']} "
          f"{'✅通过' if six_dim['passed'] else '❌未通过'}")
    print(f"{'=' * 60}")

    # ── 朋友圈文案 ──
    report_url = publish_result.get("url", "") if not skip_publish and publish_result else ""
    summary = generate_friendly_summary(
        opus_info, overall_score, six_dim, report_url
    )
    print(f"\n{'=' * 60}")
    print("📱 朋友圈打卡文案 (可直接复制):")
    print(f"{'=' * 60}")
    print(summary)
    print(f"{'=' * 60}")

    result = {
        "opus_info": opus_info,
        "page_results": page_results,
        "overall_score": overall_score,
        "all_errors": all_errors,
        "six_dimension": six_dim,
        "classified_errors": classified,
        "training": training,
        "reports": {"html": html_path, "json": json_path},
    }

    # ── Publish ──
    if not skip_publish:
        print(f"\n📤 自动发布到 GitHub Pages…")
        pub_result = publish(html_path)
        if pub_result.get("url"):
            print(f"  🔗 永久链接: {pub_result['url']}")
        result["public_url"] = pub_result.get("url")

    # ── 朋友圈文案 ──
    pub_url = result.get("public_url", "")
    summary = generate_friendly_summary(
        opus_info, overall_score, six_dim, pub_url
    )
    print(f"\n{'=' * 60}")
    print("📱 朋友圈打卡文案 (可直接复制):")
    print(f"{'=' * 60}")
    print(summary)
    print(f"{'=' * 60}")

    # ── 家族群图文版 ──
    group_parts = generate_group_message(
        opus_info, overall_score, six_dim, pub_url
    )
    print(f"\n📱 家族群分享版 (可直接发送消息):")
    print(f"{'=' * 60}")
    print(f"先把链接转给群，然后发送以下文本：")
    for p in group_parts:
        if p["type"] == "text":
            print(f"\n{p['content']}")
        elif p["type"] == "link":
            print(f"\n🔗 {p['url']}")
    print(f"{'=' * 60}")

    if not keep_audio:
        cleanup_data_dir()


    if not keep_audio:
        cleanup_data_dir()

    return result
