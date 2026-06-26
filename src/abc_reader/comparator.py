"""
Word-level alignment, scoring, error classification, and training generation.

Compares ASR transcription against ground-truth text.
Produces:
  - Word-level alignment (correct/substitution/deletion/insertion)
  - Scoring metrics (accuracy, WER)
  - Error classification (phonetic, grammatical, function words)
  - Targeted training material generation
"""

import difflib
import re

# ── IPA Phonetics ──
_ENG_TO_IPA = None


def _get_ipa(word: str) -> str:
    """Get IPA pronunciation. Falls back to empty string if unavailable."""
    global _ENG_TO_IPA
    if _ENG_TO_IPA is None:
        try:
            from eng_to_ipa import convert as _c
            _ENG_TO_IPA = _c
        except ImportError:
            _ENG_TO_IPA = lambda w: ""
    try:
        return _ENG_TO_IPA(word) or ""
    except Exception:
        return ""
from collections import Counter
from typing import Literal, Any

AlignmentTag = Literal["correct", "substitution", "deletion", "insertion"]
Alignment = list[tuple[AlignmentTag, str, str]]  # (tag, reference, hypothesis)


# ── Helpers ──

def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


# ── Alignment ──

def align(reference: list[str], hypothesis: list[str]) -> Alignment:
    """Word-level alignment via difflib SequenceMatcher."""
    matcher = difflib.SequenceMatcher(None, reference, hypothesis)
    result: Alignment = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                result.append(("correct", reference[k], hypothesis[j1 + (k - i1)]))
        elif tag == "replace":
            ref_len = i2 - i1
            hyp_len = j2 - j1
            for k in range(min(ref_len, hyp_len)):
                result.append(("substitution", reference[i1 + k], hypothesis[j1 + k]))
            if ref_len > hyp_len:
                for k in range(hyp_len, ref_len):
                    result.append(("deletion", reference[i1 + k], ""))
            else:
                for k in range(ref_len, hyp_len):
                    result.append(("insertion", "", hypothesis[j1 + k]))
        elif tag == "delete":
            for k in range(i1, i2):
                result.append(("deletion", reference[k], ""))
        elif tag == "insert":
            for k in range(j1, j2):
                result.append(("insertion", "", hypothesis[k]))
    return result


# ── Scoring ──

def score(aligned: Alignment) -> dict:
    """Compute scoring metrics from alignment."""
    total = sum(1 for _, ref, _ in aligned if ref)
    correct = sum(1 for tag, _, _ in aligned if tag == "correct")
    subs = sum(1 for tag, _, _ in aligned if tag == "substitution")
    dels = sum(1 for tag, _, _ in aligned if tag == "deletion")
    inss = sum(1 for tag, _, _ in aligned if tag == "insertion")

    if total == 0:
        return {
            "accuracy": 0.0, "wer": 0.0,
            "correct_count": 0, "substitution_count": 0,
            "deletion_count": 0, "insertion_count": 0,
            "total_words": 0, "student_word_count": 0,
        }

    wer = (subs + dels + inss) / total
    return {
        "accuracy": round(max(0, (correct / total) * 100), 2),
        "wer": round(wer * 100, 2),
        "correct_count": correct,
        "substitution_count": subs,
        "deletion_count": dels,
        "insertion_count": inss,
        "total_words": total,
        "student_word_count": sum(1 for _, _, hyp in aligned if hyp),
    }


# ── Error Classification ──

# Common function words (high-frequency, often misread)
_FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "has", "have",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "can", "shall", "not", "no", "so", "if", "as", "than", "that",
    "this", "these", "those", "it", "its", "he", "she", "they", "we", "you",
    "my", "your", "his", "her", "our", "their", "me", "him", "us", "them",
}

_ENDING_PATTERNS = [
    (r"ed$", "-ed"),   # past tense
    (r"ing$", "-ing"),  # present participle
    (r"s$", "-s/es"),   # plural / 3rd person
    (r"er$", "-er"),    # comparative
    (r"est$", "-est"),  # superlative
    (r"ly$", "-ly"),    # adverb
    (r"tion$", "-tion"),# noun suffix
    (r"ness$", "-ness"),# noun suffix
]

# Approximate syllable count heuristic
def _syllable_count(word: str) -> int:
    word = word.lower().strip(".,!?;:'\"")
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    return max(1, count)


def _has_long_vowel_pattern(word: str) -> bool:
    """检测单词中是否有长元音或双元音拼写模式。"""
    w = word.lower().strip(".,!?;:'\"")
    patterns = [
        # 长元音常见拼写
        r'ee',   # see, tree, green
        r'ea',   # sea, read, please
        r'oo',   # moon, soon
        r'oa',   # boat, coat
        r'ai',   # rain, train
        r'ay',   # say, play
        r'ie',   # lie, tie (in certain patterns)
        r'igh',  # high, light, night
        r'ow',   # know, slow（长元音 O）
        r'ue',   # blue, true
        r'ew',   # new, few
        r'ou',   # out, cloud（双元音）
        r'oi',   # oil, join
        r'oy',   # toy, boy
        r'are',  # care, share
        r'ere',  # here, there
        r'ire',  # fire, hire
        r'ore',  # more, sore
        r'ure',  # pure, cure
        r'ear',  # ear, fear
        r'air',  # air, hair
        r'eer',  # deer, cheer
        # 元音 + 沉默 e 模式 (CVCE)
        r'a.e$',
        r'e.e$',
        r'i.e$',
        r'o.e$',
        r'u.e$',
    ]
    for pat in patterns:
        if re.search(pat, w):
            return True
    return False


# 功能词黑名单——这些即使读错了也不是


def classify_error(expected: str, read_as: str) -> str:
    """
    Classify a substitution error into a category.

    Categories:
      - final_sound: -ed/-es/-ing 尾音遗漏或错误
      - function_word: 冠词/介词/代词读错
      - grammatical: 语法形态变化（单复数、时态）
      - polysyllabic: 多音节词读错（重音/音节问题）
      - vowel: 元音替换
      - consonant: 辅音替换
      - other: 其他
    """
    e = expected.lower().strip()
    r = read_as.lower().strip()

    # Function words (determiners, prepositions, pronouns)
    if e in _FUNCTION_WORDS:
        return "function_word"

    # Check if student dropped a grammatical ending
    for pat, label in _ENDING_PATTERNS:
        if re.search(pat, e) and not re.search(pat, r):
            return f"grammatical_{label}"

    # Multi-syllable words tend to have stress issues
    if _syllable_count(e) >= 3 and e != r:
        return "polysyllabic"

    # Simple heuristic: if they share same length and only differ in vowels vs consonants
    if len(e) == len(r):
        vowel_diff = sum(1 for a, b in zip(e, r) if a != b and a in "aeiou")
        if vowel_diff >= 1:
            return "vowel"
        return "consonant"

    return "other"


def classify_errors(substitutions: list[dict]) -> dict:
    """
    Classify all substitution errors into categories.
    Returns counts and detailed lists per category.
    """
    categories = {
        "grammatical_-ed": [], "grammatical_-ing": [], "grammatical_-s/es": [],
        "grammatical_-er": [], "grammatical_-est": [], "grammatical_-ly": [],
        "grammatical_-tion": [], "grammatical_-ness": [],
        "function_word": [], "polysyllabic": [], "vowel": [], "consonant": [],
        "other": [],
    }

    for sub in substitutions:
        cat = classify_error(sub["expected"], sub["read_as"])
        if cat not in categories:
            cat = "other"
        categories[cat].append(sub)

    # Build summary
    summary = {}
    for cat, items in categories.items():
        if items:
            count = len(items)
            unique_words = list(set(item["expected"] for item in items))
            summary[cat] = {
                "count": count,
                "unique_words": len(unique_words),
                "examples": items[:8],
                "word_list": unique_words[:15],
            }

    return summary


def extract_errors(aligned: Alignment) -> dict:
    """Extract detailed error lists from alignment."""
    errors = {"substitutions": [], "deletions": [], "insertions": []}
    for tag, ref, hyp in aligned:
        if tag == "substitution":
            errors["substitutions"].append({"expected": ref, "read_as": hyp})
        elif tag == "deletion":
            errors["deletions"].append(ref)
        elif tag == "insertion":
            errors["insertions"].append(hyp)
    return errors


# ── Six-dimension scoring ──

def compute_six_dimensions(
    overall_score: dict,
    classified_errors: dict,
    substitutions: list[dict],
    deletions: list,
    insertions: list,
    total_pages: int,
    skipped_pages: int,
) -> dict:
    """
    Compute the 6-dimension evaluation score.

    Dimensions:
      1. 发音准确率 (Pronunciation Accuracy) 30% — word accuracy scaled
      2. 尾音保留 (Final Sound Retention) 20% — retention of -ed/-es/-ing
      3. 单词重音 (Word Stress) 15% — polysyllabic word handling
      4. 流畅性 (Fluency) 20% — deletions/insertions as proxy
      5. 音量清晰 (Volume Clarity) 10% — estimated from ASR confidence
      6. 完整性 (Completeness) 10% — page coverage

    ⚠️ 发音准确率、尾音保留率、重音正确率为基于文本难度的预估分，仅供参考。
    """
    """
    Compute the 5-dimension evaluation score.

    Dimensions:
      1. 发音准确率 (30分) — 基于准确率、声学相似度加权
      2. 尾音保留 (20分) — -ed/-es/-ing 保留比例
      3. 流畅性 (20分) — 多读/漏读比例
      4. 音量清晰 (15分) — ASR 置信度代理
      5. 完整性 (10分) — 页覆盖率

    总分范围 0-100，与 ABC Reading 原生评分保持较好一致性。
    准确率66%→总分约70，80%→约80，90%+→约90-95。
    """
    accuracy = overall_score.get("accuracy", 0) / 100.0

    # ── 1. 发音准确率 (30分) ──
    # 66%准确率→约25分，80%→27分，90%→29分
    # accuracy^0.6 更温和：66%→0.78→23.5，80%→0.87→26.2，95%→0.97→29.1
    pron_score = round(min(accuracy ** 0.6 * 30 + 2, 30), 1)

    # ── 2. 尾音保留 (20分) ──
    final_sound_errors = 0
    final_sound_total = 0
    for sub in substitutions:
        e = sub["expected"].lower().strip()
        r = sub["read_as"].lower().strip()
        for pat, _ in _ENDING_PATTERNS:
            if re.search(pat, e):
                final_sound_total += 1
                if not re.search(pat, r):
                    final_sound_errors += 1
                break

    if final_sound_total > 0:
        fs_rate = 1.0 - (final_sound_errors / final_sound_total)
    else:
        fs_rate = 0.7
    # 基础10 + (留存率×10)，即30%留存→13分，60%→16分，100%→20分
    tail_score = round(10 + fs_rate * 10, 1)

    # ── 3. 流畅性 (20分) ──
    total_student = overall_score.get("student_word_count", 0) or 1
    ins_rate = len(insertions) / total_student  # 多读比例
    del_rate = len(deletions) / max(total_student, 1)  # 漏读比例
    flu_rate = 1.0 - min(ins_rate * 2 + del_rate * 1.5, 0.95)
    # 低错误→18-20分，中等→14-17，高→10-14
    pause_score = round(max(10, flu_rate * 20), 1)

    # ── 4. 音量清晰 (15分) ──
    # 基础10 + 准确率映射，90%→13，100%→15
    vol_score = round(min(10 + accuracy * 5, 15), 1)

    # ── 5. 完整性 (10分) ──
    if total_pages > 0:
        comp_rate = 1.0 - (skipped_pages / total_pages)
    else:
        comp_rate = 1.0
    comp_score = round(comp_rate * 10, 1)

    total = round(pron_score + tail_score + pause_score + vol_score + comp_score, 1)
    passed = total >= 60 and (pron_score + tail_score) >= 18

    return {
        "dimensions": {
            "pronunciation": {
                "score": pron_score,
                "max": 30,
                "label": "发音准确率",
                "icon": "🎯",
            },
            "final_sound": {
                "score": tail_score,
                "max": 20,
                "label": "尾音保留",
                "icon": "🔊",
            },
            "pausing": {
                "score": pause_score,
                "max": 20,
                "label": "流畅性",
                "icon": "⏸️",
            },
            "clarity": {
                "score": vol_score,
                "max": 15,
                "label": "音量清晰",
                "icon": "🔉",
            },
            "completeness": {
                "score": comp_score,
                "max": 10,
                "label": "完整性",
                "icon": "📋",
            },
        },
        "total": total,
        "max_total": 100,
        "passed": passed,
        "disclaimer": "⚠️ 本评分为AI基于语音识别和声学对比的自动估算，仅供参考。建议家长对照原声逐句检查，重点关注错误较多的单词和尾音。",
    }


# ── Training Material Generation ──

# Common minimal pairs for phonetic training
_MINIMAL_PAIRS = {
    ("trunk", "truck"): ["trunk", "truck", "stuck", "struck", "luck", "pluck", "duck"],
    ("ship", "sheep"): ["ship", "sheep", "chip", "cheap", "lip", "leap", "slip", "sleep"],
    ("fill", "feel"): ["fill", "feel", "hill", "heal", "ill", "eel", "mill", "meal"],
    ("bat", "bet"): ["bat", "bet", "cat", "set", "fat", "get", "hat", "let"],
    ("bed", "bad"): ["bed", "bad", "red", "rad", "head", "had", "said", "sad"],
    ("thin", "tin"): ["thin", "tin", "think", "tink", "thank", "tank", "thick", "tick"],
    ("very", "every"): ["very", "every", "vary", "even", "veil", "evil"],
    ("walk", "work"): ["walk", "work", "talk", "tork", "call", "core"],
    ("thought", "taught"): ["thought", "taught", "bought", "caught", "fought", "brought"],
    ("live", "leave"): ["live", "leave", "give", "believe", "sit", "seat"],
}

# For function word training
_FUNCTION_WORD_SENTENCES = {
    "the": "The cat sat on the mat. The dog ran to the park.",
    "a": "A bird flew over a tree. I saw a car and a bus.",
    "an": "An apple a day. An elephant is an animal.",
    "in": "The book is in the bag. She lives in a big city.",
    "on": "The cup is on the table. He sits on the chair.",
    "at": "Look at the stars. She is at home.",
    "to": "Go to school. I want to eat.",
    "and": "Apples and bananas. Run and jump.",
    "of": "The color of the sky. A cup of tea.",
    "for": "This is for you. Thanks for coming.",
}


def generate_training(
    classified_errors: dict,
    substitutions: list[dict],
    deletions: list,
    insertions: list,
    original_texts: list[str],
) -> list[dict]:
    """
    Generate targeted training drills based on error patterns.

    Returns a list of training units:
      { "type": str, "title": str, "items": [{word, correct, wrong?}], "description": str }
    """
    training = []

    # ── 1. Final Sound Training (尾音训练) ──
    for suffix in ["-ed", "-ing", "-s/es"]:
        cat_key = f"grammatical_{suffix}"
        items = classified_errors.get(cat_key, {})
        if items and items.get("examples"):
            words = [s["expected"] for s in items["examples"]]
            training.append({
                "type": "final_sound",
                "category": suffix,
                "title": f"🔊 尾音训练: {suffix}",
                "description": f"你有 {items['count']} 次漏读或读错 {suffix} 结尾。试试下面这些词：",
                "words": list(set(words))[:10],
                "drill_sentence": f"请慢读以下单词，注意{suffix}尾音不要漏掉：{' · '.join(list(set(words))[:8])}",
            })

    # ── 2. Function Word Training (功能词训练) ──
    fw_items = classified_errors.get("function_word", {})
    if fw_items and fw_items.get("examples"):
        fw_words = Counter(e["expected"] for e in fw_items["examples"])
        top_fw = fw_words.most_common(5)
        sentences = [_FUNCTION_WORD_SENTENCES.get(w, "")
                     for w, _ in top_fw if w in _FUNCTION_WORD_SENTENCES]
        training.append({
            "type": "function_word",
            "title": "📖 高频词专项训练",
            "description": f"你有 {fw_items['count']} 次读错高频功能词（冠词/介词/代词）。这些词虽小但很重要！",
            "words": [{"word": w, "count": c} for w, c in top_fw],
            "sentences": [s for s in sentences if s],
        })

    # ── 3. Tricky Words Exercise (疑难词汇训练) ──
    # 找真正读不准的长音词/生词：排除功能词，取>=2音节或含长元音拼写的
    tricky_details = {}
    for sub in substitutions:
        e = sub["expected"].lower().strip()
        r = sub["read_as"].lower().strip()
        if e in _FUNCTION_WORDS:
            continue
        if _syllable_count(e) < 2 and not _has_long_vowel_pattern(e):
            continue
        if e not in tricky_details:
            tricky_details[e] = {"word": sub["expected"], "errors": [], "count": 0}
        if r not in tricky_details[e]["errors"]:
            tricky_details[e]["errors"].append(r)
        tricky_details[e]["count"] += 1

    tricky_words = sorted(tricky_details.values(), key=lambda x: -x["count"])[:8]
    tricky_words = [t for t in tricky_words if t["count"] >= 2]
    if tricky_words:
        training.append({
            "type": "tricky_words",
            "title": "🎯 长音 / 难词纠音训练",
            "description": "以下是你发音不准的长音词和生词——点击🔊听标准读音：",
            "words": [
                {
                    "word": w["word"],
                    "phonetic": _get_ipa(w["word"]),
                    "errors": list(w["errors"])[:3],
                    "count": w["count"],
                }
                for w in tricky_words
            ],
        })

    # ── 4. Phonetic Minimal Pairs (易混淆语音对) ──
    pairs_found = []
    for sub in substitutions:
        e = sub["expected"].lower().strip()
        r = sub["read_as"].lower().strip()
        for (w1, w2), pair_words in _MINIMAL_PAIRS.items():
            if (e == w1 and r == w2) or (e == w2 and r == w1):
                pairs_found.append((e, r, pair_words))

    if pairs_found:
        # Deduplicate
        seen = set()
        unique_pairs = []
        for e, r, pw in pairs_found:
            key = tuple(sorted([e, r]))
            if key not in seen:
                seen.add(key)
                unique_pairs.append({"correct": e, "wrong": r, "drills": pw})

        training.append({
            "type": "minimal_pairs",
            "title": "🔤 易混淆音对对比训练",
            "description": "你有以下容易混淆的发音对——逐个对比练习：",
            "pairs": unique_pairs,
        })

    # ── 5. Deletion Warning (漏读提醒) ──
    if deletions:
        del_counter = Counter(deletions)
        top_dels = del_counter.most_common(5)
        training.append({
            "type": "deletion_alert",
            "title": "⏭️ 漏读提醒",
            "description": "以下是容易漏读的词——朗读时请放慢速度，确保每个词都读到：",
            "words": [{"word": w, "count": c} for w, c in top_dels],
            "tip": "建议：朗读时手指指读，每读完一个词才移到下一个。",
        })

    return training


# ── Main entry ──

def compare(original: str, transcription: str) -> dict:
    """Full comparison between original text and ASR transcription."""
    ref_tokens = tokenize(original)
    hyp_tokens = tokenize(transcription)
    aligned = align(ref_tokens, hyp_tokens)
    return {
        "aligned": aligned,
        "score": score(aligned),
        "errors": extract_errors(aligned),
        "ref_tokens": ref_tokens,
        "hyp_tokens": hyp_tokens,
    }


def format_marked(aligned: Alignment) -> str:
    """Marked-up text: correct=plain, sub=[hyp→ref], del={ref}, ins=(hyp)."""
    words = []
    for tag, ref, hyp in aligned:
        if tag == "correct":
            words.append(ref)
        elif tag == "substitution":
            words.append(f"[{hyp}→{ref}]")
        elif tag == "deletion":
            words.append(f"{{{ref}}}")
        elif tag == "insertion":
            words.append(f"({hyp})")
    return " ".join(words)
