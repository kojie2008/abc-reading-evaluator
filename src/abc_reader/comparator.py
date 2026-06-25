"""
Word-level alignment and scoring using difflib.

Compares the ASR transcription against the ground-truth text.
"""

import difflib
import re
from typing import Literal

AlignmentTag = Literal["correct", "substitution", "deletion", "insertion"]
Alignment = list[tuple[AlignmentTag, str, str]]  # (tag, reference, hypothesis)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def align(reference: list[str], hypothesis: list[str]) -> Alignment:
    """
    Word-level alignment via difflib SequenceMatcher.

    Returns a list of (tag, ref_word, hyp_word) tuples.
    """
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


def score(aligned: Alignment) -> dict:
    """
    Compute scoring metrics from an alignment.

    Returns:
        accuracy, wer, correct/substitution/deletion/insertion counts, total_words
    """
    total = sum(1 for _, ref, _ in aligned if ref)
    correct = sum(1 for tag, _, _ in aligned if tag == "correct")
    subs = sum(1 for tag, _, _ in aligned if tag == "substitution")
    dels = sum(1 for tag, _, _ in aligned if tag == "deletion")
    inss = sum(1 for tag, _, _ in aligned if tag == "insertion")

    if total == 0:
        return {
            "accuracy": 0.0,
            "wer": 0.0,
            "correct_count": 0,
            "substitution_count": 0,
            "deletion_count": 0,
            "insertion_count": 0,
            "total_words": 0,
            "student_word_count": 0,
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


def extract_errors(aligned: Alignment) -> dict:
    """
    Extract detailed error lists.
    """
    errors = {"substitutions": [], "deletions": [], "insertions": []}
    for tag, ref, hyp in aligned:
        if tag == "substitution":
            errors["substitutions"].append({"expected": ref, "read_as": hyp})
        elif tag == "deletion":
            errors["deletions"].append(ref)
        elif tag == "insertion":
            errors["insertions"].append(hyp)
    return errors


def compare(original: str, transcription: str) -> dict:
    """
    One-page comparison: original text vs ASR transcription.

    Returns:
        {"aligned": [...], "score": {...}, "errors": {...}, ...}
    """
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
    """
    Generate a human-readable marked-up text.
        correct → as-is
        substitution → [read→expected]
        deletion → {expected}
        insertion → (read)
    """
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
