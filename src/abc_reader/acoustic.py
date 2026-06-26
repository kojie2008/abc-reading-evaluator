"""
Acoustic analysis engine — phonetic comparison and fluency evaluation.

Features:
  1. Acoustic similarity — MFCC + DTW alignment between student and reference audio
  2. Fluency analysis — speaking rate, pause patterns, rhythm consistency
  3. Word-level segment extraction using forced alignment

Dependencies: librosa, numpy, scipy
"""

import os
import numpy as np
import librosa
from scipy.spatial.distance import cdist
from typing import Optional


# ── Constants ──
SAMPLE_RATE = 16000
N_MFCC = 13
FRAME_LENGTH = 1024
HOP_LENGTH = 512


# ── Feature Extraction ──

def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    import os
    """Load audio file, convert to mono, resample to target sample rate."""
    if not os.path.exists(path):
        return np.array([])
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def extract_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract MFCC features. Returns (n_mfcc, n_frames)."""
    if len(y) < sr * 0.1:  # too short
        return np.zeros((N_MFCC, 1))
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    # Normalize per coefficient
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
    return mfcc


def extract_pitch(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract pitch contour using librosa pyin. Returns (n_frames,) or empty."""
    if len(y) < sr * 0.1:
        return np.array([])
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=2093, sr=sr,
                                           frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        f0[~voiced_flag] = 0
        return f0
    except Exception:
        return np.array([])


def extract_energy(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract RMS energy contour. Returns (n_frames,)."""
    if len(y) < sr * 0.1:
        return np.zeros(10)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    return rms[0]


# ── VAD (Voice Activity Detection) ──

def vad_segments(y: np.ndarray, sr: int = SAMPLE_RATE,
                 energy_threshold: float = 0.01,
                 min_silence_duration: float = 0.3) -> list[dict]:
    """
    Detect speech/silence segments using energy-based VAD.

    Returns:
        [{"start": float, "end": float, "type": "speech"|"silence"}, ...]
    """
    if len(y) == 0:
        return []

    # Frame-level energy
    hop_time = HOP_LENGTH / sr
    energy = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    is_speech = energy > (energy.mean() * energy_threshold)

    # Merge frames into segments
    segments = []
    current_type = "speech" if is_speech[0] else "silence"
    start_frame = 0

    for i in range(1, len(is_speech)):
        new_type = "speech" if is_speech[i] else "silence"
        if new_type != current_type:
            segments.append({
                "start": start_frame * hop_time,
                "end": i * hop_time,
                "type": current_type,
            })
            current_type = new_type
            start_frame = i

    segments.append({
        "start": start_frame * hop_time,
        "end": len(is_speech) * hop_time,
        "type": current_type,
    })

    # Merge short silence gaps (< min_silence_duration) into speech
    merged = []
    for seg in segments:
        if merged and seg["type"] == "silence" and (seg["end"] - seg["start"]) < min_silence_duration:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)

    return merged


# ── DTW Alignment ──

def dtw_distance(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    # x: (n_mfcc, n_frames_x), y: (n_mfcc, n_frames_y)
    x_t = x.T  # (n_frames_x, n_mfcc)
    y_t = y.T  # (n_frames_y, n_mfcc)

    n, m = x_t.shape[0], y_t.shape[0]

    # Compute cost matrix
    cost = cdist(x_t, y_t, metric="cosine")
    cost = np.nan_to_num(cost, nan=1.0)

    # DTW without band constraint (too restrictive when durations differ a lot)
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0

    # Use Sakoe-Chiba band: 60% of max length
    band = max(n, m) * 3 // 5
    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m + 1, i + band)
        for j in range(j_start, j_end):
            d[i, j] = cost[i - 1, j - 1] + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])

    dist = d[n, m]
    if np.isinf(dist):
        # Fallback: try without band
        d2 = np.full((n + 1, m + 1), np.inf)
        d2[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(max(1, i - n), min(m + 1, i + m)):
                d2[i, j] = cost[i - 1, j - 1] + min(d2[i - 1, j], d2[i, j - 1], d2[i - 1, j - 1])
        dist = d2[n, m]
    
    # Normalize by path length
    normalizer = n + m
    dist = dist / normalizer if normalizer > 0 else 1.0

    # Backtrack
    i, j = n, m
    path_x, path_y = [], []
    while i > 0 or j > 0:
        path_x.append(i - 1)
        path_y.append(j - 1)
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([d[i - 1, j - 1], d[i - 1, j], d[i, j - 1]])
            if step == 0:
                i -= 1
                j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1

    return dist, np.array(path_x[::-1]), np.array(path_y[::-1])

def acoustic_similarity(student_path: str, reference_path: str) -> dict:
    """
    Compare student audio vs reference audio using acoustic features.

    Returns:
        {
            "similarity": float (0-100),
            "mfcc_distance": float,
            "pitch_correlation": float,
            "duration_ratio": float (student_dur / ref_dur),
            "frames_aligned": int,
        }
    """
    y_stu = load_audio(student_path)
    y_ref = load_audio(reference_path)

    if len(y_stu) == 0 or len(y_ref) == 0:
        return {"similarity": 50, "mfcc_distance": 0, "pitch_correlation": 0,
                "duration_ratio": 1.0, "frames_aligned": 0}
        return {"similarity": 0, "mfcc_distance": 1.0, "pitch_correlation": 0,
                "duration_ratio": 0, "frames_aligned": 0}

    # MFCC features
    mfcc_stu = extract_mfcc(y_stu)
    mfcc_ref = extract_mfcc(y_ref)

    if mfcc_stu.shape[1] < 3 or mfcc_ref.shape[1] < 3:
        return {"similarity": 50, "mfcc_distance": 0.5, "pitch_correlation": 0,
                "duration_ratio": len(y_stu) / max(len(y_ref), 1), "frames_aligned": 0}

    # DTW alignment
    dist, path_x, path_y = dtw_distance(mfcc_stu, mfcc_ref)

    # Convert DTW distance to similarity (0-100)
    similarity = max(0, min(100, (1.0 - dist) * 100))

    # Pitch correlation
    pitch_stu = extract_pitch(y_stu)
    pitch_ref = extract_pitch(y_ref)

    pitch_corr = 0.0
    if len(pitch_stu) > 10 and len(pitch_ref) > 10:
        # Interpolate to same length
        n = min(len(pitch_stu), len(pitch_ref), 500)
        p_stu = pitch_stu[:n]
        p_ref = pitch_ref[:n]
        # Only use voiced frames
        voiced = (p_stu > 0) & (p_ref > 0)
        if voiced.sum() > 5:
            p_stu_v = p_stu[voiced]
            p_ref_v = p_ref[voiced]
            # Normalize
            p_stu_v = (p_stu_v - p_stu_v.mean()) / (p_stu_v.std() + 1e-8)
            p_ref_v = (p_ref_v - p_ref_v.mean()) / (p_ref_v.std() + 1e-8)
            pitch_corr = float(np.corrcoef(p_stu_v, p_ref_v)[0, 1])
            pitch_corr = max(0, (pitch_corr + 1) / 2) * 100  # map [-1,1] to [0,100]

    duration_ratio = len(y_stu) / max(len(y_ref), 1)

    return {
        "similarity": float(round(float(similarity), 1)),
        "mfcc_distance": float(round(float(dist), 4)),
        "pitch_correlation": float(round(float(pitch_corr), 1)),
        "duration_ratio": float(round(float(duration_ratio), 2)),
        "frames_aligned": int(len(path_x)),
    }


# ── Fluency Analysis ──

def fluency_analysis(student_path: str, reference_path: str,
                     original_text: str | None = None) -> dict:
    """
    Analyze reading fluency from student audio.

    Metrics:
      - speaking_rate: syllables per second
      - speed_variance: std of syllable duration (lower = more consistent)
      - pause_frequency: pauses per minute
      - avg_pause_duration: average silence duration between speech segments
      - longest_pause: longest silence
      - duration_ratio: student audio length / reference audio length
      - flow_score: combined fluency score (0-100)

    Returns dict with all metrics.
    """
    y_stu = load_audio(student_path)
    y_ref = load_audio(reference_path)

    if len(y_stu) == 0:
        return {"speaking_rate": 0, "speed_variance": 0, "pause_frequency": 0,
                "avg_pause_duration": 0, "longest_pause": 0, "duration_ratio": 1.0,
                "flow_score": 50, "speech_segments": 0}

    # VAD segmentation
    segments = vad_segments(y_stu)
    speech_segs = [s for s in segments if s["type"] == "speech"]
    silence_segs = [s for s in segments if s["type"] == "silence"]

    total_duration = len(y_stu) / SAMPLE_RATE
    total_speech = sum(s["end"] - s["start"] for s in speech_segs)
    total_silence = sum(s["end"] - s["start"] for s in silence_segs)

    n_speech = len(speech_segs)
    n_silence = len(silence_segs)

    # 1. Speaking rate (estimated syllables from energy peaks)
    if total_speech > 0:
        # Rough syllable detection: energy envelope peaks
        energy = extract_energy(y_stu)
        # Smooth and find peaks
        from scipy.signal import find_peaks
        if len(energy) > 5:
            smoothed = np.convolve(energy, np.ones(5) / 5, mode="same")
            peaks, _ = find_peaks(smoothed, height=smoothed.mean() * 0.5, distance=5)
            est_syllables = max(1, len(peaks))
            speaking_rate = est_syllables / total_speech
        else:
            speaking_rate = 0
            est_syllables = 0
    else:
        speaking_rate = 0
        est_syllables = 0

    # 2. Speed variance (variation in inter-peak intervals)
    speed_variance = 0
    if est_syllables > 3:
        try:
            energy = extract_energy(y_stu)
            if len(energy) > 5:
                smoothed = np.convolve(energy, np.ones(5) / 5, mode="same")
                peaks, props = find_peaks(smoothed, height=smoothed.mean() * 0.5, distance=5)
                if len(peaks) > 2:
                    intervals = np.diff(peaks) * (HOP_LENGTH / SAMPLE_RATE)
                    speed_variance = float(np.std(intervals))
        except Exception:
            pass

    # 3. Pause metrics
    pause_frequency = (n_silence / total_duration * 60) if total_duration > 0 else 0
    pause_durations = [s["end"] - s["start"] for s in silence_segs]
    avg_pause = np.mean(pause_durations) if pause_durations else 0
    longest_pause = max(pause_durations) if pause_durations else 0
    duration_ratio = total_duration / max(len(y_ref) / SAMPLE_RATE, 0.1) if len(y_ref) > 0 else 1.0

    # 4. Flow score (0-100)
    # Ideal: speaking rate 2-4 syllables/sec, low variance, few short pauses
    rate_score = min(100, (speaking_rate / 3.0) * 100) if speaking_rate > 0 else 50
    variance_score = max(0, 100 - speed_variance * 50)  # lower variance = better
    pause_score = max(0, 100 - pause_frequency * 5)  # fewer pauses = better
    pause_dur_score = max(0, 100 - avg_pause * 100)  # shorter pauses = better
    rhythm_score = min(100, (1.0 / max(duration_ratio, 0.1)) * 100) if duration_ratio < 2.0 else 50

    flow_score = round(
        rate_score * 0.30 +
        variance_score * 0.15 +
        pause_score * 0.20 +
        pause_dur_score * 0.15 +
        rhythm_score * 0.20,
        1,
    )

    return {
        "speaking_rate": float(round(speaking_rate, 2)),
        "speed_variance": float(round(speed_variance, 3)),
        "pause_frequency": float(round(pause_frequency, 1)),
        "avg_pause_duration": float(round(float(avg_pause), 2)),
        "longest_pause": float(round(longest_pause, 2)),
        "duration_ratio": float(round(duration_ratio, 2)),
        "flow_score": float(round(float(flow_score), 1)),
        "speech_segments": int(n_speech),
        "estimated_syllables": int(est_syllables),
        "total_duration": float(round(total_duration, 2)),
    }


# ── Word-level segment extraction ──

def extract_word_timing(y: np.ndarray, text: str,
                        sr: int = SAMPLE_RATE) -> list[dict]:
    """
    Estimate word-level timing from audio using energy-based segmentation.

    This is a simplified forced alignment. For production use, consider
    Montreal Forced Aligner or CTC-based alignment.

    Returns:
        [{"word": str, "start": float, "end": float}, ...]
    """
    words = text.lower().strip().split()
    if not words or len(y) == 0:
        return [{"word": w, "start": 0, "end": 0} for w in words]

    # VAD segments
    segs = vad_segments(y)
    speech_segs = [s for s in segs if s["type"] == "speech"]

    if not speech_segs:
        return [{"word": w, "start": 0, "end": 0} for w in words]

    # Distribute words evenly across speech segments
    total_speech_time = sum(s["end"] - s["start"] for s in speech_segs)
    word_duration = total_speech_time / max(len(words), 1)

    result = []
    current_seg = 0
    time_in_seg = 0

    for i, word in enumerate(words):
        # Find which speech segment contains this word
        while current_seg < len(speech_segs):
            seg = speech_segs[current_seg]
            seg_dur = seg["end"] - seg["start"]
            if time_in_seg + word_duration <= seg_dur or current_seg == len(speech_segs) - 1:
                start = seg["start"] + time_in_seg
                end = min(start + word_duration, seg["end"])
                result.append({"word": word, "start": round(start, 2), "end": round(end, 2)})
                time_in_seg += word_duration
                break
            else:
                current_seg += 1
                time_in_seg = 0

    return result
