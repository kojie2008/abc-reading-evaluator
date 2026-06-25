"""
Speech recognition using faster-whisper.

Globally caches the model for reuse across pages.
"""

import os
import subprocess
from typing import Optional

from .config import ASR_MODEL

# Lazy-loaded model singleton
_model: Optional["WhisperModel"] = None  # noqa: F821


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        print(f"[ASR] 加载模型: {ASR_MODEL} …")
        _model = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
        print("[ASR] ✅ 模型加载完成")
    return _model


def convert_to_wav(input_path: str) -> str:
    """
    Convert audio to 16 kHz mono WAV (optimal for whisper).
    Returns the path to the converted file (caches on disk).
    """
    output = input_path + "_16k.wav"
    if os.path.exists(output):
        return output
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", output,
            ],
            capture_output=True,
            check=True,
        )
    except Exception as e:
        print(f"[ASR] ⚠ 转换失败 ({input_path}): {e}")
        return input_path
    return output


def transcribe(audio_path: str, language: str = "en") -> dict:
    """
    Transcribe audio with faster-whisper.

    Returns:
        {"text": str, "segments": list, "language": str, "duration": float}
    """
    if not os.path.exists(audio_path):
        return {"text": "", "segments": [], "language": language, "error": "文件不存在"}

    try:
        model = get_model()
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        seg_list = []
        full_text = ""
        for seg in segments:
            seg_list.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            full_text += seg.text + " "

        return {
            "text": full_text.strip(),
            "segments": seg_list,
            "language": info.language if info else language,
            "duration": info.duration if info else 0,
        }
    except Exception as e:
        print(f"[ASR] ✗ 识别失败: {e}")
        return {"text": "", "segments": [], "language": language, "error": str(e)}
