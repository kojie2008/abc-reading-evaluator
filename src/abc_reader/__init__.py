"""
abc_reader — ABC Reading 学生朗读评测系统

流水线:
    分享链接 → 数据抓取 → 音频下载 → ASR 识别 → 逐词对比 → 评测报告
"""

__version__ = "2.0.0"

from .pipeline import run
