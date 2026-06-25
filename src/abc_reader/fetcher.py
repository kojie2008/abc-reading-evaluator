"""
Data fetching — connect to browser CDP, capture API response.
"""

import asyncio
import json
from datetime import datetime
from typing import Any
from urllib.parse import urlparse, parse_qs

from playwright.async_api import async_playwright


def parse_share_url(url: str) -> dict:
    """
    Extract member_id and opus_id from an ABC Reading share URL.

    Example:
        https://abctime.com/prod/share/picturebook/?member_id=13046294&id=10191404
    Returns:
        {"opus_uid": int, "opus_id": int}
    """
    params = parse_qs(urlparse(url).query)
    opus_uid = int(params.get("member_id", [0])[0])
    opus_id = int(params.get("id", [0])[0])
    if not opus_uid or not opus_id:
        raise ValueError(f"Cannot parse member_id/id from URL: {url}")
    return {"opus_uid": opus_uid, "opus_id": opus_id}


async def fetch_opus_data(cdp_url: str, share_url: str) -> dict:
    """
    Open the share page in a CDP-connected browser and intercept the API response.
    Returns the full opus data dict.
    """
    params = parse_share_url(share_url)
    print(f"[抓取] opus_id={params['opus_id']}, member_id={params['opus_uid']}")

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(cdp_url)
        context = browser.contexts[0]
        page = next(
            (pg for pg in context.pages if "abctime" in pg.url or "picturebook" in pg.url),
            None,
        ) or await context.new_page()

        api_data: dict = {}

        async def on_response(response):
            nonlocal api_data
            if "/v5/study/opus_share_page" in response.url:
                try:
                    data = await response.json()
                    if data.get("code") == "200":
                        api_data = data["data"]
                        print(
                            f"[抓取] ✓ 学生: {api_data.get('name')}, "
                            f"绘本: {api_data['book_info']['pictureBookName']}, "
                            f"评分: {api_data.get('score')}"
                        )
                except Exception:
                    pass

        page.on("response", on_response)
        await page.goto(share_url, wait_until="networkidle")
        await asyncio.sleep(3)

        if not api_data:
            print("[抓取] API 未捕获，尝试刷新…")
            await page.reload(wait_until="networkidle")
            await asyncio.sleep(3)

        await browser.close()

    if not api_data:
        raise RuntimeError("无法获取作品 API 数据，请检查链接有效性")
    return api_data


def extract_content_list(api_data: dict) -> list[dict]:
    """
    Extract non-empty page entries from the API data.

    Returns a list of dicts:
        page_num, text, translation, student_audio_url, reference_audio_url
    """
    content_list = api_data.get("book_info", {}).get("contentList", [])
    video_urls = api_data.get("video_urls", [])

    pages = []
    for i, content in enumerate(content_list):
        text = (content.get("pageContent") or "").strip()
        if not text:
            continue

        pages.append(
            {
                "page_num": content.get("pageNum", i + 1),
                "text": text,
                "translation": (content.get("pageTranslate") or "").strip(),
                "student_audio_url": video_urls[i] if i < len(video_urls) else "",
                "reference_audio_url": content.get("pageContentAudio", ""),
            }
        )
    return pages
