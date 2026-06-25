"""
CLI entry point: python -m abc_reader <share_url> [options]
"""

import argparse
import asyncio
import sys

from .pipeline import run


def cli():
    parser = argparse.ArgumentParser(
        description="ABC Reading 学生朗读评测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    abc-eval "https://abctime.com/prod/share/picturebook/?member_id=xxx&id=xxx"
    abc-eval --keep-audio --no-publish "https://abctime.com/..."
        """,
    )
    parser.add_argument("url", help="ABC Reading 分享链接")
    parser.add_argument(
        "--keep-audio", action="store_true", default=True,
        help="保留下载的音频文件（默认保留）",
    )
    parser.add_argument(
        "--no-audio", dest="keep_audio", action="store_false",
        help="评测完成后删除音频文件",
    )
    parser.add_argument(
        "--no-publish", action="store_true",
        help="跳过 GitHub Pages 发布",
    )
    return parser


def main():
    parser = cli()
    args = parser.parse_args()

    try:
        result = asyncio.run(run(args.url, keep_audio=args.keep_audio, skip_publish=args.no_publish))
        url = result.get("public_url")
        if url:
            print(f"\n✅ 永久链接: {url}")
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
