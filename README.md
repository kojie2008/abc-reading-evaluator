# ABC Reading 学生朗读评测系统 📖🎤

给定一个 ABC Reading 分享链接，自动完成朗读评测全流程，生成**永久可分享的评测报告**（GitHub Pages）。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
playwright install chromium

# 2. 启动 Chrome CDP
google-chrome --headless --remote-debugging-port=9222 &

# 3. 运行评测
abc-eval "https://abctime.com/prod/share/picturebook/?member_id=xxx&id=xxx"
```

## 评测流程

```
分享链接 → 数据抓取 → 音频下载 → ASR识别 → 逐词对比 → 评测报告
                                                 ↓
                                          GitHub Pages 自动发布
```

## 输出

| 产物 | 格式 | 用途 |
|------|------|------|
| HTML 报告 | `.html` | 可视化评测报告，微信可直接打开 |
| JSON 报告 | `.json` | 结构化数据，供程序分析 |
| 永久链接 | URL | 自动发布到 GitHub Pages，永不失效 |

## 命令行选项

```bash
abc-eval --help

# 跳过 GitHub Pages 发布
abc-eval --no-publish "https://..."

# 评测后删除音频文件
abc-eval --no-audio "https://..."
```

## 配置

通过环境变量配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ABC_ASR_MODEL` | `tiny` | Whisper 模型 (tiny/base/small/medium/large-v3) |
| `ABC_CDP_URL` | `http://localhost:9222` | Chrome DevTools 地址 |
| `ABC_GITHUB_TOKEN` | — | GitHub Personal Access Token |
| `ABC_GITHUB_REPO` | `kojie2008/abc-reading-reports` | 发布仓库 |
| `ABC_DATA_DIR` | `./data` | 数据目录 |

## 项目结构

```
abc-reader/
├── src/abc_reader/
│   ├── __init__.py       # 包入口
│   ├── __main__.py       # CLI 入口 (abc-eval 命令)
│   ├── pipeline.py       # 核心流水线编排 ⭐
│   ├── config.py         # 全局配置
│   ├── fetcher.py        # 浏览器数据抓取
│   ├── downloader.py     # 音频下载
│   ├── asr.py            # 语音识别
│   ├── comparator.py     # 文本对比 & 评分
│   ├── reporter.py       # 报告生成 (HTML+JSON)
│   └── publisher.py      # GitHub Pages 发布
├── data/                 # 运行时数据 (gitignored)
│   ├── downloads/        # 下载的音频
│   └── reports/          # 评测报告
├── tests/                # 测试
├── pyproject.toml        # 项目元信息 + entry point
├── requirements.txt
└── README.md
```

## 评测指标

| 指标 | 含义 |
|------|------|
| 单词准确率 | 正确识别的单词数 / 原文总词数 × 100% |
| WER | (替换+漏读+多读) / 原文总词数 × 100% |
| 替换 | 单词读成了另一个词 |
| 漏读 | 单词未朗读 |
| 多读 | 原文没有但学生读了 |
