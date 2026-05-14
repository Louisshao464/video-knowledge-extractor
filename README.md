# 中文视频知识抽取器 🎬→📝

把视频、音频内容自动转写成结构化的知识资产，适合沉淀到 Obsidian、Notion 等知识库。

## 安装

### 通过 ClawHub（推荐）

```bash
openclaw skills install video-knowledge-extractor
```

### 从 GitHub 克隆

```bash
git clone https://github.com/Louisshao464/video-knowledge-extractor.git
```

## 快速开始

```bash
# 从视频链接抽取知识
python3 scripts/process_media.py https://www.youtube.com/watch?v=xxx

# 处理本地文件
python3 scripts/process_media.py ./meeting.mp4

# 批量处理文件夹
python3 scripts/process_media.py ./videos/

# 处理播放列表
python3 scripts/process_media.py https://youtube.com/playlist?list=xxx
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `summary.md` | 内容摘要 |
| `notes.md` | 要点笔记 |
| `chapters.md` | 时间轴章节 |
| `knowledge.json` | 结构化知识数据 |
| `manifest.json` | 处理记录清单 |

## 处理流程

1. **识别输入** — 自动区分链接、本地文件、文件夹、播放列表
2. **提取音频** — 下载或直接读取媒体文件
3. **Whisper 转写** — 支持中英文等多种语言
4. **分块总结** — 长内容自动切片，避免上下文溢出
5. **LLM 后处理** — 生成摘要、要点、章节、行动项
6. **输出入库** — 固定格式 Markdown + JSON

## 依赖

- Python >= 3.10
- whisper（OpenAI Whisper 或 faster-whisper）
- requests

## 许可证

MIT
