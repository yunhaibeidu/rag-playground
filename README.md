# rag-playground

> 专注 RAG 相关实验，一个从零开始、逐步覆盖 RAG 各个环节的学习型项目。

个人 AI 学习实验仓库，monorepo 形式收纳多个学习性质的小项目。核心目的是理解 RAG（检索增强生成）、本地 LLM 工程实践等原理。其中 **paper-rag** 已做成可直接下载使用的桌面程序，欢迎试用和反馈。

## 项目列表

| 项目 | 状态 | 简介 |
|---|---|---|
| [paper-rag](./paper-rag) | 基本可用 | 基于本地 Ollama + Chroma 的论文问答 / 写作助手。有早期 Streamlit 网页版和当前主力的桌面程序版（PySide6，可打包成 exe），用真实文献练手 RAG 全流程 |
| [code-assistant](./code-assistant) | 实验中 | 本地代码助手的网页版原型 |
| character-chat | 规划中 | 基于二次元设定的可对话模型（尚未开始，暂无目录） |

## 目录结构

```
rag-playground/
├── paper-rag/
│   ├── streamlit版本/              # 早期版本：单项目、Streamlit 网页界面
│   │   ├── main_process1.py        # 扫描文档 → 切块 → embedding 入库 → ask()/draft()
│   │   └── app.py                  # Streamlit 界面，复用 main_process1 的逻辑
│   └── exe版本/                    # 当前主版本：多项目管理、PySide6 桌面界面
│       ├── 论文问答助手_PySide6版_已集成模型切换.py   # 程序入口（GUI）
│       ├── main_process1_多项目版.py                 # 核心逻辑：索引、混合检索、问答
│       └── 项目管理_db.py                            # SQLite 多项目管理
├── code-assistant/
│   └── code-assistant-web.py
└── README.md
```

## 快速开始

### 方式一：直接下载桌面程序（推荐）

到 [Releases](https://github.com/yunhaibeidu/rag-playground/releases) 页面下载最新的 exe，解压后双击运行，无需安装 Python。

前提：本机已安装 [Ollama](https://ollama.com) 并拉取模型：

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

程序运行后会在 exe 同级目录生成 `data/` 文件夹，论文、向量库、项目配置都放在里面。

### 方式二：从源码运行

1. 安装 Ollama 并拉取模型（同上）
2. 安装 Python 依赖（建议 Python ≥ 3.11，单独建一个虚拟环境）：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行其中一个版本：
   ```bash
   # 早期 Streamlit 网页版：先把 PDF 放进 paper-rag/streamlit版本/data/papers/
   streamlit run paper-rag/streamlit版本/app.py

   # 桌面程序版
   python paper-rag/exe版本/论文问答助手_PySide6版_已集成模型切换.py
   ```

具体说明见 [paper-rag/README.md](./paper-rag/README.md)。

## 为什么是 monorepo

单人学习项目数量不多，拆成多个仓库反而增加管理成本。放在一起也方便日后整理成一套连贯的学习记录。

## 技术栈（会随项目增加更新）

- 本地推理：Ollama（`llama3`、`nomic-embed-text`；桌面版会自动检测本机已安装的模型，可在界面内切换）
- 向量库：Chroma
- 检索：向量检索 + BM25 混合检索（桌面版）
- 框架：LangChain
- 界面：Streamlit（早期版）/ PySide6（桌面版）
- 语言：Python

## 关于"教程化"

目前每个子项目的 README 里会记录踩坑过程和关键设计决策，作为轻量级的学习笔记。如果内容积累到一定程度，可能会整理进 `docs/` 目录做成更系统的教程或时间线，暂不强求一步做完整。

## 注意事项

- 各子项目的原始数据（论文 PDF、Chroma 向量库目录、索引指纹文件等）已通过 `.gitignore` 排除，不会提交到仓库。
- exe 打包产物不进仓库，统一通过 Releases 发布。
- 全部推理在本地 Ollama 完成，不依赖任何云端 API，代码中也不会硬编码任何密钥。

## License

MIT（若后续引入需要遵守其他协议的第三方内容，会在对应子目录单独说明）
