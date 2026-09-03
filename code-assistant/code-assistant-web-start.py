"""
本地编程辅助工具 - Streamlit 可视化版
------------------------------------------------
在命令行版 code-assistant.py 的基础上，做成网页聊天界面：
- 左侧：像聊天一样输入需求 / 修改要求
- 右侧：当前代码 + 真实执行结果
- 支持历史版本列表，点击即可回退到任意一版
- 支持一键下载当前代码

依赖：
    pip install streamlit requests

前提：
    本地已启动 Ollama，且已 pull 好对应模型
    (ollama pull qwen2.5-coder:7b)

运行：
    streamlit run code-assistant-web.py
"""

import re
import subprocess
import sys
import tempfile
import os
import json
import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5-coder:7b"
MAX_RETRY = 2


# ---------------- 核心逻辑（与命令行版一致） ----------------

def call_ollama(prompt: str, model: str) -> str:
    """调用本地 Ollama API，非流式返回完整回复（网页里流式展示更复杂，先用简单版）"""
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("无法连接到 Ollama，请确认已执行 `ollama serve` 或 Ollama 已在后台运行。")
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            raise RuntimeError(
                f"模型 '{model}' 不存在，请检查名称是否正确，或先在命令行执行 `ollama pull {model}` 拉取该模型。"
            )
        raise RuntimeError(f"Ollama 请求失败：{e}")


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def run_code(code: str):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "[执行超时 30s，可能存在死循环或阻塞操作]"
    finally:
        os.unlink(tmp_path)


def build_initial_prompt(requirement: str) -> str:
    return (
        f"用Python实现以下需求，只输出完整可运行的代码（包含必要的测试用例，"
        f"用 print 或 assert 验证逻辑），代码放在一个 ```python 代码块中：\n\n"
        f"{requirement}"
    )


def build_retry_prompt(requirement: str, code: str, error: str) -> str:
    return (
        f"之前生成的Python代码执行时报错了，请修复问题并重新给出完整代码。\n\n"
        f"【原始需求】\n{requirement}\n\n"
        f"【之前的代码】\n```python\n{code}\n```\n\n"
        f"【实际报错信息】\n{error}\n\n"
        f"请只输出修复后的完整代码，放在一个 ```python 代码块中。"
    )


def build_modify_prompt(code: str, instruction: str) -> str:
    return (
        f"下面是当前的完整Python代码，请在此基础上按照【修改要求】进行修改，"
        f"不要重写不相关的部分，尽量保持原有结构和已实现的功能不变，"
        f"只输出修改后的完整代码（包含原有的测试用例，如需要可补充新的验证），"
        f"放在一个 ```python 代码块中：\n\n"
        f"【当前代码】\n```python\n{code}\n```\n\n"
        f"【修改要求】\n{instruction}"
    )


def generate_and_verify(prompt: str, model: str, requirement: str, log):
    """生成 -> 执行验证 -> 失败自动重试；log 是一个可调用对象，用来把过程信息写进网页界面"""
    attempt = 0
    code = ""
    while attempt <= MAX_RETRY:
        log(f"**第 {attempt + 1} 次生成（模型：{model}）...**")
        reply = call_ollama(prompt, model)
        code = extract_python_code(reply)

        success, output = run_code(code)

        if success:
            log(f"✅ 执行成功，输出：\n```\n{output.strip() or '(无输出)'}\n```")
            return code, True, output
        else:
            log(f"❌ 执行失败，报错：\n```\n{output}\n```")
            attempt += 1
            if attempt > MAX_RETRY:
                log(f"已自动重试 {MAX_RETRY} 次仍失败，已采用最后一次结果，你可以继续提修改要求让模型修。")
                return code, False, output
            log("正在根据报错信息自动重新生成...")
            prompt = build_retry_prompt(requirement, code, output)

    return code, False, output


# ---------------- Streamlit 界面 ----------------

st.set_page_config(page_title="本地编程辅助工具", layout="wide")

# 初始化 session state
if "history" not in st.session_state:
    st.session_state.history = []  # [{code, note}]
if "cur_index" not in st.session_state:
    st.session_state.cur_index = -1
if "requirement" not in st.session_state:
    st.session_state.requirement = ""
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # 展示用的过程日志/对话记录
if "last_output" not in st.session_state:
    st.session_state.last_output = ""

st.title("🛠 本地编程辅助工具（Ollama + 自动执行校验 + 对话式修改）")

with st.sidebar:
    st.header("设置")
    model = st.text_input("使用模型", value=DEFAULT_MODEL)
    st.caption("提示：可在命令行用 `ollama list` 查看本机已有模型")

    st.divider()
    st.header("历史版本")
    if st.session_state.history:
        for i, item in enumerate(st.session_state.history):
            label = f"v{i+1} - {item['note']}"
            if i == st.session_state.cur_index:
                st.markdown(f"**➡ {label}（当前）**")
            else:
                if st.button(label, key=f"hist_{i}"):
                    st.session_state.cur_index = i
                    st.rerun()
    else:
        st.caption("暂无历史版本")

    st.divider()
    if st.session_state.history and st.session_state.cur_index >= 0:
        current_code = st.session_state.history[st.session_state.cur_index]["code"]
        st.download_button(
            "下载当前代码", current_code, file_name="output.py", mime="text/x-python"
        )


def log_to_chat(msg: str):
    st.session_state.chat_log.append(("assistant", msg))


col_chat, col_code = st.columns([1, 1])

with col_chat:
    st.subheader("💬 对话")

    # 展示历史对话
    for role, msg in st.session_state.chat_log:
        with st.chat_message(role):
            st.markdown(msg)

    placeholder = (
        "首次使用：直接描述你的编程需求（例如：写一个LRU缓存，实现get/put）"
        if not st.session_state.history
        else "继续提修改要求（例如：加个capacity=0的边界处理 / 把注释改成英文）"
    )
    user_input = st.chat_input(placeholder)

    if user_input:
        st.session_state.chat_log.append(("user", user_input))

        try:
            if not st.session_state.history:
                # 第一轮：初始生成
                st.session_state.requirement = user_input
                prompt = build_initial_prompt(user_input)
                code, ok, output = generate_and_verify(
                    prompt, model, user_input, log_to_chat
                )
                st.session_state.history.append(
                    {"code": code, "note": "初始生成" if ok else "初始生成（有报错）"}
                )
                st.session_state.cur_index = len(st.session_state.history) - 1
            else:
                # 后续轮次：基于当前代码继续改
                current_code = st.session_state.history[st.session_state.cur_index]["code"]
                prompt = build_modify_prompt(current_code, user_input)
                code, ok, output = generate_and_verify(
                    prompt, model, st.session_state.requirement, log_to_chat
                )
                note = user_input if len(user_input) <= 20 else user_input[:20] + "..."
                st.session_state.history.append(
                    {"code": code, "note": note if ok else note + "（有报错）"}
                )
                st.session_state.cur_index = len(st.session_state.history) - 1

            st.session_state.last_output = output
        except RuntimeError as e:
            st.session_state.chat_log.append(("assistant", f"⚠️ {e}"))

        st.rerun()

with col_code:
    st.subheader("📄 当前代码")
    if st.session_state.history and st.session_state.cur_index >= 0:
        current_code = st.session_state.history[st.session_state.cur_index]["code"]
        st.code(current_code, language="python")
    else:
        st.caption("还没有代码，先在左边输入你的需求吧。")
