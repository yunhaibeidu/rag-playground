import streamlit as st
import main_process1 as core  # 直接复用main_process1.py里已经跑通的建库逻辑和ask()函数

st.set_page_config(page_title="论文问答助手", page_icon="📚", layout="wide")

# ========== 侧边栏:显示已入库论文 + 清空历史按钮 ==========
with st.sidebar:
    st.header("📚 已入库论文")
    for p in core.papers:
        st.markdown(f"- {p['source']}")

    st.divider()
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        # main_process1模块里的chat_history/chat_summary是模块级全局变量,
        # 直接在这里重置,下一次调用ask()时指代消解就不会再参考旧历史。
        core.chat_history = []
        core.chat_summary = ""
        st.rerun()

st.title("🧠 论文问答助手")
st.caption("基于本地Ollama + Chroma向量库的RAG问答,只依据已入库论文内容回答")

# ========== 用session_state维护网页会话里的展示消息 ==========
# 注意:这里的messages只用于网页界面展示历史对话气泡,
# 真正参与指代消解/prompt拼接的是main_process1.chat_history,
# 两者是分开的,但内容逻辑上是同步增长的。
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 查看参考来源片段"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[来源{i} - {src['source']}]**")
                    st.markdown(src["content"][:300] + "...")

# ========== 输入框 ==========
user_input = st.chat_input("输入你的问题...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("检索论文内容并生成回答中..."):
            try:
                # 记录调用前的历史长度,便于之后单独取回这一轮检索到的来源
                history_before = list(core.chat_history)

                # 直接调用main_process1.py里原封不动的ask()函数,
                # 建库、检索、指代消解、多轮历史压缩等逻辑完全复用,不重复写一遍。
                answer = core.ask(user_input)

                # ask()内部print的检索来源信息在网页端看不到,
                # 这里重新做一次检索只是为了把来源片段展示在界面上(不影响已生成的answer)。
                query = core.get_search_query(user_input, history_before)
                results = core.vectorstore.similarity_search(query, k=3)
                sources = [
                    {"source": doc.metadata.get("source", "未知来源"), "content": doc.page_content}
                    for doc in results
                ]
            except Exception as e:
                answer = f"出错了,这一轮没有得到回答:{e}\n\n如果是连接错误,检查一下Ollama是否还在运行(ollama serve)。"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("📎 查看参考来源片段"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**[来源{i} - {src['source']}]**")
                    st.markdown(src["content"][:300] + "...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
