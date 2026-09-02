from pypdf import PdfReader
from docx import Document  # 需要 pip install python-docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
import re
import os
import shutil
import gc
import time
import hashlib
import json

# ========== 第一步:扫描文件夹,自动发现所有要入库的文件 ==========
papers_dir = "G:/test-localai/papers"

SUPPORTED_EXTS = {".pdf", ".docx", ".txt"}

source_names = {
    "Plasticity of the binding pocket in peptide.pdf": "DtpB结合口袋论文(Cell Reports 2023)",
}


def scan_papers(folder):
    papers = []
    for filename in sorted(os.listdir(folder)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTS:
            continue
        full_path = os.path.join(folder, filename)
        source = source_names.get(filename, os.path.splitext(filename)[0])
        papers.append({"path": full_path, "source": source})
    return papers


papers = scan_papers(papers_dir)
print(f"扫描到 {len(papers)} 个文件待入库: {[p['source'] for p in papers]}\n")


def extract_text_from_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            full_text += (text or "") + "\n"
        return full_text

    elif ext == ".docx":
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)

    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"暂不支持的文件格式: {ext} (文件: {path})")


embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="llama3")

persist_dir = "G:/test-localai/chroma_db"
# 指纹文件:记录上次建库时,每个论文文件的内容md5。
# 用来判断"papers文件夹内容有没有变化",没变化就跳过整个重建流程。
manifest_path = "G:/test-localai/indexed_manifest.json"


def compute_manifest(papers):
    """
    给当前papers列表算一份指纹:{文件路径: 文件内容的md5}。
    用内容md5而不是文件名+修改时间,是因为后者在复制粘贴、
    Windows时间戳误差等情况下容易失真,md5是直接对字节内容算的,
    只要文件内容真的没变,md5就必然不变,判断更可靠。
    """
    manifest = {}
    for paper in papers:
        with open(paper["path"], "rb") as f:
            content = f.read()
        manifest[paper["path"]] = hashlib.md5(content).hexdigest()
    return manifest


def load_old_manifest(path):
    """读取上次记录的指纹,第一次运行时文件不存在就当作空字典"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(path, manifest):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def safe_remove_dir(path):
    """
    安全删除向量库目录。
    Windows下Chroma的sqlite文件句柄如果还被之前的vectorstore变量占用着,
    直接shutil.rmtree容易报错或卡死,必须先释放引用+触发垃圾回收+
    留一点时间让系统真正释放文件句柄,再删,并加ignore_errors兜底。
    """
    global vectorstore
    if "vectorstore" in globals():
        del vectorstore
    gc.collect()
    time.sleep(0.5)
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


current_manifest = compute_manifest(papers)
old_manifest = load_old_manifest(manifest_path)
need_rebuild = (current_manifest != old_manifest) or (not os.path.exists(persist_dir))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

if need_rebuild:
    print("检测到论文内容有变化(或向量库不存在),开始重建向量库...")

    safe_remove_dir(persist_dir)
    print(f"已清空旧向量库: {persist_dir}")

    # ========== 第二步:逐篇论文提取文字 + 切块,并记录来源 ==========
    all_chunks = []
    all_metadatas = []

    for paper in papers:
        full_text = extract_text_from_file(paper["path"])
        chunks = splitter.split_text(full_text)
        print(f"{paper['source']}: 提取 {len(full_text)} 字符, 切成 {len(chunks)} 块")
        all_chunks.extend(chunks)
        all_metadatas.extend([{"source": paper["source"]} for _ in chunks])

    print(f"\n所有论文总共切成 {len(all_chunks)} 块")

    # ========== 第三步:embedding入库 ==========
    vectorstore = Chroma.from_texts(
        all_chunks,
        embeddings,
        metadatas=all_metadatas,
        persist_directory=persist_dir
    )
    print(f"已完成embedding并存入Chroma,共 {len(all_chunks)} 个文本块")
    print(f"向量库保存在: {persist_dir}")

    # 重建成功后,把这次的指纹存下来,供下次运行比对
    save_manifest(manifest_path, current_manifest)

else:
    # 内容没变,直接连接已有向量库,跳过提取/切块/embedding,秒开
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    print("papers文件夹内容未变化,直接加载已有向量库,跳过重建")


# ========== 中文检测 ==========
def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))


# ========== 多轮对话历史 ==========
# 注意:这两个模块级变量在被import时会被Streamlit那边通过
# main_process1.chat_history 这种方式直接读写,以维持网页会话的对话状态。
chat_history = []
chat_summary = ""

MAX_RAW_TURNS = 6
KEEP_RECENT_TURNS = 3


def format_history(history, summary):
    """
    这个函数目前只在get_search_query()里被调用,用于指代消解。
    生成最终答案的prompt不再使用它,避免历史摘要里的旧结论
    被当成对当前问题的答案直接复述出来。
    """
    parts = []
    if summary:
        parts.append(f"[更早之前对话的摘要]\n{summary}")
    if history:
        lines = []
        for i, (q, a) in enumerate(history, 1):
            lines.append(f"第{i}轮 - 问:{q}\n第{i}轮 - 答:{a}")
        parts.append("[最近几轮对话原文]\n" + "\n\n".join(lines))
    if not parts:
        return "(无)"
    return "\n\n".join(parts)


def compress_history_if_needed():
    global chat_history, chat_summary

    if len(chat_history) <= MAX_RAW_TURNS:
        return

    old_turns = chat_history[:-KEEP_RECENT_TURNS]
    old_text = "\n\n".join(
        [f"问:{q}\n答:{a}" for q, a in old_turns]
    )

    prompt = f"""请把以下对话历史压缩成一段简洁的摘要,用中文书写。
要求:保留所有关键事实、数字、结论、专业术语,去掉寒暄和重复内容,尽量精简。

[已有的早期摘要(可能为空)]
{chat_summary if chat_summary else "(无)"}

[需要合并进摘要的新增历史]
{old_text}

请输出合并后的新摘要(直接输出摘要正文,不要加"摘要:"之类的前缀):"""

    chat_summary = llm.invoke(prompt).strip()
    chat_history = chat_history[-KEEP_RECENT_TURNS:]

    print(f"[历史已压缩] 早期内容已合并进摘要,chat_history现在只保留最近{KEEP_RECENT_TURNS}轮")


def get_search_query(question, history):
    """
    合并"指代消解"和"翻译成英文"为一步。
    chat_summary只在这里用于帮助理解指代,不会进入最终答案生成环节。
    """
    history_text = format_history(history, chat_summary)
    prompt = f"""You are rewriting a user's question into a search query for a biochemistry/pharmacology paper about peptide transporters, molecular docking, and binding pockets. Use accurate scientific terminology (e.g. "peptide" not "dot", "docking" not "connection", "binding pocket", "tripeptide", "dipeptide", etc.).

Below is the recent conversation history (may be in Chinese) and the user's latest question (may be in Chinese, and may contain pronouns like "this number" / "it" that depend on the history to understand).

Rewrite the latest question into a single, standalone, complete question in ENGLISH, suitable for semantic search against the paper. Only output the rewritten English question, nothing else.

[History]
{history_text}

[Latest question]
{question}

[Rewritten English search question]
"""
    rewritten = llm.invoke(prompt).strip()
    return rewritten


def ask(question, k=3, source_filter=None):
    global chat_history

    query = get_search_query(question, chat_history)
    print(f"[检索用英文query]: {query}")

    if source_filter:
        results = vectorstore.similarity_search(
            query, k=k, filter={"source": source_filter}
        )
    else:
        results = vectorstore.similarity_search(query, k=k)

    context_parts = []
    for doc in results:
        src = doc.metadata.get("source", "未知来源")
        context_parts.append(f"[来源: {src}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # 关键改动:生成最终答案的prompt不再塞入chat_summary。
    # chat_summary只用于get_search_query()里的指代消解,
    # 避免模型把历史摘要里的旧结论当成对当前问题的答案直接复述。
    prompt = f"""请只根据下面的[检索到的论文内容]回答用户的最新问题,用中文回答。

严格遵守:
1. 只依据[检索到的论文内容]作答,不要编造内容中没有的信息。
2. 如果用户的最新问题是在做判断、确认或纠错(比如"是不是分开的""是8500吗"),
   必须直接以"是"或"不是"这两个字开头作为第一句话,之后才展开解释原因。
   禁止先复述用户问题里的数字或说法再纠正,开头必须是明确的判断结论,不能自相矛盾。
3. 当用户问"结论是什么""有没有关于XX的结论/发现"时,不要求原文必须出现"conclusion"这个词才算数。
   只要[检索到的论文内容]里出现了具体的发现、数据、现象描述或作者的判断性陈述,
   都应视为可以回答的结论性内容,直接总结给用户,不要因为没看到"结论"字样就说"没有提及"。
4. 只有当[检索到的论文内容]确实完全没有涉及问题所问的主题时,才说明"资料中没有提及",不要绕圈子。

[检索到的论文内容]
{context}

[用户最新问题]
{question}

[回答]
"""

    answer = llm.invoke(prompt).strip()

    chat_history.append((question, answer))
    compress_history_if_needed()

    print("=" * 50)
    print(f"问题: {question}")
    print("=" * 50)
    print("回答:")
    print(answer)
    print("=" * 50)
    print("参考来源(检索到的原文片段):")
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "未知来源")
        print(f"\n[来源{i} - {src}] {doc.page_content[:150]}...")

    return answer


# ========== 命令行交互式对话循环 ==========
# 关键改动:整段命令行交互逻辑包在 __name__ == "__main__" 下面。
# 这样直接 `python main_process1.py` 运行时,行为和以前完全一样;
# 但被 app.py 用 `import main_process1` 时,不会触发这个while循环卡住,
# 只会执行上面"建库+定义ask()函数"的部分,可以安全地把ask()拿去给Streamlit用。
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("论文问答助手已就绪,输入问题开始提问(输入 退出 结束)")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n检测到中断,已退出对话。")
            break

        if not user_input:
            continue

        if user_input.lower() in ("退出", "exit", "quit"):
            print("已退出对话,再见。")
            break

        try:
            ask(user_input)
        except Exception as e:
            print(f"\n[出错了,这一轮没有得到回答]: {e}")
            print("如果是连接错误,检查一下Ollama是否还在运行(ollama serve)。\n")

        print()
