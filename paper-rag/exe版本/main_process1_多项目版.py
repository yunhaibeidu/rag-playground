"""
论文问答助手 —— 多项目版

这一版相对之前的改动,核心是三件事:
    1. 路径不再写死(不再是 G:/test-localai/...),而是基于程序自身位置动态计算,
       打包成exe后依然能正确找到数据文件夹。
    2. 不再只有一个papers文件夹/一个向量库,而是"选一个项目"之后,
       动态拼出这个项目专属的papers目录和chroma_db目录。
    3. manifest(论文指纹)和chat_history(对话历史)不再是json文件/内存列表,
       改成用ProjectDB(SQLite)存取,程序重启、切换项目都不会丢数据。

本版改动:
    项目选择/加载不再在"import这个模块"的那一刻就自动执行阻塞式input()。
    改成拆出一组函数式API(list_projects/create_project/delete_project/load_project),
    GUI和CLI都调用这些函数来完成"选项目->进入项目"的流程,互不冲突。
    CLI原有的交互式菜单挪到了文件末尾的 if __name__ == "__main__" 块里。

依赖:
    需要项目管理_db.py和本文件放在同一个目录下(或者能被import到)。
"""

from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
import re
import os
import sys
import shutil
import gc
import time
import hashlib
import bm25s

from 项目管理_db import ProjectDB


# ========== 第零步:动态路径处理(打包exe后依然能用) ==========
# 关键判断:sys.frozen是PyInstaller打包后自动加在sys上的标记。
#   - 直接跑源码(python main_process1.py)时,没有这个属性 -> 用脚本文件所在目录。
#   - 打包成exe跑起来之后,__file__会指向PyInstaller解压的临时目录(不能存持久数据),
#     必须改用sys.executable(exe文件本身所在目录)才对。
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
PROJECTS_ROOT = os.path.join(DATA_DIR, "projects")
DB_PATH = os.path.join(DATA_DIR, "projects.db")

os.makedirs(PROJECTS_ROOT, exist_ok=True)

db = ProjectDB(DB_PATH)

SUPPORTED_EXTS = {".pdf", ".docx", ".txt"}

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="llama3")


# ========== 第一步:项目管理API(不再阻塞式input,GUI/CLI都能调) ==========
# 这类词大概率是用户想"退出程序"而手误敲进了创建项目的输入框,不应该被当成正常项目名直接建库
_RESERVED_EXIT_WORDS = {"exit", "quit", "退出", "结束", "q"}


def list_projects():
    """返回所有项目列表,每个元素含id/name/created_at。CLI和GUI都用这个函数查项目。"""
    return db.list_projects()


def get_project_by_name(name):
    return db.get_project_by_name(name)


def create_project(name):
    """
    创建(或复用同名)项目,不做任何交互式input,方便GUI直接调用。
    返回 (project_id, project_name, is_new)。is_new=False表示同名项目已存在,直接复用了。
    """
    existing = db.get_project_by_name(name)
    if existing:
        return existing["id"], existing["name"], False
    new_id = db.create_project(name)
    return new_id, name, True


def delete_project(project_id):
    """删除指定项目:数据库记录 + 磁盘上的papers/chroma_db文件夹。返回是否成功删除。"""
    target = next((p for p in db.list_projects() if p["id"] == project_id), None)
    if not target:
        return False
    papers_d, persist_d = get_project_paths(project_id, target["name"])
    project_dir = os.path.dirname(papers_d)
    db.delete_project(project_id)
    shutil.rmtree(project_dir, ignore_errors=True)
    return True


def sanitize_folder_name(name):
    """
    把项目名清洗成安全的文件夹名片段:
    去掉Windows/Mac都不允许的字符,去掉首尾空格和点,限制长度,空名兜底。
    """
    cleaned = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    cleaned = cleaned.strip(". ")
    if not cleaned:
        cleaned = "未命名项目"
    return cleaned[:50]


def get_project_paths(project_id, project_name):
    """
    给定项目id和项目名,拼出这个项目专属的papers目录和chroma_db目录。
    文件夹名格式为"{id}_{清洗后的项目名}"(如"1_测试论文1"),
    既靠id保证唯一不冲突,又能在磁盘上直接看出是哪个项目,不用再去数据库反查id对应的名字。

    兼容处理:早期版本文件夹名只用纯id(如"1"),如果检测到这种旧文件夹还存在、
    而新格式文件夹还不存在,就自动重命名迁移过去,不需要手动搬数据。
    """
    safe_name = sanitize_folder_name(project_name)
    folder_name = f"{project_id}_{safe_name}"
    project_dir = os.path.join(PROJECTS_ROOT, folder_name)

    old_style_dir = os.path.join(PROJECTS_ROOT, str(project_id))
    if (
        old_style_dir != project_dir
        and os.path.isdir(old_style_dir)
        and not os.path.exists(project_dir)
    ):
        try:
            os.rename(old_style_dir, project_dir)
            print(f"[已迁移] 检测到旧版纯id文件夹,已重命名为: {folder_name}")
        except (PermissionError, OSError) as e:
            # 迁移只是便利性功能,失败不应该中断主流程。
            # 常见原因:文件夹被资源管理器/其他进程占用,或被云同步/杀毒软件临时锁定。
            print(f"[迁移跳过] 无法将旧文件夹「{old_style_dir}」重命名为「{folder_name}」({e})")
            print("           可能是该文件夹正被其他程序占用,继续使用旧文件夹,不影响本次使用。")
            project_dir = old_style_dir

    papers_dir = os.path.join(project_dir, "papers")
    persist_dir = os.path.join(project_dir, "chroma_db")
    os.makedirs(papers_dir, exist_ok=True)
    return papers_dir, persist_dir


def scan_papers(folder):
    """
    扫描项目专属papers文件夹。
    source_name这里先简单用文件名去掉后缀,以后如果想要自定义显示名,
    可以在这里加一个"文件名->自定义名称"的映射表,或者接入界面让用户手动改。
    """
    papers = []
    for filename in sorted(os.listdir(folder)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTS:
            continue
        full_path = os.path.join(folder, filename)
        source = os.path.splitext(filename)[0]
        papers.append({"path": full_path, "filename": filename, "source": source})
    return papers


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


def compute_manifest(papers):
    """给当前papers列表算指纹:{filename: md5}。注意key用filename而不是完整路径,
    因为完整路径里含有project_id,换项目路径必然不同,但这不是我们关心的变化,
    我们只关心"这个文件名对应的文件内容有没有变"。"""
    manifest = {}
    for paper in papers:
        with open(paper["path"], "rb") as f:
            content = f.read()
        manifest[paper["filename"]] = hashlib.md5(content).hexdigest()
    return manifest


def safe_remove_dir(path):
    """安全删除向量库目录,释放文件句柄后再删,避免Windows下删除报错。"""
    global vectorstore
    if "vectorstore" in globals():
        del vectorstore
    gc.collect()
    time.sleep(0.5)
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# ========== 当前项目状态:模块级全局变量,初始为空,调用load_project()后才被填充 ==========
# GUI和CLI的流程都是: 先用list_projects()/create_project()选出一个项目,
# 再调load_project(pid, pname)完成实际的扫描/建库/建索引/读历史,不会在import这个文件时自动执行。
current_project_id = None
current_project_name = None
papers_dir = None
persist_dir = None
papers = []
vectorstore = None
bm25_retriever = None
bm25_corpus = []
bm25_metadatas = []
chat_history = []
chat_summary = ""

MAX_RAW_TURNS = 6
KEEP_RECENT_TURNS = 3


def load_project(project_id, project_name, log=print):
    """
    加载(或切换到)指定项目:扫描论文、必要时重建向量库、建BM25索引、加载历史对话。
    GUI和CLI都调用这一个函数完成"进入项目"的全部准备工作。
    log参数可以传一个回调函数(比如GUI里用来更新状态栏文字),默认直接print到控制台。
    返回扫描到的papers列表。
    """
    global current_project_id, current_project_name, papers_dir, persist_dir
    global papers, vectorstore, bm25_retriever, bm25_corpus, bm25_metadatas
    global chat_history, chat_summary

    current_project_id = project_id
    current_project_name = project_name
    papers_dir, persist_dir = get_project_paths(project_id, project_name)
    log(f"\n当前项目:「{current_project_name}」(id={current_project_id})")
    log(f"论文目录: {papers_dir}")
    log(f"向量库目录: {persist_dir}\n")

    papers = scan_papers(papers_dir)
    log(f"扫描到 {len(papers)} 个文件待入库: {[p['source'] for p in papers]}\n")

    current_manifest = compute_manifest(papers)
    old_manifest = db.get_paper_hashes(current_project_id)
    need_rebuild = (current_manifest != old_manifest) or (not os.path.exists(persist_dir))

    if need_rebuild:
        log("检测到论文内容有变化(或向量库不存在),开始重建向量库...")
        safe_remove_dir(persist_dir)
        log(f"已清空旧向量库: {persist_dir}")

        all_chunks = []
        all_metadatas = []
        for paper in papers:
            full_text = extract_text_from_file(paper["path"])
            chunks = _splitter.split_text(full_text)
            log(f"{paper['source']}: 提取 {len(full_text)} 字符, 切成 {len(chunks)} 块")
            all_chunks.extend(chunks)
            all_metadatas.extend([{"source": paper["source"]} for _ in chunks])

        log(f"\n所有论文总共切成 {len(all_chunks)} 块")

        if all_chunks:
            vectorstore = Chroma.from_texts(
                all_chunks, embeddings, metadatas=all_metadatas, persist_directory=persist_dir
            )
            log(f"已完成embedding并存入Chroma,共 {len(all_chunks)} 个文本块")
        else:
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            log("该项目papers文件夹为空,向量库暂时为空")

        db.clear_papers(current_project_id)
        for paper in papers:
            db.add_paper(
                current_project_id,
                filename=paper["filename"],
                source_name=paper["source"],
                file_hash=current_manifest[paper["filename"]],
            )
        log("论文指纹已同步写入数据库")
    else:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        log("papers文件夹内容未变化,直接加载已有向量库,跳过重建")

    log("正在从向量库中取出全部文档,建立BM25索引...")
    _all_docs = vectorstore.get(include=["documents", "metadatas"])
    bm25_corpus = _all_docs["documents"]
    bm25_metadatas = _all_docs["metadatas"]

    if bm25_corpus:
        _bm25_tokens = bm25s.tokenize(bm25_corpus, stopwords="en")
        bm25_retriever = bm25s.BM25()
        bm25_retriever.index(_bm25_tokens)
        log(f"BM25索引建立完成,共 {len(bm25_corpus)} 个文本块")
    else:
        bm25_retriever = None
        log("当前项目没有任何文本块,BM25索引跳过")

    _loaded_history = db.list_chat_history(current_project_id, limit=6)
    chat_history = [(row["question"], row["answer"]) for row in _loaded_history]
    chat_summary = ""

    return papers


def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def extract_numbers(text):
    return re.findall(r'\d[\d,]*\d|\d', text)


def normalize_num(s):
    return s.replace(",", "")


def is_confirmation_question(question):
    return bool(re.search(r'是不是|是否|吗|对吗|对不对', question))


def verify_number_consistency(question, context, answer):
    q_nums = {normalize_num(n) for n in extract_numbers(question)}
    q_nums = {n for n in q_nums if len(n) >= 3}
    if not q_nums:
        return answer, False

    ctx_nums = {normalize_num(n) for n in extract_numbers(context)}
    mismatched = q_nums - ctx_nums
    if not mismatched:
        return answer, False

    if not is_confirmation_question(question):
        return answer, False

    if not answer.strip().startswith("是"):
        return answer, False

    candidates = sorted({n for n in ctx_nums if len(n) >= 3}, key=len)
    if candidates:
        hint = f"原文中实际出现的相关数字包括: {', '.join(candidates)}。"
    else:
        hint = "原文中未找到与之对应的具体数字记载。"

    corrected = (
        f"不是。你提到的数字({'、'.join(sorted(mismatched))})在检索到的原文中没有找到完全对应的记载,"
        f"{hint}\n\n"
        f"[系统自动校验提示:模型原本的回答以\"是\"开头,经数字核对后发现与原文不符,已由代码自动更正为\"不是\",请以此为准。]"
    )
    return corrected, True


class SimpleDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def hybrid_search(query, k=3, source_filter=None, vector_weight=0.5, bm25_weight=0.5):
    """向量检索 + BM25检索加权融合。逻辑与之前版本一致,未改动。"""
    fetch_n = max(k * 4, 10)

    if source_filter:
        vector_results = vectorstore.similarity_search_with_score(
            query, k=fetch_n, filter={"source": source_filter}
        )
    else:
        vector_results = vectorstore.similarity_search_with_score(query, k=fetch_n)

    vector_hits = {doc.page_content: (doc.metadata, dist) for doc, dist in vector_results}

    if vector_hits:
        dists = [d for _, d in vector_hits.values()]
        min_d, max_d = min(dists), max(dists)
        span = (max_d - min_d) or 1e-9
        vector_norm = {
            text: 1 - (dist - min_d) / span
            for text, (_, dist) in vector_hits.items()
        }
    else:
        vector_norm = {}

    if bm25_retriever is not None:
        query_tokens = bm25s.tokenize([query], stopwords="en")
        bm25_idx_arr, bm25_score_arr = bm25_retriever.retrieve(query_tokens, k=fetch_n)
        bm25_idx_arr = bm25_idx_arr[0]
        bm25_score_arr = bm25_score_arr[0]

        bm25_hits = {}
        for idx, score in zip(bm25_idx_arr, bm25_score_arr):
            text = bm25_corpus[idx]
            meta = bm25_metadatas[idx]
            if source_filter and meta.get("source") != source_filter:
                continue
            bm25_hits[text] = (meta, float(score))
    else:
        bm25_hits = {}

    if bm25_hits:
        scores = [s for _, s in bm25_hits.values()]
        min_s, max_s = min(scores), max(scores)
        span = (max_s - min_s) or 1e-9
        bm25_norm = {
            text: (s - min_s) / span
            for text, (_, s) in bm25_hits.items()
        }
    else:
        bm25_norm = {}

    # 合并两路结果:文本内容作为key对齐,分数按权重加权求和
    all_texts = set(vector_norm) | set(bm25_norm)
    merged = []
    for text in all_texts:
        v_score = vector_norm.get(text, 0.0)
        b_score = bm25_norm.get(text, 0.0)
        final_score = v_score * vector_weight + b_score * bm25_weight
        meta = (vector_hits.get(text) or bm25_hits.get(text))[0]
        merged.append((text, meta, final_score))

    merged.sort(key=lambda x: x[2], reverse=True)
    top_k = merged[:k]

    return [SimpleDoc(text, meta) for text, meta, _ in top_k]


def format_history(history, summary):
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
    """
    注意:这里压缩的只是内存里chat_history这个列表(用于控制传给LLM做指代消解的token量),
    数据库里的完整历史记录不受影响、永久保留在add_chat()写入的那份原始数据里,
    不会因为内存端摘要了就把数据库的记录删掉或改写。
    """
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
    注意:这里不能给LLM预设任何具体学科/主题的硬编码描述(比如"这是一篇讲peptide transporter的论文"),
    因为现在是多项目场景,每个项目里的论文主题完全不同。之前的版本把主题写死在prompt里,
    导致哪怕用户问的是完全不相关的问题,LLM也会被这句系统设定带偏,把改写结果强行往
    那个固定主题上靠,造成检索query本身就跑偏,后面查得再准也没用。
    这里改成只告诉LLM"当前项目库里有哪些论文"作为弱背景提示(帮助消解代词/指代),
    但明确要求不要因为历史话题而缩小或改变新问题本身的范围。
    """
    history_text = format_history(history, chat_summary)
    paper_titles = "; ".join(sorted({p["source"] for p in papers})) if papers else "(无)"

    prompt = f"""You are rewriting a user's question into a standalone, precise search query in ENGLISH, to be used for semantic search against a personal paper database.

The database currently contains these papers (for background context only, do NOT assume the question is about any of them unless clearly implied): {paper_titles}

Below is the recent conversation history (may be in Chinese) and the user's latest question (may be in Chinese, and may contain pronouns like "this number" / "it" / "this paper" that depend on the history to understand).

Rewrite the latest question into a single, standalone, complete question in ENGLISH, suitable for semantic search. Rules:
1. Preserve the original topic and scope of the latest question as faithfully as possible.
2. Only use the conversation history to resolve pronouns/references that clearly depend on it (e.g. "this number", "that transporter", "it").
3. Do NOT narrow, redirect, or bias the question toward any specific paper's topic or terminology just because the previous conversation happened to be about it. If the latest question is on a different or more general topic than before, keep it that way.
4. SPECIAL CASE - explicit paper references: if the latest question explicitly refers to "another paper" / "the other paper" / "this paper" / "that paper" (e.g. 另一篇论文, 这篇论文, 另外那篇), you MUST resolve it to one specific, concrete paper title from the list above, based on which paper(s) were discussed in the conversation history. Name that paper's actual topic/title explicitly in the rewritten query instead of leaving it as a vague reference. Do not drop the reference and turn it into a generic question — a vague question defeats the purpose of the reference.
   Example: if history discussed "Paper A about environmental adaptation" and the paper list also contains "Paper B about peptide transporters", and the user asks "另一篇论文讲了什么" (what does the other paper discuss), the rewritten query should be like "What does the paper about peptide transporters discuss?" — NOT "What does a paper discuss?".
5. Only output the rewritten English question, nothing else.

[History]
{history_text}

[Latest question]
{question}

[Rewritten English search question]
"""
    rewritten = llm.invoke(prompt).strip()

    # 清洗:去掉模型可能附带的开场白/说明性前缀,只保留真正的检索问句
    # 常见情况如: "Here is the rewritten English search question:\n\nWhat..."
    lines = [l.strip() for l in rewritten.split("\n") if l.strip()]
    if len(lines) > 1:
        # 优先取最后一行非空内容作为真正的query(开场白通常在前,问句在后)
        # 但要防止最后一行本身也是空话,这里用简单规则:取最后一行
        rewritten = lines[-1]
    else:
        rewritten = lines[0] if lines else rewritten

    # 去除常见前缀短语残留(防止开场白和问句被写在同一行的情况)
    prefix_patterns = [
        r"^here('s| is) (the )?rewritten.*?:\s*",
        r"^rewritten (english )?(search )?(query|question)\s*:\s*",
        r"^search query\s*:\s*",
        r"^query\s*:\s*",
    ]
    for pat in prefix_patterns:
        rewritten = re.sub(pat, "", rewritten, flags=re.IGNORECASE).strip()

    # 去除可能包裹的引号
    rewritten = rewritten.strip('"').strip("'").strip()

    return rewritten


def is_meta_question(question):
    """
    只拦截"精确计数/列举"这一类问题(库里有几篇/都是哪些论文/列出论文清单)。
    这类问题的答案本质是"数数"和"读列表",而LLM即使拿到了论文清单也不擅长精确计数
    (容易数错、漏数、重复数),所以仍然绕开检索和模型生成,直接从papers列表拼答案返回。

    注意:"另一篇/这篇/除了这篇还有别的吗"这类指代性问题不在这里拦截——
    现在ask()的最终回答prompt里已经注入了论文清单作为背景,模型自己就能正确理解这类指代,
    不需要再靠正则规则去猜测每一种指代表达方式。
    """
    patterns = [
        r'(库|项目|系统|数据库)(里|中)?.*(有|共有|一共).*(几篇|多少篇|几个|多少个).*(论文|文献|文章|文件)',
        r'(有|共有|一共)\s*(几篇|多少篇|几个|多少个)\s*(论文|文献|文章|文件)',
        r'(论文|文献|文章|文件).*(有|共有|一共)\s*(几篇|多少篇|几个|多少个)',
        r'(都|一共)?(有|包含)?(哪些|什么).*(论文|文献|文章|文件)',
        r'(论文|文献|文章|文件).*(列表|清单|目录)',
        r'^(论文|文献)(列表|清单)',
    ]
    return any(re.search(p, question) for p in patterns)


def answer_meta_question(question):
    """直接基于当前项目实际扫描到的papers列表回答元数据问题,不经过检索/LLM生成。"""
    if not papers:
        answer = "当前项目「" + current_project_name + "」的papers文件夹里还没有放入任何论文。"
    else:
        names = [p["source"] for p in papers]
        lines = "\n".join(f"{i}. {name}" for i, name in enumerate(names, 1))
        answer = f"当前项目「{current_project_name}」库中共有 {len(names)} 篇论文:\n{lines}"

    chat_history.append((question, answer))
    db.add_chat(current_project_id, question, answer)
    compress_history_if_needed()

    print("=" * 50)
    print(f"问题: {question}")
    print("=" * 50)
    print("回答(直接来自项目文件清单,未经过向量检索):")
    print(answer)
    print("=" * 50)

    return answer


def ask(question, k=6, source_filter=None):
    global chat_history

    if current_project_id is None:
        raise RuntimeError("还没有加载任何项目,请先调用load_project()进入一个项目")

    if is_meta_question(question):
        return answer_meta_question(question)

    query = get_search_query(question, chat_history)
    print(f"[检索用英文query]: {query}")

    results = hybrid_search(query, k=k, source_filter=source_filter)

    context_parts = []
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "未知来源")
        context_parts.append(f"[来源{i} | 出自论文《{src}》]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    paper_titles = "; ".join(sorted({p["source"] for p in papers})) if papers else "(无)"
    source_paper_map = "; ".join(
        f"来源{i}->《{doc.metadata.get('source', '未知来源')}》" for i, doc in enumerate(results, 1)
    )
    distinct_papers_in_results = sorted({doc.metadata.get("source", "未知来源") for doc in results})

    prompt = f"""[当前项目论文清单(仅作背景参考,帮助你理解"这篇/另一篇/某篇论文"等指代具体指向哪一篇,以及库里实际有多少篇论文;不代表下面片段一定覆盖了它们全部)]
{paper_titles}

[重要:来源编号与论文的对应关系]
下面[检索到的论文内容]里的"来源1""来源2"等编号,只是检索到的文本片段的编号,不是论文的编号。
多个不同的来源编号完全可能出自同一篇论文的不同段落。本次检索结果的对应关系如下:
{source_paper_map}
本次检索结果实际涉及的不同论文共 {len(distinct_papers_in_results)} 篇: {', '.join(distinct_papers_in_results)}。
生成回答时,如果两个来源编号对应的论文名相同,必须视为同一篇论文的不同段落来描述
(例如"该论文在别处还提到……"),严禁说成"另一篇论文也提到"或暗示它们是两篇不同的论文。
只有当来源编号对应的论文名确实不同时,才能说"另一篇论文"。

请只根据下面的[检索到的论文内容]回答用户的最新问题,用中文回答。

0. 每一句包含具体事实、数据、方法或结论的话,句末必须标注对应的来源编号,格式为"[来源N]",
   N对应下面片段标注的编号(如[来源1 | 出自论文《论文A》])。如果一句话综合了多个片段,写"[来源1][来源2]"。
   这样用户之后如果要把这段回答用于写作,可以直接知道每句话分别来自哪篇论文,不用再去核对。

严格遵守:
1. 只依据[检索到的论文内容]作答,不要编造内容中没有的信息。
2. 如果用户的最新问题是在做判断、确认或纠错(比如"是不是分开的""是8500吗"):
   a) 先判断问题里是否包含具体的、可核实的事实断言(数字、名称、日期、是否发生某事等)。
   b) 如果包含,必须先在[检索到的论文内容]里逐字查找对应的事实,逐一核对是否完全一致
      (数字要完全相等,名称要完全对应,不能因为"数量级接近"或"看起来像"就算一致)。
   c) 只有完全一致才能回答"是";只要有任何不一致(哪怕只是数字、单位、名称有差异),
      必须回答"不是",并明确指出原文中真正的数字/名称/事实是什么。
   d) 如果[检索到的论文内容]里根本没有出现用户问题中提到的那个具体事实,不能默认为"是",
      而要回答"资料中没有找到这个具体说法",然后再补充资料中实际提到的相关内容。
   e) 回答必须直接以"是"或"不是"这两个字开头作为第一句话,之后才展开解释原因,
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

    answer, corrected = verify_number_consistency(question, context, answer)
    if corrected:
        print("[数字校验] 检测到模型判断与原文数字不一致,已自动纠正为\"不是\"")

    # 内存列表(供指代消解用)和数据库(永久记录)两边都要写
    chat_history.append((question, answer))
    db.add_chat(current_project_id, question, answer)
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

    distinct_sources = sorted({doc.metadata.get("source", "未知来源") for doc in results})
    print(f"\n本次回答共涉及 {len(distinct_sources)} 篇论文: {', '.join(distinct_sources)}")

    return answer


def draft(topic, k=6, source_filter=None, style="学术综述"):
    if current_project_id is None:
        raise RuntimeError("还没有加载任何项目,请先调用load_project()进入一个项目")

    query = get_search_query(topic, [])
    print(f"[检索用英文query]: {query}")

    results = hybrid_search(query, k=k, source_filter=source_filter)

    context_parts = []
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "未知来源")
        context_parts.append(f"[来源{i}: {src}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""你是一名学术写作助手,请根据下面的[检索到的文献片段],针对指定主题写一段{style}性质的文字,用中文书写。

严格遵守:
1. 只依据[检索到的文献片段]里的内容组织文字,不要编造片段之外的信息、数据或结论。
2. 每一句陈述性的话(尤其是包含具体数据、方法、结论的句子)后面,必须标注对应的来源编号,
   格式为句末加"[来源N]",N对应下面片段标注的编号。如果一句话综合了多个来源,
   可以写"[来源1][来源3]"。不允许出现没有来源标注的事实性陈述。
3. 如果[检索到的文献片段]中的信息不足以支撑某个角度,不要为了行文完整而编造,
   宁可少写或明确说明"现有资料未涉及此方面",也不能虚构。
4. 文字要连贯成段,符合{style}的书面语体,不要写成问答或列点形式,除非主题本身要求列点。
5. 结尾不需要额外总结"以上来源于XX文献"这类元话语,来源标注已经在句内完成。
6. 严禁在结尾添加脱离[检索到的文献片段]的升华式结论或意义阐述。
7. 只生成一段最终文字,不要先输出"草稿"或"示例"再输出"正式版本"。

[检索到的文献片段]
{context}

[写作主题/论点]
{topic}

[生成的文字]
"""

    generated = llm.invoke(prompt).strip()

    print("=" * 50)
    print(f"主题: {topic}")
    print("=" * 50)
    print("生成内容:")
    print(generated)
    print("=" * 50)
    print("来源编号对照:")
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "未知来源")
        print(f"[来源{i} - {src}] {doc.page_content[:150]}...")

    return generated


# ========== CLI交互入口 ==========
# 只有直接运行这个文件(python main_process1_多项目版.py)才会执行下面的命令行菜单;
# 被GUI当作模块import时不会触发,GUI自己调用list_projects/create_project/load_project等函数。
def _select_or_create_project_cli():
    """命令行下的交互式选项目菜单,内部调用上面的函数式API。"""
    projects = list_projects()

    print("\n" + "=" * 50)
    print("现有项目:")
    if not projects:
        print("  (暂无项目)")
    for p in projects:
        print(f"  [{p['id']}] {p['name']}  (创建于 {p['created_at']})")
    print("=" * 50)

    choice = input(
        "输入项目编号进入已有项目;输入新项目名称直接新建;"
        "输入\"删除N\"可删除编号N的项目: "
    ).strip()

    del_match = re.match(r'^删除\s*(\d+)$', choice)
    if del_match:
        pid = int(del_match.group(1))
        target = next((p for p in projects if p["id"] == pid), None)
        if not target:
            print("没有找到这个编号对应的项目,请检查输入。")
            return _select_or_create_project_cli()
        confirm = input(f"确认要删除项目「{target['name']}」(id={pid})吗?此操作不可恢复,输入\"是\"确认: ").strip()
        if confirm != "是":
            print("已取消删除。")
            return _select_or_create_project_cli()
        delete_project(pid)
        print(f"已删除项目「{target['name']}」及其磁盘数据。")
        return _select_or_create_project_cli()

    if choice.isdigit():
        pid = int(choice)
        for p in projects:
            if p["id"] == pid:
                return p["id"], p["name"]
        print("没有找到这个编号对应的项目,请检查输入。")
        return _select_or_create_project_cli()

    if choice.lower() in _RESERVED_EXIT_WORDS:
        confirm = input(
            f"「{choice}」看起来像是想退出程序,而不是一个项目名。"
            f"如果真的要创建名为「{choice}」的项目,请输入\"是\"确认,否则直接回车取消: "
        ).strip()
        if confirm != "是":
            print("已取消,请重新输入。")
            return _select_or_create_project_cli()

    pid, pname, is_new = create_project(choice)
    if is_new:
        print(f"已创建新项目「{pname}」(id={pid})")
    else:
        print(f"项目「{pname}」已经存在,直接进入该项目。")
    return pid, pname


if __name__ == "__main__":
    _pid, _pname = _select_or_create_project_cli()
    load_project(_pid, _pname)

    print("\n" + "=" * 50)
    print(f"论文问答助手已就绪(当前项目:{current_project_name})")
    print("输入问题开始提问(输入 退出 结束)")
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
