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


# ========== 数字一致性机械校验 ==========
# 目的:不再单纯依赖prompt让模型"自己判断数字是否一致",
# 而是用代码把问题里提到的数字和检索原文里出现的数字直接做字符串比对,
# 如果发现模型给出了错误的"是"判断,直接在代码层强制纠正,不给模型犯错的空间。

def extract_numbers(text):
    """提取文本里形如 8400 / 8,400 的数字(带千位逗号也能识别)"""
    return re.findall(r'\d[\d,]*\d|\d', text)


def normalize_num(s):
    """去掉千位逗号,方便统一比较"""
    return s.replace(",", "")


def is_confirmation_question(question):
    """判断问题是不是那种\"是不是/是否/对吗\"式的确认句"""
    return bool(re.search(r'是不是|是否|吗|对吗|对不对', question))


def verify_number_consistency(question, context, answer):
    """
    核对逻辑:
    1. 从用户问题里提取长度>=3位的数字(过滤掉\"1\"\"2\"这种太短、容易误判的小数字)。
       如果问题里根本没提到具体数字,不需要校验,原样返回。
    2. 从检索到的[context]原文里提取所有出现过的数字。
    3. 如果问题里的数字在原文中完全找不到对应(说明用户猜的数字很可能是错的),
       且这是一个\"是不是/吗\"式的确认句,且模型的回答却以\"是\"开头,
       说明模型判断错了——直接在代码层强制把回答改成\"不是\",
       并把原文中实际出现的、位数相近的数字作为提示附上,不再依赖模型自己纠错。
    4. 只要问题数字在原文里能找到,或者模型本来就回答了\"不是\",就不做任何改动,
       避免校验逻辑反而画蛇添足。
    返回 (最终答案, 是否发生了纠正: bool)
    """
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
        # 模型本来就没说"是",没有犯之前那种自相矛盾的错,不需要动
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

   参考示例(务必模仿这个推理方式,不要被"倾向说是"的语言习惯带偏):
   示例1 -
   用户问:"是8500条吗"
   [检索到的论文内容]里写的是:"all 8,400 possible di- and tripeptides"
   正确回答应该是:"不是。原文中提到的数字是8,400条,而不是8500条。"
   (注意:8400和8500虽然接近,但数字不同就是不同,绝对不能说"这与8500完全一致"或"是")

   示例2 -
   用户问:"是不是8400呢"
   [检索到的论文内容]里写的是:"all 8,400 possible di- and tripeptides"
   正确回答应该是:"是。原文中提到的正是8,400条di-和tripeptides。"
   (数字完全相等,才能回答"是")

   在生成回答前,请先在心里默念一遍:"用户说的数字/名称,和原文里的数字/名称,是不是逐字符完全一样?"
   只要不是完全一样,答案就必须是"不是",不允许因为数量级接近或看起来像就模糊处理。
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

    # 代码层机械校验:防止模型对数字类事实做出自相矛盾的"是"判断
    answer, corrected = verify_number_consistency(question, context, answer)
    if corrected:
        print("[数字校验] 检测到模型判断与原文数字不一致,已自动纠正为\"不是\"")

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


def draft(topic, k=6, source_filter=None, style="学术综述"):
    """
    写作助手模式:根据检索到的多篇文献片段,生成一段可直接用于论文/文档的文字,
    并在句子后面标注对应的来源编号,方便使用者核对和替换成正式引用格式。

    和ask()的区别:
    - ask()是回答一个具体问题,篇幅短、语气是对话式的;
    - draft()是围绕一个主题/论点生成一段连续的书面文字,篇幅更长、语气是书面化的,
      并且必须在陈述句后标来源编号(如"...[来源1]"),而不是像ask()那样只在最后列来源。

    参数:
        topic: 要写的主题或论点,比如"DtpB结合口袋的柔性如何支持底物多样性"
        k: 检索片段数量,写作场景通常需要比问答更多的上下文,默认6
        source_filter: 可选,限定只从某一篇文献里检索
        style: 写作风格提示,比如"学术综述""简明摘要""方法部分"等,可自由传入

    返回:
        生成的文字(str),同时打印出来方便直接复制
    """
    # 写作场景不需要指代消解(没有多轮对话依赖),但同样需要把中文主题转成
    # 适合语义检索的英文query,复用get_search_query,history传空即可。
    query = get_search_query(topic, [])
    print(f"[检索用英文query]: {query}")

    if source_filter:
        results = vectorstore.similarity_search(
            query, k=k, filter={"source": source_filter}
        )
    else:
        results = vectorstore.similarity_search(query, k=k)

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
6. 严禁在结尾添加脱离[检索到的文献片段]的升华式结论或意义阐述
   (例如"这对XX领域的整体理解具有重要意义""有助于维持XX系统的完整性"等)。
   每一句话(包括最后一句)都必须能在[检索到的文献片段]里找到直接对应的依据并标注来源,
   不能因为要让段落显得完整、有总结感,就添加片段里没有明确支持的引申性结论。
7. 只生成一段最终文字,不要先输出"草稿"或"示例"再输出"正式版本",
   也不要重复生成两遍相近的内容,只输出一个最终版本即可。

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
