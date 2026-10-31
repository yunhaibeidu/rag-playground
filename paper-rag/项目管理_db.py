"""
多论文库/多项目管理 —— SQLite版本

设计思路:
    SQLite本身不存论文原文、不存向量,它只做"账本"的工作——
    记录"有哪些项目""每个项目下有哪些论文文件""每个项目的对话历史"。
    真正的论文原文、切块、向量库、BM25索引,还是按项目分文件夹存在磁盘上
    (比如 papers/项目名/、chroma_db/项目名/ 这种结构),
    SQLite只是帮你快速查"这个项目有没有建过库""上次问了什么"。

三张表:
    projects      -- 项目本身(比如"DtpB转运蛋白研究""XX药物机制研究")
    papers        -- 每篇论文,挂在某个project_id下
    chat_history  -- 每一轮问答,挂在某个project_id下

使用方式:
    from 项目管理_db import ProjectDB
    db = ProjectDB("projects.db")   # 会自动建表,文件不存在就新建

    proj_id = db.create_project("DtpB转运蛋白研究")
    db.add_paper(proj_id, filename="xxx.pdf", source_name="DtpB论文", file_hash="abc123")
    db.add_chat(proj_id, question="...", answer="...")

    projects = db.list_projects()
    papers = db.list_papers(proj_id)
    history = db.list_chat_history(proj_id)
"""

import sqlite3
from datetime import datetime
from contextlib import contextmanager


class ProjectDB:
    def __init__(self, db_path="projects.db"):
        self.db_path = db_path
        self._init_tables()

    # ---------- 基础连接管理 ----------
    @contextmanager
    def _connect(self):
        """
        每次操作开一个新连接,操作完自动关闭并提交。
        用contextmanager包一层是为了不用在每个函数里重复写
        try/finally/commit/close,减少出错概率(比如忘记commit)。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 让查询结果可以像字典一样用列名取值,而不是只能用下标
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_tables(self):
        """
        建表。用 IF NOT EXISTS 保证多次运行/多次实例化都不会报错,
        已存在的表不会被清空或重建。
        """
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)

    # ---------- projects 相关 ----------
    def create_project(self, name):
        """新建一个项目,返回新项目的id。如果同名项目已存在会报错(name设了UNIQUE)。"""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO projects (name, created_at) VALUES (?, ?)",
                (name, datetime.now().isoformat())
            )
            return cursor.lastrowid

    def list_projects(self):
        """返回所有项目,每个是一个dict:{id, name, created_at}"""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def get_project_by_name(self, name):
        """按名字查项目,找不到返回None。常用于"这个项目名是不是已经存在过了"的判断。"""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE name = ?", (name,)).fetchone()
            return dict(row) if row else None

    def delete_project(self, project_id):
        """
        删除项目本身,以及它名下的papers和chat_history记录。
        注意:这只删SQLite里的记录,不会删磁盘上的论文文件/向量库文件夹,
        那部分需要你在调用这个函数的地方额外用shutil.rmtree处理。
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM chat_history WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM papers WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    # ---------- papers 相关 ----------
    def add_paper(self, project_id, filename, source_name, file_hash):
        """记录一篇论文属于哪个项目。file_hash就是你原来算的md5,用来判断文件内容变没变。"""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO papers (project_id, filename, source_name, file_hash, added_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (project_id, filename, source_name, file_hash, datetime.now().isoformat())
            )
            return cursor.lastrowid

    def list_papers(self, project_id):
        """返回某个项目下所有论文记录"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM papers WHERE project_id = ? ORDER BY added_at", (project_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_paper_hashes(self, project_id):
        """
        返回 {filename: file_hash} 字典,方便直接替换你原来compute_manifest()里
        和old_manifest比对的逻辑——现在manifest不再存json文件了,直接从数据库查。
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT filename, file_hash FROM papers WHERE project_id = ?", (project_id,)
            ).fetchall()
            return {row["filename"]: row["file_hash"] for row in rows}

    def clear_papers(self, project_id):
        """清空某个项目下的论文记录,通常在"重新扫描该项目文件夹并重建"之前调用。"""
        with self._connect() as conn:
            conn.execute("DELETE FROM papers WHERE project_id = ?", (project_id,))

    # ---------- chat_history 相关 ----------
    def add_chat(self, project_id, question, answer):
        """记录一轮问答"""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_history (project_id, question, answer, created_at) "
                "VALUES (?, ?, ?, ?)",
                (project_id, question, answer, datetime.now().isoformat())
            )

    def list_chat_history(self, project_id, limit=None):
        """
        返回某个项目的历史问答,按时间正序(老的在前,方便直接拼进对话上下文)。
        limit可选,只取最近N条时用(比如只要最近6轮)。
        """
        with self._connect() as conn:
            sql = "SELECT * FROM chat_history WHERE project_id = ? ORDER BY created_at"
            if limit:
                sql = ("SELECT * FROM (SELECT * FROM chat_history WHERE project_id = ? "
                       "ORDER BY created_at DESC LIMIT ?) ORDER BY created_at")
                rows = conn.execute(sql, (project_id, limit)).fetchall()
            else:
                rows = conn.execute(sql, (project_id,)).fetchall()
            return [dict(row) for row in rows]

    def clear_chat_history(self, project_id):
        """清空某个项目的对话历史,对应你原来"换一批论文就该清空chat_history"的需求"""
        with self._connect() as conn:
            conn.execute("DELETE FROM chat_history WHERE project_id = ?", (project_id,))


# ========== 简单自测,直接运行这个文件看效果 ==========
if __name__ == "__main__":
    import os

    test_db_path = "test_projects.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    db = ProjectDB(test_db_path)

    print("--- 创建两个项目 ---")
    p1 = db.create_project("DtpB转运蛋白研究")
    p2 = db.create_project("某药物机制研究")
    print(f"项目1 id={p1}, 项目2 id={p2}")

    print("\n--- 给项目1添加论文 ---")
    db.add_paper(p1, filename="Plasticity of the binding pocket in peptide.pdf",
                 source_name="DtpB结合口袋论文(Cell Reports 2023)", file_hash="fakehash123")

    print("--- 给项目1添加一轮对话 ---")
    db.add_chat(p1, question="DtpB的结合口袋有什么特点?", answer="具有很强的柔性...")

    print("\n--- 查看所有项目 ---")
    for proj in db.list_projects():
        print(proj)

    print("\n--- 查看项目1的论文 ---")
    for paper in db.list_papers(p1):
        print(paper)

    print("\n--- 查看项目1的对话历史 ---")
    for chat in db.list_chat_history(p1):
        print(chat)

    print("\n--- 项目1的文件指纹字典(替代原来的json manifest) ---")
    print(db.get_paper_hashes(p1))

    os.remove(test_db_path)
    print("\n自测完成,测试数据库已删除。")