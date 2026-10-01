"""
论文问答助手 - PySide6 桌面版(多项目版)

复用 main_process1_多项目版.py 里已经跑通的项目管理API和问答/写作逻辑:
    core.list_projects()            -> list[dict],每个元素含 id/name/created_at
    core.create_project(name)       -> (project_id, project_name, is_new)
    core.delete_project(project_id) -> bool
    core.load_project(pid, pname, log=callback) -> list[dict] (papers)
        调用后会设置好 core.current_project_id / current_project_name /
        papers / vectorstore / chat_history 等全局状态
    core.papers                     -> list[dict],每个元素至少有 "source" 字段
    core.chat_history                -> list,模块级全局变量
    core.ask(user_input)             -> str,问答模式主函数
    core.draft(topic, k=.., style=..) -> str,写作模式主函数
    core.get_search_query(question, history) -> str,用于展示时重新检索来源
    core.hybrid_search(query, k=..) -> list[SimpleDoc],向量+BM25混合检索

依赖安装:
    pip install PySide6

架构说明:
    - 启动后先弹出"项目选择"对话框:可以进入已有项目、新建项目、删除项目。
      选中项目后,后台线程执行 core.load_project()(可能要重建向量库,耗时较久),
      加载完成才进入主聊天窗口,加载过程中对话框显示"正在加载..."并禁止操作。
    - 主窗口里可以随时点"切换项目"按钮重新弹出项目选择对话框,不需要重启程序。
    - 耗时的 ask()/draft()/load_project() 调用都放在 QThread 子线程里执行,
      避免调用期间界面卡死、无法响应点击/拖动。
    - 主线程只负责收发 Qt 信号、更新界面,不直接调用耗时函数。
"""

import sys
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QScrollArea, QFrame,
    QRadioButton, QButtonGroup, QComboBox, QSlider, QListWidget,
    QListWidgetItem, QSplitter, QSizePolicy, QMessageBox, QDialog,
    QInputDialog, QFileDialog
)
import os
import shutil

# ========== 复用已跑通的后端逻辑 ==========
try:
    import main_process1_多项目版 as core
    USE_DUMMY = False
except Exception as e:
    print(f"[警告] 无法导入 main_process1_多项目版,将使用模拟数据跑界面: {e}")
    USE_DUMMY = True

    class _FakeDoc:
        def __init__(self, source, content):
            self.metadata = {"source": source}
            self.page_content = content

    class DummyCore:
        papers = [{"source": "示例论文A"}, {"source": "示例论文B"}]
        chat_history = []
        chat_summary = ""
        current_project_id = 1
        current_project_name = "模拟项目"
        _RESERVED_EXIT_WORDS = {"exit", "quit", "退出", "结束", "q"}
        _projects = [{"id": 1, "name": "模拟项目", "created_at": "2026-01-01"}]

        @staticmethod
        def list_projects():
            return DummyCore._projects

        @staticmethod
        def create_project(name):
            for p in DummyCore._projects:
                if p["name"] == name:
                    return p["id"], p["name"], False
            new_id = (max((p["id"] for p in DummyCore._projects), default=0)) + 1
            DummyCore._projects.append({"id": new_id, "name": name, "created_at": "now"})
            return new_id, name, True

        @staticmethod
        def delete_project(pid):
            before = len(DummyCore._projects)
            DummyCore._projects = [p for p in DummyCore._projects if p["id"] != pid]
            return len(DummyCore._projects) < before

        @staticmethod
        def load_project(pid, pname, log=print):
            import time
            log(f"正在加载项目「{pname}」...")
            time.sleep(0.5)
            DummyCore.current_project_id = pid
            DummyCore.current_project_name = pname
            log("加载完成")
            return DummyCore.papers

        @staticmethod
        def get_project_paths(pid, pname):
            base = os.path.join(os.getcwd(), "_dummy_data", f"{pid}_{pname}")
            papers_dir = os.path.join(base, "papers")
            os.makedirs(papers_dir, exist_ok=True)
            return papers_dir, os.path.join(base, "chroma_db")

        @staticmethod
        def ask(user_input, k=6):
            import time
            time.sleep(1.2)
            return f"(模拟回答) 关于「{user_input}」的问答结果。"

        @staticmethod
        def draft(topic, k=6, style="学术综述"):
            import time
            time.sleep(1.2)
            return f"(模拟写作,风格={style},k={k}) 关于「{topic}」的写作段落。"

        @staticmethod
        def get_search_query(question, history):
            return question

        @staticmethod
        def hybrid_search(query, k=3):
            return [_FakeDoc(f"示例论文{i}", f"这是关于「{query}」的模拟检索片段内容 " * 5)
                    for i in range(1, k + 1)]

    core = DummyCore()


# ========== 后台线程:加载/切换项目(可能要重建向量库,比较耗时) ==========
class ProjectLoadWorker(QObject):
    log = Signal(str)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, project_id, project_name):
        super().__init__()
        self.project_id = project_id
        self.project_name = project_name

    def run(self):
        try:
            papers = core.load_project(self.project_id, self.project_name, log=self.log.emit)
            self.finished.emit(papers)
        except Exception as e:
            self.error.emit(f"加载项目失败: {e}")


# ========== 后台线程:添加论文后重新建库 ==========
class AddPapersWorker(QObject):
    log = Signal(str)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, project_id, project_name, file_paths):
        super().__init__()
        self.project_id = project_id
        self.project_name = project_name
        self.file_paths = file_paths

    def run(self):
        try:
            papers_dir, _ = core.get_project_paths(self.project_id, self.project_name)
            copied = 0
            for src_path in self.file_paths:
                filename = os.path.basename(src_path)
                dst_path = os.path.join(papers_dir, filename)
                if os.path.abspath(src_path) == os.path.abspath(dst_path):
                    continue
                self.log.emit(f"正在复制: {filename}")
                shutil.copy2(src_path, dst_path)
                copied += 1
            self.log.emit(f"已复制 {copied} 个文件,正在重新建库(可能需要一些时间)...")
            papers = core.load_project(self.project_id, self.project_name, log=self.log.emit)
            self.finished.emit(papers)
        except Exception as e:
            self.error.emit(f"添加论文失败: {e}")


# ========== 项目选择对话框 ==========
class ProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择项目")
        self.resize(420, 380)
        self.selected_project = None  # (id, name),对话框accept时设置

        self.thread = None
        self.worker = None

        layout = QVBoxLayout(self)

        title = QLabel("📚 论文问答助手 - 选择要进入的项目")
        f = QFont()
        f.setBold(True)
        f.setPointSize(12)
        title.setFont(f)
        layout.addWidget(title)

        self.project_list = QListWidget()
        self.project_list.itemDoubleClicked.connect(self._on_enter)
        layout.addWidget(self.project_list, 1)

        hint = QLabel("提示: 新建项目后是空的,进入项目后可以点\"➕ 添加论文\"导入PDF。")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        btn_row1 = QHBoxLayout()
        self.enter_btn = QPushButton("进入选中项目")
        self.enter_btn.clicked.connect(self._on_enter)
        self.new_btn = QPushButton("新建项目")
        self.new_btn.clicked.connect(self._on_new)
        self.delete_btn = QPushButton("删除选中项目")
        self.delete_btn.clicked.connect(self._on_delete)
        btn_row1.addWidget(self.enter_btn)
        btn_row1.addWidget(self.new_btn)
        btn_row1.addWidget(self.delete_btn)
        layout.addLayout(btn_row1)

        self._refresh_list()

    def _refresh_list(self):
        self.project_list.clear()
        for p in core.list_projects():
            item = QListWidgetItem(f"[{p['id']}] {p['name']}  (创建于 {p['created_at']})")
            item.setData(Qt.UserRole, (p["id"], p["name"]))
            self.project_list.addItem(item)

    def _set_busy(self, busy, status_text=""):
        self.enter_btn.setEnabled(not busy)
        self.new_btn.setEnabled(not busy)
        self.delete_btn.setEnabled(not busy)
        self.project_list.setEnabled(not busy)
        self.status_label.setText(status_text)

    def _on_new(self):
        name, ok = QInputDialog.getText(self, "新建项目", "输入新项目的名称:")
        if not ok or not name.strip():
            return
        name = name.strip()

        if name.lower() in core._RESERVED_EXIT_WORDS:
            reply = QMessageBox.question(
                self, "确认项目名",
                f"「{name}」看起来像是想退出程序,而不是一个项目名。\n"
                f"确定要创建名为「{name}」的项目吗?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        pid, pname, is_new = core.create_project(name)
        self._refresh_list()
        if not is_new:
            QMessageBox.information(self, "提示", f"项目「{pname}」已经存在,已为你选中。")
        # 新建/找到后直接定位选中它,方便用户直接点"进入选中项目"
        for i in range(self.project_list.count()):
            item = self.project_list.item(i)
            if item.data(Qt.UserRole)[0] == pid:
                self.project_list.setCurrentItem(item)
                break

    def _on_delete(self):
        item = self.project_list.currentItem()
        if not item:
            QMessageBox.information(self, "提示", "请先在列表中选中一个项目。")
            return
        pid, pname = item.data(Qt.UserRole)
        reply = QMessageBox.question(
            self, "确认删除",
            f"确认要删除项目「{pname}」吗?此操作会同时删除该项目下的所有论文和对话记录,不可恢复。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        core.delete_project(pid)
        self._refresh_list()

    def _on_enter(self):
        item = self.project_list.currentItem()
        if not item:
            QMessageBox.information(self, "提示", "请先在列表中选中一个项目。")
            return
        pid, pname = item.data(Qt.UserRole)

        self._set_busy(True, "正在加载项目(首次加载或论文有变化时会重建向量库,可能需要一些时间)...")

        self.thread = QThread()
        self.worker = ProjectLoadWorker(pid, pname)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(lambda msg: self.status_label.setText(msg))
        self.worker.finished.connect(self._on_load_finished)
        self.worker.error.connect(self._on_load_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.start()

    def _on_load_finished(self, papers):
        item = self.project_list.currentItem()
        pid, pname = item.data(Qt.UserRole)
        self.selected_project = (pid, pname)
        self.accept()

    def _on_load_error(self, error_msg):
        self._set_busy(False, "")
        QMessageBox.critical(self, "加载失败", error_msg)


# ========== 后台工作线程:执行耗时的 ask/draft 调用 ==========
class Worker(QObject):
    finished = Signal(str, list)   # (answer, sources)
    error = Signal(str)

    def __init__(self, mode, user_input, draft_style="学术综述", draft_k=6):
        super().__init__()
        self.mode = mode
        self.user_input = user_input
        self.draft_style = draft_style
        self.draft_k = draft_k

    def run(self):
        try:
            if self.mode == "qa":
                history_before = list(core.chat_history)
                answer = core.ask(self.user_input)
                query = core.get_search_query(self.user_input, history_before)
                results = core.hybrid_search(query, k=3)
            else:
                answer = core.draft(self.user_input, k=self.draft_k, style=self.draft_style)
                results = core.hybrid_search(self.user_input, k=self.draft_k)

            sources = [
                {"source": doc.metadata.get("source", "未知来源"), "content": doc.page_content}
                for doc in results
            ]
            self.finished.emit(answer, sources)
        except Exception as e:
            self.error.emit(f"出错了,这一轮没有得到结果: {e}\n\n如果是连接错误,检查一下Ollama是否还在运行(ollama serve)。")


# ========== 聊天气泡组件 ==========
class ChatBubble(QFrame):
    def __init__(self, role, content, sources=None, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 4)

        bubble = QFrame()
        bubble.setStyleSheet(self._bubble_style(role))
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 8, 12, 8)

        role_label = QLabel("🧑 你" if role == "user" else "🤖 助手")
        role_font = QFont()
        role_font.setBold(True)
        role_font.setPointSize(9)
        role_label.setFont(role_font)
        bubble_layout.addWidget(role_label)

        content_label = QLabel(content)
        content_label.setWordWrap(True)
        content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble_layout.addWidget(content_label)

        if sources:
            self.toggle_btn = QPushButton(f"📎 查看参考来源片段 ({len(sources)})")
            self.toggle_btn.setCheckable(True)
            self.toggle_btn.clicked.connect(self._toggle_sources)
            bubble_layout.addWidget(self.toggle_btn)

            self.sources_widget = QWidget()
            src_layout = QVBoxLayout(self.sources_widget)
            src_layout.setContentsMargins(4, 4, 4, 4)
            for i, src in enumerate(sources, 1):
                head = QLabel(f"【来源{i} - {src['source']}】")
                head.setStyleSheet("font-weight: bold; color: #555;")
                body_text = src["content"][:300] + "..."
                body = QLabel(body_text)
                body.setWordWrap(True)
                body.setStyleSheet("color: #333; background:#f5f5f5; padding:6px; border-radius:4px;")
                src_layout.addWidget(head)
                src_layout.addWidget(body)
            self.sources_widget.setVisible(False)
            bubble_layout.addWidget(self.sources_widget)

        outer_h = QHBoxLayout()
        outer_h.setContentsMargins(0, 0, 0, 0)
        if role == "user":
            outer_h.addStretch(1)
            outer_h.addWidget(bubble, 3)
        else:
            outer_h.addWidget(bubble, 3)
            outer_h.addStretch(1)
        outer.addLayout(outer_h)

    def _toggle_sources(self, checked):
        self.sources_widget.setVisible(checked)

    @staticmethod
    def _bubble_style(role):
        if role == "user":
            return "background-color: #DCF0FF; border-radius: 10px;"
        return "background-color: #F0F0F0; border-radius: 10px;"


# ========== 主窗口 ==========
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 论文问答助手 (PySide6版)" + ("  [模拟数据模式]" if USE_DUMMY else ""))
        self.resize(1100, 720)

        self.thread = None
        self.worker = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ---------- 左侧侧边栏 ----------
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)

        self.project_label = QLabel("")
        pf = QFont()
        pf.setBold(True)
        pf.setPointSize(11)
        self.project_label.setFont(pf)
        self.project_label.setWordWrap(True)
        sidebar_layout.addWidget(self.project_label)

        self.switch_project_btn = QPushButton("🔀 切换项目")
        self.switch_project_btn.clicked.connect(self._on_switch_project)
        sidebar_layout.addWidget(self.switch_project_btn)

        line0 = QFrame()
        line0.setFrameShape(QFrame.HLine)
        sidebar_layout.addWidget(line0)

        sidebar_layout.addWidget(self._title_label("🔀 模式切换", 12))
        self.radio_qa = QRadioButton("💬 问答模式")
        self.radio_write = QRadioButton("✍️ 写作模式")
        self.radio_qa.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.radio_qa)
        mode_group.addButton(self.radio_write)
        sidebar_layout.addWidget(self.radio_qa)
        sidebar_layout.addWidget(self.radio_write)

        # 写作模式专属选项
        self.write_options = QWidget()
        write_opt_layout = QVBoxLayout(self.write_options)
        write_opt_layout.setContentsMargins(0, 6, 0, 6)

        write_opt_layout.addWidget(QLabel("写作风格"))
        self.style_combo = QComboBox()
        self.style_combo.addItems(["学术综述", "简明摘要", "方法部分"])
        write_opt_layout.addWidget(self.style_combo)

        write_opt_layout.addWidget(QLabel("检索片段数量(k)"))
        k_row = QHBoxLayout()
        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setMinimum(3)
        self.k_slider.setMaximum(10)
        self.k_slider.setValue(6)
        self.k_value_label = QLabel("6")
        self.k_slider.valueChanged.connect(lambda v: self.k_value_label.setText(str(v)))
        k_row.addWidget(self.k_slider)
        k_row.addWidget(self.k_value_label)
        write_opt_layout.addLayout(k_row)

        self.write_options.setVisible(False)
        sidebar_layout.addWidget(self.write_options)

        self.radio_qa.toggled.connect(self._on_mode_changed)
        self.radio_write.toggled.connect(self._on_mode_changed)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        sidebar_layout.addWidget(line1)

        sidebar_layout.addWidget(self._title_label("📚 已入库论文", 12))
        self.paper_list = QListWidget()
        sidebar_layout.addWidget(self.paper_list, 1)

        self.add_paper_btn = QPushButton("➕ 添加论文(PDF)")
        self.add_paper_btn.clicked.connect(self._on_add_papers)
        sidebar_layout.addWidget(self.add_paper_btn)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        sidebar_layout.addWidget(line2)

        self.clear_btn = QPushButton("🗑️ 清空对话历史")
        self.clear_btn.clicked.connect(self._clear_history)
        sidebar_layout.addWidget(self.clear_btn)

        sidebar.setMaximumWidth(280)
        splitter.addWidget(sidebar)

        # ---------- 右侧主聊天区 ----------
        chat_container = QWidget()
        chat_layout = QVBoxLayout(chat_container)

        header = QLabel("🧠 论文问答助手")
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header.setFont(header_font)
        chat_layout.addWidget(header)

        caption = QLabel("基于本地 Ollama + Chroma 向量库的 RAG 问答,只依据当前项目已入库论文内容回答")
        caption.setStyleSheet("color: gray;")
        chat_layout.addWidget(caption)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.chat_area = QWidget()
        self.chat_area_layout = QVBoxLayout(self.chat_area)
        self.chat_area_layout.addStretch(1)
        self.scroll_area.setWidget(self.chat_area)
        chat_layout.addWidget(self.scroll_area, 1)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        chat_layout.addWidget(self.status_label)

        input_row = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("输入你的问题...")
        self.input_box.returnPressed.connect(self._on_send)
        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self.input_box, 1)
        input_row.addWidget(self.send_btn)
        chat_layout.addLayout(input_row)

        splitter.addWidget(chat_container)
        splitter.setStretchFactor(1, 1)

        self._on_mode_changed()
        self.refresh_project_info()

    # ---------- UI辅助 ----------
    @staticmethod
    def _title_label(text, size):
        lbl = QLabel(text)
        f = QFont()
        f.setBold(True)
        f.setPointSize(size)
        lbl.setFont(f)
        return lbl

    def refresh_project_info(self):
        """项目加载/切换完成后调用:刷新侧边栏的项目名、论文列表,清空聊天区(切到新项目)。"""
        self.project_label.setText(f"📁 当前项目:{core.current_project_name}")
        self.paper_list.clear()
        if core.papers:
            for p in core.papers:
                self.paper_list.addItem(p["source"])
        else:
            placeholder = QListWidgetItem("(当前项目还没有论文,点击下方\"添加论文\"导入PDF)")
            placeholder.setFlags(Qt.NoItemFlags)
            self.paper_list.addItem(placeholder)
        self._clear_chat_bubbles()

    def _on_mode_changed(self):
        is_write = self.radio_write.isChecked()
        self.write_options.setVisible(is_write)
        self.input_box.setPlaceholderText(
            "输入你想写作的主题(例如:DtpB结合口袋的柔性如何支持底物多样性识别)..."
            if is_write else "输入你的问题..."
        )

    def _add_bubble(self, role, content, sources=None):
        bubble = ChatBubble(role, content, sources)
        self.chat_area_layout.insertWidget(self.chat_area_layout.count() - 1, bubble)
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def _clear_chat_bubbles(self):
        while self.chat_area_layout.count() > 1:
            item = self.chat_area_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _clear_history(self):
        self._clear_chat_bubbles()
        # 只清空当前内存里的对话历史(用于指代消解),不影响数据库里的永久记录
        core.chat_history = []
        core.chat_summary = ""
        self.status_label.setText("对话历史已清空(数据库中的历史记录仍保留)")

    def _on_add_papers(self):
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "提示", "上一轮还在处理中,请稍候再添加论文。")
            return

        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择要导入的论文PDF", "", "PDF文件 (*.pdf)"
        )
        if not file_paths:
            return

        self.input_box.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.switch_project_btn.setEnabled(False)
        self.add_paper_btn.setEnabled(False)
        self.status_label.setText(f"正在导入 {len(file_paths)} 个文件...")

        self.thread = QThread()
        self.worker = AddPapersWorker(core.current_project_id, core.current_project_name, file_paths)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(lambda msg: self.status_label.setText(msg))
        self.worker.finished.connect(self._on_add_papers_finished)
        self.worker.error.connect(self._on_add_papers_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.start()

    def _on_add_papers_finished(self, papers):
        self.refresh_project_info()
        self.status_label.setText(f"导入完成,当前项目共有 {len(papers)} 篇论文")
        self.input_box.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.switch_project_btn.setEnabled(True)
        self.add_paper_btn.setEnabled(True)

    def _on_add_papers_error(self, error_msg):
        self.status_label.setText("")
        self.input_box.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.switch_project_btn.setEnabled(True)
        self.add_paper_btn.setEnabled(True)
        QMessageBox.critical(self, "导入失败", error_msg)

    def _on_switch_project(self):
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "提示", "上一轮还在处理中,请稍候再切换项目。")
            return

        dialog = ProjectDialog(self)
        if dialog.exec() == QDialog.Accepted and dialog.selected_project:
            self.refresh_project_info()

    def _on_send(self):
        user_input = self.input_box.text().strip()
        if not user_input:
            return
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "提示", "上一轮还在处理中,请稍候...")
            return

        mode = "qa" if self.radio_qa.isChecked() else "draft"
        self._add_bubble("user", user_input)
        self.input_box.clear()
        self.input_box.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.switch_project_btn.setEnabled(False)
        self.status_label.setText(
            "检索论文内容并生成回答中..." if mode == "qa" else "检索论文内容并生成写作段落中..."
        )

        self.thread = QThread()
        self.worker = Worker(
            mode, user_input,
            draft_style=self.style_combo.currentText(),
            draft_k=self.k_slider.value(),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.start()

    def _on_worker_finished(self, answer, sources):
        self._add_bubble("assistant", answer, sources)
        self.status_label.setText("")
        self.input_box.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.switch_project_btn.setEnabled(True)
        self.input_box.setFocus()

    def _on_worker_error(self, error_msg):
        self._add_bubble("assistant", error_msg)
        self.status_label.setText("")
        self.input_box.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.switch_project_btn.setEnabled(True)


def main():
    app = QApplication(sys.argv)

    # 启动时先弹出项目选择对话框,选好项目、load_project()完成后才打开主窗口
    dialog = ProjectDialog()
    if dialog.exec() != QDialog.Accepted or not dialog.selected_project:
        sys.exit(0)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
