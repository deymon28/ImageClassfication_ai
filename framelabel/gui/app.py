"""Main window: sessions and review queues on the left, the grid / box editor in the centre, classes and the frame's
boxes on the right. All decisions are saved immediately (labels/<stem>.txt and review.json)."""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
from PyQt6.QtCore import QEvent, QObject, QSettings, Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QColor, QIcon, QKeySequence, QPixmap
from PyQt6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox,
                             QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                             QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QStackedWidget, QTableWidget,
                             QVBoxLayout, QWidget)

from framelabel.core.boxes import Box
from framelabel.core.project import Project, folder_project
from framelabel.core.propagate import propagate
from framelabel.core.queues import FrameInfo
from framelabel.core.session import Session
from framelabel.core.workspace import Workspace
from framelabel.perf import setup_logging, timed
from .canvas import Canvas
from .grid import Grid
from .keys import Keymap, key_name
from .workers import Job, start_job

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("framelabel")
EDIT_QUEUES = [("edit", "To review (proposals / marked)"), ("uncertain", "Uncertain proposals first"),
               ("small", "Small (far) objects first"), ("disagree", "Model disagrees with labels"),
               ("unreviewed", "Not reviewed"), ("todo", "Marked in grid"), ("pos", "Labeled"), ("neg", "Empty"),
               ("skip", "Skipped"), ("all", "All frames")]
GRID_QUEUES = [("triage", "Model found nothing"), ("ui", "Suspected interface screens"), ("neg", "Saved as empty"),
               ("skip", "Skipped"), ("unreviewed", "Not reviewed"), ("all", "All frames")]
STATUS_COLORS = {"pos": "#5fd35f", "neg": "#8a8a8a", "skip": "#c08040", "todo": "#ffd000", None: "#ffffff"}


def color_icon(color: str, size: int = 12) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(color))
    return QIcon(pm)


class MainWindow(QMainWindow):
    def __init__(self, target: Path | None = None):
        super().__init__()
        self.settings = QSettings("framelabel", "framelabel")
        self.keymap = Keymap(json.loads(self.settings.value("keymap", "{}") or "{}"))
        self.project: Project | None = None
        self.sessions: list[Session] = []
        self.ws: Workspace | None = None
        self.mode = "grid"
        self.queue: list[FrameInfo] = []
        self.idx = 0
        self.cur_path: Path | None = None
        self.prefetch: dict[Path, Future] = {}
        self.pool = ThreadPoolExecutor(2)
        self.fitter = None
        self.job = None
        self.setWindowTitle("framelabel")
        self.resize(1600, 950)
        self.build_ui()
        self.build_menus()
        QApplication.instance().installEventFilter(self)
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        if target is not None:
            QTimer.singleShot(0, lambda: self.open_target(Path(target)))
        elif (recent := self.recent()) and Path(recent[0]).exists():
            QTimer.singleShot(0, lambda: self.open_target(Path(recent[0])))

    # ================================================================ UI
    def build_ui(self) -> None:
        self.canvas = Canvas()
        self.grid = Grid()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.grid)
        self.stack.addWidget(self.canvas)
        self.setCentralWidget(self.stack)
        self.canvas.changed.connect(self.on_boxes_changed)
        self.canvas.selection_changed.connect(self.on_selection)
        self.canvas.hover_info.connect(lambda s: self.pos_label.setText(s))
        self.canvas.sam_requested.connect(self.sam_fit)
        try:
            self.canvas.loupe_zoom = float(self.settings.value("loupe_zoom", 3.0))
        except (TypeError, ValueError):
            pass
        self.canvas.loupe_changed.connect(lambda z: self.settings.setValue("loupe_zoom", z))
        self.grid.marks_changed.connect(self.update_status)
        self.grid.page_changed.connect(self.update_status)

        # left: session, queue, frames
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        self.session_box = QComboBox()
        self.session_box.currentIndexChanged.connect(self.on_session_changed)
        self.queue_box = QComboBox()
        self.queue_box.currentIndexChanged.connect(self.on_queue_changed)
        self.mode_btn = QPushButton("Grid / Editor  (Tab)")
        self.mode_btn.clicked.connect(self.switch_mode)
        self.frame_list = QListWidget()
        self.frame_list.setUniformItemSizes(True)
        self.frame_list.itemActivated.connect(self.on_frame_clicked)
        self.frame_list.itemClicked.connect(self.on_frame_clicked)
        self.counts_label = QLabel()
        self.counts_label.setWordWrap(True)
        for w in (QLabel("Session"), self.session_box, QLabel("Queue"), self.queue_box, self.mode_btn, self.frame_list,
                  self.counts_label):
            lv.addWidget(w)
        dock = QDockWidget("Frames", self)
        dock.setObjectName("frames")
        dock.setWidget(left)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        # right: classes, boxes, options
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)
        rv.addWidget(QLabel("Classes (1-9)"))
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(170)
        self.class_list.currentRowChanged.connect(self.on_class_row)
        rv.addWidget(self.class_list)
        rv.addWidget(QLabel("Boxes on this frame"))
        self.box_table = QTableWidget(0, 3)
        self.box_table.setHorizontalHeaderLabels(["class", "conf", "size px"])
        self.box_table.verticalHeader().setVisible(False)
        self.box_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.box_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.box_table.cellClicked.connect(lambda r, _c: self.canvas.select(r, center=False))
        self.box_table.cellDoubleClicked.connect(lambda r, _c: self.canvas.select(r, center=True))
        self.box_table.horizontalHeader().setStretchLastSection(True)
        rv.addWidget(self.box_table)
        row = QHBoxLayout()
        row.addWidget(QLabel("Show proposals from conf"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(float(self.settings.value("conf", 0.3)))
        self.conf_spin.valueChanged.connect(self.on_conf_changed)
        row.addWidget(self.conf_spin)
        rv.addLayout(row)
        self.weak_check = QCheckBox("Show weak proposals (click to accept)  W")
        self.weak_check.toggled.connect(self.on_weak_toggled)
        self.sam_check = QCheckBox("Fit new boxes with SAM")
        self.sam_check.toggled.connect(lambda v: setattr(self.canvas, "sam_on_draw", v))
        self.carry_check = QCheckBox("Carry boxes to frames without proposals")
        self.carry_check.setToolTip("When the next frame has no proposals and is not reviewed, the previous frame's boxes "
                                    "are tracked onto it")
        for w in (self.weak_check, self.sam_check, self.carry_check):
            rv.addWidget(w)
        self.help_label = QLabel()
        self.help_label.setWordWrap(True)
        self.help_label.setStyleSheet("color: #9a9a9a")
        rv.addWidget(self.help_label)
        rv.addStretch(1)
        dock = QDockWidget("Labels", self)
        dock.setObjectName("labels")
        dock.setWidget(right)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        self.status_label = QLabel()
        self.pos_label = QLabel()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_job)
        sb = self.statusBar()
        sb.addWidget(self.status_label, 1)
        sb.addPermanentWidget(self.pos_label)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.cancel_btn)

    def act(self, menu, text: str, slot, action: str | None = None, checkable: bool = False, shortcut: str | None = None):
        a = QAction(text, self)
        a.triggered.connect(slot)
        a.setCheckable(checkable)
        keys = self.keymap.map.get(action, []) if action else []
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        elif keys:
            # keys are dispatched by the event filter (layout independent); the menu only shows the first one
            a.setText(f"{text}\t{keys[0]}")
        menu.addAction(a)
        return a

    def build_menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        self.act(m, "Open project...", self.open_project_dialog, shortcut="Ctrl+O")
        self.act(m, "Open folder...", self.open_folder_dialog)
        self.act(m, "New project...", self.new_project)
        self.recent_menu = m.addMenu("Recent")
        self.recent_menu.aboutToShow.connect(self.fill_recent)
        m.addSeparator()
        self.act(m, "Project settings...", self.project_settings)
        self.act(m, "Keyboard shortcuts...", self.keyboard_dialog)
        m.addSeparator()
        self.act(m, "Export YOLO dataset...", self.export_dialog, shortcut="Ctrl+E")
        m.addSeparator()
        self.act(m, "Quit", self.close, shortcut="Ctrl+Q")

        m = mb.addMenu("&Edit")
        self.act(m, "Accept and next", lambda: self.do("accept_next"), "accept_next")
        self.act(m, "Empty frame and next", lambda: self.do("empty_next"), "empty_next")
        self.act(m, "Skip frame and next", lambda: self.do("skip_next"), "skip_next")
        m.addSeparator()
        self.act(m, "Undo", lambda: self.do("undo"), "undo")
        self.act(m, "Redo", lambda: self.do("redo"), "redo")
        self.act(m, "Delete selected box", lambda: self.do("delete_box"), "delete_box")
        self.act(m, "Toggle class of box", lambda: self.do("toggle_class"), "toggle_class")
        self.act(m, "Accept all proposals as labels", lambda: self.do("accept_proposals"), "accept_proposals")
        self.act(m, "Reset to model proposals", lambda: self.do("reset_proposals"), "reset_proposals")
        m.addSeparator()
        self.act(m, "Copy boxes from previous frame", lambda: self.do("copy_prev"), "copy_prev")
        self.act(m, "Track boxes from previous frame", lambda: self.do("propagate_prev"), "propagate_prev")
        self.act(m, "Fit selected box with SAM (Alt+click: from a point)", lambda: self.do("sam_fit"), "sam_fit")

        m = mb.addMenu("&View")
        self.act(m, "Grid / editor", lambda: self.do("switch_mode"), "switch_mode")
        self.act(m, "Next grid view", lambda: self.do("grid_view"), "grid_view")
        self.act(m, "Fit image", lambda: self.do("fit_view"), "fit_view")
        self.act(m, "Show / hide boxes", lambda: self.do("toggle_boxes"), "toggle_boxes")
        self.act(m, "Loupe", lambda: self.do("toggle_loupe"), "toggle_loupe")
        self.act(m, "Loupe: stronger (also Ctrl+wheel)", lambda: self.do("loupe_in"), "loupe_in")
        self.act(m, "Loupe: weaker", lambda: self.do("loupe_out"), "loupe_out")
        self.act(m, "Fullscreen", lambda: self.do("fullscreen"), "fullscreen")
        gm = m.addMenu("Grid size")
        grp = QActionGroup(self)
        for cols, rows in ((2, 2), (3, 3), (4, 3), (4, 4), (5, 4)):
            a = QAction(f"{cols} x {rows}", self, checkable=True)
            a.triggered.connect(lambda _=False, c=cols, r=rows: self.set_grid_size(c, r))
            grp.addAction(a)
            gm.addAction(a)
            if f"{cols}x{rows}" == self.settings.value("grid", "3x3"):
                a.setChecked(True)
        c, r = map(int, str(self.settings.value("grid", "3x3")).split("x"))
        self.grid.cols, self.grid.rows = c, r

        m = mb.addMenu("&Auto-label")
        self.act(m, "Prelabel with models...", self.prelabel_dialog, shortcut="Ctrl+L")
        self.act(m, "Retrain a labeler on my labels...", self.retrain_dialog)
        self.act(m, "Check GPU / game status", self.gpu_status)
        self.update_help()

    def update_help(self) -> None:
        k = self.keymap.first
        if self.mode == "grid":
            text = (f"GRID: LMB has objects | MMB / Ctrl+LMB skip | RMB full view | {k('commit_page')} commit page | "
                    f"{k('prev_page')}/{k('next_page')} pages | {k('grid_view')} next view | {k('switch_mode')} editor")
        else:
            text = (f"EDITOR: drag = new box, drag box/corner/edge = move/resize, RMB delete, wheel zoom, MMB pan | "
                    f"1-9 class | {k('accept_next')} accept | {k('empty_next')} empty | {k('skip_next')} skip | "
                    f"{k('prev')}/{k('next')} prev/next | {k('next_unreviewed')} next new | {k('undo')} undo | "
                    f"{k('copy_prev')} copy prev | {k('propagate_prev')} track prev | {k('sam_fit')} SAM fit, "
                    f"Alt+click SAM | {k('accept_proposals')} accept proposals | W weak proposals | "
                    f"{k('toggle_loupe')} loupe, Ctrl+wheel or +/- its zoom")
        self.help_label.setText(text)

    # ================================================================ projects
    def recent(self) -> list[str]:
        try:
            return list(json.loads(self.settings.value("recent", "[]") or "[]"))
        except ValueError:
            return []

    def add_recent(self, path: Path) -> None:
        items = [str(path)] + [p for p in self.recent() if p != str(path)]
        self.settings.setValue("recent", json.dumps(items[:10]))

    def fill_recent(self) -> None:
        self.recent_menu.clear()
        for p in self.recent():
            a = self.recent_menu.addAction(p)
            a.triggered.connect(lambda _=False, q=p: self.open_target(Path(q)))

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project", str(ROOT / "projects"), "Project (*.yaml *.yml)")
        if path:
            self.open_target(Path(path))

    def open_folder_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open a session folder, a folder of sessions or of images")
        if path:
            self.open_target(Path(path))

    def new_project(self) -> None:
        from .dialogs import ProjectDialog

        folder = QFileDialog.getExistingDirectory(self, "Folder with the images or sessions to label")
        if not folder:
            return
        proj = folder_project(Path(folder))
        dlg = ProjectDialog(proj, self, new=True)
        if dlg.exec():
            path, _ = QFileDialog.getSaveFileName(self, "Save project file", str(ROOT / "projects" / f"{proj.name}.yaml"),
                                                  "Project (*.yaml)")
            if path:
                dlg.apply()
                proj.save(Path(path))
                self.open_target(Path(path))

    def open_target(self, target: Path) -> None:
        try:
            if target.suffix.lower() in (".yaml", ".yml"):
                proj = Project.load(target)
            elif target.is_dir():
                proj = folder_project(target)
            else:
                raise FileNotFoundError(target)
        except Exception as e:
            QMessageBox.critical(self, "Cannot open", f"{target}\n\n{e}")
            return
        self.commit_current()
        self.project = proj
        self.sessions = proj.sessions()
        if proj.path:
            self.add_recent(proj.path)
        elif target.is_dir():
            self.add_recent(target)
        self.setWindowTitle(f"framelabel - {proj.name}")
        self.canvas.set_classes(proj.class_names, [c.color for c in proj.classes])
        self.grid.class_colors = [QColor(c.color) for c in proj.classes]
        self.grid.ui_threshold = proj.ui_threshold
        self.fill_classes()
        if "show_conf" in proj.prelabel:  # the project's recommended threshold for its labeler
            self.conf_spin.blockSignals(True)
            self.conf_spin.setValue(float(proj.prelabel["show_conf"]))
            self.conf_spin.blockSignals(False)
        self.session_box.blockSignals(True)
        self.session_box.clear()
        for s in self.sessions:
            self.session_box.addItem(s.name)
        self.session_box.blockSignals(False)
        if not self.sessions:
            self.canvas.message = "No frames found in the project's sources"
            self.canvas.update()
            return
        last = self.settings.value(f"session/{proj.path or target}", "")
        i = next((k for k, s in enumerate(self.sessions) if s.name == last), 0)
        self.session_box.setCurrentIndex(i)
        self.on_session_changed(i)

    def fill_classes(self) -> None:
        self.class_list.blockSignals(True)
        self.class_list.clear()
        for i, c in enumerate(self.project.classes):
            it = QListWidgetItem(color_icon(c.color, 14), f"{i + 1}  {c.name}")
            self.class_list.addItem(it)
        self.class_list.setCurrentRow(self.canvas.cls)
        self.class_list.blockSignals(False)

    def on_session_changed(self, i: int) -> None:
        if not (0 <= i < len(self.sessions)):
            return
        self.commit_current()
        s = self.sessions[i]
        self.status_label.setText(f"Opening {s.name}...")
        QApplication.processEvents()
        self.ws = Workspace(self.project, s, self.conf_spin.value())
        self.settings.setValue(f"session/{self.project.path or s.root}", s.name)
        if not s.auto_dir.exists():
            self.status_label.setText("No model proposals yet: Auto-label > Prelabel to get them. All frames go to the grid.")
        triage = self.ws.queue("triage")
        unreviewed_in_triage = any(f.status is None for f in triage)
        self.set_mode("grid" if unreviewed_in_triage else "edit", queue="triage" if unreviewed_in_triage else "edit")

    # ================================================================ modes and queues
    def set_mode(self, mode: str, queue: str | None = None) -> None:
        self.commit_current()
        self.mode = mode
        self.queue_box.blockSignals(True)
        self.queue_box.clear()
        items = GRID_QUEUES if mode == "grid" else EDIT_QUEUES + [
            (f"class:{i}", f"Has class {c.name}") for i, c in enumerate(self.project.classes)]
        for key, label in items:
            self.queue_box.addItem(label, key)
        k = next((n for n, (key, _) in enumerate(items) if key == queue), 0)
        self.queue_box.setCurrentIndex(k)
        self.queue_box.blockSignals(False)
        self.stack.setCurrentWidget(self.grid if mode == "grid" else self.canvas)
        if mode == "grid":
            self.box_table.setRowCount(0)
            self.cur_path = None
        self.update_help()
        self.on_queue_changed(k)

    def switch_mode(self) -> None:
        if self.ws is None:
            return
        self.set_mode("edit" if self.mode == "grid" else "grid")

    def on_queue_changed(self, _i: int) -> None:
        if self.ws is None:
            return
        self.commit_current()
        name = self.queue_box.currentData()
        self.queue = self.ws.queue(name)
        with timed("fill_frame_list", 50):
            self.fill_frame_list()
        if self.mode == "grid":
            first = next((i for i, f in enumerate(self.queue) if f.status is None), 0) if name == "triage" else 0
            self.grid.set_items(self.queue, first // self.grid.per_page,
                                initial_mark=lambda fi, v=name: self.ws.initial_mark(fi, v))
            self.grid.setFocus()
        else:
            if not self.queue:
                self.cur_path = None
                self.canvas.img = None
                self.canvas.message = "This queue is empty"
                self.canvas.update()
                self.update_status()
                return
            first = next((i for i, f in enumerate(self.queue) if f.status in (None, "todo")), 0)
            self.load_frame(first)
            self.canvas.setFocus()
        self.update_status()

    def fill_frame_list(self) -> None:
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        for fi in self.queue:
            it = QListWidgetItem(self.frame_label(fi))
            it.setForeground(QColor(STATUS_COLORS.get(fi.status, "#ffffff")))
            self.frame_list.addItem(it)
        self.frame_list.blockSignals(False)

    def frame_label(self, fi: FrameInfo) -> str:
        tag = {"pos": "labeled", "neg": "empty", "skip": "skip", "todo": "queued"}.get(fi.status or "", "new")
        extra = f"  {fi.max_conf:.2f}" if fi.max_conf else ""
        return f"{fi.stem}  [{tag}]{extra}"

    def update_list_item(self, i: int) -> None:
        if 0 <= i < self.frame_list.count():
            fi = self.queue[i]
            it = self.frame_list.item(i)
            it.setText(self.frame_label(fi))
            it.setForeground(QColor(STATUS_COLORS.get(fi.status, "#ffffff")))

    def on_frame_clicked(self, item: QListWidgetItem) -> None:
        row = self.frame_list.row(item)
        if self.mode == "grid":
            self.grid.set_page(row // self.grid.per_page)
        else:
            self.go(row)

    # ================================================================ editor frames
    def read_image(self, path: Path):
        f = self.prefetch.pop(path, None)
        img = f.result() if f is not None else cv2.imread(str(path))
        for q in self.queue[self.idx + 1:self.idx + 3]:  # read ahead
            if q.path not in self.prefetch:
                self.prefetch[q.path] = self.pool.submit(cv2.imread, str(q.path))
        for k in list(self.prefetch):
            if len(self.prefetch) > 6:
                del self.prefetch[k]
        return img

    def load_frame(self, idx: int, carried: list[Box] | None = None) -> None:
        with timed("load_frame", 50):
            self._load_frame(idx, carried)

    def _load_frame(self, idx: int, carried: list[Box] | None = None) -> None:
        self.idx = min(max(idx, 0), len(self.queue) - 1)
        fi = self.queue[self.idx]
        img = self.read_image(fi.path)
        if img is None:
            self.canvas.message = f"Cannot read {fi.path}"
            self.canvas.img = None
            self.canvas.update()
            return
        h, w = img.shape[:2]
        boxes, weak = self.ws.load_boxes(fi.path, w, h)
        self.cur_path = fi.path
        self.canvas.message = ""
        self.canvas.set_frame(img, boxes, weak, keep_view=True)
        if carried is not None and not boxes and fi.status in (None, "todo"):
            self.canvas.replace_boxes(carried)
            self.canvas.message = "boxes carried over from the previous frame: check them"
        self.frame_list.blockSignals(True)
        self.frame_list.setCurrentRow(self.idx)
        self.frame_list.blockSignals(False)
        self.fill_box_table()
        self.update_status()

    def commit_current(self, status: str | None = None, force: bool = False) -> None:
        """Save the editor's frame when it was edited (or always with force / an explicit status)."""
        if self.ws is None or self.mode != "edit" or self.cur_path is None or self.canvas.img is None:
            return
        if not (self.canvas.dirty or force or status):
            return
        with timed("commit_current", 50):
            self.ws.save(self.cur_path, self.canvas.boxes, self.canvas.iw, self.canvas.ih, status)
        self.canvas.dirty = False
        if 0 <= self.idx < len(self.queue) and self.queue[self.idx].path == self.cur_path:
            self.queue[self.idx] = self.ws.index[self.ws.by_stem[self.cur_path.stem]]
            self.update_list_item(self.idx)

    def go(self, idx: int, carry: bool = False) -> None:
        if not self.queue:
            return
        prev_img, prev_boxes = self.canvas.img, self.canvas.boxes_copy()
        self.commit_current()
        if idx >= len(self.queue):
            self.status_label.setText("End of this queue.")
            idx = len(self.queue) - 1
        carried = None
        if carry and self.carry_check.isChecked() and prev_img is not None and prev_boxes and 0 <= idx < len(self.queue):
            nxt = self.queue[idx]
            if nxt.status in (None, "todo") and nxt.max_conf < self.conf_spin.value():
                img = self.read_image(nxt.path)
                if img is not None and img.shape[:2] == prev_img.shape[:2]:
                    self.prefetch[nxt.path] = self.pool.submit(lambda im=img: im)
                    carried = propagate(prev_img, img, prev_boxes, [])
        self.load_frame(idx, carried)

    def fill_box_table(self) -> None:
        from PyQt6.QtWidgets import QTableWidgetItem

        t = self.box_table
        t.setRowCount(len(self.canvas.boxes))
        for r, b in enumerate(self.canvas.boxes):
            name = self.project.class_names[b.cls] if 0 <= b.cls < len(self.project.classes) else str(b.cls)
            it = QTableWidgetItem(name)
            it.setIcon(color_icon(self.project.color(b.cls)))
            t.setItem(r, 0, it)
            t.setItem(r, 1, QTableWidgetItem("" if b.conf is None else f"{b.conf:.2f}"))
            n = b.norm()
            t.setItem(r, 2, QTableWidgetItem(f"{n.w:.0f} x {n.h:.0f}"))
        if 0 <= self.canvas.sel < t.rowCount():
            t.selectRow(self.canvas.sel)

    def on_boxes_changed(self) -> None:
        self.fill_box_table()
        self.update_status()

    def on_selection(self, i: int) -> None:
        if 0 <= i < self.box_table.rowCount():
            self.box_table.selectRow(i)
            b = self.canvas.boxes[i]
            self.class_list.blockSignals(True)
            self.class_list.setCurrentRow(b.cls)
            self.class_list.blockSignals(False)
        else:
            self.box_table.clearSelection()

    def on_class_row(self, row: int) -> None:
        if row < 0:
            return
        self.canvas.cls = row
        if self.canvas.sel >= 0:
            self.canvas.set_class_of(self.canvas.sel, row)

    def on_conf_changed(self, v: float) -> None:
        self.settings.setValue("conf", v)
        if self.ws is not None:
            self.commit_current()
            self.ws.conf = v
            if self.mode == "edit" and self.cur_path is not None:
                self.load_frame(self.idx)

    def on_weak_toggled(self, v: bool) -> None:
        self.canvas.show_weak = v
        self.canvas.update()

    def set_grid_size(self, cols: int, rows: int) -> None:
        self.settings.setValue("grid", f"{cols}x{rows}")
        first = self.grid.page * self.grid.per_page
        self.grid.cols, self.grid.rows = cols, rows
        self.grid.cache.clear()
        self.grid.set_page(first // self.grid.per_page)

    def update_status(self) -> None:
        if self.ws is None:
            return
        self.counts_label.setText(self.ws.stats_line())
        if self.mode == "grid":
            g = self.grid
            n_marked = sum(1 for m in g.marks.values() if m == "people")
            n_skip = sum(1 for m in g.marks.values() if m == "skip")
            self.status_label.setText(f"GRID  {self.queue_box.currentText()}: {len(self.queue)} frames, page "
                                      f"{g.page + 1}/{g.n_pages}  |  marked {n_marked}, skip {n_skip}")
        elif self.queue and self.cur_path is not None:
            fi = self.queue[self.idx]
            st = fi.status or "new"
            ui = "  UI? (X to skip)" if fi.ui < self.project.ui_threshold else ""
            cls = self.project.class_names[self.canvas.cls]
            self.status_label.setText(f"EDIT {self.idx + 1}/{len(self.queue)}  {fi.stem}  [{st}{'*' if self.canvas.dirty else ''}]"
                                      f"{ui}  class {self.canvas.cls + 1}={cls}  boxes {len(self.canvas.boxes)}  "
                                      f"zoom {self.canvas.view.zoom:.1f}x")

    # ================================================================ keys
    def eventFilter(self, obj: QObject, e: QEvent) -> bool:
        if e.type() != QEvent.Type.KeyPress or not self.isActiveWindow():
            return False
        fw = QApplication.focusWidget()
        if isinstance(fw, (QLineEdit, QAbstractSpinBox, QPlainTextEdit)) or (isinstance(fw, QComboBox) and fw.isEditable()):
            return False
        if fw is not None and fw.window() is not self:
            return False
        name = key_name(e.key(), e.nativeVirtualKey(), e.modifiers())
        if not name:
            return False
        return self.handle_key(name)

    def handle_key(self, name: str) -> bool:
        if self.ws is None:
            return False
        if self.mode == "edit" and name.isdigit() and name != "0":
            k = int(name) - 1
            if k < len(self.project.classes):
                self.class_list.setCurrentRow(k)
                return True
        if name == "W":
            self.weak_check.toggle()
            return True
        if name == "Esc":
            if self.mode == "grid" and self.grid.preview is not None:
                self.grid.close_preview()
                return True
            if self.isFullScreen():
                self.showNormal()
                return True
            return False
        if self.mode == "grid" and self.grid.preview is not None:
            if name == "Z":
                self.grid.preview_view.reset()
                self.grid.update()
            else:
                self.grid.close_preview()
            return True
        grid_actions = ("commit_page", "prev_page", "next_page", "grid_view", "switch_mode", "fullscreen")
        for action in self.keymap.actions_for(name):
            if self.mode == "grid" and action not in grid_actions:
                continue
            if self.mode == "edit" and action in ("commit_page", "prev_page", "next_page", "grid_view"):
                continue
            self.do(action)
            return True
        return False

    def do(self, action: str) -> None:
        if self.ws is None:
            return
        c = self.canvas
        if action == "switch_mode":
            self.switch_mode()
        elif action == "fullscreen":
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif self.mode == "grid":
            self.grid_action(action)
        elif not self.queue or self.cur_path is None:
            return
        elif action == "accept_next":
            self.commit_current(force=True)
            self.go(self.idx + 1, carry=True)
        elif action == "empty_next":
            c.boxes = []
            self.commit_current(status="neg")
            self.go(self.idx + 1)
        elif action == "skip_next":
            self.commit_current(status="skip")
            self.go(self.idx + 1)
        elif action == "prev":
            self.go(self.idx - 1)
        elif action == "next":
            self.go(self.idx + 1, carry=True)
        elif action == "next_unreviewed":
            nxt = next((i for i in range(self.idx + 1, len(self.queue)) if self.queue[i].status in (None, "todo")), None)
            if nxt is None:
                self.status_label.setText("No unreviewed frames after this one.")
            else:
                self.go(nxt)
        elif action == "undo":
            c.undo()
        elif action == "redo":
            c.redo()
        elif action == "delete_box":
            c.delete_selected()
        elif action == "toggle_class":
            t = c.target_box()
            if t >= 0 and len(self.project.classes) > 1:
                c.set_class_of(t, (c.boxes[t].cls + 1) % len(self.project.classes))
        elif action == "reset_proposals":
            c.replace_boxes(self.ws.proposals(self.cur_path, c.iw, c.ih))
        elif action == "accept_proposals":
            c.accept_all()
        elif action == "toggle_boxes":
            c.show_boxes = not c.show_boxes
            c.update()
        elif action == "toggle_loupe":
            c.loupe = not c.loupe
            c.update()
        elif action in ("loupe_in", "loupe_out") and c.loupe:
            c.change_loupe_zoom(1.25 if action == "loupe_in" else 0.8)
        elif action == "fit_view":
            c.view.reset()
            c.update()
        elif action in ("copy_prev", "propagate_prev"):
            self.from_previous(track=action == "propagate_prev")
        elif action == "sam_fit":
            t = c.target_box()
            if t >= 0:
                self.sam_fit(c.boxes[t], index=t)
        elif action == "save":
            self.commit_current(force=True)
        self.update_status()

    def grid_action(self, action: str) -> None:
        g = self.grid
        if action == "grid_view":
            n = self.queue_box.count()
            self.queue_box.setCurrentIndex((self.queue_box.currentIndex() + 1) % n)
        elif action == "commit_page":
            self.ws.commit_page(g.stems_on_page(), g.marks)
            for i in range(g.page * g.per_page, min((g.page + 1) * g.per_page, len(self.queue))):
                self.queue[i] = self.ws.index[self.ws.by_stem[self.queue[i].stem]]
                self.update_list_item(i)
            if g.page + 1 >= g.n_pages:
                if self.queue_box.currentData() == "triage":
                    self.status_label.setText("Grid finished: switching to the editor.")
                    self.set_mode("edit")
                    return
                self.status_label.setText("End of this view.")
                g.set_page(g.page)
            else:
                g.set_page(g.page + 1)
        elif action == "prev_page":
            g.set_page(g.page - 1)
        elif action == "next_page":
            g.set_page(g.page + 1)
        self.frame_list.setCurrentRow(min(g.page * g.per_page, max(0, len(self.queue) - 1)))

    def from_previous(self, track: bool) -> None:
        """Boxes of the previous frame in time order (not the queue): copied, or tracked onto this frame."""
        stems = list(self.ws.by_stem)
        i = self.ws.by_stem.get(self.cur_path.stem, 0)
        prev = next((self.ws.index[j] for j in range(i - 1, -1, -1) if self.ws.index[j].status == "pos"), None)
        if prev is None:
            self.status_label.setText("No labeled frame before this one.")
            return
        img = cv2.imread(str(prev.path))
        if img is None:
            return
        h, w = img.shape[:2]
        boxes, _ = self.ws.load_boxes(prev.path, w, h)
        c = self.canvas
        if track and img.shape[:2] == c.img.shape[:2]:
            props = [b for b in self.ws.proposals(self.cur_path, c.iw, c.ih)]
            boxes = propagate(img, c.img, boxes, props)
        c.replace_boxes(c.boxes_copy() + boxes if not track else boxes)
        self.status_label.setText(f"{len(boxes)} boxes from {prev.stem}" + (" (tracked)" if track else ""))
        del stems

    # ================================================================ SAM
    def sam_fit(self, prompt, index: int | None = None) -> None:
        from framelabel.ml.guard import GuardError, check_gpu_allowed
        from framelabel.ml.models import find_sam
        from framelabel.ml.sam import BoxFitter

        c = self.canvas
        if c.img is None:
            return
        try:
            check_gpu_allowed(self.project.game_processes or ["UAGame.exe"])
        except GuardError as e:
            self.status_label.setText(str(e))
            return
        if self.fitter is None:
            weights = find_sam(self.model_dirs())
            if weights is None:
                QMessageBox.information(self, "SAM", "No SAM weights (e.g. sam2.1_b.pt) found in the model folders.")
                return
            self.fitter = BoxFitter(weights)
            self.status_label.setText("Loading SAM...")
            QApplication.processEvents()
        if isinstance(prompt, Box):
            if index is None:
                index = len(c.boxes) - 1  # the box just drawn
            new = self.fitter.fit(c.img, box=prompt)
            if new is None:
                self.status_label.setText("SAM found nothing there.")
                return
            new.cls = c.boxes[index].cls if 0 <= index < len(c.boxes) else c.cls
            c.snapshot()
            c.boxes[index] = new
            c.edited()
        else:
            new = self.fitter.fit(c.img, point=prompt)
            if new is None:
                self.status_label.setText("SAM found nothing there.")
                return
            new.cls = c.cls
            c.add_box(new)
        self.status_label.setText(f"SAM box {new.w:.0f} x {new.h:.0f} px")

    # ================================================================ auto-label, export, training
    def model_dirs(self) -> list[Path]:
        dirs = list(self.project.model_dirs) if self.project else []
        return dirs + [ROOT / "models", ROOT / "train_runs"]

    def gpu_status(self) -> None:
        from framelabel.ml.guard import gpu_compute_processes, running_processes

        game = running_processes(self.project.game_processes if self.project else ["UAGame.exe"])
        procs = gpu_compute_processes()
        lines = [f"Game running: {', '.join(game) if game else 'no'}", "", "GPU compute processes:"]
        lines += [f"  {pid} {name} {mem}" for pid, name, mem in procs] or ["  none"]
        QMessageBox.information(self, "GPU / game", "\n".join(lines))

    def gpu_preflight(self) -> bool:
        from framelabel.ml.guard import GuardError, check_gpu_allowed, gpu_compute_processes

        try:
            check_gpu_allowed(self.project.game_processes or ["UAGame.exe"])
        except GuardError as e:
            QMessageBox.critical(self, "GPU work blocked", str(e))
            return False
        others = [p for p in gpu_compute_processes() if "python" in p[1].lower()]
        if others:
            txt = "\n".join(f"{pid} {name}" for pid, name, _ in others)
            r = QMessageBox.question(self, "GPU busy", f"Other Python GPU processes are running:\n{txt}\n\n"
                                     "Running GPU jobs in parallel can exhaust memory. Continue anyway?")
            return r == QMessageBox.StandardButton.Yes
        return True

    def prelabel_dialog(self) -> None:
        from .dialogs import PrelabelDialog

        if self.project is None:
            return
        dlg = PrelabelDialog(self.project, self.model_dirs(), self.sessions, self.session_box.currentIndex(), self)
        if not dlg.exec() or not self.gpu_preflight():
            return
        cfg, sessions, redo = dlg.result_config()
        self.commit_current()
        from framelabel.ml.predict import prelabel_sessions

        def run(progress, cancelled):
            from framelabel.ml.models import model_class_names

            cfg.class_maps = []
            for m in cfg.models:
                names = model_class_names(str(m))
                cfg.class_maps.append({i: self.project.map_model_class(n, i, len(names)) for i, n in enumerate(names)})
            return prelabel_sessions(sessions, cfg, redo, progress, cancelled, self.project.ignore_zones)

        self.run_job(Job(run), "Prelabeling", self.after_prelabel)

    def after_prelabel(self, stats: dict) -> None:
        QMessageBox.information(self, "Prelabel", f"{stats['done']} of {stats['frames']} frames, "
                                f"{stats['with_proposals']} with proposals, {stats['seconds']:.0f} s")
        if self.ws is not None:
            self.ws.reload()
            self.set_mode(self.mode, self.queue_box.currentData())

    def run_job(self, job: Job, title: str, done) -> None:
        if self.job is not None:
            QMessageBox.information(self, "Busy", "Another job is running.")
            return
        self.job = job
        self.progress.setVisible(True)
        self.cancel_btn.setVisible(True)
        self.progress.setRange(0, 0)

        def prog(i, n, msg):
            self.progress.setRange(0, max(n, 1))
            self.progress.setValue(i)
            self.status_label.setText(f"{title}: {i}/{n}  {msg}")

        def finish(result):
            self.job = None
            self.progress.setVisible(False)
            self.cancel_btn.setVisible(False)
            done(result)

        def fail(msg):
            self.job = None
            self.progress.setVisible(False)
            self.cancel_btn.setVisible(False)
            QMessageBox.critical(self, title, msg)

        job.progress.connect(prog)
        job.finished.connect(finish)
        job.failed.connect(fail)
        start_job(self, job)

    def cancel_job(self) -> None:
        if self.job is not None:
            self.job.cancel_requested = True
            self.status_label.setText("Cancelling after the current batch...")

    def export_dialog(self) -> None:
        from .dialogs import ExportDialog

        if self.project is None:
            return
        self.commit_current()
        dlg = ExportDialog(self.project, self.sessions, self)
        if dlg.exec():
            res = dlg.result
            if res is not None:
                QMessageBox.information(self, "Export", res.summary(self.project.class_names))

    def retrain_dialog(self) -> None:
        from .dialogs import RetrainDialog

        if self.project is None:
            return
        self.commit_current()
        RetrainDialog(self.project, self.sessions, self.model_dirs(), self).exec()

    def project_settings(self) -> None:
        from .dialogs import ProjectDialog

        if self.project is None:
            return
        dlg = ProjectDialog(self.project, self)
        if dlg.exec():
            dlg.apply()
            if self.project.path:
                self.project.save()
            self.open_target(self.project.path or self.project.sources[0])

    def keyboard_dialog(self) -> None:
        from .dialogs import KeymapDialog

        dlg = KeymapDialog(self.keymap, self)
        if dlg.exec():
            self.keymap = dlg.keymap
            self.settings.setValue("keymap", json.dumps(dlg.overrides()))
            self.update_help()

    def closeEvent(self, e) -> None:
        self.commit_current()
        self.settings.setValue("geometry", self.saveGeometry())
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(e)


def install_excepthooks(app) -> None:
    """Log unhandled exceptions (PyQt6 aborts the process on one inside a slot) and tell the user."""
    def hook(tp, value, tb):
        log.error("unhandled exception", exc_info=(tp, value, tb))
        if threading.current_thread() is threading.main_thread():  # dialogs only from the GUI thread
            QMessageBox.critical(None, "framelabel error", f"{tp.__name__}: {value}\n\nDetails are in the log.")

    sys.excepthook = hook
    threading.excepthook = lambda a: log.error("unhandled exception in thread %s", a.thread and a.thread.name,
                                               exc_info=(a.exc_type, a.exc_value, a.exc_traceback))


def run_app(target: Path | None = None) -> int:
    log_dir = setup_logging()
    app = QApplication.instance() or QApplication(sys.argv)
    install_excepthooks(app)
    app.setApplicationName("framelabel")
    app.setStyle("Fusion")
    from PyQt6.QtGui import QPalette

    pal = QPalette()
    for role, color in ((QPalette.ColorRole.Window, "#2b2b2b"), (QPalette.ColorRole.WindowText, "#e6e6e6"),
                        (QPalette.ColorRole.Base, "#1f1f1f"), (QPalette.ColorRole.AlternateBase, "#262626"),
                        (QPalette.ColorRole.Text, "#e6e6e6"), (QPalette.ColorRole.Button, "#353535"),
                        (QPalette.ColorRole.ButtonText, "#e6e6e6"), (QPalette.ColorRole.Highlight, "#3d6fb4"),
                        (QPalette.ColorRole.HighlightedText, "#ffffff"), (QPalette.ColorRole.ToolTipBase, "#2b2b2b"),
                        (QPalette.ColorRole.ToolTipText, "#e6e6e6")):
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    w = MainWindow(target)
    w.show()
    watchdog = None
    if os.environ.get("FRAMELABEL_WATCHDOG", "1") != "0":
        from .watchdog import Watchdog

        watchdog = Watchdog(log_dir)
    code = app.exec()
    if watchdog is not None:
        watchdog.stop()
    return code
