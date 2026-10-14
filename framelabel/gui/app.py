"""Main window: sessions and review queues on the left, the grid / box editor in the centre, classes and the frame's
boxes on the right. All decisions are saved immediately (labels/<stem>.txt and review.json)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
from PyQt6.QtCore import QEvent, QObject, QSettings, QSortFilterProxyModel, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QActionGroup, QColor, QIcon, QKeySequence, QPainter, QPixmap
from PyQt6.QtWidgets import (QMenu, QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox,
                             QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListView,
                             QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
                             QPushButton, QSizePolicy, QSplitter, QStackedWidget, QTableWidget, QToolButton, QVBoxLayout,
                             QWidget)

from framelabel.core.boxes import Box, grow_box
from framelabel.core.images import imread
from framelabel.core.project import Project, folder_project
from framelabel.core.propagate import propagate
from framelabel.core.queues import FrameInfo
from framelabel.core.session import Session, session_overview
from framelabel.core.pace import Pace
from framelabel.core.queues import SORTS
from framelabel.core.views import BY_KEY, all_views, status_counts
from framelabel.core.workspace import Workspace
from framelabel.core import index_cache
from framelabel.perf import setup_logging, timed
from .canvas import Canvas, to_qimage
from .frame_list import FrameListModel
from .peek import FramePeek
from .grid import Grid
from .keys import ACTION_INFO, Keymap, key_name
from .theme import T, apply_theme
from .palette import CommandPalette, Entry
from .panels import (CoachCard, Central, ContextBar, GRID_SIZES, SwitchBox, ViewTree, grid_size_text, note_label,
                     section_label)
from .process_job import ProcessJob
from .workers import Job, start_job, stop_jobs
from .workflow import STAGE_KEYS, WorkflowMixin

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("framelabel")
SIDE_WIDTH_LEFT, SIDE_WIDTH_RIGHT = 224, 236   # the side panels stay narrow: the photos get the rest
STAGE_BY_ACTION = {action: stage for stage, action in STAGE_KEYS.items()}
COUNTS_DEBOUNCE_MS = 400   # the counts next to the views are recomputed this long after the last change
CONF_PRESETS = (0.10, 0.20, 0.30, 0.50)
GRID_ONLY_KEYS = ("arrow_left", "arrow_right", "arrow_up", "arrow_down", "tile_mark", "tile_skip")
GRID_FOCUS_KEYS = GRID_ONLY_KEYS   # only while the grid has the keyboard focus: a list or the tree keeps its arrows and Enter
ARROW_MODES = (("tiles", "Move between tiles", "Left / Right / Up / Down move the cursor over the tiles of the page; "
                "pages are flipped with A / D or PageUp / PageDown"),
               ("pages", "Flip pages", "Left / Up: previous page, Right / Down: next page; the tile cursor is off"))
ARROW_TEXT = {k: t for k, t, _h in ARROW_MODES}


def read_frame(path: Path):
    """(image, QImage) or (None, None); runs in the prefetch threads (QImage, unlike QPixmap, is safe there)."""
    img = imread(str(path))
    return (img, to_qimage(img)) if img is not None else (None, None)


def color_icon(color: str, size: int = 12) -> QIcon:
    """A round colour dot (class colours in the lists)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(color))
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pm)


def make_settings() -> QSettings:
    """The user's settings (registry on Windows); FRAMELABEL_SETTINGS_FILE selects an INI file instead (tests)."""
    path = os.environ.get("FRAMELABEL_SETTINGS_FILE")
    if path:
        return QSettings(path, QSettings.Format.IniFormat)
    return QSettings("framelabel", "framelabel")


class MainWindow(WorkflowMixin, QMainWindow):
    session_loaded = pyqtSignal()  # a session's workspace was applied (also used by tests)

    def __init__(self, target: Path | None = None):
        super().__init__()
        self.settings = make_settings()
        self.keymap = Keymap(json.loads(self.settings.value("keymap", "{}") or "{}"))
        self.prefetch_ahead = 2
        self.apply_perf_settings()
        self.project: Project | None = None
        self.sessions: list[Session] = []
        self.ws: Workspace | None = None
        self.mode = "grid"
        self.panels_visible = True
        self.view_key = "triage"  # the view (set of frames) shown; the mode says how it is shown
        self.sort_key = str(self.settings.value("sort", "time"))  # the order of the Label stage
        if self.sort_key not in dict(SORTS):
            self.sort_key = "time"
        self.pace = Pace()
        self.lead: str | None = None  # what was just done to a frame: the end-of-view message starts with it
        self.undo_pages: list[dict] = []  # saved grid pages that can be taken back, newest last
        self.queue: list[FrameInfo] = []
        self.idx = 0
        self.cur_path: Path | None = None
        self.prefetch: dict[Path, Future] = {}
        self.pool = ThreadPoolExecutor(2)
        self.sam_worker = None  # SamWorker: its own Python thread, started on the first use
        self.sam_weights: Path | None = None
        self.sam_seq = 0
        self.sam_pending: dict[int, dict] = {}
        self.job = None
        self.job_title = ""
        self.closing = False
        self.loader: Job | None = None  # the session being loaded in the background
        self.reloader: Job | None = None  # proposals / new frames being re-read in the background
        self.overview_job: Job | None = None  # frame and review counts for the session box
        self.counts_job: Job | None = None  # frame counts of the views, computed in the background
        self.counts_by_view: dict[str, int] = {}
        self.counts_timer = QTimer(self)  # recompute the counts once the decisions stop coming
        self.counts_timer.setSingleShot(True)
        self.counts_timer.setInterval(COUNTS_DEBOUNCE_MS)
        self.counts_timer.timeout.connect(self.refresh_counts)
        self.conf_timer = QTimer(self)  # apply the shown confidence once the spin box stops changing
        self.conf_timer.setSingleShot(True)
        self.conf_timer.setInterval(250)
        self.conf_timer.timeout.connect(self.apply_conf)
        self.log_dir: Path | None = None  # where run_app keeps the logs (the freeze watchdog writes there)
        self.watchdog = None
        self.setWindowTitle("framelabel")
        self.resize(1600, 950)
        self.build_ui()
        self.build_menus()
        QApplication.instance().installEventFilter(self)
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        state = self.settings.value("dock_state")
        self.docks_sized = bool(state) and self.restoreState(state)  # the panel widths you left them at
        if not self.docks_sized:
            self.size_docks()
        if target is not None:
            QTimer.singleShot(0, lambda: self.open_target(Path(target)))
        elif (recent := self.recent()) and Path(recent[0]).exists():
            QTimer.singleShot(0, lambda: self.open_target(Path(recent[0])))

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if not getattr(self, "docks_sized", False):
            self.docks_sized = True
            QTimer.singleShot(0, self.size_docks)  # the dock widths can only be set once the window is laid out

    def setting_bool(self, key: str, default: bool) -> bool:
        v = self.settings.value(key, default)
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")

    def apply_watchdog(self) -> None:
        """Start or stop the freeze watchdog as the performance setting says (it needs run_app's log folder)."""
        want = (self.log_dir is not None and os.environ.get("FRAMELABEL_WATCHDOG", "1") != "0"
                and self.setting_bool("perf/watchdog", True))
        if want and self.watchdog is None:
            from .watchdog import Watchdog

            self.watchdog = Watchdog(self.log_dir, self)
        elif not want and self.watchdog is not None:
            self.watchdog.stop()
            self.watchdog.deleteLater()
            self.watchdog = None

    def apply_perf_settings(self) -> None:
        """Performance preferences (File > Performance settings); the watchdog is read at startup by run_app."""
        try:
            index_cache.IO_WORKERS = min(max(int(self.settings.value("perf/io_threads", 8)), 1), 32)
            self.prefetch_ahead = min(max(int(self.settings.value("perf/prefetch", 2)), 0), 8)
        except (TypeError, ValueError):
            pass

    # ================================================================ UI
    def build_ui(self) -> None:
        self.canvas = Canvas()
        self.grid = Grid()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.grid)
        self.stack.addWidget(self.canvas)
        self.context_bar = ContextBar()
        self.central = Central(self.context_bar, self.stack)
        self.setCentralWidget(self.central)
        self.mode_switch = self.context_bar.mode_switch
        self.mode_switch.mode_chosen.connect(self.on_mode_chosen)
        self.context_bar.grid_size_chosen.connect(self.set_grid_size)
        self.context_bar.sort_chosen.connect(self.on_sort_chosen)
        self.context_bar.set_sort(self.sort_key)
        self.grid.open_requested.connect(self.open_in_editor)
        self.grid.locked.connect(lambda _stem: self.toast(
            f"Labeled frame: double-click it (or press {self.keymap.first('open_frame')}) to edit its boxes"))
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
        self.build_left_dock()
        self.peek = FramePeek(self.frame_list, self.frame_at_index, self.grid, self)
        self.peek.set_enabled(str(self.settings.value("peek", "true")).lower() in ("1", "true", "yes"))
        self.build_frame_dock()
        self.build_tools_dock()
        self.build_status_bar()

    def build_left_dock(self) -> None:
        """Session, how far you are, grid / editor, the views with their counts, and the frames of the current view."""
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 8, 10, 6)
        lv.setSpacing(6)
        self.session_box = QComboBox()
        self.session_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.session_box.setMinimumContentsLength(6)
        self.session_box.currentIndexChanged.connect(self.on_session_changed)
        self.report_btn = QToolButton()
        self.report_btn.setText("Report")
        self.report_btn.setProperty("quiet", True)
        self.report_btn.setToolTip("Session report: counts of everything hidden from the normal views (recorder, interface "
                                   "screens, ignore zones, weak proposals, dropped classes), each with a view")
        self.report_btn.clicked.connect(self.session_report)
        row = QHBoxLayout()
        row.addWidget(self.session_box, 1)
        row.addWidget(self.report_btn)
        lv.addLayout(row)
        self.done_bar = QProgressBar()
        self.done_bar.setTextVisible(False)
        self.done_bar.setToolTip("Frames with a decision (labeled, empty or skipped) out of all frames of the session")
        self.counts_label = QLabel()
        self.counts_label.setWordWrap(True)
        # long summaries must wrap, not widen the dock
        self.counts_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self.counts_label.setProperty("role", "note")
        self.counts_label.setTextFormat(Qt.TextFormat.RichText)
        self.pace_label = QLabel()
        self.pace_label.setProperty("role", "note")
        self.pace_label.setVisible(False)
        lv.addWidget(self.done_bar)
        lv.addWidget(self.counts_label)
        lv.addWidget(self.pace_label)

        self.view_tree = ViewTree()
        self.view_tree.view_chosen.connect(self.on_view_chosen)
        self.frame_model = FrameListModel(self)
        self.frame_proxy = QSortFilterProxyModel(self)  # the filter box above the list
        self.frame_proxy.setSourceModel(self.frame_model)
        self.frame_proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.frame_filter = QLineEdit()
        self.frame_filter.setPlaceholderText("Filter: 1024, new, empty")
        self.frame_filter.setClearButtonEnabled(True)
        self.frame_filter.textChanged.connect(self.frame_proxy.setFilterFixedString)
        self.frame_list = QListView()
        self.frame_list.setModel(self.frame_proxy)
        self.frame_list.setUniformItemSizes(True)
        self.frame_list.setEditTriggers(QListView.EditTrigger.NoEditTriggers)
        self.frame_list.activated.connect(self.on_frame_clicked)
        self.frame_list.clicked.connect(self.on_frame_clicked)
        self.frames_title = section_label("Frames")
        self.coach = CoachCard()
        lower = QWidget()
        lw = QVBoxLayout(lower)
        lw.setContentsMargins(0, 0, 0, 0)
        for w in (self.coach, self.frames_title, self.frame_filter, self.frame_list):
            lw.addWidget(w)
        split = self.left_split = QSplitter(Qt.Orientation.Vertical)
        self.view_tree.expanded.connect(lambda _i: self.fit_view_tree())
        self.view_tree.collapsed.connect(lambda _i: self.fit_view_tree())
        split.addWidget(self.view_tree)
        split.addWidget(lower)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        lv.addWidget(split, 1)
        dock = QDockWidget("Frames", self)
        dock.setObjectName("frames")
        dock.setTitleBarWidget(QWidget())
        dock.setWidget(left)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.left_dock = dock

    def fit_view_tree(self) -> None:
        """Give the view list the height of all its rows (no scrolling), the rest of the panel to the frame list."""
        total = sum(self.left_split.sizes()) or self.left_split.height()
        h = min(self.view_tree.sizeHint().height(), max(total - 160, 120))
        self.left_split.setSizes([h, max(total - h, 120)])

    def build_frame_dock(self) -> None:
        """Right, top (editor only): the class for new boxes and the boxes of this frame. No title bar: the panel says
        what it is by its first heading, and every pixel not spent on chrome goes to the photo."""
        self.frame_dock = QDockWidget("This frame", self)
        self.frame_dock.setObjectName("labels")
        self.frame_dock.setTitleBarWidget(QWidget())
        ep = QWidget()
        rv = QVBoxLayout(ep)
        rv.setContentsMargins(10, 4, 10, 6)
        rv.setSpacing(2)
        rv.addWidget(section_label("Class for new boxes  (1-9)"))
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(96)
        self.class_list.currentRowChanged.connect(self.on_class_row)
        rv.addWidget(self.class_list)
        rv.addWidget(section_label("Boxes on this frame"))
        self.box_table = QTableWidget(0, 3)
        self.box_table.setHorizontalHeaderLabels(["class", "conf", "size px"])
        self.box_table.verticalHeader().setVisible(False)
        self.box_table.verticalHeader().setDefaultSectionSize(22)
        self.box_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.box_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.box_table.setShowGrid(False)
        self.box_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.box_table.customContextMenuRequested.connect(self.box_table_delete)
        self.box_table.cellClicked.connect(lambda r, _c: self.canvas.select(r, center=False))
        self.box_table.cellDoubleClicked.connect(lambda r, _c: self.canvas.select(r, center=True))
        hh = self.box_table.horizontalHeader()
        hh.setSectionResizeMode(0, hh.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, hh.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, hh.ResizeMode.ResizeToContents)
        self.box_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rv.addWidget(self.box_table, 1)
        self.frame_dock.setWidget(ep)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.frame_dock)

    KEY_OF = {"undo_page": "undo"}  # a button whose action is named differently from the key it shows
    ACTION_TITLES = {"undo_page": "Undo page", "open_frame": "Open in editor", "mark_checked": "Mark checked",
                     "accept_next": "Save + next", "empty_next": "Empty", "skip_next": "Skip",
                     "prev": "Previous", "next": "Next", "undo": "Undo", "copy_prev": "Copy prev",
                     "propagate_prev": "Track now", "sam_fit": "SAM fit",
                     "accept_proposals": "Accept all", "commit_page": "Commit page",
                     "prev_page": "Previous page", "next_page": "Next page", "grid_view": "Next stage",
                     "reload": "Reload"}

    def action_page(self, actions: tuple[str, ...], wide: str) -> QWidget:
        """Buttons that do what the keys do, two per row; the `wide` one (the main action) spans both columns."""
        page = QWidget()
        lay = QGridLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        row = col = 0
        for action in actions:
            b = QPushButton()
            b.setProperty("primary", action == wide)  # the main action of the mode is the coloured one
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # the keys keep working on the canvas / grid
            b.clicked.connect(lambda _=False, a=action: self.do(a))
            self.action_buttons[action] = b
            if action == wide:
                lay.addWidget(b, row, 0, 1, 2)
                row, col = row + 1, 0
                continue
            lay.addWidget(b, row, col)
            col += 1
            if col == 2:
                row, col = row + 1, 0
        return page

    def build_tools_dock(self) -> None:
        """Right, bottom: one-click actions for the current mode and the settings that change what you see."""
        tools = QWidget()
        tv = QVBoxLayout(tools)
        tv.setContentsMargins(10, 4, 10, 8)
        tv.setSpacing(4)
        tv.addWidget(section_label("Actions"))
        self.action_buttons: dict[str, QPushButton] = {}
        self.action_pages = SwitchBox()
        self.action_pages.addWidget(self.action_page(("commit_page", "prev_page", "next_page", "undo_page", "open_frame",
                                                      "grid_view", "reload"), "commit_page"))
        self.action_pages.addWidget(self.action_page(("accept_next", "empty_next", "skip_next", "prev", "next", "undo",
                                                      "mark_checked", "accept_proposals", "copy_prev", "propagate_prev",
                                                      "sam_fit"), "accept_next"))
        tv.addWidget(self.action_pages)

        tv.addWidget(section_label("Proposals from confidence"))
        row = QHBoxLayout()
        row.setSpacing(4)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(float(self.settings.value("conf", 0.3)))
        self.conf_spin.setToolTip("The model's boxes at or above this confidence are the ones you edit; frames where none "
                                  "reaches it are 'model found nothing' (triage). Applied at once, remembered per project.")
        self.conf_spin.valueChanged.connect(self.on_conf_changed)
        self.conf_spin.setFixedWidth(64)
        row.addWidget(self.conf_spin)
        for v in CONF_PRESETS:
            b = QPushButton(f"{v:.2f}")
            b.setProperty("chip", True)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setToolTip(f"Show proposals from confidence {v:.2f}")
            b.clicked.connect(lambda _=False, x=v: self.conf_spin.setValue(x))
            row.addWidget(b, 1)
        tv.addLayout(row)
        self.weak_check = QCheckBox("Weak proposals  (W)")
        self.weak_check.setToolTip("Draw the proposals below the confidence faintly; click one to accept it")
        self.weak_check.toggled.connect(self.on_weak_toggled)
        tv.addWidget(self.weak_check)

        tv.addWidget(section_label("Editor helpers"))
        self.sam_check = QCheckBox("SAM-fit new boxes")
        self.sam_check.setToolTip("Every box you draw is snapped to the object by SAM (needs the GPU, game closed)")
        self.sam_check.toggled.connect(lambda v: setattr(self.canvas, "sam_on_draw", v))
        self.carry_check = QCheckBox("Auto-track to next frame")
        self.carry_check.setToolTip("The on/off switch for tracking. On: moving to a new frame without proposals tracks "
                                    "the previous frame's boxes onto it by itself. Off: nothing is carried over (the "
                                    "'Track now' button and T still track once, on request)")
        self.carry_check.setChecked(str(self.settings.value("auto_track", "false")).lower() in ("1", "true", "yes"))
        self.carry_check.toggled.connect(self.on_carry_toggled)
        tv.addWidget(self.sam_check)
        tv.addWidget(self.carry_check)
        tv.addStretch(1)
        links = QHBoxLayout()
        for text, slot, tip in (("Search  Ctrl+K", lambda: self.do("command_palette"), "Find a view, an action or a frame by typing"),
                                ("Help  F1", self.help_dialog, "Keys and mouse for each mode")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setProperty("quiet", True)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(slot)
            links.addWidget(b)
        tv.addLayout(links)
        dock = QDockWidget("Actions and settings", self)
        dock.setObjectName("tools")
        dock.setTitleBarWidget(QWidget())
        dock.setWidget(tools)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.tools_dock = dock

    def refresh_action_texts(self) -> None:
        for action, b in self.action_buttons.items():
            key = self.keymap.first(self.KEY_OF.get(action, action)).replace("Ctrl+", "^").replace("Shift+", "⇧")
            b.setText(f"{self.ACTION_TITLES[action]}   {key}" if key else self.ACTION_TITLES[action])
            b.setToolTip(ACTION_INFO[self.KEY_OF.get(action, action)][1] if self.KEY_OF.get(action, action) in ACTION_INFO else "")
        if getattr(self, "view_key", None) == "checked":  # there, saving the page means giving the frames back to Review
            b = self.action_buttons["commit_page"]
            b.setText(b.text().replace("Commit page", "Return to Review"))
            b.setToolTip("Return the ticked tiles (the whole page if none is ticked) to Review")

    def apply_panels(self) -> None:
        """Side panels on / off (Ctrl+B); the box list only exists for the editor."""
        on = self.panels_visible
        self.left_dock.setVisible(on)
        self.tools_dock.setVisible(on)
        self.frame_dock.setVisible(on and self.mode == "edit")

    def size_docks(self) -> None:
        self.resizeDocks([self.left_dock, self.tools_dock], [SIDE_WIDTH_LEFT, SIDE_WIDTH_RIGHT], Qt.Orientation.Horizontal)

    def build_status_bar(self) -> None:
        self.status_label = QLabel()
        self.pos_label = QLabel()
        # a fixed width: the text changes on every mouse move and would re-layout the status bar each time
        self.pos_label.setMinimumWidth(self.pos_label.fontMetrics().horizontalAdvance("new box 99999 x 99999 px") + 8)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.load_progress = QProgressBar()
        self.load_progress.setMaximumWidth(200)
        self.load_progress.setVisible(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_job)
        self.keys_label = QLabel()
        sb = self.statusBar()
        sb.addWidget(self.status_label, 1)
        sb.addPermanentWidget(self.keys_label)
        sb.addPermanentWidget(self.pos_label)
        sb.addPermanentWidget(self.load_progress)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.cancel_btn)

    def toast(self, text: str, action: tuple | None = None) -> None:
        """Say what an action just did, over the work area (the status bar line is easy to miss). `action` is a
        (button text, callable) such as ('Undo', self.undo_page)."""
        self.central.toast.show_message(text, action=action)

    def act(self, menu, text: str, slot, action: str | None = None, checkable: bool = False, shortcut: str | None = None):
        a = QAction(text, self)
        a.triggered.connect(slot)
        a.setCheckable(checkable)
        if action:
            a.setData(action)  # key-driven: the command palette lists it from the key map, not from the menu
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
        self.act(m, "Export YOLO dataset...", self.export_dialog, shortcut="Ctrl+E")
        m.addSeparator()
        self.act(m, "Quit", self.close, shortcut="Ctrl+Q")

        m = mb.addMenu("&Edit")
        self.act(m, "Save and next", lambda: self.do("accept_next"), "accept_next")
        self.act(m, "Empty frame and next", lambda: self.do("empty_next"), "empty_next")
        self.act(m, "Skip frame and next", lambda: self.do("skip_next"), "skip_next")
        self.act(m, "Mark frame as checked (hides it from Review)", lambda: self.do("mark_checked"), "mark_checked")
        m.addSeparator()
        self.act(m, "Undo (boxes, or the last saved grid page)", lambda: self.do("undo"), "undo")
        self.act(m, "Undo the last saved grid page", self.undo_page)
        self.act(m, "Redo", lambda: self.do("redo"), "redo")
        self.act(m, "Delete selected box", lambda: self.do("delete_box"), "delete_box")
        self.act(m, "Toggle class of box", lambda: self.do("toggle_class"), "toggle_class")
        self.act(m, "Accept all proposals as labels", lambda: self.do("accept_proposals"), "accept_proposals")
        self.act(m, "Reset to model proposals", lambda: self.do("reset_proposals"), "reset_proposals")
        self.act(m, "Go to frame...", lambda: self.do("goto_frame"), "goto_frame")
        m.addSeparator()
        self.act(m, "Copy boxes from previous frame", lambda: self.do("copy_prev"), "copy_prev")
        self.act(m, "Track boxes from previous frame (once, this frame)", lambda: self.do("propagate_prev"), "propagate_prev")
        self.act(m, "Auto-track to next frames on / off", lambda: self.do("toggle_carry"), "toggle_carry")
        self.act(m, "Fit selected box with SAM (Alt+click: from a point)", lambda: self.do("sam_fit"), "sam_fit")
        self.act(m, "Fit all boxes on this frame with SAM", lambda: self.do("sam_fit_all"))

        m = mb.addMenu("&View")
        self.act(m, "Search views, actions and frames...", lambda: self.do("command_palette"), "command_palette")
        m.addSeparator()
        self.act(m, "Go to Triage", lambda: self.do("go_triage"), "go_triage")
        self.act(m, "Go to Label", lambda: self.do("go_label"), "go_label")
        self.act(m, "Go to Review", lambda: self.do("go_review"), "go_review")
        self.act(m, "Next stage that has frames", lambda: self.do("grid_view"), "grid_view")
        m.addSeparator()
        self.act(m, "Switch grid / editor", lambda: self.do("switch_mode"), "switch_mode")
        self.act(m, "Fit image", lambda: self.do("fit_view"), "fit_view")
        self.act(m, "Show / hide boxes", lambda: self.do("toggle_boxes"), "toggle_boxes")
        self.act(m, "Loupe", lambda: self.do("toggle_loupe"), "toggle_loupe")
        self.act(m, "Loupe: stronger (also Ctrl+wheel)", lambda: self.do("loupe_in"), "loupe_in")
        self.act(m, "Loupe: weaker", lambda: self.do("loupe_out"), "loupe_out")
        self.act(m, "Fullscreen", lambda: self.do("fullscreen"), "fullscreen")
        self.act(m, "Hide / show side panels", lambda: self.do("toggle_panels"), "toggle_panels")
        m.addSeparator()
        gm = m.addMenu("Grid tiles per page")
        grp = QActionGroup(self)
        for cols, rows in GRID_SIZES:
            a = QAction(grid_size_text(cols, rows), self, checkable=True)
            a.triggered.connect(lambda _=False, c=cols, r=rows: self.set_grid_size(c, r))
            grp.addAction(a)
            gm.addAction(a)
            if f"{cols}x{rows}" == self.settings.value("grid", "3x3"):
                a.setChecked(True)
        c, r = map(int, str(self.settings.value("grid", "3x3")).split("x"))
        self.grid.cols, self.grid.rows = c, r
        self.context_bar.set_grid_size(c, r)
        self.grid_size_actions = grp
        m.addSeparator()
        self.act(m, "Session report: what is filtered out...", self.session_report)
        self.act(m, "Reload proposals and new frames", lambda: self.do("reload"), "reload")

        m = mb.addMenu("&Auto-label")
        self.act(m, "Prelabel with models...", self.prelabel_dialog, shortcut="Ctrl+L")
        self.act(m, "Retrain a labeler on my labels...", self.retrain_dialog)
        self.act(m, "Check GPU / game status", self.gpu_status)

        m = mb.addMenu("&Settings")
        self.act(m, "Project (classes, sources, zones)...", self.project_settings)
        self.act(m, "Keyboard shortcuts...", self.keyboard_dialog)
        self.act(m, "Performance...", self.performance_dialog)
        self.act(m, "SAM fit correction...", self.sam_adjust_dialog)
        am = m.addMenu("Arrow keys in the grid")
        self.arrow_mode = str(self.settings.value("arrows", "tiles"))
        if self.arrow_mode not in ARROW_TEXT:
            self.arrow_mode = "tiles"
        arrow_group = QActionGroup(self)
        for key, text, tip in ARROW_MODES:
            a = QAction(text, self, checkable=True)
            a.setToolTip(tip)
            a.setStatusTip(tip)
            a.setChecked(key == self.arrow_mode)
            a.triggered.connect(lambda _=False, k=key: self.set_arrow_mode(k))
            arrow_group.addAction(a)
            am.addAction(a)
        self.arrow_actions = arrow_group
        a = QAction("Preview a frame when the mouse rests on the list", self, checkable=True)
        a.setChecked(str(self.settings.value("peek", "true")).lower() in ("1", "true", "yes"))
        a.setToolTip("A small photo with its boxes appears beside the frame list")
        a.toggled.connect(self.set_peek)
        m.addAction(a)

        m = mb.addMenu("&Help")
        self.act(m, "Keyboard and mouse help", self.help_dialog, shortcut="F1")
        self.update_context()

    def update_context(self) -> None:
        """Key hints in the mode switch and the action buttons follow the keymap; then the context bar."""
        self.mode_switch.set_hint(self.keymap.first("switch_mode"))
        self.refresh_action_texts()
        self.update_status()

    def mode_hint(self) -> str:
        """The keys of the current mode, short, for the status bar."""
        k = self.keymap.first
        if self.mode == "grid":
            arrows = "arrows: tiles, Enter: tick, X: skip" if self.arrow_mode == "tiles" else "arrows: pages"
            if self.view_key == "checked":
                return (f"click: tick to return  |  {k('commit_page')}: return the ticked (or the page) to Review  |  "
                        f"double-click: editor  |  {k('undo')}: undo  |  {arrows}")
            if self.grid.sticky:
                return (f"click: has objects  |  Ctrl+click: skip  |  double-click: editor  |  wheel: zoom  |  "
                        f"{k('commit_page')}: save + next  |  {arrows}")
            return (f"click: has objects  |  Ctrl+click: skip  |  double-click: editor  |  {k('commit_page')}: save page  |  "
                    f"{k('undo')}: undo page  |  {arrows}  |  {k('switch_mode')}: editor")
        return (f"drag: new box  |  RMB: delete  |  wheel: zoom  |  {k('accept_next')}: save + next  |  {k('empty_next')}: empty  |  "
                f"{k('skip_next')}: skip  |  {k('switch_mode')}: grid")

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
        self.cancel_loader()
        self.shutdown_sam()  # the model folders may differ
        self.project = proj
        self.sessions = proj.sessions()
        if proj.path:
            self.add_recent(proj.path)
        elif target.is_dir():
            self.add_recent(target)
        self.setWindowTitle(f"framelabel - {proj.name}")
        self.canvas.set_classes(proj.class_names, [c.color for c in proj.classes])
        self.grid.class_colors = [QColor(c.color) for c in proj.classes]
        self.grid.class_names = list(proj.class_names)
        self.grid.ui_threshold = proj.ui_threshold
        self.fill_classes()
        self.view_tree.set_views(all_views(proj.class_names), [c.color for c in proj.classes])
        self.refresh_tree()
        QTimer.singleShot(0, self.fit_view_tree)
        # The shown confidence decides which frames the grid ("model found nothing") and the editor list get, so it must
        # not change between runs: the value you chose is remembered per project; the project's show_conf (the
        # recommendation for its labeler) is only the first default.
        conf = self.saved_conf(proj)
        if conf is None and "show_conf" in proj.prelabel:
            conf = float(proj.prelabel["show_conf"])
        if conf is not None:
            self.conf_spin.blockSignals(True)
            self.conf_spin.setValue(conf)
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
        self.start_overview()
        last = self.settings.value(f"session/{proj.path or target}", "")
        i = next((k for k, s in enumerate(self.sessions) if s.name == last), 0)
        self.session_box.setCurrentIndex(i)
        self.on_session_changed(i)

    def start_overview(self) -> None:
        """Fill the session box with frame / reviewed counts (a listing per session: done in a thread)."""
        sessions = list(self.sessions)
        job = Job(lambda progress, cancelled: [session_overview(s) for s in sessions])
        self.overview_job = job
        job.finished.connect(self.on_overview)
        start_job(self, job)

    @pyqtSlot(object)
    def on_overview(self, rows) -> None:
        if self.sender() is not self.overview_job:
            return
        self.overview_job = None
        for i, (s, o) in enumerate(zip(self.sessions, rows)):
            if i < self.session_box.count():
                done = o["pos"] + o["neg"] + o["skip"]
                text = (f"{s.name}  ({o['frames']} frames, review.json damaged)" if o.get("damaged") else
                        f"{s.name}  ({o['frames']} frames, {done} decided, {o['pos']} labeled)")
                self.session_box.setItemText(i, text)
                self.session_box.setItemData(i, text, Qt.ItemDataRole.ToolTipRole)

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
        self.settings.setValue(f"session/{self.project.path or s.root}", s.name)
        self.start_loading(s)

    def start_loading(self, s: Session) -> None:
        """Build the session's workspace (index, interface-screen scores) in a background thread."""
        self.cancel_loader()
        self.reloader = None
        self.ws, self.queue, self.cur_path = None, [], None  # handlers return early while there is no workspace
        self.frame_model.set_queue([])
        self.grid.message = f"Loading {s.name}..."
        self.grid.set_items([], 0)
        self.canvas.img = None
        self.canvas.message = f"Loading {s.name}..."
        self.canvas.update()
        self.status_label.setText(f"Loading {s.name}...")
        self.counts_label.setText("")
        self.done_bar.setRange(0, 1)
        self.done_bar.setValue(0)
        self.done_bar.setFormat("")
        self.counts_by_view = {}
        self.view_tree.set_counts({})
        self.counts_timer.stop()
        if self.counts_job is not None:
            self.counts_job.cancel()
            self.counts_job = None  # its result no longer matches self.counts_job and is dropped
        self.pace.reset()
        self.undo_pages.clear()
        self.pace_label.setVisible(False)
        self.load_progress.setRange(0, 0)
        self.load_progress.setVisible(True)
        project, conf = self.project, self.conf_spin.value()
        job = Job(lambda progress, cancelled: Workspace(project, s, conf, progress, cancelled))
        self.loader = job
        job.progress.connect(self.on_load_progress)
        job.finished.connect(self.on_loaded)
        job.failed.connect(self.on_load_failed)
        start_job(self, job)

    def cancel_loader(self) -> None:
        if self.loader is not None:
            self.loader.cancel()  # a late result is ignored: it no longer matches self.loader
            self.loader = None
            self.load_progress.setVisible(False)

    @pyqtSlot(int, int, str)
    def on_load_progress(self, i: int, n: int, msg: str) -> None:
        if self.sender() is self.loader:
            self.load_progress.setRange(0, max(n, 1))
            self.load_progress.setValue(i)
            self.status_label.setText(f"Loading: {msg} {i}/{n}")

    @pyqtSlot(object)
    def on_loaded(self, ws) -> None:
        if self.sender() is not self.loader:
            return  # a stale load: another session was chosen meanwhile
        self.loader = None
        self.load_progress.setVisible(False)
        if ws.review.classes_conflict:
            choice = self.ask_class_conflict(ws)
            if choice == "cancel":
                msg = (f"Not opened: the class list differs from the one {ws.session.name} was labeled with "
                       "(Settings > Project).")
                self.grid.message = self.canvas.message = msg
                self.canvas.update()
                self.status_label.setText(msg)
                return
            if choice == "adopt":
                ws.review.adopt_classes()
        self.ws = ws
        self.grid.message = ""
        if not ws.session.auto_dir.exists():
            self.status_label.setText("No model proposals yet: Auto-label > Prelabel to get them. All frames go to the grid.")
        # start where the work is: Triage, else Label, else Review, else the library (nothing is left to decide)
        start = next((k for k in ("triage", "edit", "review") if ws.view(k, self.sort_key)), None)
        if start is None:
            self.set_mode(self.mode_of("all"), queue="all")
            self.toast("Every frame of this session has a decision")
        else:
            self.set_mode(self.mode_of(start), queue=start)
        self.refresh_counts()
        if self.setting_bool("perf/sam_preload", False) and self.ensure_sam(quiet=True):
            self.sam_worker.preload()
        self.status_label.setText(f"{ws.session.name}: {len(ws.index)} frames. " + ws.hidden_summary())
        self.session_loaded.emit()

    @pyqtSlot(str)
    def on_load_failed(self, msg: str) -> None:
        if self.sender() is not self.loader:
            return
        self.loader = None
        self.load_progress.setVisible(False)
        self.grid.message = ""
        self.canvas.message = f"Cannot open the session: {msg}"
        self.canvas.update()
        self.status_label.setText(self.canvas.message)

    def goto_frame(self) -> None:
        """Jump to a frame of the current queue by 1-based number or by (the start of) its name."""
        if self.ws is None or not self.queue:
            return
        text, ok = QInputDialog.getText(self, "Go to frame", f"Frame number (1-{len(self.queue)}) or name:")
        if not ok or not text.strip():
            return
        text = text.strip()
        if text.isdigit() and 1 <= int(text) <= len(self.queue):
            i = int(text) - 1
        else:  # not a valid number: a name, e.g. 100006 for the frame 100006_000
            i = next((k for k, f in enumerate(self.queue) if f.stem.startswith(text)), -1)
        if not 0 <= i < len(self.queue):
            self.status_label.setText(f"No frame '{text}' in this queue.")
            return
        self.jump_to(i)

    def jump_to(self, i: int) -> None:
        """Show the i-th frame of the current queue (its page in the grid)."""
        if self.mode == "grid":
            self.grid.set_page(i // self.grid.per_page)
            self.select_frame_row(i)
        else:
            self.go(i)

    # ---- re-reading proposals and new frames without leaving the frame
    def reload_session(self) -> None:
        if self.ws is None or self.loader is not None:
            return
        if self.reloader is not None:
            self.reloader.cancel()
        ws = self.ws
        job = Job(lambda progress, cancelled: ws.build_fresh(progress, cancelled))
        job.ws = ws
        self.reloader = job
        self.status_label.setText("Re-reading proposals...")
        job.finished.connect(self.on_reloaded)
        job.failed.connect(self.on_reload_failed)
        start_job(self, job)

    @pyqtSlot(object)
    def on_reloaded(self, fresh) -> None:
        job = self.sender()
        if job is not self.reloader or self.ws is not job.ws:
            return  # another reload or another session replaced this one
        self.reloader = None
        stem = None
        if self.mode == "edit" and self.cur_path is not None:
            stem = self.cur_path.stem
        elif self.mode == "grid" and self.grid.items:
            stem = self.grid.page_items()[0].stem
        self.commit_current()
        self.ws.apply_reload(fresh)
        self.apply_queue(keep_stem=stem)
        self.refresh_counts()
        self.status_label.setText("Proposals reloaded.")

    @pyqtSlot(str)
    def on_reload_failed(self, msg: str) -> None:
        if self.sender() is self.reloader:
            self.reloader = None
            self.status_label.setText(f"Reload failed: {msg}")

    def fill_frame_list(self) -> None:
        self.frame_model.set_queue(self.queue)

    def update_list_item(self, i: int) -> None:
        self.frame_model.row_changed(i)

    def select_frame_row(self, row: int) -> None:
        """Highlight a row of the frame list without triggering anything."""
        if 0 <= row < self.frame_model.rowCount():
            ix = self.frame_proxy.mapFromSource(self.frame_model.index(row))
            if ix.isValid():  # the frame may be hidden by the filter
                self.frame_list.setCurrentIndex(ix)
                self.frame_list.scrollTo(ix)

    def on_frame_clicked(self, index) -> None:
        row = self.frame_proxy.mapToSource(index).row()  # the list may be filtered
        if self.mode == "grid":
            self.grid.set_page(row // self.grid.per_page)
        else:
            self.go(row)

    # ================================================================ editor frames
    def read_image(self, path: Path):
        """(image, QImage) of a frame: from the read-ahead if it was prefetched, else read now. Keeps the next two
        frames and the previous one prefetching (the worker also converts to QImage); drops the rest."""
        f = self.prefetch.pop(path, None)
        img, qimg = f.result() if f is not None else read_frame(path)
        wanted = [self.queue[i].path for i in (*range(self.idx + 1, self.idx + 1 + self.prefetch_ahead), self.idx - 1)
                  if 0 <= i < len(self.queue)]
        for p in [p for p in self.prefetch if p not in wanted]:
            self.prefetch.pop(p).cancel()
        for p in wanted:
            if p not in self.prefetch and p != path:
                self.prefetch[p] = self.pool.submit(read_frame, p)
        return img, qimg

    def load_frame(self, idx: int, carried: list[Box] | None = None) -> None:
        with timed("load_frame", 50):
            self._load_frame(idx, carried)

    def _load_frame(self, idx: int, carried: list[Box] | None = None) -> None:
        self.idx = min(max(idx, 0), len(self.queue) - 1)
        fi = self.queue[self.idx]
        img, qimg = self.read_image(fi.path)
        if img is None:
            self.canvas.message = f"Cannot read {fi.path}"
            self.canvas.img = None
            self.canvas.update()
            return
        h, w = img.shape[:2]
        boxes, weak = self.ws.load_boxes(fi.path, w, h)
        self.cur_path = fi.path
        self.remember_position(fi.stem)
        self.canvas.message = ""
        self.canvas.set_frame(img, boxes, weak, keep_view=True, qimg=qimg, ghosts=self.ws.ghosts(fi.path, w, h))
        if carried is not None and not boxes and fi.status in (None, "todo"):
            self.canvas.replace_boxes(carried)
            self.canvas.message = "boxes carried over from the previous frame: check them"
        self.select_frame_row(self.idx)
        self.fill_box_table()
        self.update_status()

    def commit_current(self, status: str | None = None, force: bool = False) -> None:
        """Save the editor's frame when it was edited (or always with force / an explicit status)."""
        if self.ws is None or self.mode != "edit" or self.cur_path is None or self.canvas.img is None:
            return
        if not (self.canvas.dirty or force or status):
            return
        was = self.ws.review.get(self.cur_path.stem)
        with timed("commit_current", 50):
            now = self.ws.save(self.cur_path, self.canvas.boxes, self.canvas.iw, self.canvas.ih, status)
        if was in (None, "todo") and now in ("pos", "neg", "skip"):
            self.record_decisions(1)  # a frame that waited for a decision has one now
        self.canvas.dirty = False
        self.schedule_counts()
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
            self.say_end_of_view()
            idx = len(self.queue) - 1
        carried = None
        if carry and self.carry_check.isChecked() and prev_img is not None and prev_boxes and 0 <= idx < len(self.queue):
            nxt = self.queue[idx]
            if nxt.status in (None, "todo") and nxt.max_conf < self.conf_spin.value():
                img, qimg = self.read_image(nxt.path)
                if img is not None and img.shape[:2] == prev_img.shape[:2]:
                    done = Future()
                    done.set_result((img, qimg))
                    self.prefetch[nxt.path] = done  # load_frame takes it right away
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
        self.sync_class_list()
        self.update_status()

    def on_selection(self, i: int) -> None:
        if 0 <= i < self.box_table.rowCount():
            self.box_table.selectRow(i)
        else:
            self.box_table.clearSelection()
        self.sync_class_list()

    def sync_class_list(self) -> None:
        """The class list shows the selected box's class, else the class new boxes get. Only the list changes."""
        c = self.canvas
        row = c.boxes[c.sel].cls if 0 <= c.sel < len(c.boxes) else c.cls
        if 0 <= row < self.class_list.count() and self.class_list.currentRow() != row:
            self.class_list.blockSignals(True)
            self.class_list.setCurrentRow(row)
            self.class_list.blockSignals(False)

    def on_class_row(self, row: int) -> None:
        if row >= 0:
            self.choose_class(row)

    def choose_class(self, row: int) -> None:
        """Class `row` for new boxes and for the selected box. Always applied, even when the list already shows it
        (a key press must not depend on what the list happened to show)."""
        if not 0 <= row < len(self.project.classes):
            return
        self.canvas.cls = row
        if self.canvas.sel >= 0:
            self.canvas.set_class_of(self.canvas.sel, row)
        self.sync_class_list()
        self.update_status()

    def box_table_delete(self, pos) -> None:
        """Right click on a row of the box table deletes that box at once (Ctrl+Z brings it back)."""
        row = self.box_table.rowAt(pos.y())
        if not 0 <= row < len(self.canvas.boxes):
            return
        self.canvas.select(row, center=False)
        self.canvas.delete_selected()
        self.toast("Box deleted (Ctrl+Z undoes)")

    def conf_key(self, proj: Project) -> str:
        ident = str(proj.path or (proj.sources[0] if proj.sources else proj.name)).lower().encode()
        return f"conf_by_project/{hashlib.sha1(ident).hexdigest()[:12]}"

    def saved_conf(self, proj: Project) -> float | None:
        try:
            v = self.settings.value(self.conf_key(proj), None)
            return None if v in (None, "") else float(v)
        except (TypeError, ValueError):
            return None

    def on_conf_changed(self, v: float) -> None:
        """The spin box moved: remember the value now, apply it (rebuild the views) when it stops moving."""
        self.settings.setValue("conf", v)
        if self.project is not None:
            self.settings.setValue(self.conf_key(self.project), v)
        if self.ws is not None:
            self.commit_current()
            self.ws.conf = v
            self.conf_timer.start()
            self.update_status()

    def apply_conf(self) -> None:
        if self.ws is None:
            return
        self.ws.conf = self.conf_spin.value()
        stem = None
        if self.mode == "edit" and self.cur_path is not None:
            stem = self.cur_path.stem
        elif self.mode == "grid" and self.grid.items:
            stem = self.grid.page_items()[0].stem
        self.apply_queue(keep_stem=stem)
        self.refresh_counts()
        self.toast(f"Showing proposals from confidence {self.ws.conf:.2f}: {len(self.queue)} frames in this view")

    def on_carry_toggled(self, v: bool) -> None:
        self.settings.setValue("auto_track", "true" if v else "false")
        self.toast("Auto-track " + ("ON: new frames get the previous boxes" if v else "OFF: nothing is carried over"))
        self.update_status()

    def on_weak_toggled(self, v: bool) -> None:
        self.canvas.show_weak = v
        self.canvas.update()

    def set_grid_size(self, cols: int, rows: int) -> None:
        self.settings.setValue("grid", f"{cols}x{rows}")
        self.context_bar.set_grid_size(cols, rows)
        for a in self.grid_size_actions.actions():
            a.setChecked(a.text() == grid_size_text(cols, rows))
        first = self.grid.page * self.grid.per_page
        self.grid.cols, self.grid.rows = cols, rows
        self.grid.clear_cache()
        self.grid.set_page(first // self.grid.per_page)

    def set_arrow_mode(self, mode: str) -> None:
        """What the arrow keys do in the grid: move a tile cursor, or flip pages. Only one of the two is on."""
        self.arrow_mode = mode
        self.settings.setValue("arrows", mode)
        for a in self.arrow_actions.actions():
            a.setChecked(a.text() == ARROW_TEXT[mode])
        self.grid.kbd = False
        self.grid.hover = -1
        self.grid.update()
        self.update_status()
        self.toast("Arrow keys: " + ARROW_TEXT[mode].lower())

    def set_peek(self, on: bool) -> None:
        self.settings.setValue("peek", "true" if on else "false")
        self.peek.set_enabled(on)

    def frame_at_index(self, ix):
        """The frame of a row of the (filtered) frame list, or None."""
        row = self.frame_proxy.mapToSource(ix).row()
        return self.queue[row] if 0 <= row < len(self.queue) else None

    def arrow(self, action: str) -> None:
        g = self.grid
        if self.arrow_mode == "pages":
            g.set_page(g.page + (-1 if action in ("arrow_left", "arrow_up") else 1))
            self.select_frame_row(min(g.page * g.per_page, max(0, len(self.queue) - 1)))
            return
        dx, dy = {"arrow_left": (-1, 0), "arrow_right": (1, 0), "arrow_up": (0, -1), "arrow_down": (0, 1)}[action]
        g.move_cursor(dx, dy)

    STATUS_WORDS = {"pos": "labeled", "neg": "empty", "skip": "skipped", "todo": "marked", None: "new"}

    def grid_stats(self) -> str:
        """What saving the grid page would do, as coloured chips for the context bar: tiles ticked 'has objects', SKIP,
        and the empty ones. In the Triage stage always; elsewhere only the changes that are pending."""
        g = self.grid
        items = g.page_items()
        n_obj = sum(1 for fi in items if g.pending(fi) == "people")
        if g.return_mode:  # Library > Checked: a tick means 'return to Review'
            n = n_obj or len(items)
            return f"<span style='color:{T['accent_hi']}'><b>{n}</b> to return{'' if n_obj else ' (whole page)'}</span>"
        n_skip = sum(1 for fi in items if g.pending(fi) == "skip")
        n_empty = sum(1 for fi in items if g.pending(fi) == "empty" or (fi.status is None and g.marks.get(fi.stem) is None))
        triage = self.current_stage() == "triage"
        parts = []
        if n_obj or triage:
            parts.append(f"<span style='color:{T['accent_hi']}'><b>{n_obj}</b> has objects</span>")
        if n_skip or triage:
            parts.append(f"<span style='color:{T['text2']}'><b>{n_skip}</b> skip</span>")
        if n_empty and (triage or n_obj or n_skip):
            parts.append(f"<span style='color:{T['text']}'><b>{n_empty}</b> empty</span>")
        return "&nbsp;&nbsp;&nbsp;".join(parts)

    def update_status(self) -> None:
        """Everything that says where you are: the context bar, the hint card, the progress, the status bar."""
        if self.ws is None:
            if self.project is not None and not self.sessions:
                self.context_bar.set_text("No frames", "", "The project's sources contain no frames.")
            return
        ws = self.ws
        c = status_counts(ws.index)
        n = len(ws.index)
        decided = c["labeled"] + c["empty"] + c["skipped"]
        self.done_bar.setRange(0, max(n, 1))
        self.done_bar.setValue(decided)
        self.counts_label.setText(f"<b>{decided}</b> of {n} decided ({100 * decided // max(n, 1)}%)")
        pace = self.pace_text()
        self.pace_label.setText(pace)
        self.pace_label.setVisible(bool(pace))
        vd = self.view_def()
        where_name = self.view_name(vd.key)
        stats = ""
        if self.mode == "grid":
            g = self.grid
            n_marked = sum(1 for fi in g.page_items() if g.pending(fi) == "people")
            n_skip = sum(1 for fi in g.page_items() if g.pending(fi) == "skip")
            changes = f"{n_marked} to return" if g.return_mode else f"{n_marked} has objects, {n_skip} skip"
            thr = self.project.ui_threshold
            n_sugg = sum(1 for fi in g.page_items() if g.marks.get(fi.stem) == "skip" and fi.status is None and fi.ui < thr)
            sugg = f" ({n_sugg} pre-marked as interface screens: click to undo)" if n_sugg else ""
            where = (f"Grid  |  photo {g.page + 1} of {len(self.queue)}" if g.sticky else
                     f"Grid  |  page {g.page + 1} of {g.n_pages}  |  {len(self.queue)} frames")
            self.status_label.setText(f"GRID  {where_name} (shown conf {self.conf_spin.value():.2f}): "
                                      f"{len(self.queue)} frames, page {g.page + 1}/{g.n_pages}  |  to change: "
                                      f"{changes}{sugg}")
            stats = self.grid_stats()
        elif self.queue and self.cur_path is not None:
            fi = self.queue[self.idx]
            st = self.STATUS_WORDS.get(fi.status, "new")
            if fi.status in ("pos", "neg") and fi.stem in ws.checked:
                st += ", checked"
            ui = "  UI? (X to skip)" if fi.ui < self.project.ui_threshold else ""
            more = f"  model sees {len(self.canvas.ghosts)} more (amber)" if self.canvas.ghosts else ""
            cls = self.project.class_names[self.canvas.cls]
            where = f"Editor  |  frame {self.idx + 1} of {len(self.queue)}  |  {fi.stem}  |  {st}{' (unsaved edits)' if self.canvas.dirty else ''}"
            self.status_label.setText(f"EDIT {self.idx + 1}/{len(self.queue)}  {fi.stem}  [{st}{'*' if self.canvas.dirty else ''}]"
                                      f"{ui}  class {self.canvas.cls + 1}={cls}  boxes {len(self.canvas.boxes)}{more}  "
                                      f"zoom {self.canvas.view.zoom:.1f}x  |  auto-track "
                                      f"{'ON' if self.carry_check.isChecked() else 'off'} (Y)")
        else:
            where = "Editor  |  no frames"
        self.setWindowTitle(f"framelabel - {self.project.name} - {where_name}")
        title = BY_KEY[vd.stage].label if vd.stage else vd.label
        self.context_bar.set_text(title, where, vd.hint, stats, vd.label if vd.parent else "")
        self.keys_label.setText(self.mode_hint())
        card = self.coach_text()
        if card != getattr(self, "coach_shown", None):
            self.coach_shown = card
            self.coach.show_hint(*card)

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
                self.choose_class(k)
                return True
        if name == "W":
            self.weak_check.toggle()
            return True
        if name == "Esc":
            if self.mode == "grid" and self.grid.modal_preview:
                self.grid.close_preview()
                return True
            if self.isFullScreen():
                self.showNormal()
                return True
            return False
        if self.mode == "grid" and self.grid.preview is not None and (name == "Z" or self.grid.modal_preview):
            if name == "Z":
                self.grid.preview_view.reset()
                self.grid.update()
            else:
                self.grid.close_preview()
            return True
        grid_actions = ("commit_page", "prev_page", "next_page", "grid_view", "switch_mode", "fullscreen", "reload", "goto_frame",
                        "toggle_panels", "go_triage", "go_label", "go_review", "command_palette", "open_frame", "undo",
                        "arrow_left", "arrow_right", "arrow_up", "arrow_down", "tile_mark", "tile_skip")
        for action in self.keymap.actions_for(name):
            if self.mode == "grid" and action not in grid_actions:
                continue
            if self.mode == "edit" and action in ("commit_page", "prev_page", "next_page", "open_frame", *GRID_ONLY_KEYS):
                continue
            if action in GRID_FOCUS_KEYS and QApplication.focusWidget() is not self.grid:
                continue  # the arrow keys and Enter belong to a list or the tree while one has the focus
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
        elif action == "reload":
            self.reload_session()
        elif action == "grid_view":
            self.next_view()
        elif action == "toggle_panels":
            self.panels_visible = not self.panels_visible
            self.apply_panels()
        elif action == "toggle_carry":
            self.carry_check.toggle()
        elif action == "goto_frame":
            self.goto_frame()
        elif action in STAGE_BY_ACTION:
            self.on_view_chosen(STAGE_BY_ACTION[action])
        elif action == "command_palette":
            self.command_palette()
        elif self.mode == "grid":
            self.grid_action(action)
        elif not self.queue or self.cur_path is None:
            return
        elif action == "accept_next":
            n = len(c.boxes)
            self.commit_current(force=True)
            checked = self.check_in_review()
            self.finish_frame((f"Saved: {n} box{'es' if n != 1 else ''}" if n else "Saved as empty (no boxes)")
                              + (", checked" if checked else ""), carry=True)
        elif action == "empty_next":
            c.boxes = []
            self.commit_current(status="neg")
            checked = self.check_in_review()
            self.finish_frame("Saved as empty" + (", checked" if checked else ""))
        elif action == "skip_next":
            self.commit_current(status="skip")
            self.finish_frame("Skipped (not exported)")
        elif action == "mark_checked":
            self.commit_current()  # edits are saved first: a save takes the check away
            if self.ws.review.get(self.cur_path.stem) in ("pos", "neg"):
                self.check_in_review(force=True)
                self.finish_frame("Checked: it is hidden from Review now")
            else:
                self.toast("Only saved frames can be checked")
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
        elif action == "sam_fit_all":
            for i in range(len(c.boxes)):  # one request per box; each answer replaces its own box (undoable)
                self.sam_fit(c.boxes[i], index=i)
        elif action == "save":
            self.commit_current(force=True)
        self.update_status()

    def finish_frame(self, said: str, carry: bool = False) -> None:
        """Say what was done to this frame and go on to the next one. At the end of the view the message leads the
        end-of-view one, so a confirmation is never lost."""
        self.lead = said
        self.go(self.idx + 1, carry=carry)
        if self.lead is not None:
            self.toast(said)
            self.lead = None

    def check_in_review(self, force: bool = False) -> bool:
        """In the Review stage a frame you saved (or marked) is checked: it leaves Review. True when it was checked."""
        if self.ws is None or self.cur_path is None or (self.current_stage() != "review" and not force):
            return False
        stem = self.cur_path.stem
        if self.ws.review.get(stem) not in ("pos", "neg"):
            return False
        self.ws.check([stem])
        self.schedule_counts()
        return True

    def grid_action(self, action: str) -> None:
        g = self.grid
        if action == "commit_page":
            self.commit_grid_page()
            return
        if action in ("undo", "undo_page"):
            self.undo_page()
            return
        if action == "open_frame":
            self.open_in_editor(g.hovered_stem())
            return
        if action in ("tile_mark", "tile_skip"):
            g.mark_at_cursor("people" if action == "tile_mark" else "skip", start=self.arrow_mode == "tiles")
            return
        if action.startswith("arrow_"):
            self.arrow(action)
            return
        if action == "prev_page":
            g.set_page(g.page - 1)
        elif action == "next_page":
            g.set_page(g.page + 1)
        self.select_frame_row(min(g.page * g.per_page, max(0, len(self.queue) - 1)))

    def commit_grid_page(self) -> None:
        """Save the page: ticked tiles go to Label, SKIP tiles are skipped, the unmarked ones become empty. In Review the
        page is also marked as checked. A toast with Undo says what happened."""
        g = self.grid
        if self.view_key == "checked":
            self.return_to_review()
            return
        items = g.page_items()
        n_obj = sum(1 for fi in items if g.pending(fi) == "people")
        n_skip = sum(1 for fi in items if g.pending(fi) == "skip")
        stems, marks = g.stems_on_page(), dict(g.marks)
        undo = self.ws.commit_page(stems, g.marks)
        n_empty = sum(1 for _stem, (_old, _t, new) in undo.items() if new == "neg")
        review = self.current_stage() == "review"
        newly = [x for x in stems if x not in self.ws.checked] if review else []
        if review:
            self.ws.check(stems)
        pushed = bool(undo or newly)
        if pushed:
            self.undo_pages.append({"view": self.view_key, "stem": items[0].stem if items else None, "undo": undo,
                                    "marks": marks, "checked": newly})
            del self.undo_pages[:-20]
        if undo:  # a tick ('todo') is not a decision: the frame is decided later, in the editor
            self.record_decisions(sum(1 for _old, _t, new in undo.values() if new != "todo"))
        for stem in stems:
            g.marks.pop(stem, None)  # saved: the tile shows the status now, not a pending mark
        self.schedule_counts()
        for i in range(g.page * g.per_page, min((g.page + 1) * g.per_page, len(self.queue))):
            self.queue[i] = self.ws.index[self.ws.by_stem[self.queue[i].stem]]
            self.update_list_item(i)
        if not pushed:
            text = "Nothing to change on this page"
        elif review:
            text = f"Page checked ({len(undo)} changed)" if undo else "Page checked"
        else:
            text = f"Page saved: {n_empty} empty, {n_obj} to Label, {n_skip} skipped"
        action = ("Undo", self.undo_page) if pushed else None
        if g.page + 1 >= g.n_pages:
            if self.after_triage_page(action):
                return
            self.status_label.setText("End of this view.")
            self.toast(text, action)
            g.set_page(g.page)
        else:
            self.toast(text, action)
            g.set_page(g.page + 1)
        self.select_frame_row(min(g.page * g.per_page, max(0, len(self.queue) - 1)))

    def return_to_review(self) -> None:
        """Library > Checked: the ticked tiles (the whole page when none is ticked) are no longer checked, so Review shows
        them again if they still deserve a look. The toast has Undo."""
        g = self.grid
        stems = g.stems_on_page()
        pick = [s for s in stems if g.marks.get(s) == "people"] or stems
        if not pick:
            self.toast("No frames to return")
            return
        page = g.page
        self.commit_current()
        self.ws.uncheck(pick)
        self.undo_pages.append({"view": "checked", "stem": pick[0], "undo": {}, "marks": {}, "checked": [], "unchecked": pick})
        del self.undo_pages[:-20]
        self.schedule_counts()
        self.apply_queue()
        g.set_page(page)
        self.toast(f"{len(pick)} frame{'s' if len(pick) != 1 else ''} returned to Review", ("Undo", self.undo_page))

    def undo_page(self) -> None:
        """Take back the last saved grid page: statuses and label files as they were, the page shown again with its marks."""
        if self.ws is None or not self.undo_pages:
            self.toast("Nothing to undo")
            return
        entry = self.undo_pages.pop()
        self.commit_current()
        restored, left = self.ws.undo_page(entry["undo"])
        self.pace.add(-sum(1 for stem in restored if entry["undo"][stem][2] != "todo"))
        self.ws.uncheck(entry["checked"])
        self.ws.check(entry.get("unchecked", []))
        self.schedule_counts()
        view = entry["view"] if entry["view"] in self.view_tree.items else "triage"
        self.set_mode("grid", queue=view, keep_stem=entry["stem"])
        on_page = {fi.stem for fi in self.grid.page_items()}
        for stem in on_page:
            self.grid.marks.pop(stem, None)
        self.grid.marks.update({k: m for k, m in entry["marks"].items() if k in on_page and k not in left})
        self.grid.update()
        self.update_status()
        n = len(restored) + len(entry["checked"]) + len(entry.get("unchecked", []))
        text = f"Page restored ({n} frame{'s' if n != 1 else ''})"
        if left:
            text += f"; {len(left)} decided again since, left as they are"
        self.toast(text)

    # ================================================================ the command palette
    def palette_entries(self) -> list[Entry]:
        """Everything that can be run without typing a frame: the views, the key-driven actions that fit the current mode,
        and the menu items that have no key."""
        entries: list[Entry] = []
        for v in all_views(self.project.class_names):
            n = self.counts_by_view.get(v.key)
            key = self.keymap.first(STAGE_KEYS[v.key]) if v.is_stage else ""
            entries.append(Entry(f"Go to {self.view_name(v.key)}", "" if n is None else f"{n:,} frames".replace(",", " "), key,
                                 lambda k=v.key: self.on_view_chosen(k), "view"))
        group_of_mode = "Grid" if self.mode == "grid" else "Editor"
        for action, (group, text) in ACTION_INFO.items():
            if group in ("Everywhere", group_of_mode) and action not in STAGE_BY_ACTION and action != "command_palette":
                entries.append(Entry(text, "", self.keymap.first(action), lambda a=action: self.do(a), "action"))
        for menu in self.menuBar().findChildren(QMenu):
            for a in menu.actions():
                if a.isSeparator() or a.menu() is not None or a.data() is not None or not a.isEnabled():
                    continue
                title = a.text().split("\t")[0].replace("&", "")
                if title and "Recent" not in title:
                    entries.append(Entry(title.rstrip(".") if title.endswith("...") else title, "", a.shortcut().toString(),
                                         a.trigger, "action"))
        return entries

    def palette_frames(self, query: str) -> list[Entry]:
        """Frames of the current view that the typed text names: a number, or the start of a name."""
        q = query.strip()
        out: list[Entry] = []
        if not self.queue or not q:
            return out
        if q.isdigit() and 1 <= int(q) <= len(self.queue):
            i = int(q) - 1
            out.append(Entry(f"Go to frame {q}: {self.queue[i].stem}", "", "", lambda i=i: self.jump_to(i), "frame"))
        for i, f in enumerate(self.queue):
            if f.stem.startswith(q) and len(out) < 8:
                out.append(Entry(f"Go to frame {f.stem}", self.STATUS_WORDS.get(f.status, "new"), "",
                                 lambda i=i: self.jump_to(i), "frame"))
        return out

    def command_palette(self) -> None:
        if self.project is None or self.ws is None:
            return
        dlg = CommandPalette(self.palette_entries(), self.palette_frames, self)
        if dlg.exec() and dlg.chosen is not None and dlg.chosen.run is not None:
            QTimer.singleShot(0, dlg.chosen.run)  # after the palette is gone, so focus returns to the work area

    def from_previous(self, track: bool) -> None:
        """Boxes of the previous labeled frame in time order (not the queue): copied, or tracked onto this frame."""
        i = self.ws.by_stem.get(self.cur_path.stem, 0)
        prev = next((self.ws.index[j] for j in range(i - 1, -1, -1) if self.ws.index[j].status == "pos"), None)
        if prev is None:
            self.toast("No labeled frame before this one")
            return
        img = imread(str(prev.path))
        if img is None:
            self.toast(f"Cannot read {prev.path.name}")
            return
        h, w = img.shape[:2]
        boxes, _ = self.ws.load_boxes(prev.path, w, h)
        c = self.canvas
        if track and img.shape[:2] == c.img.shape[:2]:
            props = [b for b in self.ws.proposals(self.cur_path, c.iw, c.ih)]
            boxes = propagate(img, c.img, boxes, props)
        c.replace_boxes(c.boxes_copy() + boxes if not track else boxes)
        n = f"{len(boxes)} box{'es' if len(boxes) != 1 else ''}"
        self.toast(f"{n} {'tracked' if track else 'copied'} from {prev.stem} ({self.keymap.first('undo')} undoes)")

    # ================================================================ SAM
    def ensure_sam(self, quiet: bool = False) -> bool:
        """Start the SAM thread (once per weights file); False when there are no SAM weights."""
        from framelabel.ml.models import find_sam

        from .sam_worker import SamWorker

        weights = find_sam(self.model_dirs())
        if weights is None:
            if not quiet:
                QMessageBox.information(self, "SAM", "No SAM weights (e.g. sam2.1_b.pt) found in the model folders.")
            return False
        if self.sam_worker is not None and weights == self.sam_weights:
            return True
        self.shutdown_sam()
        self.sam_weights = weights
        self.sam_worker = SamWorker(weights, self.project.game_processes or ["UAGame.exe"])
        self.sam_worker.ready.connect(self.on_sam_ready)
        self.sam_worker.failed.connect(self.on_sam_failed)
        self.sam_worker.status.connect(self.status_label.setText)
        self.sam_worker.start()
        return True

    def shutdown_sam(self) -> None:
        if self.sam_worker is not None and not self.sam_worker.stop():
            log.warning("the SAM thread did not stop (it is a daemon thread and ends with the process)")
        self.sam_worker = self.sam_weights = None
        self.sam_pending.clear()

    def sam_fit(self, prompt, index: int | None = None) -> None:
        """Ask the SAM thread to fit a rough Box (the selected / just drawn box) or a click point (x, y)."""
        c = self.canvas
        if c.img is None or not self.ensure_sam():
            return
        self.sam_seq += 1
        rid = self.sam_seq
        if isinstance(prompt, Box):
            if index is None:
                index = len(c.boxes) - 1  # the box just drawn
            target = c.boxes[index] if 0 <= index < len(c.boxes) else None
            self.sam_pending[rid] = {"path": self.cur_path, "box": target}
            self.sam_worker.submit(rid, c.img, prompt.norm(), None)
        else:
            self.sam_pending[rid] = {"path": self.cur_path, "box": None, "cls": c.cls}
            self.sam_worker.submit(rid, c.img, None, tuple(prompt))
        self.status_label.setText("SAM...")

    @pyqtSlot(int, object)
    def on_sam_ready(self, rid: int, new) -> None:
        req = self.sam_pending.pop(rid, None)
        c = self.canvas
        if req is None or req["path"] != self.cur_path or c.img is None:
            self.status_label.setText("SAM result dropped: the frame changed.")
            return
        if new is None:
            self.status_label.setText("SAM found nothing there.")
            return
        adjust = self.sam_adjust_px()
        if adjust:
            new = grow_box(new, adjust, c.img.shape[1], c.img.shape[0])
        if req["box"] is not None:
            # edits move boxes in place, so the identity survives them; undo / redo / a delete replace or remove it
            i = next((k for k, b in enumerate(c.boxes) if b is req["box"]), -1)
            if i < 0:
                self.status_label.setText("SAM result dropped: the box changed.")
                return
            new.cls = c.boxes[i].cls
            c.snapshot()
            c.boxes[i] = new
            c.edited()
        else:
            new.cls = req["cls"]
            c.add_box(new)
        self.status_label.setText(f"SAM box {new.w:.0f} x {new.h:.0f} px" + (f" (corrected {adjust:+d} px per side)" if adjust else ""))

    @pyqtSlot(int, str)
    def on_sam_failed(self, rid: int, msg: str) -> None:
        self.sam_pending.pop(rid, None)
        self.status_label.setText(msg)

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
        if not self.sessions:
            QMessageBox.information(self, "Prelabel", "The project's sources contain no frames.")
            return
        dlg = PrelabelDialog(self.project, self.model_dirs(), self.sessions, self.session_box.currentIndex(), self)
        if not dlg.exec() or not self.gpu_preflight():
            return
        cfg, sessions, redo = dlg.result_config()
        self.commit_current()
        target = self.project.path or self.project.sources[0]
        args = ["-m", "framelabel", "prelabel", str(target), "--models", *map(str, cfg.models), "--imgsz", str(cfg.imgsz),
                "--conf", str(cfg.conf), "--tta" if cfg.tta else "--no-tta", "--batch", str(cfg.batch),
                "--max-vram-gb", str(cfg.max_vram_gb), "--session", *[s.name for s in sessions],
                "--progress-json", "--cancel-file", "{cancel_file}"]
        if redo:
            args.append("--redo")
        self.run_job(ProcessJob(args), "Prelabeling", self.after_prelabel)

    def after_prelabel(self, stats: dict) -> None:
        lines = [f"{stats['done']} of {stats['frames']} frames, {stats['with_proposals']} with proposals, "
                 f"{stats['seconds']:.0f} s" + (" (cancelled)" if stats.get("cancelled") else "")]
        if "without_proposals" in stats:
            lines.append(f"No detection at all: {stats['without_proposals']} frames.")
            lines.append(f"Detections only inside an ignore zone (the frame looks empty): {stats['only_in_ignore_zone']} frames "
                         "(Triage > Only in ignore zone).")
            dropped = stats.get("dropped_classes") or {}
            if dropped:
                lines.append("Dropped, no project class: " + ", ".join(f"{k} {v}" for k, v in list(dropped.items())[:6])
                             + " (set class_map in the project to keep one).")
        lines.append("Details: View > Session report.")
        QMessageBox.information(self, "Prelabel", "\n".join(lines))
        self.reload_session()

    def run_job(self, job, title: str, done) -> None:
        """Run a Job (thread) or a ProcessJob (child process) with the progress bar and the Cancel button."""
        if self.job is not None:
            QMessageBox.information(self, "Busy", "Another job is running.")
            return
        self.job = job
        self.job_title = title
        self.progress.setVisible(True)
        self.cancel_btn.setVisible(True)
        self.progress.setRange(0, 0)

        def prog(i, n, msg):
            self.progress.setRange(0, max(n, 1))
            self.progress.setValue(i)
            self.status_label.setText(f"{title}: {i}/{n}  {msg}")

        def end():
            job, self.job = self.job, None
            self.progress.setVisible(False)
            self.cancel_btn.setVisible(False)
            if isinstance(job, ProcessJob):  # we are inside its signal: let the event loop delete it afterwards
                job.setParent(self)
                job.deleteLater()

        def finish(result):
            end()
            if not self.closing:
                done(result)

        def fail(msg):
            end()
            if not self.closing:
                QMessageBox.critical(self, title, msg)

        def abort():
            end()
            self.status_label.setText(f"{title}: cancelled.")

        job.progress.connect(prog)
        job.finished.connect(finish)
        job.failed.connect(fail)
        if hasattr(job, "aborted"):
            job.aborted.connect(abort)
        if isinstance(job, ProcessJob):
            job.start()
        else:
            start_job(self, job)

    def cancel_job(self) -> None:
        if self.job is not None:
            self.job.cancel()
            self.status_label.setText("Cancelling after the current batch...")

    def export_dialog(self) -> None:
        from .dialogs import ExportDialog

        if self.project is None:
            return
        self.commit_current()
        dlg = ExportDialog(self.project, self.sessions, self)
        if dlg.exec():
            summary = self.project.class_names
            self.run_job(dlg.make_job(), "Exporting", lambda res: QMessageBox.information(self, "Export", res.summary(summary)))

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
        if not dlg.exec():
            return
        path = self.project.path
        if path is None:  # a folder opened without a project file: the settings live in one, so it is written now
            name, _ = QFileDialog.getSaveFileName(self, "Save the settings as a project file",
                                                  str(ROOT / "projects" / f"{self.project.name}.yaml"), "Project (*.yaml)")
            if not name:
                self.toast("Not applied: a folder needs a project file to keep its settings")
                return
            path = Path(name)
        dlg.apply()
        self.project.save(path)
        self.open_target(path)

    def session_report(self) -> None:
        """A window with counts of everything filtered out of the normal views and a view for each."""
        from framelabel.core.report import capture_rows, frame_rows, prelabel_rows

        from .dialogs import ReportDialog

        if self.ws is None:
            return
        ws = self.ws
        sections = [("What the recorder dropped before saving (capture.json)", capture_rows(ws.capture)),
                    ("Model proposals", prelabel_rows(ws.session)),
                    ("Frames and where to find them", frame_rows(ws.index, ws.conf, self.project.ui_threshold,
                                                                 self.project.class_names, ws.checked))]
        dlg = ReportDialog(f"Session report: {ws.session.name}", sections, self)
        dlg.open_view.connect(self.open_view)
        dlg.exec()

    def open_view(self, mode: str, queue: str) -> None:
        self.set_mode(mode, queue=queue)

    def sam_adjust_px(self) -> int:
        """The correction (signed pixels per side; positive = bigger) applied to every SAM result."""
        try:
            return int(self.settings.value("sam/adjust_px", 0))
        except (TypeError, ValueError):
            return 0

    def sam_adjust_dialog(self) -> None:
        from .dialogs import SamAdjustDialog

        dlg = SamAdjustDialog(self.sam_adjust_px(), self)
        if dlg.exec():
            self.settings.setValue("sam/adjust_px", dlg.value())
            v = dlg.value()
            self.toast("SAM fit correction: " + (f"{abs(v)} px {'outward' if v > 0 else 'inward'}" if v else "off"))

    def ask_class_conflict(self, ws) -> str:
        """The project's classes are not the ones this session was labeled with: 'keep' (open, the session's list stays in
        its file), 'adopt' (rewrite the session's list) or 'cancel' (do not open)."""
        def show(names):
            return ", ".join(f"{i} {n}" for i, n in enumerate(names or [])) or "none"

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Class list differs")
        box.setText(f"The classes of this project are not the ones {ws.session.name} was labeled with.")
        box.setInformativeText(
            f"Session: {show(ws.review.stored_classes)}\nProject: {show(ws.review.classes)}\n\n"
            "Labels store the class number, so if the numbers no longer mean the same, existing boxes would change class. "
            "Fix the list in Settings > Project, or choose:")
        keep = box.addButton("Open, keep the session's list", QMessageBox.ButtonRole.AcceptRole)
        adopt = box.addButton("Use the project's list", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("Do not open", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        return "keep" if clicked is keep else "adopt" if clicked is adopt else "cancel"

    def performance_dialog(self) -> None:
        from .dialogs import PerformanceDialog

        dlg = PerformanceDialog(self.settings, self)
        if dlg.exec():
            dlg.apply()
            self.apply_perf_settings()
            self.apply_watchdog()
            self.status_label.setText("Performance settings saved.")
            self.toast("Performance settings applied")

    def keyboard_dialog(self) -> None:
        from .dialogs import KeymapDialog

        dlg = KeymapDialog(self.keymap, self)
        if dlg.exec():
            self.keymap = dlg.keymap
            self.settings.setValue("keymap", json.dumps(dlg.overrides()))
            self.update_context()

    def help_dialog(self) -> None:
        from .dialogs import HelpDialog

        HelpDialog(self.keymap, self).exec()

    def closeEvent(self, e) -> None:
        if self.job is not None:
            r = QMessageBox.question(self, "Quit", f"{self.job_title} is still running. Stop it and quit?")
            if r != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.closing = True
            self.job.cancel()
            proc = getattr(self.job, "proc", None)  # a child process: give it time to stop after its batch
            if proc is not None and not proc.waitForFinished(10000):
                proc.kill()
                proc.waitForFinished(5000)
        self.closing = True
        self.peek.hide()
        self.counts_timer.stop()
        self.conf_timer.stop()
        self.commit_current()
        self.cancel_loader()
        self.reloader = self.overview_job = None
        stop_jobs(self)
        self.shutdown_sam()
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("dock_state", self.saveState())
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.grid.shutdown()
        self.settings.sync()
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
    apply_theme(app)
    w = MainWindow(target)
    w.log_dir = log_dir
    w.show()
    w.apply_watchdog()
    code = app.exec()
    if w.watchdog is not None:
        w.watchdog.stop()
    return code
