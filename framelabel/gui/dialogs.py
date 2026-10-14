"""Dialogs: prelabel (model picker, ensemble, TTA), export, retrain, project settings, keyboard shortcuts."""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                             QPlainTextEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTextBrowser,
                             QVBoxLayout)

from framelabel.core.project import PALETTE, ClassDef, Project
from framelabel.core.report import ReportRow
from framelabel.core.session import Session
from framelabel.ml.models import scan_models
from .keys import ACTION_GROUPS, ACTION_INFO, DEFAULT_KEYMAP, Keymap
from .workers import Job, start_job, stop_jobs

ROOT = Path(__file__).resolve().parents[2]


def buttons(dlg: QDialog, ok_text: str = "OK") -> QDialogButtonBox:
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    bb.button(QDialogButtonBox.StandardButton.Ok).setText(ok_text)
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    return bb


class PrelabelDialog(QDialog):
    """Pick one model (or several: an ensemble fused with WBF), the inference size, TTA and the sessions."""

    def __init__(self, project: Project, model_dirs: list[Path], sessions: list[Session], current: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prelabel with models")
        self.resize(720, 560)
        self.sessions, self.current = sessions, current
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Models (check several for an ensemble; their boxes are merged with weighted box fusion):"))
        self.models = QListWidget()
        defaults = {str(p.resolve()) for p in project.default_models}
        for m in scan_models(model_dirs, project.default_models):
            it = QListWidgetItem(f"{m.label}    {m.info}    {m.path}")
            it.setData(Qt.ItemDataRole.UserRole, str(m.path))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if str(m.path.resolve()) in defaults else Qt.CheckState.Unchecked)
            self.models.addItem(it)
        if self.models.count() and not defaults:
            self.models.item(0).setCheckState(Qt.CheckState.Checked)
        lay.addWidget(self.models)
        form = QFormLayout()
        pre = project.prelabel
        self.imgsz = QComboBox()
        for s in (960, 1280, 1536, 1920):
            self.imgsz.addItem(str(s), s)
        self.imgsz.setCurrentText(str(pre.get("imgsz", 1280)))
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.01, 0.9)
        self.conf.setSingleStep(0.01)
        self.conf.setValue(float(pre.get("conf", 0.05)))
        self.tta = QCheckBox("Test-time augmentation (flip + larger scale: ~4x slower, finds more small objects)")
        self.tta.setChecked(bool(pre.get("tta", False)))
        self.redo = QCheckBox("Redo frames that already have proposals")
        self.scope = QComboBox()
        self.scope.addItem("Current session", "current")
        self.scope.addItem("All sessions of the project", "all")
        self.vram = QDoubleSpinBox()
        self.vram.setRange(1, 16)
        self.vram.setValue(float(pre.get("max_vram_gb", 8)))
        self.batch = QSpinBox()
        self.batch.setRange(1, 32)
        self.batch.setValue(int(pre.get("batch", 4)))
        form.addRow("Inference size", self.imgsz)
        form.addRow("Lowest confidence stored", self.conf)
        form.addRow("", self.tta)
        form.addRow("", self.redo)
        form.addRow("Sessions", self.scope)
        form.addRow("VRAM cap, GB", self.vram)
        form.addRow("Batch", self.batch)
        lay.addLayout(form)
        lay.addWidget(QLabel("Model classes are matched to project classes by name (or the project's class_map); a "
                             "one-class model labels class 1. The game must be closed."))
        lay.addWidget(buttons(self, "Run"))

    def accept(self) -> None:
        if not self.checked_models():
            QMessageBox.warning(self, "Prelabel", "Check at least one model.")
            return
        super().accept()

    def checked_models(self) -> list[Path]:
        return [Path(self.models.item(i).data(Qt.ItemDataRole.UserRole)) for i in range(self.models.count())
                if self.models.item(i).checkState() == Qt.CheckState.Checked]

    def result_config(self):
        from framelabel.ml.predict import PredictConfig

        cfg = PredictConfig(models=self.checked_models(), imgsz=int(self.imgsz.currentData()), conf=self.conf.value(),
                            tta=self.tta.isChecked(), max_vram_gb=self.vram.value(), batch=self.batch.value())
        current = self.current if 0 <= self.current < len(self.sessions) else 0
        sessions = self.sessions if self.scope.currentData() == "all" else self.sessions[current:current + 1]
        return cfg, sessions, self.redo.isChecked()


def export_with(classes: list[str], zones, out: Path, p: dict, progress=None, cancelled=None):
    """core.export.export_dataset with the dialog's settings (plain values: safe in a worker thread)."""
    from framelabel.core.export import export_dataset

    return export_dataset(p["sessions"], classes, out, p["neg"], p["val"], p["seed"], p["split"], p["block"], zones,
                          progress, cancelled)


class ExportDialog(QDialog):
    def __init__(self, project: Project, sessions: list[Session], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export YOLO dataset")
        self.project, self.sessions, self.result = project, sessions, None
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Sessions:"))
        self.list = QListWidget()
        for s in sessions:
            it = QListWidgetItem(s.name)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if s.review_path.exists() else Qt.CheckState.Unchecked)
            self.list.addItem(it)
        lay.addWidget(self.list)
        form = QFormLayout()
        exp = project.export
        out = project.export_out or (ROOT / "datasets" / f"{project.name.replace(' ', '_')}_export")
        self.out = QLineEdit(str(out))
        pick = QPushButton("...")
        pick.clicked.connect(self.pick_out)
        row = QHBoxLayout()
        row.addWidget(self.out)
        row.addWidget(pick)
        self.neg = QDoubleSpinBox()
        self.neg.setRange(0, 5)
        self.neg.setSingleStep(0.05)
        self.neg.setValue(float(exp.get("neg_ratio", 0.15)))
        self.val = QDoubleSpinBox()
        self.val.setRange(0, 0.5)
        self.val.setSingleStep(0.05)
        self.val.setValue(float(exp.get("val_fraction", 0.15)))
        self.split = QComboBox()
        self.split.addItem("Random time blocks of every session, class-balanced", "blocks")
        self.split.addItem("Last part of the timeline", "time")
        self.block = QDoubleSpinBox()
        self.block.setRange(5, 3600)
        self.block.setValue(float(exp.get("block_s", 120)))
        self.seed = QSpinBox()
        self.seed.setValue(int(exp.get("seed", 0)))
        form.addRow("Output folder (replaced)", row)
        form.addRow("Empty frames per labeled frame", self.neg)
        form.addRow("Validation share", self.val)
        form.addRow("Validation split", self.split)
        form.addRow("Block length, s", self.block)
        form.addRow("Seed", self.seed)
        lay.addLayout(form)
        self.bb = buttons(self, "Export")
        lay.addWidget(self.bb)

    def pick_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.out.text())
        if d:
            self.out.setText(d)

    def chosen(self) -> list[Session]:
        return [s for i, s in enumerate(self.sessions) if self.list.item(i).checkState() == Qt.CheckState.Checked]

    def read_params(self) -> dict:
        """The export settings as plain values, read on the GUI thread before the job starts."""
        return {"sessions": self.chosen(), "neg": self.neg.value(), "val": self.val.value(), "seed": self.seed.value(),
                "split": self.split.currentData(), "block": self.block.value()}

    def export_job(self, out: Path, p: dict) -> Job:
        """A job that exports with the settings in `p`; it holds plain values only, never a widget."""
        classes, zones = list(self.project.class_names), list(self.project.ignore_zones)
        return Job(lambda progress, cancelled: export_with(classes, zones, out, p, progress, cancelled))

    def validate_out(self, title: str = "Export", ask_replace: bool = True) -> Path | None:
        out = Path(self.out.text())
        if out.exists() and not (out / "data.yaml").exists():
            QMessageBox.critical(self, title, f"{out} exists and is not an exported dataset.")
            return None
        if ask_replace and out.exists() and QMessageBox.question(self, title, f"Replace {out}?") != QMessageBox.StandardButton.Yes:
            return None
        return out

    def accept(self) -> None:
        """Validate only: the export itself runs as a background job (make_job) started by the caller."""
        self.out_path = self.validate_out()
        if self.out_path is not None:
            self.params = self.read_params()
            super().accept()

    def make_job(self) -> Job:
        return self.export_job(self.out_path, self.params)


class RetrainDialog(ExportDialog):
    """Active learning: export the current labels and fine-tune a labeler model on them in a background process."""

    def __init__(self, project: Project, sessions: list[Session], model_dirs: list[Path], parent=None):
        super().__init__(project, sessions, parent)
        self.setWindowTitle("Retrain a labeler on my labels")
        self.proc: QProcess | None = None
        self.running_export: Job | None = None
        self.out.setText(str(ROOT / "datasets" / f"{project.name.replace(' ', '_')}_retrain"))
        lay = self.layout()
        form = QFormLayout()
        self.base = QComboBox()
        for m in scan_models(model_dirs, project.default_models):
            self.base.addItem(f"{m.label}  ({m.path})", str(m.path))
        self.name = QLineEdit(f"{project.name.replace(' ', '_')}_labeler")
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(60)
        self.imgsz = QComboBox()
        for s in (640, 960, 1280, 1536):
            self.imgsz.addItem(str(s), s)
        self.imgsz.setCurrentText("1280")
        self.bs = QSpinBox()
        self.bs.setRange(1, 64)
        self.bs.setValue(4)
        self.freeze = QSpinBox()
        self.freeze.setRange(0, 30)
        self.freeze.setValue(10)
        self.vram = QDoubleSpinBox()
        self.vram.setRange(2, 16)
        self.vram.setValue(13)
        form.addRow("Start from model", self.base)
        form.addRow("Run name (train_runs/<name>)", self.name)
        form.addRow("Epochs", self.epochs)
        form.addRow("Image size", self.imgsz)
        form.addRow("Batch", self.bs)
        form.addRow("Freeze first layers", self.freeze)
        form.addRow("VRAM cap, GB", self.vram)
        lay.insertLayout(lay.count() - 1, form)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setMinimumHeight(220)
        lay.insertWidget(lay.count() - 1, self.log)
        self.resize(820, 820)

    def accept(self) -> None:
        if self.proc is not None:
            QMessageBox.information(self, "Retrain", "Training is running; close this window to keep it running in the "
                                    "background log, or wait.")
            return
        if self.running_export is not None:
            return
        if not self.parent().gpu_preflight():
            return
        out = self.validate_out("Retrain", ask_replace=False)
        if out is None:
            return
        self.params = self.read_params()
        self.bb.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.log.appendPlainText("Exporting the labeled frames...")
        job = self.export_job(out, self.params)
        self.running_export, self.export_out = job, out
        job.progress.connect(self.on_export_progress)
        job.finished.connect(self.on_exported)
        job.failed.connect(self.on_export_failed)
        job.aborted.connect(self.on_export_aborted)
        start_job(self, job)

    def on_export_progress(self, i: int, n: int, _msg: str) -> None:
        if i == n or i % 500 == 0:
            self.log.appendPlainText(f"exported {i}/{n}")

    def on_exported(self, res) -> None:
        self.running_export = None
        self.log.appendPlainText(res.summary(self.project.class_names))
        self.start_training(self.export_out)

    def on_export_failed(self, msg: str) -> None:
        self.running_export = None
        self.bb.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        QMessageBox.critical(self, "Retrain", msg)

    def on_export_aborted(self) -> None:
        self.running_export = None
        self.bb.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self.log.appendPlainText("export cancelled")

    def start_training(self, out: Path) -> None:
        args = ["-m", "framelabel.ml.train", "--data", str(out / "data.yaml"), "--model", self.base.currentData(),
                "--name", self.name.text(), "--epochs", str(self.epochs.value()), "--imgsz", self.imgsz.currentText(),
                "--batch", str(self.bs.value()), "--freeze", str(self.freeze.value()),
                "--max-vram-gb", str(self.vram.value()), "--game", *(self.project.game_processes or ["UAGame.exe"])]
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(ROOT))
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.read_output)
        self.proc.finished.connect(self.on_finished)
        self.log.appendPlainText(f"$ python {' '.join(args)}")
        self.proc.start(sys.executable, args)

    def read_output(self) -> None:
        text = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.replace("\r", "\n").splitlines():
            line = line.replace("\x1b[K", "").rstrip()
            if line and ("it/s" not in line or "100%" in line):  # progress-bar redraws: only the finished bar
                self.log.appendPlainText(line)

    def on_finished(self, code: int, _status) -> None:
        self.log.appendPlainText(f"training finished with exit code {code}; the model is "
                                 f"train_runs/{self.name.text()}/weights/best.pt (shown in the prelabel model list)")
        self.proc = None

    def reject(self) -> None:
        if self.running_export is not None:
            self.running_export.cancel()
            stop_jobs(self)
            self.running_export = None
        if self.proc is not None and QMessageBox.question(self, "Retrain", "Stop the training?") == QMessageBox.StandardButton.Yes:
            self.proc.kill()
            self.proc.waitForFinished(5000)
            self.proc = None
        elif self.proc is not None:
            return
        super().reject()


class ProjectDialog(QDialog):
    """Name, classes (name + colour; the order defines the class ids), sources, model folders and the
    project-specific zones."""

    def __init__(self, project: Project, parent=None, new: bool = False):
        super().__init__(parent)
        self.setWindowTitle("New project" if new else "Project settings")
        self.project = project
        self.resize(760, 640)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(project.name)
        form.addRow("Name", self.name)
        lay.addLayout(form)
        lay.addWidget(QLabel("Classes (order = class id; renaming keeps existing labels, reordering or removing does "
                             "not rewrite them):"))
        self.classes = QTableWidget(0, 2)
        self.classes.setHorizontalHeaderLabels(["name", "colour"])
        self.classes.horizontalHeader().setStretchLastSection(True)
        self.classes.cellDoubleClicked.connect(self.pick_color)
        for c in project.classes:
            self.add_class_row(c.name, c.color)
        row = QHBoxLayout()
        for text, fn in (("Add class", lambda: self.add_class_row(f"class{self.classes.rowCount()}",
                                                                  PALETTE[self.classes.rowCount() % len(PALETTE)])),
                         ("Remove selected", self.remove_class)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        lay.addWidget(self.classes)
        lay.addLayout(row)
        lay.addWidget(QLabel("Sources (session folders, folders of sessions, image folders or YOLO datasets):"))
        self.sources = QListWidget()
        for s in project.sources:
            self.sources.addItem(str(s))
        lay.addWidget(self.sources)
        row = QHBoxLayout()
        for text, fn in (("Add folder", lambda: self.add_dir(self.sources)), ("Remove", lambda: self.remove(self.sources))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(QLabel("Model folders (*.pt and training runs):"))
        self.models = QListWidget()
        for s in project.model_dirs:
            self.models.addItem(str(s))
        lay.addWidget(self.models)
        row = QHBoxLayout()
        for text, fn in (("Add folder", lambda: self.add_dir(self.models)), ("Remove", lambda: self.remove(self.models))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        form = QFormLayout()
        self.ignore = QLineEdit("; ".join(" ".join(f"{v:g}" for v in z) for z in project.ignore_zones))
        self.ignore.setPlaceholderText("x1 y1 x2 y2 (normalized); several separated by ';'")
        self.ui_zone = QLineEdit(" ".join(f"{v:g}" for v in project.ui_zone) if project.ui_zone else "")
        self.ui_zone.setPlaceholderText("x1 y1 x2 y2 of a HUD element present in every gameplay frame")
        self.game = QLineEdit(" ".join(project.game_processes))
        self.game.setPlaceholderText("process names that block GPU work, e.g. UAGame.exe")
        form.addRow("Ignore model proposals in zones", self.ignore)
        form.addRow("Interface-screen detection zone", self.ui_zone)
        form.addRow("Game processes", self.game)
        lay.addLayout(form)
        lay.addWidget(buttons(self))

    def add_class_row(self, name: str, color: str) -> None:
        r = self.classes.rowCount()
        self.classes.insertRow(r)
        self.classes.setItem(r, 0, QTableWidgetItem(name))
        it = QTableWidgetItem(color)
        it.setBackground(QColor(color))
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.classes.setItem(r, 1, it)

    def pick_color(self, r: int, c: int) -> None:
        if c != 1:
            return
        col = QColorDialog.getColor(QColor(self.classes.item(r, 1).text()), self)
        if col.isValid():
            self.classes.item(r, 1).setText(col.name())
            self.classes.item(r, 1).setBackground(col)

    def remove_class(self) -> None:
        r = self.classes.currentRow()
        if r >= 0 and self.classes.rowCount() > 1:
            self.classes.removeRow(r)

    def add_dir(self, lst: QListWidget) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder")
        if d:
            lst.addItem(d)

    @staticmethod
    def remove(lst: QListWidget) -> None:
        for it in lst.selectedItems():
            lst.takeItem(lst.row(it))

    @staticmethod
    def zones(text: str) -> list[tuple[float, ...]]:
        """'x1 y1 x2 y2; ...' -> zones; ValueError naming the part that is not four numbers between 0 and 1."""
        out = []
        for part in text.split(";"):
            if not part.strip():
                continue
            try:
                vals = [float(v) for v in part.replace(",", " ").split()]
            except ValueError:
                vals = []
            if len(vals) != 4 or not all(0 <= v <= 1 for v in vals) or vals[0] >= vals[2] or vals[1] >= vals[3]:
                raise ValueError(f"'{part.strip()}' is not a zone: four numbers x1 y1 x2 y2 between 0 and 1, "
                                 "x1 < x2 and y1 < y2")
            out.append(tuple(vals))
        return out

    def problem(self) -> str:
        """Why the entered settings cannot be applied, '' when they can."""
        names = [self.classes.item(r, 0).text().strip() for r in range(self.classes.rowCount())]
        names = [n for n in names if n]
        if not names:
            return "The project needs at least one class."
        if len(set(names)) != len(names):
            return "Two classes have the same name; class names must differ."
        try:
            self.zones(self.ignore.text())
            if len(self.zones(self.ui_zone.text())) > 1:
                return "The interface-screen detection zone is one zone."
        except ValueError as e:
            return str(e)
        return ""

    def accept(self) -> None:
        msg = self.problem()
        if msg:
            QMessageBox.warning(self, self.windowTitle(), msg)
            return
        super().accept()

    def apply(self) -> None:
        p = self.project
        p.name = self.name.text().strip() or p.name
        p.classes = [ClassDef(self.classes.item(r, 0).text().strip(), self.classes.item(r, 1).text())
                     for r in range(self.classes.rowCount()) if self.classes.item(r, 0).text().strip()]
        p.sources = [Path(self.sources.item(i).text()) for i in range(self.sources.count())]
        p.model_dirs = [Path(self.models.item(i).text()) for i in range(self.models.count())]
        p.ignore_zones = self.zones(self.ignore.text())
        z = self.zones(self.ui_zone.text())
        p.ui_zone = z[0] if z else None
        p.game_processes = self.game.text().split()


def ordered_actions() -> list[str]:
    """The keymap's actions grouped (everywhere / grid / editor) the way the shortcut dialog and the help list them."""
    return [a for g in ACTION_GROUPS for a in DEFAULT_KEYMAP if ACTION_INFO.get(a, ("Editor",))[0] == g]


class KeymapDialog(QDialog):
    def __init__(self, keymap: Keymap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")
        self.resize(760, 760)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Several keys separated by commas, e.g. 'Space, Return' or 'Ctrl+Z'. Letters work in any "
                             "keyboard layout. 1-9 select classes, W toggles weak proposals, Esc leaves fullscreen. "
                             "Changes apply when you press OK."))
        self.actions = ordered_actions()
        self.table = QTableWidget(len(self.actions), 3)
        self.table.setHorizontalHeaderLabels(["where", "what it does", "keys"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        for r, action in enumerate(self.actions):
            group, text = ACTION_INFO.get(action, ("Editor", action))
            for c, value in enumerate((group, text)):
                it = QTableWidgetItem(value)
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, it)
            self.table.setItem(r, 2, QTableWidgetItem(", ".join(keymap.map[action])))
        self.table.setColumnWidth(0, 90)
        self.table.setColumnWidth(1, 400)
        lay.addWidget(self.table)
        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(self.reset)
        lay.addWidget(reset)
        lay.addWidget(buttons(self))
        self.keymap = keymap

    def reset(self) -> None:
        for r, action in enumerate(self.actions):
            self.table.item(r, 2).setText(", ".join(DEFAULT_KEYMAP[action]))

    def overrides(self) -> dict[str, list[str]]:
        out = {}
        for r, action in enumerate(self.actions):
            keys = [k.strip() for k in self.table.item(r, 2).text().split(",") if k.strip()]
            if keys != DEFAULT_KEYMAP[action]:
                out[action] = keys
        return out

    def accept(self) -> None:
        self.keymap = Keymap(self.overrides())
        super().accept()


class HelpDialog(QDialog):
    """What the mouse and the keys do in each mode, with the keys as currently set; opens on F1."""

    MOUSE = {
        "Grid": [("Click", "mark the frame 'has objects' (click again to undo)"),
                 ("Ctrl+click or middle button", "mark the frame SKIP"),
                 ("Double-click", "open the frame in the Editor (Tab comes back to its page)"),
                 ("Arrow keys, Enter, X", "move the tile cursor (or flip pages: Settings > Arrow keys in the grid); Enter ticks the "
                                          "tile, X marks it SKIP; O opens it in the editor"),
                 ("Labeled tiles", "locked: a click cannot cost a frame its boxes; open it in the Editor to change it"),
                 ("Hold the right button", "peek at the photo; it closes when you let go (a quick click keeps it open)"),
                 ("1 × 1 tiles", "the tile is the photo itself: wheel zoom, drag to pan, no right click needed")],
        "Editor": [("Drag on empty space", "new box"), ("Drag a box / corner / edge", "move / resize"),
                   ("Right click", "delete the box under the cursor"), ("Wheel", "zoom at the cursor"),
                   ("Middle drag", "pan"), ("Alt+click", "box from a point with SAM"),
                   ("Click a faint box", "accept that weak proposal (when weak proposals are shown, W)")],
    }

    def __init__(self, keymap: Keymap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard and mouse help")
        self.resize(820, 780)
        lay = QVBoxLayout(self)
        view = QTextBrowser()
        view.setHtml(self.html(keymap))
        lay.addWidget(view)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        lay.addWidget(bb)

    @classmethod
    def html(cls, keymap: Keymap) -> str:
        def table(rows: list[tuple[str, str]]) -> str:
            return "<table cellpadding='3'>" + "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows) + "</table>"

        out = ["<h3>The way through the work</h3>"
               "<p><b>1 Triage</b> (grid): frames the model found nothing in. Click the tiles that show an object, press "
               "Space: the unmarked ones are saved as empty, the ticked ones go to Label.<br>"
               "<b>2 Label</b> (editor): fix the model's boxes or draw your own, Space saves and goes on. The order can be "
               "changed (time, unsure first, small objects first).<br>"
               "<b>3 Review</b> (grid): saved frames that deserve a second look (possibly missed or wrong boxes, small "
               "objects). Fix them, or mark them as checked to hide them. Amber dashed boxes are what the model sees and "
               "the labels do not.</p>"
               "<p>Under each stage are its <i>lenses</i> (parts of the stage, with counts). The <b>Library</b> lists "
               "everything by decision or class. The icon of a stage is the way it opens: a grid of tiles, or a box with "
               "handles (the editor); the <b>Grid | Editor</b> switch at the top changes it for the view you are in.</p>"
               "<p>A frame is <b>new</b> (no decision), <b>marked</b> ('has objects' ticked in Triage, boxes still to do), "
               "<b>labeled</b>, <b>empty</b> or <b>skipped</b>. Ctrl+Z in the grid takes back the last saved page; "
               "<b>Ctrl+K</b> searches views, actions and frames.</p>"]
        for group in ACTION_GROUPS:
            rows = [(", ".join(keymap.map[a]), ACTION_INFO[a][1]) for a in DEFAULT_KEYMAP if ACTION_INFO.get(a, ("",))[0] == group]
            out.append(f"<h3>Keys: {'everywhere' if group == 'Everywhere' else group.lower()}</h3>" + table(rows))
            if group in cls.MOUSE:
                out.append(f"<h3>Mouse: {group.lower()}</h3>" + table(cls.MOUSE[group]))
        out.append("<h3>More keys</h3>" + table([("1-9", "select the class for new boxes (editor)"),
                                                 ("W", "show / hide weak proposals"), ("Esc", "close a preview, leave fullscreen")]))
        return "".join(out)


class PerformanceDialog(QDialog):
    """Preferences that trade memory / CPU for responsiveness. Values are stored under perf/ in the settings."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Performance settings")
        self.settings = settings
        lay = QVBoxLayout(self)
        form = QFormLayout()

        def boolean(key: str, default: bool) -> bool:
            v = settings.value(key, default)
            return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")

        def number(key: str, default: int) -> int:
            try:
                return int(settings.value(key, default))
            except (TypeError, ValueError):
                return default

        self.watchdog = QCheckBox("Log a stack trace when the window freezes for 2 s")
        self.watchdog.setChecked(boolean("perf/watchdog", True))
        self.io = QSpinBox()
        self.io.setRange(1, 32)
        self.io.setValue(number("perf/io_threads", 8))
        self.io.setToolTip("Threads that read label files and frames when a session opens")
        self.prefetch = QSpinBox()
        self.prefetch.setRange(0, 8)
        self.prefetch.setValue(number("perf/prefetch", 2))
        self.prefetch.setToolTip("Frames read ahead of the current one in the editor (one behind is always kept)")
        self.preload = QCheckBox("Load SAM in the background after a session opens (uses VRAM; the game must be closed)")
        self.preload.setChecked(boolean("perf/sam_preload", False))
        form.addRow("", self.watchdog)
        form.addRow("I/O threads", self.io)
        form.addRow("Frames read ahead", self.prefetch)
        form.addRow("", self.preload)
        lay.addLayout(form)
        lay.addWidget(buttons(self))

    def apply(self) -> None:
        self.settings.setValue("perf/watchdog", self.watchdog.isChecked())
        self.settings.setValue("perf/io_threads", self.io.value())
        self.settings.setValue("perf/prefetch", self.prefetch.value())
        self.settings.setValue("perf/sam_preload", self.preload.isChecked())


class SamAdjustDialog(QDialog):
    """A fixed correction applied to every box SAM fits: the fitter is often a pixel or two off on every side."""

    def __init__(self, px: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM fit correction")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Move every side of a box fitted by SAM by a fixed number of pixels (of the photo), so a\n"
                             "fitter that is a pixel or two too tight (or too loose) comes out right."))
        form = QFormLayout()
        self.px = QSpinBox()
        self.px.setRange(0, 20)
        self.px.setSuffix(" px per side")
        self.px.setValue(abs(px))
        self.direction = QComboBox()
        self.direction.addItem("Outward: the box gets bigger", 1)
        self.direction.addItem("Inward: the box gets smaller", -1)
        self.direction.setCurrentIndex(1 if px < 0 else 0)
        form.addRow("Amount", self.px)
        form.addRow("Direction", self.direction)
        lay.addLayout(form)
        lay.addWidget(buttons(self))

    def value(self) -> int:
        """Signed pixels: positive grows the box, negative shrinks it."""
        return self.px.value() * int(self.direction.currentData())


class ReportDialog(QDialog):
    """Everything that is filtered out of the normal views of a session, counted, with the view that shows it.
    Double-click a row with a view to open it."""

    open_view = pyqtSignal(str, str)  # mode ("grid" / "edit"), queue key

    def __init__(self, title: str, sections: list[tuple[str, list[ReportRow]]], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 720)
        self.sections = sections
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("What is filtered out of the normal views, and where to look at it. Double-click a row that "
                             "has a view to open it."))
        n = sum(len(rows) + 1 for _, rows in sections)
        self.table = QTableWidget(n, 4)
        self.table.setHorizontalHeaderLabels(["", "Frames", "Where / what to do", "Notes"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setWordWrap(True)
        self.views: dict[int, tuple[str, str]] = {}
        r = 0
        for head, rows in sections:
            item = QTableWidgetItem(head)
            f = QFont(item.font())
            f.setBold(True)
            item.setFont(f)
            self.table.setItem(r, 0, item)
            self.table.setSpan(r, 0, 1, 4)
            r += 1
            for row in rows:
                self.table.setItem(r, 0, QTableWidgetItem(row.label))
                cnt = QTableWidgetItem("" if row.count is None else str(row.count))
                cnt.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, 1, cnt)
                self.table.setItem(r, 2, QTableWidgetItem(row.where))
                self.table.setItem(r, 3, QTableWidgetItem(row.note))
                if row.mode and row.queue:
                    self.views[r] = (row.mode, row.queue)
                r += 1
        self.table.setColumnWidth(0, 430)
        self.table.setColumnWidth(1, 70)
        self.table.setColumnWidth(2, 250)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.resizeRowsToContents()
        self.table.cellDoubleClicked.connect(self.on_double_click)
        lay.addWidget(self.table)
        row = QHBoxLayout()
        copy = QPushButton("Copy as text")
        copy.clicked.connect(self.copy_text)
        row.addWidget(copy)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)

    def on_double_click(self, r: int, _c: int) -> None:
        if r in self.views:
            self.open_view.emit(*self.views[r])
            self.accept()

    def as_text(self) -> str:
        lines = []
        for head, rows in self.sections:
            lines.append(head)
            lines += [f"  {row.label}: {'' if row.count is None else row.count}  [{row.where}] {row.note}".rstrip()
                      for row in rows]
        return "\n".join(lines)

    def copy_text(self) -> None:
        from PyQt6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.as_text())
