"""A background job that runs in a separate Python process (prelabel): the window stays responsive, torch is not
imported into it, a CUDA error cannot take it down and the VRAM is released when the process exits.

The child prints protocol lines (see `python -m framelabel prelabel --progress-json`):
  @@progress {"done": i, "total": n, "msg": "..."}   @@result {...}   @@error message
Everything else it prints is logged."""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import deque
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from framelabel.core.index_cache import cache_dir

log = logging.getLogger("framelabel.process")
ROOT = Path(__file__).resolve().parents[2]
KILL_AFTER_CANCEL_MS = 60000


class ProcessJob(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, args: list[str], program: str | None = None, parent=None):
        """`args` may contain "{cancel_file}": a path whose existence asks the child to stop after its batch."""
        super().__init__(parent)
        self.cancel_file = cache_dir() / f"cancel_{os.getpid()}_{id(self)}"
        self.program = program or sys.executable
        self.args = [a.replace("{cancel_file}", str(self.cancel_file)) for a in args]
        self.proc: QProcess | None = None
        self.buffer = ""
        self.result = None
        self.error: str | None = None
        self.tail: deque[str] = deque(maxlen=40)

    def start(self) -> None:
        self.cancel_file.parent.mkdir(parents=True, exist_ok=True)
        self.cancel_file.unlink(missing_ok=True)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc = QProcess(self)
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(ROOT))
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.read_output)
        self.proc.finished.connect(self.on_finished)
        self.proc.errorOccurred.connect(self.on_error)
        log.info("starting %s", " ".join([self.program, *self.args]))
        self.proc.start(self.program, self.args)

    def cancel(self) -> None:
        """Ask the child to stop after the current batch; kill it if it does not within a minute."""
        try:
            self.cancel_file.parent.mkdir(parents=True, exist_ok=True)
            self.cancel_file.write_text("cancel", encoding="utf-8")
        except OSError:
            log.warning("cannot write the cancel file")
        QTimer.singleShot(KILL_AFTER_CANCEL_MS, self.kill_if_running)

    def kill_if_running(self) -> None:
        if self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning:
            log.warning("the child did not stop after cancel: killing it")
            self.proc.kill()

    def running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning

    def read_output(self) -> None:
        self.buffer += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace").replace("\r", "\n")
        *lines, self.buffer = self.buffer.split("\n")
        for line in lines:
            self.handle_line(line.rstrip())

    def handle_line(self, line: str) -> None:
        if not line:
            return
        try:
            if line.startswith("@@progress "):
                d = json.loads(line[11:])
                self.progress.emit(int(d["done"]), int(d["total"]), str(d.get("msg", "")))
                return
            if line.startswith("@@result "):
                self.result = json.loads(line[9:])
                return
        except (ValueError, KeyError, TypeError):
            log.warning("bad protocol line: %s", line[:200])
            return
        if line.startswith("@@error "):
            self.error = line[8:]
            return
        self.tail.append(line)
        log.info("child: %s", line)

    def on_finished(self, code: int, _status) -> None:
        if self.buffer.strip():
            self.handle_line(self.buffer.strip())
            self.buffer = ""
        self.cancel_file.unlink(missing_ok=True)
        if code == 0 and self.result is not None:
            self.finished.emit(self.result)
        else:
            self.failed.emit(self.error or f"the process exited with code {code}\n" + "\n".join(list(self.tail)[-10:]))

    def on_error(self, err) -> None:
        if err == QProcess.ProcessError.FailedToStart:
            self.cancel_file.unlink(missing_ok=True)
            self.failed.emit(f"cannot start {self.program}")
