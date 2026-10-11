"""Detects a blocked GUI thread: a 200 ms timer re-arms faulthandler; if the thread is stuck for FREEZE_S, the stacks
of all threads are written to freeze.log, and late ticks are logged as UI stalls."""
from __future__ import annotations

import faulthandler
import logging
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QTimer

log = logging.getLogger("framelabel.watchdog")
TICK_MS, FREEZE_S, STALL_MS = 200, 2.0, 300


class Watchdog(QObject):
    def __init__(self, log_dir: Path, parent=None):
        super().__init__(parent)
        self.file = open(log_dir / "freeze.log", "a", encoding="utf-8")  # kept open: faulthandler writes to its fd
        self.last = time.perf_counter()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(TICK_MS)
        self.tick()

    def tick(self) -> None:
        now = time.perf_counter()
        late = (now - self.last) * 1000 - TICK_MS
        if late > STALL_MS:
            log.warning("UI stalled for %.0f ms", late)
        self.last = now
        faulthandler.dump_traceback_later(FREEZE_S, repeat=False, file=self.file)  # cancels the previous arm

    def stop(self) -> None:
        self.timer.stop()
        faulthandler.cancel_dump_traceback_later()
        self.file.close()
