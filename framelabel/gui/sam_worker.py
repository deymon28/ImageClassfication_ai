"""SAM box fitting in a dedicated Python thread: the model is loaded there on the first request, requests run one at a
time from a queue.

It is a plain `threading.Thread`, not a QThread running a slot: torch's pybind11 layer caches the interpreter's thread
state of the calling thread, and a Qt-created thread gets a fresh one for every slot call, so the second call crashed
the process ("_PyThreadState_Attach: non-NULL old thread state"). A Python thread keeps the same state for its life."""
from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

log = logging.getLogger("framelabel.sam")


class SamWorker(QObject):
    """Lives in the GUI thread; its signals are emitted from the worker thread and arrive queued."""

    ready = pyqtSignal(int, object)   # request id, Box | None
    failed = pyqtSignal(int, str)
    status = pyqtSignal(str)

    def __init__(self, weights: Path, game_processes: list[str]):
        super().__init__()
        self.weights, self.game_processes, self.fitter = Path(weights), list(game_processes), None
        self.jobs: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self.loop, name="sam", daemon=True)

    # ---- called from the GUI thread
    def start(self) -> None:
        self.thread.start()

    def submit(self, rid: int, img, box, point) -> None:
        """Fit a rough box or a click on `img` (a BGR array that the GUI never modifies in place)."""
        self.jobs.put(("fit", rid, img, box, point))

    def preload(self) -> None:
        """Load the model before the first request (a preference), so the first fit is not slow."""
        self.jobs.put(("preload",))

    def stop(self, timeout: float = 10.0) -> bool:
        self.jobs.put(None)
        if self.thread.is_alive():
            self.thread.join(timeout)
        return not self.thread.is_alive()

    # ---- the worker thread
    def loop(self) -> None:
        while (job := self.jobs.get()) is not None:
            if job[0] == "preload":
                self.do_preload()
            else:
                self.do_fit(*job[1:])

    def load(self) -> None:
        from framelabel.ml.guard import check_gpu_allowed

        check_gpu_allowed(self.game_processes)
        if self.fitter is None:
            self.status.emit("Loading SAM...")
            from framelabel.ml.sam import BoxFitter

            fitter = BoxFitter(self.weights)
            fitter.load()
            self.fitter = fitter

    def do_preload(self) -> None:
        from framelabel.ml.guard import GuardError

        try:
            self.load()
            self.status.emit("SAM ready.")
        except GuardError as e:
            self.status.emit(str(e))
        except Exception:
            log.exception("SAM preload failed")
            self.status.emit("SAM preload failed (see the log).")

    def do_fit(self, rid: int, img, box, point) -> None:
        from framelabel.ml.guard import GuardError

        try:
            self.load()
            self.ready.emit(rid, self.fitter.fit(img, box=box, point=point))
        except GuardError as e:
            self.failed.emit(rid, str(e))
        except Exception as e:
            log.exception("SAM request failed")
            self.failed.emit(rid, f"SAM failed: {type(e).__name__}: {e}")
