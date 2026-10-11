"""SAM box fitting in a dedicated thread: the model is loaded there on the first request, requests run one at a time."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

log = logging.getLogger("framelabel.sam")


class SamWorker(QObject):
    ready = pyqtSignal(int, object)   # request id, Box | None
    failed = pyqtSignal(int, str)
    status = pyqtSignal(str)

    def __init__(self, weights: Path, game_processes: list[str]):
        super().__init__()
        self.weights, self.game_processes, self.fitter = Path(weights), list(game_processes), None

    @pyqtSlot()
    def preload(self) -> None:
        """Load the model before the first request (a preference), so the first fit is not slow."""
        from framelabel.ml.guard import GuardError, check_gpu_allowed

        try:
            check_gpu_allowed(self.game_processes)
            if self.fitter is None:
                self.status.emit("Loading SAM...")
                from framelabel.ml.sam import BoxFitter

                fitter = BoxFitter(self.weights)
                fitter.load()
                self.fitter = fitter
            self.status.emit("SAM ready.")
        except GuardError as e:
            self.status.emit(str(e))
        except Exception:
            log.exception("SAM preload failed")
            self.status.emit("SAM preload failed (see the log).")

    @pyqtSlot(int, object, object, object)
    def fit(self, rid: int, img, box, point) -> None:
        """Fit a rough box or a click on `img` (a BGR array that the GUI never modifies in place)."""
        from framelabel.ml.guard import GuardError, check_gpu_allowed

        try:
            check_gpu_allowed(self.game_processes)
            if self.fitter is None:
                self.status.emit("Loading SAM...")
                from framelabel.ml.sam import BoxFitter

                fitter = BoxFitter(self.weights)
                fitter.load()
                self.fitter = fitter
            self.ready.emit(rid, self.fitter.fit(img, box=box, point=point))
        except GuardError as e:
            self.failed.emit(rid, str(e))
        except Exception as e:
            log.exception("SAM request failed")
            self.failed.emit(rid, f"SAM failed: {type(e).__name__}: {e}")
