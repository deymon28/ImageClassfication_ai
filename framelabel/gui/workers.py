"""Background jobs for the UI: a QThread running a Python callable with progress and cancel support."""
from __future__ import annotations

import logging
import time

from PyQt6.QtCore import QCoreApplication, QObject, QThread, pyqtSignal, pyqtSlot

from framelabel.core.index_cache import Cancelled

log = logging.getLogger("framelabel.job")
PROGRESS_INTERVAL_S = 0.1


class Job(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    aborted = pyqtSignal()  # the callable raised Cancelled

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.cancel_requested = False
        self._last_emit = 0.0

    def cancelled(self) -> bool:
        return self.cancel_requested

    def cancel(self) -> None:
        self.cancel_requested = True

    def report(self, i: int, n: int, msg: str = "") -> None:
        """Progress callback for the worker: at most one signal per PROGRESS_INTERVAL_S (the last one always)."""
        now = time.monotonic()
        if now - self._last_emit >= PROGRESS_INTERVAL_S or i >= n:
            self._last_emit = now
            self.progress.emit(i, n, msg)

    def run(self) -> None:
        try:
            self._run()
        finally:
            # hand the object back to the GUI thread: it must not outlive its thread with a dead thread affinity
            self.moveToThread(QCoreApplication.instance().thread())

    def _run(self) -> None:
        try:
            result = self.fn(*self.args, progress=self.report, cancelled=self.cancelled, **self.kwargs)
        except Cancelled:
            self.aborted.emit()
            return
        except Exception as e:  # reported to the UI; the traceback goes to the log
            log.exception("job failed")
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.finished.emit(result)


class _Reaper(QObject):
    """Lives in the GUI thread and drops a job's references once its thread has finished (never from the worker
    thread itself: deleting Qt objects there races with the GUI thread)."""

    def __init__(self, owner):
        super().__init__(owner)
        self.jobs: list[tuple[QThread, Job]] = []

    @pyqtSlot()
    def reap(self) -> None:
        th = self.sender()
        self.jobs = [(t, j) for t, j in self.jobs if t is not th]
        th.wait(5000)  # `finished` is emitted just before the thread ends
        th.deleteLater()


def start_job(owner, job: Job) -> QThread:
    """Run job.run in a new thread owned by `owner`; the thread is kept referenced until it finishes."""
    reaper = owner.__dict__.get("_reaper")
    if reaper is None:
        reaper = owner.__dict__["_reaper"] = _Reaper(owner)
    th = QThread(owner)
    job.moveToThread(th)
    th.started.connect(job.run)
    for sig in (job.finished, job.failed, job.aborted):
        sig.connect(th.quit)
    th.finished.connect(reaper.reap)
    reaper.jobs.append((th, job))
    job.thread_ref = th
    th.start()
    return th


def stop_jobs(owner, timeout_ms: int = 10000) -> None:
    """Cancel the jobs started with start_job(owner, ...) and wait for their threads (they stop at the next cancel
    check); needed before the owner is destroyed, or Qt aborts with 'QThread destroyed while still running'."""
    reaper = owner.__dict__.get("_reaper")
    for th, job in list(reaper.jobs if reaper else []):
        job.cancel()
        th.quit()
        if not th.wait(timeout_ms):
            log.warning("a background job did not stop within %d ms", timeout_ms)
