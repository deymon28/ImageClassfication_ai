"""Background jobs for the UI: a QThread running a Python callable with progress and cancel support."""
from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, QThread, pyqtSignal

log = logging.getLogger("framelabel.job")


class Job(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.cancel_requested = False

    def cancelled(self) -> bool:
        return self.cancel_requested

    def run(self) -> None:
        try:
            result = self.fn(*self.args, progress=self.progress.emit, cancelled=self.cancelled, **self.kwargs)
        except Exception as e:  # reported to the UI; the traceback goes to the log
            log.exception("job failed")
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.finished.emit(result)


def start_job(owner, job: Job) -> QThread:
    """Run job.run in a new thread owned by `owner`; the thread is kept referenced until it finishes."""
    th = QThread(owner)
    job.moveToThread(th)
    th.started.connect(job.run)
    job.finished.connect(th.quit)
    job.failed.connect(th.quit)
    th.finished.connect(th.deleteLater)
    jobs = owner.__dict__.setdefault("_jobs", [])
    jobs.append((th, job))
    th.finished.connect(lambda: jobs.remove((th, job)) if (th, job) in jobs else None)
    th.start()
    return th
