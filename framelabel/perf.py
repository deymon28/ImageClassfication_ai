"""Logging setup and timing helpers shared by the CLI and the GUI (no Qt imports)."""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

log = logging.getLogger("framelabel")


def app_data_dir() -> Path:
    """Per-user folder for logs and caches: FRAMELABEL_HOME, else %LOCALAPPDATA%/framelabel, else ~/.framelabel."""
    env = os.environ.get("FRAMELABEL_HOME")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / "framelabel" if base else Path.home() / ".framelabel"


def setup_logging(level: int = logging.INFO) -> Path:
    """Console + rotating file log (2 MB x 3). Returns the log folder. Safe to call twice."""
    folder = app_data_dir() / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("framelabel")
    if not root.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s")
        fh = RotatingFileHandler(folder / "framelabel.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(fh)
        root.addHandler(ch)
        root.setLevel(level)
    return folder


@contextmanager
def timed(label: str, warn_ms: float = 100.0):
    """Log how long the block took: DEBUG normally, WARNING when slower than warn_ms."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - t0) * 1000
        log.log(logging.WARNING if ms > warn_ms else logging.DEBUG, "%s took %.0f ms", label, ms)
