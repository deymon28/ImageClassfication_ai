"""Parsed YOLO rows of whole label folders, cached between runs.

Opening a session needs the proposals (labels_auto) and labels of every frame. Reading thousands of small files one by
one is slow (seconds, minutes on a cold disk cache), so the folder is listed once (on Windows the directory listing
already carries size and modification time), the parsed rows are kept in a JSON file in the local cache folder, and
only files whose (mtime_ns, size) changed are read again, in parallel.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from framelabel.perf import app_data_dir
from .boxes import parse_rows

log = logging.getLogger("framelabel.cache")
CACHE_VERSION = 1
_write_lock = threading.Lock()


class Cancelled(Exception):
    """Raised by long core operations when their `cancelled()` callback returns True."""


def cache_dir() -> Path:
    """FRAMELABEL_CACHE_DIR, else <app data>/cache."""
    env = os.environ.get("FRAMELABEL_CACHE_DIR")
    return Path(env) if env else app_data_dir() / "cache"


def scan_txt(folder: Path) -> dict[str, tuple[int, int]]:
    """stem -> (mtime_ns, size) of the *.txt files directly in folder; {} if the folder does not exist."""
    out: dict[str, tuple[int, int]] = {}
    try:
        with os.scandir(folder) as it:
            for e in it:
                if e.name.endswith(".txt") and e.is_file():
                    st = e.stat()
                    out[e.name[:-4]] = (st.st_mtime_ns, st.st_size)
    except (FileNotFoundError, NotADirectoryError):
        return {}
    return out


def _cache_path(folder: Path) -> tuple[Path, str]:
    key = str(Path(folder).resolve())
    return cache_dir() / f"rows_{hashlib.sha1(key.lower().encode()).hexdigest()[:20]}.json", key


def _load(path: Path, key: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") == CACHE_VERSION and data.get("folder") == key and isinstance(data["entries"], dict):
            return data["entries"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return {}


def _rows(raw) -> list[tuple]:
    return [tuple(r) for r in raw]


def read_rows_cached(folder: Path, wanted: set[str] | None = None, progress=None, cancelled=None,
                     workers: int = 8) -> dict[str, list[tuple]]:
    """Parsed YOLO rows (core.boxes.parse_rows tuples) of every txt file in folder (only stems in `wanted` when
    given). A stem without a file is absent from the result; an empty file maps to []."""
    folder = Path(folder)
    full = scan_txt(folder)
    listing = full if wanted is None else {k: v for k, v in full.items() if k in wanted}
    path, key = _cache_path(folder)
    old = _load(path, key)
    entries: dict[str, list] = {}
    result: dict[str, list[tuple]] = {}
    to_read = []
    for stem, sig in listing.items():
        ent = old.get(stem)
        if ent is not None and len(ent) == 3 and (ent[0], ent[1]) == sig:
            entries[stem] = ent
            result[stem] = _rows(ent[2])
        else:
            to_read.append(stem)
    changed = bool(to_read)
    if to_read:
        if cancelled and cancelled():
            raise Cancelled()

        def read(stem: str):
            try:
                return stem, parse_rows((folder / f"{stem}.txt").read_text(encoding="utf-8"))
            except OSError:  # vanished or locked meanwhile
                return stem, None

        ex = ThreadPoolExecutor(workers)
        try:
            futures = [ex.submit(read, s) for s in to_read]
            for n, f in enumerate(as_completed(futures), 1):
                stem, rows = f.result()
                if rows is not None:
                    result[stem] = rows
                    entries[stem] = [*listing[stem], [list(r) for r in rows]]
                if n % 200 == 0 or n == len(to_read):
                    if cancelled and cancelled():
                        raise Cancelled()
                    if progress:
                        progress(n, len(to_read), "reading labels")
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    if wanted is not None:  # keep entries of other stems that are still valid, so alternating wanted sets do not thrash
        for stem, ent in old.items():
            if stem not in listing and stem in full and len(ent) == 3 and (ent[0], ent[1]) == full[stem]:
                entries[stem] = ent
    if len(entries) != len(old) or changed:
        _write(path, key, entries)
    return result


def _write(path: Path, key: str, entries: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({"version": CACHE_VERSION, "folder": key, "entries": entries}), encoding="utf-8")
            os.replace(tmp, path)
    except OSError as e:
        log.warning("cannot write the index cache %s: %s", path.name, e)
        tmp.unlink(missing_ok=True)
