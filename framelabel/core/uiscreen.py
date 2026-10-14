"""Interface-screen detection (inventory, map, death or extraction screens) by a HUD element that is present in every
gameplay frame: frames whose zone does not match the session's typical (median) content get a low score and are
suggested as 'skip'. For Arena Breakout Infinite the zone is the body-status silhouette icon (bottom-left)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import index_cache
from .index_cache import Cancelled, cache_dir
from .images import imread
from .session import Session

log = logging.getLogger("framelabel.ui")


def zone_crop(gray: np.ndarray, zone) -> np.ndarray:
    h, w = gray.shape
    x1, y1, x2, y2 = zone
    return gray[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)].astype(np.float32)


def _read_crops(frames: list[Path], zone, progress, cancelled, workers: int | None = None) -> dict[str, np.ndarray | None]:
    """HUD-zone crops of the frames (a reduced grayscale decode, in parallel: decoding releases the GIL)."""
    def read(p: Path):
        img = imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        return zone_crop(img, zone) if img is not None else None

    out: dict[str, np.ndarray | None] = {}
    ex = ThreadPoolExecutor(workers or index_cache.IO_WORKERS)
    try:
        for i, (p, crop) in enumerate(zip(frames, ex.map(read, frames))):
            out[p.stem] = crop
            if i % 200 == 0:
                if cancelled and cancelled():
                    raise Cancelled()
                if progress:
                    progress(i, len(frames), "interface screens")
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return out


def _score(crop, tmpl: np.ndarray) -> float:
    if crop is None or crop.shape != tmpl.shape:
        return 1.0
    return float(cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED).max())


LEGACY_ZONE = [0.0, 0.75, 0.07, 0.92]  # the zone of screendet's label_frames.py, whose ui_scores.json has no '_zone'


def _cache_key(session: Session, zone) -> str:
    """One cache per session folder and zone: projects that share a session and a zone share it, another zone never
    sees it."""
    return hashlib.sha1(f"{str(session.cache_dir.resolve()).lower()}{zone!r}".encode()).hexdigest()[:20]


def _template_path(session: Session, zone) -> Path:
    return cache_dir() / f"ui_tmpl_{_cache_key(session, zone)}.npy"


def _scores_path(session: Session, zone) -> Path:
    return cache_dir() / f"ui_scores_{_cache_key(session, zone)}.json"


def _read_scores(path: Path, zone, zone_if_missing=None) -> dict[str, float]:
    """{stem: score} of a scores file made for `zone`; {} when it is missing, unreadable or made for another zone (a file
    without '_zone' counts as made for `zone_if_missing`)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("_zone", zone_if_missing) != zone:
        return {}
    return {k: float(v) for k, v in data.items() if not k.startswith("_") and isinstance(v, (int, float))}


def ui_scores(session: Session, frames: list[Path], zone, progress=None, cancelled=None) -> dict[str, float]:
    """Similarity of each frame's zone to the session's median zone (~1 in gameplay, ~0 on interface screens).
    Cached in the app's cache folder per session and zone (a flat {stem: score} map plus '_zone'); nothing is written
    into the session. A <session>/ui_scores.json made for the same zone (by an older framelabel, or by screendet's
    label_frames.py) is read as a start and left as it is. When only frames were added, just those are scored against
    the median template saved in the cache; otherwise everything is recomputed.
    `progress(i, n, msg)` and `cancelled()` are optional; cancelling raises index_cache.Cancelled."""
    zone = [float(v) for v in zone]
    cache = _scores_path(session, zone)
    scores = _read_scores(cache, zone)
    if not scores:
        scores = _read_scores(session.cache_dir / "ui_scores.json", zone, LEGACY_ZONE)
    missing = [p for p in frames if p.stem not in scores]
    if scores and not missing:
        if not cache.exists():
            _write_scores(cache, scores, zone)  # taken over from the session's file: from now on read from the cache
        return scores
    if not frames:
        return {}
    tmpl_file = _template_path(session, zone)
    tmpl = None
    if scores:  # frames were added: score only them against the saved median template
        try:
            tmpl = np.load(tmpl_file)
        except (OSError, ValueError):
            tmpl = None
    if tmpl is not None:
        log.info("interface screens: scoring %d new frames", len(missing))
        crops = _read_crops(missing, zone, progress, cancelled)
        scores.update({k: _score(c, tmpl) for k, c in crops.items()})
    else:
        crops = _read_crops(frames, zone, progress, cancelled)
        valid = [c for c in crops.values() if c is not None and c.size]
        if not valid:
            return {k: 1.0 for k in crops}
        shape = valid[0].shape
        ref = [c for c in valid[::10] if c.shape == shape]
        tmpl = np.median(np.stack(ref), axis=0).astype(np.float32)  # most frames are gameplay
        scores = {k: _score(c, tmpl) for k, c in crops.items()}
        try:
            tmpl_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(tmpl_file, tmpl)
        except OSError:
            pass
    _write_scores(cache, scores, zone)
    return scores


def _write_scores(path: Path, scores: dict[str, float], zone) -> None:
    """Atomically; a cache that cannot be written (a read-only app folder) only costs a recount next time."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({**scores, "_zone": zone}), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        log.warning("cannot write the interface-screen cache %s: %s", path.name, e)
        tmp.unlink(missing_ok=True)
