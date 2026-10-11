"""Interface-screen detection (inventory, map, death or extraction screens) by a HUD element that is present in every
gameplay frame: frames whose zone does not match the session's typical (median) content get a low score and are
suggested as 'skip'. For Arena Breakout Infinite the zone is the body-status silhouette icon (bottom-left)."""
from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .index_cache import Cancelled, cache_dir
from .session import Session

log = logging.getLogger("framelabel.ui")


def zone_crop(gray: np.ndarray, zone) -> np.ndarray:
    h, w = gray.shape
    x1, y1, x2, y2 = zone
    return gray[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)].astype(np.float32)


def _read_crops(frames: list[Path], zone, progress, cancelled, workers: int = 8) -> dict[str, np.ndarray | None]:
    """HUD-zone crops of the frames (a reduced grayscale decode, in parallel: cv2.imread releases the GIL)."""
    def read(p: Path):
        img = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        return zone_crop(img, zone) if img is not None else None

    out: dict[str, np.ndarray | None] = {}
    ex = ThreadPoolExecutor(workers)
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


def _template_path(session: Session, zone) -> Path:
    key = f"{str(session.cache_dir.resolve()).lower()}{zone!r}"
    return cache_dir() / f"ui_tmpl_{hashlib.sha1(key.encode()).hexdigest()[:20]}.npy"


def ui_scores(session: Session, frames: list[Path], zone, progress=None, cancelled=None) -> dict[str, float]:
    """Similarity of each frame's zone to the session's median zone (~1 in gameplay, ~0 on interface screens).
    Cached in <session>/ui_scores.json (a flat {stem: score} map plus '_zone'). When only frames were added, just
    those are scored against the median template saved in the local cache; otherwise everything is recomputed.
    `progress(i, n, msg)` and `cancelled()` are optional; cancelling raises index_cache.Cancelled."""
    cache = session.cache_dir / "ui_scores.json"
    zone = [float(v) for v in zone]
    scores: dict = {}
    if cache.exists():
        try:
            scores = json.loads(cache.read_text(encoding="utf-8"))
        except ValueError:
            scores = {}
        if scores.get("_zone", zone) != zone:  # caches without '_zone' were made with the default zone
            scores = {}
    missing = [p for p in frames if p.stem not in scores]
    if scores and not missing:
        return {k: v for k, v in scores.items() if not k.startswith("_")}
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
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps({**{k: v for k, v in scores.items() if not k.startswith("_")}, "_zone": zone}),
                       encoding="utf-8")
        tmp.replace(cache)
    except OSError:
        pass
    return {k: v for k, v in scores.items() if not k.startswith("_")}
