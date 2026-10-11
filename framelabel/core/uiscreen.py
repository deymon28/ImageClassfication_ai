"""Interface-screen detection (inventory, map, death or extraction screens) by a HUD element that is present in every
gameplay frame: frames whose zone does not match the session's typical (median) content get a low score and are
suggested as 'skip'. For Arena Breakout Infinite the zone is the body-status silhouette icon (bottom-left)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .session import Session


def zone_crop(gray: np.ndarray, zone) -> np.ndarray:
    h, w = gray.shape
    x1, y1, x2, y2 = zone
    return gray[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)].astype(np.float32)


def ui_scores(session: Session, frames: list[Path], zone, progress=None) -> dict[str, float]:
    """Similarity of each frame's zone to the session's median zone (~1 in gameplay, ~0 on interface screens).
    Cached in <session>/ui_scores.json (a flat {stem: score} map plus '_zone'); recomputed when frames were added
    or the zone changed."""
    cache = session.cache_dir / "ui_scores.json"
    zone = [float(v) for v in zone]
    if cache.exists():
        try:
            scores = json.loads(cache.read_text(encoding="utf-8"))
        except ValueError:
            scores = {}
        same_zone = scores.get("_zone", zone) == zone  # caches without '_zone' were made with the default zone
        if same_zone and all(p.stem in scores for p in frames):
            return {k: v for k, v in scores.items() if not k.startswith("_")}
    if not frames:
        return {}
    crops = {}
    for i, p in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        crops[p.stem] = zone_crop(img, zone) if img is not None else None
        if progress and i % 200 == 0:
            progress(i, len(frames))
    valid = [c for c in crops.values() if c is not None and c.size]
    if not valid:
        return {k: 1.0 for k in crops}
    shape = valid[0].shape
    ref = [c for c in valid[::10] if c.shape == shape]
    tmpl = np.median(np.stack(ref), axis=0).astype(np.float32)  # most frames are gameplay
    scores = {k: float(cv2.matchTemplate(c, tmpl, cv2.TM_CCOEFF_NORMED).max()) if c is not None and c.shape == shape else 1.0
              for k, c in crops.items()}
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({**scores, "_zone": zone}), encoding="utf-8")
    except OSError:
        pass
    return scores
