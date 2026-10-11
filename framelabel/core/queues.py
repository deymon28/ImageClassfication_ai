"""Per-frame summary (status, model proposals, labels) and the review queues built from it.

Queues are ordered lists of frames to walk through:
  all         every frame in time order
  triage      the model found nothing (shown as a grid: confirm empty frames a page at a time)
  edit        frames with proposals, or marked 'has objects' in the grid, or labeled/skipped before
  uncertain   unreviewed frames with proposals in the uncertain confidence band, most uncertain first
  small       frames with small (far) objects in proposals or labels, smallest first
  disagree    reviewed frames where the model disagrees with the labels (possible missed or wrong boxes)
  unreviewed / pos / neg / skip / todo   by review status
  class:<i>   frames whose labels contain class i
  ui          suspected interface screens
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .boxes import iou_matrix, parse_rows, in_zones
from .session import Review, Session

QUEUES = ("edit", "triage", "uncertain", "small", "disagree", "all", "unreviewed", "todo", "pos", "neg", "skip", "ui")


@dataclass
class FrameInfo:
    path: Path
    status: str | None = None
    confs: list[float] = field(default_factory=list)       # proposal confidences (outside ignore zones)
    prop_h: list[float] = field(default_factory=list)      # proposal heights, normalized
    prop_xyxy: np.ndarray | None = None                    # normalized proposal boxes
    label_cls: list[int] = field(default_factory=list)
    label_h: list[float] = field(default_factory=list)
    label_xyxy: np.ndarray | None = None
    ui: float = 1.0

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def max_conf(self) -> float:
        return max(self.confs, default=0.0)


def _xyxy(rows) -> np.ndarray:
    a = np.array([(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for _, cx, cy, w, h, _ in rows], float)
    return a.reshape(-1, 4)


def frame_info(session: Session, path: Path, review: Review, ignore_zones=(), ui: float = 1.0) -> FrameInfo:
    fi = FrameInfo(path, review.get(path.stem), ui=ui)
    auto = session.auto_path(path.stem)
    if auto.exists():
        rows = [r for r in parse_rows(auto.read_text(encoding="utf-8"))
                if r[5] is not None and not in_zones(r[1], r[2], ignore_zones)]
        fi.confs = [r[5] for r in rows]
        fi.prop_h = [r[4] for r in rows]
        fi.prop_xyxy = _xyxy(rows)
    lab = session.label_path(path.stem)
    if fi.status in ("pos", "neg") and lab.exists():
        rows = parse_rows(lab.read_text(encoding="utf-8"))
        fi.label_cls = [r[0] for r in rows]
        fi.label_h = [r[4] for r in rows]
        fi.label_xyxy = _xyxy(rows)
    return fi


def build_index(session: Session, review: Review, ignore_zones=(), ui: dict[str, float] | None = None) -> list[FrameInfo]:
    ui = ui or {}
    return [frame_info(session, p, review, ignore_zones, ui.get(p.stem, 1.0)) for p in session.frames()]


def uncertainty(confs: list[float], low: float, high: float) -> float:
    """0..1, higher = more uncertain: the proposal closest to the middle of the band decides."""
    band = [c for c in confs if low <= c <= high]
    if not band:
        return 0.0
    mid, half = (low + high) / 2, max((high - low) / 2, 1e-6)
    return 1.0 - min(abs(c - mid) for c in band) / half


def disagreement(fi: FrameInfo, conf: float, iou: float = 0.4) -> float:
    """Unmatched confident proposals (likely missed objects) + unmatched labels (likely wrong or missed by the model,
    weighted lower)."""
    if fi.status not in ("pos", "neg"):
        return 0.0
    props = fi.prop_xyxy[np.array(fi.confs) >= conf] if fi.prop_xyxy is not None and len(fi.confs) else np.zeros((0, 4))
    labels = fi.label_xyxy if fi.label_xyxy is not None else np.zeros((0, 4))
    if not len(props) and not len(labels):
        return 0.0
    if not len(props) or not len(labels):
        return float(len(props)) + 0.5 * len(labels)
    m = iou_matrix(props, labels) >= iou
    return float((~m.any(1)).sum()) + 0.5 * float((~m.any(0)).sum())


def build_queue(index: list[FrameInfo], name: str, conf: float = 0.3, band: tuple[float, float] = (0.15, 0.6),
                small_h: float = 0.03, ui_threshold: float = 0.3) -> list[FrameInfo]:
    """Frames of a queue in walking order. `conf` is the proposal confidence shown as a box; `small_h` is a box
    height as a share of the image height (0.03 = 32 px at 1080p)."""
    if name == "all":
        return list(index)
    if name == "triage":
        return [f for f in index if f.max_conf < conf]
    if name == "edit":
        return [f for f in index if f.max_conf >= conf or f.status in ("todo", "pos", "skip")]
    if name == "uncertain":
        scored = [(uncertainty(f.confs, *band), f) for f in index if f.status in (None, "todo")]
        return [f for s, f in sorted((x for x in scored if x[0] > 0), key=lambda x: -x[0])]
    if name == "small":
        def smallest(f: FrameInfo) -> float:
            hs = f.label_h if f.status in ("pos", "neg") else [h for h, c in zip(f.prop_h, f.confs) if c >= band[0]]
            return min(hs, default=1.0)
        scored = [(smallest(f), f) for f in index if f.status != "skip"]
        return [f for h, f in sorted((x for x in scored if x[0] < small_h), key=lambda x: x[0])]
    if name == "disagree":
        scored = [(disagreement(f, max(conf, 0.5)), f) for f in index]
        return [f for s, f in sorted((x for x in scored if x[0] > 0), key=lambda x: -x[0])]
    if name == "unreviewed":
        return [f for f in index if f.status is None]
    if name in ("todo", "pos", "neg", "skip"):
        return [f for f in index if f.status == name]
    if name == "ui":
        return [f for f in index if f.ui < ui_threshold]
    if name.startswith("class:"):
        k = int(name.split(":", 1)[1])
        return [f for f in index if k in f.label_cls]
    raise ValueError(f"unknown queue {name!r}")
