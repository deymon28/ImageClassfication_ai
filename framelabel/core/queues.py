"""Per-frame summary (status, model proposals, labels) and the review queues built from it.

Queues are ordered lists of frames to walk through:
  all         every frame in time order
  triage      the model found nothing (shown as a grid: confirm empty frames a page at a time)
  edit        frames with proposals, or marked 'has objects' in the grid, or labeled/skipped before
  proposals   new frames with a proposal at the shown confidence
  uncertain   unreviewed frames with proposals in the uncertain confidence band, most uncertain first
  small       frames with small (far) objects in proposals or labels, smallest first
  disagree    reviewed frames where the model disagrees with the labels (possible missed or wrong boxes)
  review      decided frames that deserve a second look: missed / extra / smalllab (checked ones left out)
  missed      decided frames where a confident proposal has no label (a possibly missed object)
  extra       decided frames with a label that has no proposal (possibly wrong, or a hard object)
  smalllab    labeled frames with a small (far) object
  unreviewed / pos / neg / skip / todo   by review status
  class:<i>   frames whose labels contain class i
  ui          suspected interface screens
  zone        frames whose proposals were hidden by an ignore zone (they look empty otherwise)
  weak        unreviewed frames whose best proposal is below the shown confidence
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .boxes import iou_matrix, read_rows, in_zones
from .index_cache import read_rows_cached
from .session import Review, Session

QUEUES = ("edit", "triage", "proposals", "uncertain", "small", "disagree", "review", "missed", "extra", "smalllab", "all",
          "unreviewed", "todo", "pos", "neg", "skip", "ui", "zone", "weak")


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
    ignored_conf: float = 0.0                              # best confidence among proposals hidden by ignore zones
    ignored_xyxy: np.ndarray | None = None                 # normalized boxes of those hidden proposals

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def max_conf(self) -> float:
        return max(self.confs, default=0.0)


def _xyxy(rows) -> np.ndarray:
    a = np.array([(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for _, cx, cy, w, h, _ in rows], float)
    return a.reshape(-1, 4)


def frame_info_from_rows(path: Path, status: str | None, auto_rows, label_rows, ignore_zones=(),
                         ui: float = 1.0) -> FrameInfo:
    """FrameInfo from already parsed rows; None rows mean the file does not exist."""
    fi = FrameInfo(path, status, ui=ui)
    if auto_rows is not None:
        scored = [r for r in auto_rows if r[5] is not None]
        rows = [r for r in scored if not in_zones(r[1], r[2], ignore_zones)]
        fi.confs = [r[5] for r in rows]
        fi.prop_h = [r[4] for r in rows]
        fi.prop_xyxy = _xyxy(rows)
        hidden = [r for r in scored if in_zones(r[1], r[2], ignore_zones)]
        if hidden:
            fi.ignored_conf = max(r[5] for r in hidden)
            fi.ignored_xyxy = _xyxy(hidden)
    if status in ("pos", "neg") and label_rows is not None:
        fi.label_cls = [r[0] for r in label_rows]
        fi.label_h = [r[4] for r in label_rows]
        fi.label_xyxy = _xyxy(label_rows)
    return fi


def frame_info(session: Session, path: Path, review: Review, ignore_zones=(), ui: float = 1.0) -> FrameInfo:
    """One frame's summary read straight from its files (used after a save; the whole session uses build_index)."""
    auto, lab = session.auto_path(path.stem), session.label_path(path.stem)
    auto_rows = read_rows(auto) if auto.exists() else None
    status = review.get(path.stem)
    label_rows = read_rows(lab) if status in ("pos", "neg") and lab.exists() else None
    return frame_info_from_rows(path, status, auto_rows, label_rows, ignore_zones, ui)


def build_index(session: Session, review: Review, ignore_zones=(), ui: dict[str, float] | None = None,
                progress=None, cancelled=None) -> list[FrameInfo]:
    """Summaries of every frame; label files are read through a persistent cache and only when they changed.
    `progress(i, n, msg)` and `cancelled()` are optional; cancelling raises index_cache.Cancelled."""
    frames = session.frames()
    auto = read_rows_cached(session.auto_dir, None, progress, cancelled)
    wanted = {p.stem for p in frames if review.get(p.stem) in ("pos", "neg")}
    labels = read_rows_cached(session.labels_dir, wanted, progress, cancelled)
    ui = ui or {}
    return [frame_info_from_rows(p, review.get(p.stem), auto.get(p.stem), labels.get(p.stem), ignore_zones,
                                 ui.get(p.stem, 1.0)) for p in frames]


def uncertainty(confs: list[float], low: float, high: float) -> float:
    """0..1, higher = more uncertain: the proposal closest to the middle of the band decides."""
    band = [c for c in confs if low <= c <= high]
    if not band:
        return 0.0
    mid, half = (low + high) / 2, max((high - low) / 2, 1e-6)
    return 1.0 - min(abs(c - mid) for c in band) / half


def disagreement_parts(fi: FrameInfo, conf: float, iou: float = 0.4) -> tuple[int, int]:
    """(proposals without a label, labels without a proposal) of a decided frame: the first are objects the labels
    may have missed, the second boxes the model does not see (misplaced, or just hard objects)."""
    if fi.status not in ("pos", "neg"):
        return 0, 0
    props = fi.prop_xyxy[np.array(fi.confs) >= conf] if fi.prop_xyxy is not None and len(fi.confs) else np.zeros((0, 4))
    labels = fi.label_xyxy if fi.label_xyxy is not None else np.zeros((0, 4))
    if not len(props) and not len(labels):
        return 0, 0
    if not len(props) or not len(labels):
        return len(props), len(labels)
    m = iou_matrix(props, labels) >= iou
    return int((~m.any(1)).sum()), int((~m.any(0)).sum())


def disagreement(fi: FrameInfo, conf: float, iou: float = 0.4) -> float:
    """Unmatched confident proposals (likely missed objects) + unmatched labels (likely wrong or missed by the model,
    weighted lower)."""
    missed, extra = disagreement_parts(fi, conf, iou)
    return float(missed) + 0.5 * extra


def unmatched_proposals(fi: FrameInfo, conf: float, iou: float = 0.4) -> np.ndarray:
    """Normalized xyxy rows of the proposals at or above `conf` that no label of a decided frame covers (what the model
    sees and the labels do not); every proposal at or above `conf` of a frame without labels."""
    if fi.prop_xyxy is None or not len(fi.confs):
        return np.zeros((0, 4))
    strong = fi.prop_xyxy[np.array(fi.confs) >= conf]
    if fi.status not in ("pos", "neg") or not len(strong) or fi.label_xyxy is None or not len(fi.label_xyxy):
        return strong
    return strong[~(iou_matrix(strong, fi.label_xyxy) >= iou).any(1)]


def smallest_height(f: FrameInfo, band: tuple[float, float] = (0.15, 0.6)) -> float:
    """Height (share of the image) of the smallest object of a frame: of its labels once it is decided, else of its
    proposals from the band's lower edge up; 1.0 when it has none."""
    hs = f.label_h if f.status in ("pos", "neg") else [h for h, c in zip(f.prop_h, f.confs) if c >= band[0]]
    return min(hs, default=1.0)


SORTS = (("time", "Time order"), ("unsure", "Unsure first"), ("small", "Small objects first"))


def sort_frames(frames: list[FrameInfo], sort: str, band: tuple[float, float] = (0.15, 0.6)) -> list[FrameInfo]:
    """Another walking order for the same frames: 'unsure' puts the frames whose proposals are closest to the middle of
    the confidence band first, 'small' the ones with the smallest (farthest) objects; ties keep the time order."""
    if sort == "unsure":
        return sorted(frames, key=lambda f: -uncertainty(f.confs, *band))
    if sort == "small":
        return sorted(frames, key=lambda f: smallest_height(f, band))
    return list(frames)


def review_flags(index: list[FrameInfo], conf: float, checked=frozenset()) -> dict[str, tuple[int, int, float]]:
    """stem -> (missed, extra, smallest label height) of every decided frame that was not marked as checked."""
    thr = max(conf, 0.5)
    out = {}
    for f in index:
        if f.status in ("pos", "neg") and f.stem not in checked:
            missed, extra = disagreement_parts(f, thr)
            out[f.stem] = (missed, extra, min(f.label_h, default=1.0) if f.status == "pos" else 1.0)
    return out


GRID_PENDING = ("triage", "ui", "zone")        # grid views of frames still to decide: any status means it was decided
EDITOR_PENDING = ("edit", "small", "proposals")  # editor views: labeled / empty / skipped frames are done, 'todo' still waits
REVIEW_QUEUES = ("review", "missed", "extra", "smalllab")  # decided frames that deserve a second look


def hide_reviewed_frames(frames: list[FrameInfo], name: str) -> list[FrameInfo]:
    """Drop the frames that were already decided from the queues that are meant for work still to do (the views of
    a status, 'all' and the review queues - which are about decided frames - are left alone)."""
    if name in GRID_PENDING:
        return [f for f in frames if f.status is None]
    if name in EDITOR_PENDING:
        return [f for f in frames if f.status in (None, "todo")]
    return frames


SMALL_H = 0.03  # a box height as a share of the image height: 0.03 is 32 px at 1080p, a far object


def build_queue(index: list[FrameInfo], name: str, conf: float = 0.3, band: tuple[float, float] = (0.15, 0.6),
                small_h: float = SMALL_H, ui_threshold: float = 0.3, hide_reviewed: bool = False, checked=frozenset(),
                flags: dict | None = None) -> list[FrameInfo]:
    """Frames of a queue in walking order; with hide_reviewed the ones that already have a decision are left out of
    the queues for work still to do (see hide_reviewed_frames). `checked` (stems) are left out of the review queues;
    `flags` is review_flags() when the caller already has it."""
    frames = _queue(index, name, conf, band, small_h, ui_threshold, checked, flags)
    return hide_reviewed_frames(frames, name) if hide_reviewed else frames


def _queue(index: list[FrameInfo], name: str, conf: float, band: tuple[float, float], small_h: float,
           ui_threshold: float, checked=frozenset(), flags: dict | None = None) -> list[FrameInfo]:
    """`conf` is the proposal confidence shown as a box; `small_h` is a box height as a share of the image height
    (0.03 = 32 px at 1080p)."""
    if name == "all":
        return list(index)
    if name == "triage":
        return [f for f in index if f.max_conf < conf]
    if name == "edit":
        return [f for f in index if f.max_conf >= conf or f.status in ("todo", "pos", "skip")]
    if name == "proposals":
        return [f for f in index if f.status is None and f.max_conf >= conf]
    if name == "uncertain":
        scored = [(uncertainty(f.confs, *band), f) for f in index if f.status in (None, "todo")]
        return [f for s, f in sorted((x for x in scored if x[0] > 0), key=lambda x: -x[0])]
    if name == "small":
        scored = [(smallest_height(f, band), f) for f in index if f.status != "skip"]
        return [f for h, f in sorted((x for x in scored if x[0] < small_h), key=lambda x: x[0])]
    if name == "disagree":
        scored = [(disagreement(f, max(conf, 0.5)), f) for f in index]
        return [f for s, f in sorted((x for x in scored if x[0] > 0), key=lambda x: -x[0])]
    if name in REVIEW_QUEUES:
        flags = flags if flags is not None else review_flags(index, conf, checked)
        by = {f.stem: f for f in index}
        if name == "missed":
            hits = [(-m, -e, 0, s) for s, (m, e, _h) in flags.items() if m > 0]
        elif name == "extra":
            hits = [(-e, -m, 0, s) for s, (m, e, _h) in flags.items() if e > 0]
        elif name == "smalllab":
            hits = [(h, 0, 0, s) for s, (m, e, h) in flags.items() if h < small_h]
        else:  # review: disagreements first (the biggest first), then the frames that only have a small object
            hits = [(-(m + 0.5 * e), h, 0, s) for s, (m, e, h) in flags.items() if m or e or h < small_h]
        return [by[s] for *_k, s in sorted(hits)]
    if name == "checked":
        return [f for f in index if f.stem in checked]
    if name == "unreviewed":
        return [f for f in index if f.status is None]
    if name in ("todo", "pos", "neg", "skip"):
        return [f for f in index if f.status == name]
    if name == "ui":
        return [f for f in index if f.ui < ui_threshold]
    if name == "zone":
        return [f for f in index if f.ignored_conf >= conf]
    if name == "weak":
        return [f for f in index if f.status is None and 0 < f.max_conf < conf]
    if name.startswith("class:"):
        k = int(name.split(":", 1)[1])
        return [f for f in index if k in f.label_cls]
    raise ValueError(f"unknown queue {name!r}")
