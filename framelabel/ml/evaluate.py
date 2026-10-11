"""Measure labeler models (or ensembles / TTA setups) on a labeled YOLO split, with the numbers that matter for
auto-labeling: false positives per frame (each one is a box to delete) and recall on small/far objects (each miss is a
box to draw), at several confidence thresholds, plus AP50.

  python -m framelabel eval --data ../fin_snip_helper/datasets/own/data.yaml \
      --setup old=../fin_snip_helper/train_runs/enemy_y26s/weights/best.pt --setup new=train_runs/x/weights/best.pt

Model classes are matched to the dataset's classes by name (a one-class model maps to class 0). Categories:
'person' = every class together (class-agnostic), then each class of the dataset.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from framelabel.core.boxes import in_zones, iou_matrix, parse_rows
from framelabel.core.session import IMG_EXT

SIZE_BUCKETS = ((0, 20), (20, 50), (50, 10**9))  # box height in pixels at 1080p
CONFS = (0.2, 0.3, 0.4, 0.5)


def match(pred: np.ndarray, conf: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Greedy matching in descending confidence -> (true-positive flag per prediction, matched flag per label)."""
    tp, hit = np.zeros(len(pred), bool), np.zeros(len(gt), bool)
    if len(pred) and len(gt):
        ious = iou_matrix(pred, gt)
        for i in np.argsort(-conf):
            cand = np.where(~hit & (ious[i] >= thr))[0]
            if len(cand):
                j = cand[np.argmax(ious[i, cand])]
                tp[i] = hit[j] = True
    return tp, hit


def average_precision(conf: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """COCO-style 101-point interpolated AP from all predictions of a category."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-conf)
    ctp = np.cumsum(tp[order])
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(ctp) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    pts = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, pts, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


@dataclass
class Split:
    images: list[Path]
    labels_dir: Path
    names: list[str]


def load_split(data_yaml: Path, split: str = "val") -> Split:
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    base = Path(cfg.get("path") or Path(data_yaml).parent)
    if not base.is_absolute():
        base = (Path(data_yaml).parent / base).resolve()
    img_dir = base / cfg[split]
    names = cfg["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    return Split(images, Path(str(img_dir).replace("images", "labels")), names)


def gt_boxes(lab: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    rows = parse_rows(lab.read_text(encoding="utf-8")) if lab.exists() else []
    if not rows:
        return np.zeros((0, 4)), np.zeros(0, int)
    a = np.array([(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2) for _, cx, cy, bw, bh, _ in rows]) * [w, h, w, h]
    return a, np.array([r[0] for r in rows], int)


@dataclass
class Acc:
    conf: list = field(default_factory=list)
    tp: list = field(default_factory=list)
    n_gt: int = 0
    # per operating confidence: tp, fp, hits by size, labels by size
    op: dict = field(default_factory=lambda: {c: {"tp": 0, "fp": 0, "hit": np.zeros(3), "n": np.zeros(3)} for c in CONFS})


def evaluate_predictions(split: Split, preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                         ignore_zones=()) -> dict:
    """preds: image stem -> (xyxy pixels, conf, dataset class). Predictions centred in ignore zones are dropped."""
    cats = {"person": None, **{n: i for i, n in enumerate(split.names)}}
    acc = {c: Acc() for c in cats}
    n_img = 0
    for img in split.images:
        with Image.open(img) as im:  # header only
            w, h = im.size
        gt, gcls = gt_boxes(split.labels_dir / f"{img.stem}.txt", w, h)
        pxy, pconf, pcls = preds.get(img.stem, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
        if ignore_zones and len(pxy):
            keep = np.array([not in_zones((b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h, ignore_zones) for b in pxy])
            pxy, pconf, pcls = pxy[keep], pconf[keep], pcls[keep]
        n_img += 1
        for c, k in cats.items():
            g = gt if k is None else gt[gcls == k]
            sel = np.ones(len(pxy), bool) if k is None else pcls == k
            p, pc = pxy[sel], pconf[sel]
            a = acc[c]
            tp, _ = match(p, pc, g)
            a.conf.append(pc)
            a.tp.append(tp)
            a.n_gt += len(g)
            hpx = (g[:, 3] - g[:, 1]) * 1080 / h
            for t in CONFS:
                m = pc >= t
                op_tp, hit = match(p[m], pc[m], g)
                o = a.op[t]
                o["tp"] += int(op_tp.sum())
                o["fp"] += int((~op_tp).sum())
                for i, (lo, hi) in enumerate(SIZE_BUCKETS):
                    s = (hpx >= lo) & (hpx < hi)
                    o["n"][i] += s.sum()
                    o["hit"][i] += hit[s].sum()
    out = {"images": n_img, "categories": {}}
    for c, a in acc.items():
        conf, tp = (np.concatenate(a.conf), np.concatenate(a.tp)) if a.conf else (np.zeros(0), np.zeros(0, bool))
        ops = {}
        for t, o in a.op.items():
            p = o["tp"] / max(o["tp"] + o["fp"], 1)
            r = o["tp"] / max(a.n_gt, 1)
            ops[str(t)] = {"precision": p, "recall": r, "fp": o["fp"], "fp_per_frame": o["fp"] / max(n_img, 1),
                           "recall_by_height": {f"{lo}-{hi if hi < 10**9 else ''}px": (float(hh / n) if n else None, int(n))
                                                for (lo, hi), hh, n in zip(SIZE_BUCKETS, o["hit"], o["n"])}}
        out["categories"][c] = {"labels": a.n_gt, "ap50": average_precision(conf, tp, a.n_gt), "at_conf": ops}
    return out


def class_map_by_name(model_names: list[str], data_names: list[str]) -> dict[int, int | None]:
    if len(model_names) == 1 and model_names[0] not in data_names:
        return {0: 0}
    return {i: (data_names.index(n) if n in data_names else None) for i, n in enumerate(model_names)}


def predict_split(split: Split, models: list[Path], imgsz: int, tta: bool, batch: int = 4, max_vram_gb: float = 6.0,
                  progress=None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from .models import model_class_names
    from .predict import PredictConfig, Predictor

    maps = [class_map_by_name(model_class_names(str(m)), split.names) for m in models]
    cfg = PredictConfig(models=list(models), imgsz=imgsz, conf=0.01, tta=tta, max_vram_gb=max_vram_gb, batch=batch,
                        max_det=300, class_maps=maps)
    pred = Predictor(cfg)
    out = {}
    for start in range(0, len(split.images), batch):
        chunk = split.images[start:start + batch]
        ims = [cv2.imread(str(p)) for p in chunk]
        for p, r in zip(chunk, pred.predict(ims)):
            out[p.stem] = r
        if progress:
            progress(min(start + batch, len(split.images)), len(split.images))
    return out


def format_report(name: str, res: dict, confs=CONFS) -> str:
    lines = [f"{name}  ({res['images']} images)"]
    lines.append("  category  labels   AP50 | per confidence: precision, recall, false positives per frame, "
                 "[recall for boxes <20 px, 20-50 px, 50+ px tall at 1080p]")
    for c, r in res["categories"].items():
        parts = []
        for t in confs:
            o = r["at_conf"][str(t)]
            rb = [v[0] for v in o["recall_by_height"].values()]
            rbs = " ".join("  - " if v is None else f"{v:.2f}" for v in rb)
            parts.append(f"@{t}: P {o['precision']:.3f} R {o['recall']:.3f} FP/f {o['fp_per_frame']:.3f} [{rbs}]")
        lines.append(f"  {c:8} {r['labels']:6d} {r['ap50']:6.3f} | " + " | ".join(parts))
    return "\n".join(lines)


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
