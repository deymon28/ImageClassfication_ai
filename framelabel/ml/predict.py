"""Model proposals for frames: one or several YOLO models, optional test-time augmentation (horizontal flip, a larger
second scale), merged with weighted boxes fusion, mapped to the project's classes, written to labels_auto/.

GPU work: callers check the game is closed (guard.check_gpu_allowed) first; the VRAM cap is applied here.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from framelabel.core.boxes import in_zones
from framelabel.core.session import Session
from .fusion import weighted_boxes_fusion
from .guard import cap_vram


@dataclass
class PredictConfig:
    models: list[Path]
    imgsz: int = 1280
    conf: float = 0.05          # lowest confidence stored
    tta: bool = False           # + horizontal flip and a 1.3x larger scale
    fuse_iou: float = 0.55
    max_vram_gb: float = 8.0
    half: bool = True
    batch: int = 8
    device: int | str = 0
    max_det: int = 100
    class_maps: list[dict[int, int | None]] = field(default_factory=list)  # per model: model class -> project class

    def meta(self) -> dict:
        return {"weights": [str(m) for m in self.models], "imgsz": self.imgsz, "conf": self.conf, "tta": self.tta,
                "fuse_iou": self.fuse_iou, "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def tta_sizes(imgsz: int, tta: bool) -> list[int]:
    return [imgsz, int(round(imgsz * 1.3 / 32) * 32)] if tta else [imgsz]


class Predictor:
    """Loads the models once; `predict(images)` returns per image (xyxy pixels, conf, project class)."""

    def __init__(self, cfg: PredictConfig):
        import torch
        from ultralytics import YOLO

        self.cfg = cfg
        if torch.cuda.is_available() and cfg.device != "cpu":
            cap_vram(cfg.max_vram_gb, int(cfg.device) if str(cfg.device).isdigit() else 0)
        else:
            self.cfg.device, self.cfg.half = "cpu", False
        self.models = [YOLO(str(m)) for m in cfg.models]
        if not cfg.class_maps:
            cfg.class_maps = [{i: i for i in m.names} for m in self.models]

    def _passes(self, model, images: list[np.ndarray], imgsz: int, flip: bool):
        src = [cv2.flip(im, 1) for im in images] if flip else images
        extra = {"quantize": 16} if self.cfg.half else {}
        res = model.predict(source=src, imgsz=imgsz, conf=self.cfg.conf, device=self.cfg.device,
                            max_det=self.cfg.max_det, verbose=False, **extra)
        out = []
        for im, r in zip(images, res):
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy().astype(float)
            if flip:
                w = im.shape[1]
                xyxy[:, [0, 2]] = w - xyxy[:, [2, 0]]
            out.append((xyxy, b.conf.cpu().numpy().astype(float), b.cls.cpu().numpy().astype(int)))
        return out

    def predict(self, images: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        per_image: list[list] = [[] for _ in images]
        for model, cmap in zip(self.models, self.cfg.class_maps):
            for imgsz in tta_sizes(self.cfg.imgsz, self.cfg.tta):
                for flip in ((False, True) if self.cfg.tta else (False,)):
                    for i, (xyxy, conf, cls) in enumerate(self._passes(model, images, imgsz, flip)):
                        mapped = np.array([cmap.get(int(c)) if cmap.get(int(c)) is not None else -1 for c in cls], int)
                        keep = mapped >= 0
                        per_image[i].append((xyxy[keep], conf[keep], mapped[keep]))
        # a single pass is returned as is (the model's own NMS output); several passes are fused
        return [sets[0] if len(sets) == 1 else weighted_boxes_fusion(sets, self.cfg.fuse_iou) for sets in per_image]


def rows_for(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, w: int, h: int, min_conf: float) -> list[str]:
    rows = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        if c < min_conf:
            continue
        x1, x2 = max(0.0, x1), min(float(w), x2)
        y1, y2 = max(0.0, y1), min(float(h), y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        rows.append(f"{int(k)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f} {c:.3f}")
    return rows


def prelabel_sessions(sessions: list[Session], cfg: PredictConfig, redo: bool = False,
                      progress: Callable[[int, int, str], None] | None = None,
                      cancelled: Callable[[], bool] | None = None, ignore_zones=()) -> dict:
    """Write labels_auto/<stem>.txt for every frame without proposals (all frames with redo). Returns counts."""
    todo: list[tuple[Session, Path]] = []
    for s in sessions:
        s.auto_dir.mkdir(parents=True, exist_ok=True)
        todo += [(s, p) for p in s.frames() if redo or not s.auto_path(p.stem).exists()]
    stats = {"frames": len(todo), "done": 0, "with_proposals": 0, "seconds": 0.0}
    if not todo:
        return stats
    t0 = time.perf_counter()
    pred = Predictor(cfg)
    import torch

    for start in range(0, len(todo), cfg.batch):
        if cancelled and cancelled():
            break
        chunk = todo[start:start + cfg.batch]
        images = [cv2.imread(str(p)) for _, p in chunk]
        ok = [(sp, im) for sp, im in zip(chunk, images) if im is not None]
        results = pred.predict([im for _, im in ok]) if ok else []
        for ((s, p), im), (xyxy, conf, cls) in zip(ok, results):
            h, w = im.shape[:2]
            rows = rows_for(xyxy, conf, cls, w, h, cfg.conf)
            tmp = s.auto_path(p.stem).with_suffix(".tmp")
            tmp.write_text("\n".join(rows), encoding="utf-8")
            tmp.replace(s.auto_path(p.stem))
            stats["with_proposals"] += any(not in_zones(*map(float, r.split()[1:3]), ignore_zones) for r in rows)
        stats["done"] += len(chunk)
        if progress:
            vram = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            progress(stats["done"], len(todo), f"{stats['with_proposals']} with proposals, VRAM peak {vram:.2f} GB")
    stats["seconds"] = time.perf_counter() - t0
    for s in sessions:
        (s.auto_dir / "_meta.json").write_text(json.dumps(cfg.meta()), encoding="utf-8")
    return stats
