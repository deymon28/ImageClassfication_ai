"""Weighted boxes fusion (Solovyev et al.): merge predictions of several models / test-time augmentations of one
image. Clusters of same-class boxes overlapping above `iou_thr` become one box with confidence-weighted coordinates;
the fused confidence is the mean confidence scaled by the share of predictions sets that found the object, so an
object seen by only one of four passes is down-weighted."""
from __future__ import annotations

import numpy as np

from framelabel.core.boxes import iou_matrix


def weighted_boxes_fusion(sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]], iou_thr: float = 0.55,
                          weights: list[float] | None = None, skip_conf: float = 0.0,
                          conf_type: str = "avg") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sets: per model/pass (xyxy (N,4), conf (N,), cls (N,)) -> fused (xyxy, conf, cls), sorted by confidence."""
    n_sets = len(sets)
    if n_sets == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    weights = list(weights) if weights is not None else [1.0] * n_sets
    rows = []  # (cls, conf * weight, weight, set index, x1, y1, x2, y2)
    for k, (xyxy, conf, cls) in enumerate(sets):
        xyxy, conf, cls = np.asarray(xyxy, float).reshape(-1, 4), np.asarray(conf, float), np.asarray(cls, int)
        for b, c, k_cls in zip(xyxy, conf, cls):
            if c >= skip_conf:
                rows.append((int(k_cls), float(c) * weights[k], weights[k], k, *b))
    if not rows:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    a = np.array(rows, float)
    out_b, out_c, out_k = [], [], []
    total_w = sum(weights)
    for k_cls in np.unique(a[:, 0]).astype(int):
        r = a[a[:, 0] == k_cls]
        r = r[np.argsort(-r[:, 1])]
        clusters: list[list[np.ndarray]] = []
        fused: list[np.ndarray] = []
        for row in r:
            box = row[4:8]
            j = -1
            if fused:
                ious = iou_matrix(box[None], np.array(fused))[0]
                best = int(np.argmax(ious))
                if ious[best] > iou_thr:
                    j = best
            if j < 0:
                clusters.append([row])
                fused.append(box.copy())
            else:
                clusters[j].append(row)
                c = np.array(clusters[j])
                fused[j] = (c[:, 4:8] * c[:, 1:2]).sum(0) / c[:, 1].sum()
        for c, box in zip(clusters, fused):
            c = np.array(c)
            score = c[:, 1].max() if conf_type == "max" else c[:, 1].sum() / len(c)
            present_w = sum(weights[int(s)] for s in np.unique(c[:, 3]))  # each set counted once
            score = score * min(present_w, total_w) / total_w
            out_b.append(box)
            out_c.append(score)
            out_k.append(k_cls)
    order = np.argsort(-np.array(out_c))
    return np.array(out_b)[order], np.array(out_c)[order], np.array(out_k, int)[order]
