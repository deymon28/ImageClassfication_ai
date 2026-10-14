"""Export reviewed frames of one or more sessions as a YOLO dataset (images/{train,val}, labels/{train,val}, data.yaml).

Compatible with the dataset that screendet's tools/train.py fine-tunes on (datasets/own): file names are
'<session>_<stem>', all labeled positives plus a share of reviewed negatives (half of them the model's hardest
false positives), and a validation split made of whole time blocks of every session, balanced per class.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .boxes import parse_rows, read_rows
from .index_cache import Cancelled
from .split import balanced_block_split, choose_negatives, split_by_time
from .session import Session


def max_conf(path: Path, ignore_zones=()) -> float:
    """Highest proposal confidence in a labels_auto file (0 if none), ignoring proposals centred in ignore zones."""
    from .boxes import in_zones

    if not path.exists():
        return 0.0
    confs = [conf for _, cx, cy, _, _, conf in read_rows(path)
             if conf is not None and not in_zones(cx, cy, ignore_zones)]
    return max(confs, default=0.0)


@dataclass
class ExportResult:
    out: Path
    positives: int
    negatives: int
    negatives_available: int
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    boxes: dict[str, list[int]] = field(default_factory=dict)

    def summary(self, classes: list[str]) -> str:
        per = {split: ", ".join(f"{n} {c}" for n, c in zip(classes, counts)) for split, counts in self.boxes.items()}
        return (f"positives {self.positives}, negatives {self.negatives} of {self.negatives_available}; "
                f"train {len(self.train)} / val {len(self.val)} images; boxes train: {per.get('train', '')}; "
                f"val: {per.get('val', '')}  ->  {self.out}")


def export_dataset(sessions: list[Session], classes: list[str], out: Path, neg_ratio: float = 0.15,
                   val_fraction: float = 0.15, seed: int = 0, split: str = "blocks", block_s: float = 120.0,
                   ignore_zones=(), progress=None, cancelled=None) -> ExportResult:
    pos, negs, src = [], [], {}
    for s in sessions:
        review = s.review()
        for p in s.frames():
            key = f"{s.name.replace('/', '_')}_{p.stem}"  # session and frame names are timestamps: keys sort by time
            st = review.get(p.stem)
            if st == "pos":
                pos.append(key)
            elif st == "neg":
                negs.append((key, max_conf(s.auto_path(p.stem), ignore_zones)))
            else:
                continue
            src[key] = (p, s.label_path(p.stem))
    if not pos:
        raise ValueError("no labeled positives yet")
    chosen = choose_negatives(negs, round(len(pos) * neg_ratio), seed)
    if split == "blocks":
        box_classes = {k: [r[0] for r in read_rows(src[k][1])]
                       for k in pos if src[k][1].exists()}
        train, val = balanced_block_split(pos + chosen, box_classes, val_fraction, seed, block_s, n_classes=len(classes))
    else:
        train, val = split_by_time(pos + chosen, val_fraction)
    out = Path(out)
    if out.exists() and not (out / "data.yaml").exists():
        raise FileExistsError(f"{out} exists and is not an exported dataset; refusing to overwrite it")
    # everything is built next to the target and swapped in at the end: a cancelled or failed export keeps the old one
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        if any(tmp.iterdir()) and not (tmp / "data.yaml").exists() and not (tmp / "images").exists():
            raise FileExistsError(f"{tmp} exists and is not a partial export; remove it first")
        shutil.rmtree(tmp)
    counts = {"train": [0] * len(classes), "val": [0] * len(classes)}
    total, done = len(train) + len(val), 0
    try:
        for name, keys in (("train", train), ("val", val)):
            (tmp / "images" / name).mkdir(parents=True)
            (tmp / "labels" / name).mkdir(parents=True)
            for key in keys:
                img, lab = src[key]
                shutil.copy2(img, tmp / "images" / name / f"{key}{img.suffix}")
                rows = lab.read_text(encoding="utf-8", errors="replace") if lab.exists() else ""
                (tmp / "labels" / name / f"{key}.txt").write_text(rows, encoding="utf-8")
                for cls, *_ in parse_rows(rows):
                    if 0 <= cls < len(classes):
                        counts[name][cls] += 1
                done += 1
                if done % 50 == 0 or done == total:
                    if cancelled and cancelled():
                        raise Cancelled()
                    if progress:
                        progress(done, total, "exporting images")
        # plain names come out as '  0: enemy'; a name YAML would misread ('no: helmet', '#2') is quoted
        names = yaml.safe_dump(dict(enumerate(classes)), allow_unicode=True, default_flow_style=False, sort_keys=False)
        names = "".join(f"  {line}" for line in names.splitlines(True))
        (tmp / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}",
                                       encoding="utf-8")
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return ExportResult(out, len(pos), len(chosen), len(negs), train, val, counts)
