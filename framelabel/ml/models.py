"""Model files offered for auto-labeling: *.pt files in the project's model folders, including ultralytics training
runs (<run>/weights/best.pt), with their class names read on demand."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    path: Path
    label: str          # short display name
    info: str = ""      # e.g. the run's best mAP50 from results.csv


def _run_info(run_dir: Path) -> str:
    res = run_dir / "results.csv"
    if not res.exists():
        return ""
    try:
        with res.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        key = next((k for k in rows[0] if k.strip() == "metrics/mAP50(B)"), None)
        if not rows or key is None:
            return ""
        best = max(float(r[key]) for r in rows)
        return f"{len(rows)} epochs, best mAP50 {best:.3f}"
    except (OSError, ValueError, KeyError, IndexError, StopIteration):
        return ""


def scan_models(dirs: list[Path], extra: list[Path] | None = None) -> list[ModelFile]:
    out, seen = [], set()

    def add(p: Path, label: str, info: str = "") -> None:
        key = p.resolve()
        if key not in seen and p.is_file():
            seen.add(key)
            out.append(ModelFile(p, label, info))

    for p in extra or []:
        add(Path(p), Path(p).stem)
    for d in dirs:
        d = Path(d)
        if d.is_file() and d.suffix == ".pt":
            add(d, d.stem)
            continue
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.pt")):
            if not p.name.startswith(("sam", "mobile_sam")):
                add(p, p.stem)
        for p in sorted(d.glob("*/weights/best.pt")):  # training runs
            add(p, f"{p.parent.parent.name} (best)", _run_info(p.parent.parent))
    return out


def find_sam(dirs: list[Path]) -> Path | None:
    for d in dirs:
        d = Path(d)
        for pat in ("sam2.1_b.pt", "sam2.1_*.pt", "sam2_*.pt", "mobile_sam.pt", "sam_*.pt"):
            hits = sorted(d.glob(pat)) if d.is_dir() else []
            if hits:
                return hits[0]
    return None


@lru_cache(maxsize=64)
def model_class_names(path: str) -> list[str]:
    """Class names stored in a checkpoint (loads it on the CPU; cached). A sidecar <name>.classes.json, if present,
    avoids loading."""
    side = Path(path).with_suffix(".classes.json")
    if side.exists():
        return list(json.loads(side.read_text(encoding="utf-8")))
    from ultralytics import YOLO

    names = YOLO(path).names
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
