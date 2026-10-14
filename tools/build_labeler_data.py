"""Build the one-class 'person' training set for the labeler model from public Arena Breakout Infinite sets plus the
train split of the user's own labeled frames.

  python tools/build_labeler_data.py --raw ../fin_snip_helper/datasets/raw ../fin_snip_helper/datasets/eval_raw \
      --own ../fin_snip_helper/datasets/own --out datasets/abi_person

Sources and the reasons for excluding the others are in tools/abi_sources.yaml. Polygon labels become their bounding
box, every kept class becomes class 0, exact/near duplicates (difference hash) are dropped and Roboflow augmented
copies of one original are capped. Only the own set's *train* split is used, so its val split stays an honest test.
A small random share of the public images forms this set's val split, for training curves only.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from framelabel.core.boxes import parse_rows  # noqa: E402
from framelabel.core.images import imread  # noqa: E402
from framelabel.core.session import IMG_EXT  # noqa: E402

RF_RE = re.compile(r"^(.*?)_(?:jpg|jpeg|png|JPG|PNG)\.rf\.[0-9a-f]+$")  # Roboflow '<orig>_jpg.rf.<hash>'


def dhash(img) -> str:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    return hashlib.md5((g[:, 1:] > g[:, :-1]).tobytes()).hexdigest()


def orig_key(stem: str) -> str:
    m = RF_RE.match(stem)
    return m[1] if m else stem


def class_ids(names: list[str], keep) -> set[int]:
    return set(range(len(names))) if keep == "*" else {i for i, n in enumerate(names) if str(n) in set(map(str, keep))}


def person_rows(label: Path, ok: set[int], min_px: float, w: int, h: int) -> tuple[list[str], int]:
    """Kept rows as '0 cx cy w h' and the number of dropped rows."""
    rows = parse_rows(label.read_text(encoding="utf-8")) if label.exists() else []
    kept, dropped = [], 0
    for cls, cx, cy, bw, bh, _ in rows:
        if cls in ok and bw * w >= min_px and bh * h >= min_px:
            kept.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        else:
            dropped += 1
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, nargs="+", required=True, help="folders that contain the downloaded sets")
    ap.add_argument("--own", type=Path, help="own exported dataset (its images/train and labels/train are used)")
    ap.add_argument("--sources", type=Path, default=ROOT / "tools/abi_sources.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets/abi_person")
    ap.add_argument("--neg-ratio", type=float, default=0.10, help="public empty images per positive (default 0.10)")
    ap.add_argument("--max-copies", type=int, default=2, help="augmented copies kept per original image (default 2)")
    ap.add_argument("--val-fraction", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.sources.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    seen: set[str] = set()
    copies: Counter = Counter()
    pos, neg = [], []
    stats: Counter = Counter()
    for name, keep in cfg["keep"].items():
        d = next((r / name for r in args.raw if (r / name).is_dir()), None)
        if d is None:
            print(f"  missing: {name}")
            continue
        meta = yaml.safe_load((d / "data.yaml").read_text(encoding="utf-8"))
        names = meta["names"]
        names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        ok = class_ids(names, keep)
        drop_prefixes = tuple(cfg.get("drop_prefixes", {}).get(name, ()))
        n_before = len(pos) + len(neg)
        for split in ("train", "valid", "test"):
            for img_path in sorted((d / split / "images").glob("*")):
                if img_path.suffix.lower() not in IMG_EXT:
                    continue
                if drop_prefixes and img_path.name.startswith(drop_prefixes):
                    stats["other_game"] += 1
                    continue
                img = imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                kept, dropped = person_rows(d / split / "labels" / f"{img_path.stem}.txt", ok, 2.0, w, h)
                if not kept and dropped:
                    stats["only_other_classes"] += 1  # e.g. head-only images: not a clean negative
                    continue
                hsh = dhash(img)
                if hsh in seen:
                    stats["duplicate"] += 1
                    continue
                okey = orig_key(img_path.stem)
                if len(okey) >= 16 and copies[okey] >= args.max_copies:  # uuid-like originals shared across sets
                    stats["extra_copy"] += 1
                    continue
                seen.add(hsh)
                copies[okey] += 1
                uid = f"{name.split('__')[1][:20]}_{img_path.stem[:48]}"
                (pos if kept else neg).append((img_path, kept, uid))
        print(f"  {name}: {len(pos) + len(neg) - n_before} images")
    rng.shuffle(neg)
    neg = neg[: int(len(pos) * args.neg_ratio)]
    items = pos + neg
    rng.shuffle(items)
    n_val = int(len(items) * args.val_fraction)
    splits = [("val" if i < n_val else "train", it) for i, it in enumerate(items)]
    own = 0
    if args.own:
        for img_path in sorted((args.own / "images/train").iterdir()):
            lab = args.own / "labels/train" / f"{img_path.stem}.txt"
            rows = parse_rows(lab.read_text(encoding="utf-8")) if lab.exists() else []
            kept = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for _, cx, cy, bw, bh, _ in rows]
            splits.append(("train", (img_path, kept, f"own_{img_path.stem}")))
            own += 1
    out = args.out
    if out.exists():
        if not (out / "data.yaml").exists():
            sys.exit(f"{out} exists and is not a built dataset; refusing to overwrite it.")
        shutil.rmtree(out)
    boxes: Counter = Counter()
    for split, (src, kept, uid) in splits:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / "images" / split / f"{uid}{src.suffix.lower()}")
        (out / "labels" / split / f"{uid}.txt").write_text("\n".join(kept), encoding="utf-8")
        boxes[split] += len(kept)
    (out / "data.yaml").write_text(yaml.safe_dump({"path": out.resolve().as_posix(), "train": "images/train",
                                                   "val": "images/val", "names": {0: "person"}}), encoding="utf-8")
    print(f"public positives {len(pos)}, negatives {len(neg)}, own train frames {own}; val {n_val}")
    print(f"boxes {dict(boxes)}, skipped {dict(stats)}  ->  {out}")


if __name__ == "__main__":
    main()
