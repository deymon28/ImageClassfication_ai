"""Stage-2 data for the labeler model: the own exported set's train split, cut into train + dev by whole time blocks.

  python tools/split_own_dev.py --own ../fin_snip_helper/datasets/own --out datasets/abi_own

Training picks its best epoch on dev, so the own set's val split is never seen before the final comparison. Images
are hard-linked (copied on another drive) into <out>/images/{train,val}, so ultralytics' label caches are written
here and not into the source dataset.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from framelabel.core.boxes import parse_rows  # noqa: E402
from framelabel.core.session import IMG_EXT  # noqa: E402
from framelabel.core.split import balanced_block_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--own", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "datasets/abi_own")
    ap.add_argument("--dev-fraction", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    cfg = yaml.safe_load((args.own / "data.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]
    images = {p.stem: p.resolve() for p in (args.own / "images/train").iterdir() if p.suffix.lower() in IMG_EXT}
    classes = {k: [r[0] for r in parse_rows((args.own / "labels/train" / f"{k}.txt").read_text(encoding="utf-8"))]
               for k in images if (args.own / "labels/train" / f"{k}.txt").exists()}
    train, dev = balanced_block_split(sorted(images), classes, args.dev_fraction, args.seed, n_classes=len(names))
    if args.out.exists():
        if not (args.out / "data.yaml").exists():
            sys.exit(f"{args.out} exists and is not a built dataset; refusing to overwrite it.")
        shutil.rmtree(args.out)
    for split, keys in (("train", train), ("val", dev)):
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)
        for k in keys:
            dst = args.out / "images" / split / images[k].name
            try:
                os.link(images[k], dst)
            except OSError:
                shutil.copy2(images[k], dst)
            lab = args.own / "labels/train" / f"{k}.txt"
            (args.out / "labels" / split / f"{k}.txt").write_text(lab.read_text(encoding="utf-8") if lab.exists() else "",
                                                                 encoding="utf-8")
    (args.out / "data.yaml").write_text(yaml.safe_dump({"path": args.out.resolve().as_posix(), "train": "images/train",
                                                        "val": "images/val", "names": names}), encoding="utf-8")
    count = lambda keys: [sum(c == i for k in keys for c in classes.get(k, ())) for i in range(len(names))]
    print(f"train {len(train)} images {count(train)} boxes per class, dev {len(dev)} images {count(dev)}  ->  {args.out}")


if __name__ == "__main__":
    main()
