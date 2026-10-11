"""A negatives-only test set: frames the user reviewed as empty that are not in the exported dataset and not within
`--gap-s` seconds of any exported frame (neighbouring frames are near-duplicates). Most frames met while labeling are
empty, so false positives per frame here are the number of boxes the labeler makes the user delete.

  python tools/build_neg_eval.py --project projects/abi.yaml --own ../fin_snip_helper/datasets/own --out datasets/own_negs
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from framelabel.core.project import Project  # noqa: E402
from framelabel.core.split import frame_time  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--own", type=Path, required=True, help="the exported dataset whose frames are excluded")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets/own_negs")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--gap-s", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    proj = Project.load(args.project)
    used: dict[str, list[float]] = defaultdict(list)
    for split in ("train", "val"):
        for p in (args.own / "images" / split).iterdir():
            sess, t = frame_time(p.stem)
            used[sess].append(t)
    for v in used.values():
        v.sort()
    cands = []
    for s in proj.sessions():
        review = s.review()
        for p in s.frames():
            if review.get(p.stem) != "neg":
                continue
            key = f"{s.name}_{p.stem}"
            sess, t = frame_time(key)
            ts = used.get(sess, [])
            i = bisect_left(ts, t)
            near = any(abs(ts[j] - t) <= args.gap_s for j in (i - 1, i) if 0 <= j < len(ts))
            if not near:
                cands.append((key, p))
    random.Random(args.seed).shuffle(cands)
    chosen = sorted(cands[: args.n])
    if args.out.exists():
        if not (args.out / "data.yaml").exists():
            sys.exit(f"{args.out} exists and is not a built dataset")
        shutil.rmtree(args.out)
    (args.out / "images/val").mkdir(parents=True)
    (args.out / "labels/val").mkdir(parents=True)
    for key, p in chosen:
        try:
            os.link(p, args.out / "images/val" / f"{key}{p.suffix}")
        except OSError:
            shutil.copy2(p, args.out / "images/val" / f"{key}{p.suffix}")
        (args.out / "labels/val" / f"{key}.txt").write_text("", encoding="utf-8")
    names = {i: n for i, n in enumerate(proj.class_names)}
    (args.out / "data.yaml").write_text(yaml.safe_dump({"path": args.out.resolve().as_posix(), "train": "images/val",
                                                        "val": "images/val", "names": names}), encoding="utf-8")
    print(f"{len(cands)} eligible empty frames, {len(chosen)} chosen -> {args.out}")


if __name__ == "__main__":
    main()
