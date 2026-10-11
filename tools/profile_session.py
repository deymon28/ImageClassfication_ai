"""Timing table for opening one session of a project (read-only: no interface-screen scores, caches go to a temporary
folder).

  python tools/profile_session.py projects/abi.yaml [--session NAME]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def t(label: str, fn, n: int = 1):
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
    print(f"{label:48s} {(time.perf_counter() - t0) / n * 1000:10.1f} ms", flush=True)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", type=Path)
    ap.add_argument("--session", help="session name (default: the one with the most frames)")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["FRAMELABEL_CACHE_DIR"] = tmp  # before the cache module reads it
        from framelabel.core.project import Project
        from framelabel.core.queues import QUEUES, build_index, build_queue

        proj = Project.load(a.project)
        sessions = t("Project.sessions()", proj.sessions)
        s = next((x for x in sessions if x.name == a.session), None) if a.session else None
        s = s or max(sessions, key=lambda x: len(x.frames()))
        frames = t(f"session.frames() [{s.name}]", s.frames)
        print(f"frames: {len(frames)}")
        review = t("Review load", lambda: s.review(proj.class_names))
        t("build_index, no cache", lambda: build_index(s, review, proj.ignore_zones))
        idx = t("build_index, cached", lambda: build_index(s, review, proj.ignore_zones))
        for q in QUEUES:
            t(f"build_queue {q}", lambda q=q: build_queue(idx, q, conf=0.2), 3)


if __name__ == "__main__":
    main()
