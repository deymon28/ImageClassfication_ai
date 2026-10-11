"""Labeling state of one session, independent of the UI: frame index, review statuses, loading a frame's boxes
(reviewed labels, else model proposals), saving a decision, committing a grid page."""
from __future__ import annotations

from pathlib import Path

from .boxes import Box, read_yolo, write_yolo, yolo_lines
from .project import Project
from .queues import FrameInfo, build_index, build_queue, frame_info
from .session import Session
from .uiscreen import ui_scores


class Workspace:
    def __init__(self, project: Project, session: Session, conf: float = 0.3, progress=None):
        self.project, self.session, self.conf = project, session, conf
        self.review = session.review(project.class_names)
        frames = session.frames()
        self.ui = ui_scores(session, frames, project.ui_zone, progress) if project.ui_zone else {}
        self.index: list[FrameInfo] = build_index(session, self.review, project.ignore_zones, self.ui)
        self.by_stem = {fi.stem: i for i, fi in enumerate(self.index)}

    # ---------------- index
    def refresh(self, stem: str) -> FrameInfo:
        i = self.by_stem[stem]
        self.index[i] = frame_info(self.session, self.index[i].path, self.review, self.project.ignore_zones,
                                   self.ui.get(stem, 1.0))
        return self.index[i]

    def reload(self) -> None:
        self.review = self.session.review(self.project.class_names)
        self.index = build_index(self.session, self.review, self.project.ignore_zones, self.ui)
        self.by_stem = {fi.stem: i for i, fi in enumerate(self.index)}

    def queue(self, name: str, **kw) -> list[FrameInfo]:
        return build_queue(self.index, name, conf=self.conf, ui_threshold=self.project.ui_threshold, **kw)

    def counts(self) -> dict[str, int]:
        return self.review.counts()

    def stats_line(self) -> str:
        c = self.counts()
        done = c["pos"] + c["neg"] + c["skip"]
        return (f"reviewed {done}/{len(self.index)}  (objects {c['pos']}, empty {c['neg']}, skip {c['skip']}, "
                f"queued {c['todo']})")

    # ---------------- frames
    def proposals(self, path: Path, w: int, h: int, min_conf: float | None = None) -> list[Box]:
        return read_yolo(self.session.auto_path(path.stem), w, h, self.conf if min_conf is None else min_conf,
                         self.project.ignore_zones)

    def load_boxes(self, path: Path, w: int, h: int) -> tuple[list[Box], list[Box]]:
        """(boxes to edit, weak proposals). Reviewed frames show their labels, others the model's proposals."""
        st = self.review.get(path.stem)
        lab = self.session.label_path(path.stem)
        weak_all = read_yolo(self.session.auto_path(path.stem), w, h, 0.0, self.project.ignore_zones)
        weak = [b for b in weak_all if b.conf is not None and b.conf < self.conf]
        if st in ("pos", "neg") and lab.exists():
            return read_yolo(lab, w, h), weak
        return [b for b in weak_all if b.conf is not None and b.conf >= self.conf], weak

    def save(self, path: Path, boxes: list[Box], w: int, h: int, status: str | None = None) -> str:
        """Save a decision: 'pos'/'neg' write the label file ('pos' if any box survives), 'skip' removes it."""
        lab = self.session.label_path(path.stem)
        if status is None:
            status = "pos" if yolo_lines(boxes, w, h) else "neg"
        if status in ("pos", "neg"):
            write_yolo(lab, boxes if status == "pos" else [], w, h)
        else:
            lab.unlink(missing_ok=True)
        self.review.set(path.stem, status)
        self.refresh(path.stem)
        return status

    # ---------------- grid
    def initial_mark(self, fi: FrameInfo, view: str = "triage") -> str | None:
        st = self.review.get(fi.stem)
        if st in ("todo", "pos"):
            return "people"
        if st == "skip":
            return "skip"
        suspected = fi.ui < self.project.ui_threshold
        if suspected and (st is None or (st == "neg" and view == "ui")):
            return "skip"  # a suggestion: never overrides a decision outside the 'ui' view
        return None

    def commit_page(self, stems: list[str], marks: dict[str, str]) -> None:
        changes: dict[str, str] = {}
        for stem in stems:
            st, mark = self.review.get(stem), marks.get(stem)
            lab = self.session.label_path(stem)
            if mark == "people":
                if st not in ("todo", "pos"):
                    changes[stem] = "todo"
            elif mark == "skip":
                if st != "skip":
                    lab.unlink(missing_ok=True)
                    changes[stem] = "skip"
            elif st != "neg":
                lab.parent.mkdir(parents=True, exist_ok=True)
                lab.write_text("", encoding="utf-8")
                changes[stem] = "neg"
        if changes:
            self.review.set_many(changes)
            for stem in changes:
                self.refresh(stem)

