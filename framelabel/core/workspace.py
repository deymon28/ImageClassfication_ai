"""Labeling state of one session, independent of the UI: frame index, review statuses, loading a frame's boxes
(reviewed labels, else model proposals), saving a decision, committing a grid page."""
from __future__ import annotations

import logging
from pathlib import Path

from .boxes import Box, iou_matrix, read_yolo, to_array, write_yolo, yolo_lines
from .checked import load_checked, save_checked
from .project import Project
from .queues import FrameInfo, build_index, build_queue, frame_info
from .report import capture_info, hidden_line
from .session import Session
from .uiscreen import ui_scores
from .views import view_frames


log = logging.getLogger("framelabel")


class Workspace:
    def __init__(self, project: Project, session: Session, conf: float = 0.3, progress=None, cancelled=None):
        self.project, self.session, self.conf = project, session, conf
        self.review = session.review(project.class_names)
        frames = session.frames()
        self.ui = ui_scores(session, frames, project.ui_zone, progress, cancelled) if project.ui_zone else {}
        self.index: list[FrameInfo] = build_index(session, self.review, project.ignore_zones, self.ui, progress, cancelled)
        self.by_stem = {fi.stem: i for i, fi in enumerate(self.index)}
        self.capture = capture_info(session)  # what the recorder dropped (capture.json), if the session has it
        self.checked: set[str] = load_checked(session)  # frames marked as checked in Review (a personal note)

    # ---------------- index
    def refresh(self, stem: str) -> FrameInfo:
        i = self.by_stem[stem]
        self.index[i] = frame_info(self.session, self.index[i].path, self.review, self.project.ignore_zones,
                                   self.ui.get(stem, 1.0))
        return self.index[i]

    def reload(self, progress=None, cancelled=None) -> None:
        self.review = self.session.review(self.project.class_names)
        self.index = build_index(self.session, self.review, self.project.ignore_zones, self.ui, progress, cancelled)
        self.by_stem = {fi.stem: i for i, fi in enumerate(self.index)}

    def build_fresh(self, progress=None, cancelled=None) -> tuple[dict[str, float], list[FrameInfo]]:
        """(interface scores, index) read from disk without touching this workspace: safe to run in a thread, then
        hand the result to apply_reload on the GUI thread."""
        review = self.session.review(self.project.class_names)
        ui = ui_scores(self.session, self.session.frames(), self.project.ui_zone, progress, cancelled)             if self.project.ui_zone else {}
        return ui, build_index(self.session, review, self.project.ignore_zones, ui, progress, cancelled)

    def apply_reload(self, fresh: tuple[dict[str, float], list[FrameInfo]]) -> None:
        """Take new proposals (and new frames) from build_fresh, but keep this workspace's statuses and labels:
        decisions made while the index was being rebuilt are newer than the snapshot."""
        ui, frames = fresh
        old = {fi.stem: fi for fi in self.index}
        for fi in frames:
            o = old.get(fi.stem)
            fi.status = o.status if o is not None else self.review.get(fi.stem)
            if o is not None:
                fi.label_cls, fi.label_h, fi.label_xyxy = o.label_cls, o.label_h, o.label_xyxy
        self.ui = ui
        self.index = frames
        self.by_stem = {fi.stem: i for i, fi in enumerate(self.index)}

    def queue(self, name: str, **kw) -> list[FrameInfo]:
        """A raw queue of queues.py (any name, no lens or sort)."""
        return build_queue(self.index, name, conf=self.conf, ui_threshold=self.project.ui_threshold,
                           checked=self.checked, **kw)

    def view(self, key: str, sort: str = "time", flags: dict | None = None) -> list[FrameInfo]:
        """The frames of a view of the window (a stage, a lens or a library view), see views.py. `flags` is
        queues.review_flags() when the caller has it already (the Review views need it, and it is not cheap)."""
        return view_frames(self.index, key, self.conf, self.project.ui_threshold, sort, self.checked,
                           self.project.class_names, flags)

    # ---------------- checked (Review)
    def check(self, stems) -> None:
        stems = set(stems)
        if stems - self.checked:
            self.checked |= stems
            self.persist_checked()

    def uncheck(self, stems) -> None:
        stems = set(stems)
        if stems & self.checked:
            self.checked -= stems
            self.persist_checked()

    def persist_checked(self) -> None:
        """Write the note; a failure (a locked or read-only app folder) is logged and never blocks a decision."""
        try:
            save_checked(self.session, self.checked)
        except OSError:
            log.warning("could not write the checked-frames note", exc_info=True)

    def counts(self) -> dict[str, int]:
        return self.review.counts()

    def hidden_summary(self) -> str:
        """One line about frames that are filtered out of the normal views (see core.report)."""
        return hidden_line(self.index, self.capture, self.conf, self.project.ui_threshold)

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

    def ghosts(self, path: Path, w: int, h: int) -> list[Box]:
        """Model proposals at the shown confidence that the labels of a decided frame do not cover: what the model sees
        and the labels do not (empty for a frame without a decision, which shows its proposals as boxes anyway)."""
        if self.review.get(path.stem) not in ("pos", "neg"):
            return []
        props = read_yolo(self.session.auto_path(path.stem), w, h, self.conf, self.project.ignore_zones)
        labels = read_yolo(self.session.label_path(path.stem), w, h)
        if not props or not labels:
            return props
        covered = (iou_matrix(to_array(props), to_array(labels)) >= 0.4).any(1)
        return [b for b, hit in zip(props, covered) if not hit]

    def save(self, path: Path, boxes: list[Box], w: int, h: int, status: str | None = None) -> str:
        """Save a decision: 'pos'/'neg' write the label file ('pos' if any box survives), 'skip' removes it. A saved
        frame is no longer 'checked': it is looked at again if it still disagrees with the model."""
        lab = self.session.label_path(path.stem)
        if status is None:
            status = "pos" if yolo_lines(boxes, w, h) else "neg"
        if status in ("pos", "neg"):
            write_yolo(lab, boxes if status == "pos" else [], w, h)
        else:
            lab.unlink(missing_ok=True)
        self.review.set(path.stem, status)
        self.refresh(path.stem)
        self.uncheck([path.stem])  # last: the decision is saved whatever happens to the note
        return status

    # ---------------- grid
    def initial_mark(self, fi: FrameInfo, view: str = "triage") -> str | None:
        st = self.review.get(fi.stem)
        if view == "checked":
            return None  # nothing is ticked yet: a tick here means 'return this frame to Review'
        if st in ("todo", "pos"):
            return "people"
        if st == "skip":
            return "skip"
        suspected = fi.ui < self.project.ui_threshold
        if suspected and (st is None or (st == "neg" and view == "ui")):
            return "skip"  # a suggestion: never overrides a decision outside the 'ui' view
        return None

    def commit_page(self, stems: list[str], marks: dict[str, str]) -> dict[str, tuple[str | None, str | None, str]]:
        """Save a grid page: tiles marked 'has objects' are queued for the editor, SKIP tiles are skipped, the unmarked
        ones become empty. **A labeled frame is never changed by a page save** (the grid locks its tile, and a stale
        mark must not cost it its boxes either); it is changed in the editor. Returns what was changed as
        stem -> (old status, old label file text or None, new status), the record undo_page() needs."""
        changes: dict[str, str] = {}
        old: dict[str, tuple[str | None, str | None]] = {}

        def remember(stem: str, lab: Path) -> None:
            try:
                text = lab.read_text(encoding="utf-8") if lab.exists() else None
            except OSError:
                text = None
            old[stem] = (self.review.get(stem), text)

        for stem in stems:
            st, mark = self.review.get(stem), marks.get(stem)
            if st == "pos":
                continue
            lab = self.session.label_path(stem)
            if mark == "people":
                if st != "todo":
                    remember(stem, lab)
                    changes[stem] = "todo"
            elif mark == "skip":
                if st != "skip":
                    remember(stem, lab)
                    lab.unlink(missing_ok=True)
                    changes[stem] = "skip"
            elif st != "neg":
                remember(stem, lab)
                lab.parent.mkdir(parents=True, exist_ok=True)
                lab.write_text("", encoding="utf-8")
                changes[stem] = "neg"
        if changes:
            self.review.set_many(changes)
            for stem in changes:
                self.refresh(stem)
        return {stem: (*old[stem], new) for stem, new in changes.items()}

    def undo_page(self, undo: dict[str, tuple[str | None, str | None, str]]) -> tuple[list[str], list[str]]:
        """Put back what commit_page() changed: the statuses and the label files as they were. A frame that was decided
        again since (labeled in the editor, say) is left alone: undoing a page must never undo later work.
        Returns (restored stems, stems left alone)."""
        restored, left = [], []
        for stem, (old_status, text, new_status) in undo.items():
            if self.review.get(stem) != new_status:
                left.append(stem)
                continue
            lab = self.session.label_path(stem)
            if text is None:
                lab.unlink(missing_ok=True)
            else:
                lab.parent.mkdir(parents=True, exist_ok=True)
                lab.write_text(text, encoding="utf-8")
            restored.append(stem)
        if restored:
            self.review.set_many({stem: undo[stem][0] for stem in restored})
            for stem in restored:
                self.refresh(stem)
        return restored, left
