"""The navigation of the main window: stages (Triage, Label, Review), their lenses, the library, the way of looking
(grid or editor), the order of the Label stage, the counts next to every view and the way from one stage to the next.

A mixin of MainWindow: it only uses the window's widgets and its workspace, and keeps no widgets of its own."""
from __future__ import annotations

import hashlib

from PyQt6.QtCore import pyqtSlot

from framelabel.core.pace import Pace, pace_line
from framelabel.core.queues import SMALL_H, SORTS, review_flags
from framelabel.core.views import (BY_KEY, STAGE_ORDER, all_views, find_view, next_stage, stage_label, view_counts,
                                   view_progress)
from framelabel.perf import timed
from .workers import Job, start_job

STAGE_KEYS = {"triage": "go_triage", "edit": "go_label", "review": "go_review"}  # the key map action of each stage


def mode_setting(key: str) -> str:
    return f"viewmode/{key.replace(':', '_')}"


class WorkflowMixin:
    # ================================================================ what is shown
    def view_def(self, key: str | None = None):
        return find_view(key or self.view_key, self.project.class_names if self.project else [])

    def view_name(self, key: str | None = None) -> str:
        """Where a view lives, for people: 'Triage', 'Triage > Weak proposals', 'Class: enemy'."""
        return stage_label(key or self.view_key, self.project.class_names if self.project else [])

    def current_stage(self) -> str | None:
        return self.view_def().stage

    def mode_of(self, key: str) -> str:
        """The way a view opens: the way you looked at it last time, else the way it suits best."""
        remembered = self.settings.value(mode_setting(key), "")
        return remembered if remembered in ("grid", "edit") else self.view_def(key).mode

    def refresh_tree(self) -> None:
        """Mode icons of the stages and the tooltips of every row (they name the key and the way the view opens)."""
        if self.project is None:
            return
        views = all_views(self.project.class_names)
        self.view_tree.set_mode_icons({v.key: self.mode_of(v.key) for v in views if v.is_stage})
        tips = {}
        for v in views:
            tip = f"{self.view_name(v.key)}: {v.hint}"
            if v.is_stage:
                way = "a grid" if self.mode_of(v.key) == "grid" else "the editor"
                tip += f"\nOpens as {way}.  Key: {self.keymap.first(STAGE_KEYS[v.key])}"
            tips[v.key] = tip
        self.view_tree.set_tooltips(tips)

    # ================================================================ showing a view
    def set_mode(self, mode: str, queue: str | None = None, keep_stem: str | None = None) -> None:
        """Show a view in grid or editor mode. Without `queue` the current view stays and only the way of showing it
        changes; with it, that view is shown. `keep_stem` keeps the frame you are on (its page, or the editor frame)."""
        self.commit_current()
        if queue is None and self.view_key not in self.view_tree.items:
            queue = "triage" if mode == "grid" else "edit"
        if queue is not None:
            self.view_key = queue
        self.mode = mode
        self.settings.setValue(mode_setting(self.view_key), mode)
        self.view_tree.select(self.view_key)
        self.context_bar.mode_switch.set_mode(mode)
        self.stack.setCurrentWidget(self.grid if mode == "grid" else self.canvas)
        self.action_pages.setCurrentIndex(0 if mode == "grid" else 1)
        self.context_bar.show_tiles(mode == "grid")
        self.context_bar.show_sort(bool(self.current_stage() and BY_KEY[self.current_stage()].sortable))
        self.action_buttons["mark_checked"].setVisible(self.current_stage() == "review")
        self.apply_panels()
        if mode == "grid":
            self.box_table.setRowCount(0)
            self.cur_path = None
        self.refresh_tree()
        self.apply_queue(keep_stem=keep_stem)

    def current_stem(self) -> str | None:
        """The frame you are on: the editor's frame, or the tile under the mouse (else the first of the page)."""
        if self.mode == "edit":
            return self.cur_path.stem if self.cur_path is not None else None
        return self.grid.hovered_stem() if self.grid.items else None

    def switch_mode(self) -> None:
        if self.ws is None:
            return
        self.set_mode("edit" if self.mode == "grid" else "grid", keep_stem=self.current_stem())

    def on_mode_chosen(self, mode: str) -> None:
        if self.ws is not None and mode != self.mode:
            self.set_mode(mode, keep_stem=self.current_stem())
        else:
            self.context_bar.mode_switch.set_mode(self.mode)

    def on_view_chosen(self, key: str) -> None:
        """A click in the list (or a stage key): show that view the way you last looked at it."""
        if self.ws is None:
            return
        self.set_mode(self.mode_of(key), queue=key)

    def open_in_editor(self, stem: str | None) -> None:
        """Open a frame of the current view in the editor (double-click in the grid, or O)."""
        if self.ws is not None and stem:
            self.set_mode("edit", keep_stem=stem)

    def on_sort_chosen(self, key: str) -> None:
        self.sort_key = key
        self.settings.setValue("sort", key)
        self.apply_queue(keep_stem=self.current_stem())
        self.toast(dict(SORTS)[key])

    # ================================================================ stages
    def counts_are_fresh(self) -> bool:
        """The counts next to the views match the workspace (no change is waiting to be counted)."""
        return self.counts_job is None and not self.counts_timer.isActive()

    def stage_remaining(self, stage: str) -> int:
        """Frames left in a stage, now. Review is the slow one: its last count is used only while it is up to date."""
        if stage == "review" and "review" in self.counts_by_view and self.counts_are_fresh():
            return self.counts_by_view["review"]
        return len(self.ws.view(stage, self.sort_key))

    def next_stage_with_work(self, current: str | None) -> tuple[str | None, dict[str, int]]:
        counts = {s: self.stage_remaining(s) for s in STAGE_ORDER}
        return next_stage(current, counts), counts

    def next_view(self) -> None:
        """V: the next stage that still has frames."""
        if self.ws is None:
            return
        nxt, counts = self.next_stage_with_work(self.current_stage())
        if nxt is not None:
            self.on_view_chosen(nxt)
            return
        here = self.current_stage()
        if here and counts.get(here):
            self.toast(f"No other stage has frames. {BY_KEY[here].label} still has {counts[here]}.")
        else:
            self.toast("Nothing left to do in Triage, Label or Review")

    def review_notes(self, flags: dict) -> dict[str, str]:
        """What Review found in each frame, for the frame list: 'missed 2, wrong 1, small'."""
        notes = {}
        for stem, (missed, extra, small) in flags.items():
            parts = ([f"missed {missed}"] if missed else []) + ([f"wrong {extra}"] if extra else []) \
                + (["small"] if small < SMALL_H else [])
            if parts:
                notes[stem] = ", ".join(parts)
        return notes

    def say_end_of_view(self) -> None:
        """The walk through a view ended: say what is still to do here (frames passed over), in the stage of a lens, or in
        the next stage that has work, with a button that goes there."""
        vd = self.view_def()
        name = self.view_name()
        lead, self.lead = self.lead, None
        start = (f"{lead}. " if lead else "") + f"End of {name}."
        left_here = len(self.ws.view(self.view_key, self.sort_key)) if vd.group == "workflow" else 0
        if left_here:  # frames earlier in the list were passed over
            self.toast(f"{start} {left_here} still to do in it.", ("Back to the start", self.apply_queue))
            return
        stage_left = self.stage_remaining(vd.stage) if vd.parent else 0
        if stage_left:  # the end of a lens: the rest of its stage is elsewhere
            stage = BY_KEY[vd.stage]
            self.toast(f"{start} {stage.label} still has {stage_left} frames.", (f"Back to {stage.label}",
                                                                                lambda s=vd.stage: self.on_view_chosen(s)))
            return
        nxt, counts = self.next_stage_with_work(vd.stage)
        if nxt:
            self.toast(f"{start} Next: {BY_KEY[nxt].label}, {counts[nxt]} frames.",
                       (f"Go to {BY_KEY[nxt].label}", lambda s=nxt: self.on_view_chosen(s)))
        else:
            self.toast(start + (" Nothing else is left to do." if vd.group == "workflow" else ""))

    def after_triage_page(self, undo_action: tuple | None = None) -> bool:
        """The last page of a Triage view was saved. Pages that were passed over are shown first; when the view is done
        the way goes on to the Triage stage (after a lens) or to the next stage. True for a Triage view (the window is
        then already showing what comes next), False for any other. The toast keeps the Undo of the page just saved."""
        vd = self.view_def()
        if vd.stage != "triage":
            return False
        name = self.view_name()
        left_here = len(self.ws.view(self.view_key, self.sort_key))
        if left_here:
            self.apply_queue()
            self.toast(f"{left_here} frame{'s' if left_here != 1 else ''} left in {name}", undo_action)
            return True
        if vd.parent and self.stage_remaining("triage"):
            self.on_view_chosen("triage")
            self.toast(f"{vd.label} is done. Back in Triage.", undo_action)
            return True
        nxt, counts = self.next_stage_with_work("triage")
        if nxt is None:
            self.apply_queue()  # the empty state of Triage says the rest
            self.toast("Triage is done. Every frame has a decision.", undo_action)
            return True
        self.on_view_chosen(nxt)
        self.toast(f"Triage is done. Now: {BY_KEY[nxt].label}, {counts[nxt]} frames.", undo_action)
        return True

    def empty_message(self) -> str:
        """What the work area says when the view has no frames: what this is, why it is empty and where to go on."""
        vd = self.view_def()
        if vd.group != "workflow":
            return f"No frames in {vd.label}."
        name = self.view_name()
        text = f"Nothing left in {name}." if vd.is_stage else f"No frames in {name}."
        if vd.parent:  # an empty lens: its stage may still have frames
            stage_left = self.stage_remaining(vd.stage)
            if stage_left:
                return f"{text} {BY_KEY[vd.stage].label} still has {stage_left} frames."
        nxt, counts = self.next_stage_with_work(vd.stage)
        if nxt:
            text += f" Next: {BY_KEY[nxt].label}, {counts[nxt]} frames (press {self.keymap.first('grid_view')})."
        elif vd.stage == "review":
            text += f" Everything is checked. Export the dataset with File > Export (Ctrl+E)."
        else:
            text += " Every frame has a decision."
        return text

    def coach_text(self) -> tuple[str, str, str]:
        """(title, way of looking, what to do) for the card under the list."""
        vd = self.view_def()
        title = self.view_name(vd.key) if vd.group == "workflow" else vd.label
        grid = self.mode == "grid"
        hint = vd.hint
        if vd.stage == "review":
            hint += (" Space marks the page as checked; double-click a tile to fix it in the Editor." if grid
                     else f" Space saves and marks the frame as checked; {self.keymap.first('mark_checked')} only marks it.")
        elif vd.group == "library" and grid:
            hint += " Double-click a tile to open it in the Editor."
        return title, "Grid" if grid else "Editor", hint

    # ================================================================ the queue of the current view
    def apply_queue(self, keep_stem: str | None = None) -> None:
        """Rebuild the current view and show it; with keep_stem stay on that frame (or its grid page) if it is
        still in the view."""
        if self.ws is None:
            return
        self.commit_current()
        name = self.view_key
        flags = review_flags(self.ws.index, self.ws.conf, self.ws.checked) if self.current_stage() == "review" else None
        self.queue = self.ws.view(name, self.sort_key, flags)
        self.frame_model.notes = self.review_notes(flags) if flags is not None else {}
        with timed("fill_frame_list", 20):
            self.fill_frame_list()
        pos = next((i for i, f in enumerate(self.queue) if f.stem == keep_stem), None) if keep_stem else None
        if pos is None and keep_stem and self.mode == "edit" and keep_stem in self.ws.by_stem:
            # the frame you are on stays (in time order) even when the new filter would not list it
            at = self.ws.by_stem[keep_stem]
            pos = sum(1 for f in self.queue if self.ws.by_stem.get(f.stem, 0) < at)
            self.queue.insert(pos, self.ws.index[at])
            self.fill_frame_list()
        empty = self.empty_message() if not self.queue else ""
        self.grid.message = empty or "No frames in this view"
        if self.mode == "grid":
            if pos is not None:
                page = pos // self.grid.per_page
            elif self.current_stage() == "triage":
                page = next((i for i, f in enumerate(self.queue) if f.status is None), 0) // self.grid.per_page
            else:
                page = 0
            marks = dict(self.grid.marks) if keep_stem else {}
            was = {fi.stem: fi.status for fi in self.grid.items} if marks else {}
            self.grid.show_ignored = name == "zone"
            self.grid.conf = self.ws.conf
            self.grid.set_items(self.queue, page, initial_mark=lambda fi, v=name: self.ws.initial_mark(fi, v))
            if marks:  # marks not saved yet survive the rebuild, unless the frame was decided meanwhile (in the editor)
                now = {f.stem: f.status for f in self.grid.page_items()}
                self.grid.marks.update({k: m for k, m in marks.items() if k in now and was.get(k) == now[k]})
                self.grid.update()
            self.grid.setFocus()
        else:
            if not self.queue:
                self.cur_path = None
                self.canvas.img = None
                self.canvas.message = empty
                self.canvas.update()
                self.update_status()
                return
            if pos is None:
                pos = self.resume_position(name)
            first = pos if pos is not None else next((i for i, f in enumerate(self.queue) if f.status in (None, "todo")), 0)
            self.load_frame(first)
            self.canvas.setFocus()
        self.update_status()

    def position_key(self, queue_name: str) -> str:
        root = str(self.ws.session.root).lower().encode()
        return f"pos/{hashlib.sha1(root).hexdigest()[:12]}/{queue_name.replace(':', '_')}"

    def remember_position(self, stem: str) -> None:
        if self.ws is not None:
            self.settings.setValue(self.position_key(self.view_key), stem)

    def resume_position(self, queue_name: str) -> int | None:
        """Where the user stopped in this view; in the stages of work only if that frame still waits for a decision."""
        stem = self.settings.value(self.position_key(queue_name), "")
        i = next((k for k, f in enumerate(self.queue) if f.stem == stem), None) if stem else None
        if i is None:
            return None
        if self.current_stage() in ("triage", "edit"):
            return i if self.queue[i].status in (None, "todo") else None
        return i

    # ================================================================ counts, progress and pace
    def schedule_counts(self) -> None:
        if self.ws is not None and not self.closing:
            self.counts_timer.start()

    def refresh_counts(self) -> None:
        """Count the frames of every view in a thread (the Review views are the slow ones on very large sessions)."""
        if self.ws is None or self.closing:
            return
        if self.counts_job is not None:
            self.counts_job.cancel()
        index, names = list(self.ws.index), list(self.project.class_names)
        conf, thr, checked = self.ws.conf, self.project.ui_threshold, frozenset(self.ws.checked)

        def count(progress, cancelled):
            counts = view_counts(index, names, conf, thr, checked)
            return counts, view_progress(index, conf, counts)

        job = Job(count)
        self.counts_job = job
        job.finished.connect(self.on_counts)
        start_job(self, job)

    @pyqtSlot(object)
    def on_counts(self, result) -> None:
        if self.sender() is not self.counts_job:
            return  # a newer count is on its way
        self.counts_job = None
        counts, progress = result
        self.counts_by_view = counts
        recommended = next((s for s in STAGE_ORDER if counts.get(s, 0) > 0), None)
        self.view_tree.set_counts(counts, progress, recommended)
        self.update_status()

    def record_decisions(self, n: int) -> None:
        """Count decisions for the pace line under the progress bar."""
        self.pace.add(n)

    def pace_text(self) -> str:
        left = self.counts_by_view.get("triage", 0) + self.counts_by_view.get("edit", 0)
        return pace_line(self.pace, left)


__all__ = ["Pace", "STAGE_KEYS", "WorkflowMixin"]
