"""The views of a session: what you look at, organised by what you do.

Work goes through three **stages**, each with the way of looking that suits it:

  1 Triage  (grid)    frames the model found nothing in: tick the ones that show something, the rest become empty
  2 Label   (editor)  frames with model proposals and the ones ticked in Triage: fix the boxes, save
  3 Review  (grid)    saved frames that deserve a second look (possibly missed / wrong boxes, small objects)

A stage can be narrowed with a **lens** (a refinement that is always a subset of its stage: Triage > weak proposals,
Label > marked in Triage, Review > possibly missed). The **Library** lists everything by decision or class. The mode
(grid or editor) is independent: every view opens in the way listed here and can be shown in the other.

Vocabulary used everywhere in the UI: a frame is *new* (no decision yet), *marked* ("has objects" ticked in the grid,
boxes still to do), *labeled*, *empty* or *skipped*. Triage and Label only ever contain frames without a decision, so
what is left to do is always visible (they partition the new and marked frames: a frame is in exactly one of them);
the Library and Review show decisions.

The order of the Label stage can be changed (time, unsure proposals first, small objects first): that is a sort, not
a view.
"""
from __future__ import annotations

from dataclasses import dataclass

from .queues import FrameInfo, SORTS, build_queue, review_flags, sort_frames

GROUPS = (("workflow", "Workflow"), ("library", "Library"))
STAGE_ORDER = ("triage", "edit", "review")


@dataclass(frozen=True)
class ViewDef:
    key: str
    label: str
    group: str                   # "workflow" or "library"
    mode: str                    # the way it opens: "grid" or "edit"
    hint: str                    # what these frames are and what to do with them (one or two sentences)
    step: int | None = None      # 1..3 for a stage
    parent: str | None = None    # a lens refines this stage
    queue: str | None = None     # the queue (queues.py) when it is not the key
    sortable: bool = False

    @property
    def is_stage(self) -> bool:
        return self.step is not None

    @property
    def stage(self) -> str | None:
        """The stage a view belongs to (itself for a stage, the parent for a lens, None for the library)."""
        return self.key if self.is_stage else self.parent


VIEWS: tuple[ViewDef, ...] = (
    ViewDef("triage", "Triage", "workflow", "grid",
            "Frames the model found nothing in. Click the tiles that show an object; the rest are saved as empty.",
            step=1),
    ViewDef("weak", "Weak proposals", "workflow", "grid",
            "The model has a proposal below the shown confidence: maybe a far or partly hidden object (dotted boxes).",
            parent="triage"),
    ViewDef("zone", "Only in ignore zone", "workflow", "grid",
            "The model saw something only inside an ignore zone (a HUD icon, say), so the frame looks empty. The hidden "
            "boxes are cyan.", parent="triage"),
    ViewDef("ui", "Interface screens", "workflow", "grid",
            "Menus, maps and inventories (no HUD). They are pre-marked SKIP when you save the page.", parent="triage"),
    ViewDef("edit", "Label", "workflow", "edit",
            "Fix the model's boxes or draw your own, then save. Includes the frames you marked in Triage.", step=2,
            sortable=True),
    ViewDef("proposals", "From the model", "workflow", "edit",
            "New frames with a proposal at the shown confidence.", parent="edit"),
    ViewDef("todo", "Marked in Triage", "workflow", "edit",
            "Frames you marked 'has objects' in Triage; they have no boxes yet.", parent="edit"),
    ViewDef("review", "Review", "workflow", "grid",
            "Saved frames that deserve a second look. Fix what is wrong, or mark a frame as checked to hide it here.",
            step=3),
    ViewDef("missed", "Possibly missed", "workflow", "grid",
            "The model is confident about a box you did not draw (or the frame is empty).", parent="review"),
    ViewDef("extra", "Possibly wrong", "workflow", "grid",
            "A box has no matching proposal: misplaced, or just a hard object the model does not see.", parent="review"),
    ViewDef("smalllab", "Small objects", "workflow", "grid",
            "Labeled frames with a far (small) object: check that its box is tight.", parent="review"),
    ViewDef("all", "All frames", "library", "grid", "Every frame of the session in time order, whatever its status."),
    ViewDef("unreviewed", "New", "library", "grid", "Frames without a decision yet."),
    ViewDef("pos", "Labeled", "library", "grid", "Frames saved with boxes."),
    ViewDef("neg", "Empty", "library", "grid", "Frames saved as empty: the negative examples."),
    ViewDef("skip", "Skipped", "library", "grid", "Frames set aside as unusable; they are not exported."),
)
BY_KEY = {v.key: v for v in VIEWS}


def class_view(i: int, name: str) -> ViewDef:
    return ViewDef(f"class:{i}", f"Class: {name}", "library", "grid", f"Frames with at least one {name} box.")


def all_views(class_names: list[str]) -> list[ViewDef]:
    """Every view of a project in list order: the fixed ones, with one class view per class at the end of the library."""
    return list(VIEWS) + [class_view(i, n) for i, n in enumerate(class_names)]


def find_view(key: str, class_names: list[str] = ()) -> ViewDef:
    if key.startswith("class:"):
        i = int(key.split(":", 1)[1])
        return class_view(i, class_names[i] if 0 <= i < len(class_names) else str(i))
    return BY_KEY[key]


def stage_label(key: str, class_names: list[str] = ()) -> str:
    """'Triage', or 'Triage > Weak proposals' for a lens, 'Class: enemy' for a class view: where a view lives."""
    try:
        v = find_view(key, class_names)
    except (KeyError, ValueError, IndexError):
        return key
    if v.is_stage or v.parent is None:
        return v.label
    return f"{BY_KEY[v.parent].label} > {v.label}"


def view_frames(index: list[FrameInfo], key: str, conf: float, ui_threshold: float, sort: str = "time",
                checked=frozenset(), class_names: list[str] = (), flags: dict | None = None) -> list[FrameInfo]:
    """The frames of a view. The stages of work never contain decided frames; a lens is the part of its stage that also
    matches the lens' own queue, in the stage's order; the sort applies to sortable stages (and so to their lenses)."""
    v = find_view(key, class_names)
    kw = dict(conf=conf, ui_threshold=ui_threshold, hide_reviewed=True, checked=checked, flags=flags)
    if v.parent:
        stage = view_frames(index, v.parent, conf, ui_threshold, sort, checked, class_names, flags)
        members = {f.stem for f in build_queue(index, v.queue or key, **kw)}
        return [f for f in stage if f.stem in members]
    frames = build_queue(index, v.queue or key, **kw)
    return sort_frames(frames, sort) if v.sortable else frames


def view_counts(index: list[FrameInfo], class_names: list[str], conf: float, ui_threshold: float,
                checked=frozenset()) -> dict[str, int]:
    """Frame count of every view (shown next to its name). The stages are built once and every lens is counted against
    its stage, so the whole list costs little more than the review flags (the slow part on very large sessions)."""
    flags = review_flags(index, conf, checked)
    stages = {s: view_frames(index, s, conf, ui_threshold, "time", checked, class_names, flags) for s in STAGE_ORDER}
    counts: dict[str, int] = {}
    for v in all_views(class_names):
        if v.is_stage:
            counts[v.key] = len(stages[v.key])
        elif v.parent:
            members = {f.stem for f in build_queue(index, v.queue or v.key, conf=conf, ui_threshold=ui_threshold,
                                                   hide_reviewed=True, checked=checked, flags=flags)}
            counts[v.key] = sum(1 for f in stages[v.parent] if f.stem in members)
        else:
            counts[v.key] = len(view_frames(index, v.key, conf, ui_threshold, "time", checked, class_names, flags))
    return counts


def view_progress(index: list[FrameInfo], conf: float, counts: dict[str, int]) -> dict[str, tuple[int, int]]:
    """stage -> (frames left, frames that went through the stage): the progress bar of Triage and Label."""
    triage_total = sum(1 for f in index if f.max_conf < conf)
    label_total = sum(1 for f in index if f.max_conf >= conf or f.status in ("todo", "pos"))
    return {"triage": (counts.get("triage", 0), triage_total), "edit": (counts.get("edit", 0), label_total)}


def next_stage(current: str | None, counts: dict[str, int]) -> str | None:
    """The next stage after `current` (cyclic, in work order) that still has frames; None when no other stage has."""
    i = STAGE_ORDER.index(current) if current in STAGE_ORDER else -1
    for step in range(1, len(STAGE_ORDER) + 1):
        key = STAGE_ORDER[(i + step) % len(STAGE_ORDER)]
        if key != current and counts.get(key, 0) > 0:
            return key
    return None


def status_counts(index: list[FrameInfo]) -> dict[str, int]:
    """Frames per decision: new / marked / labeled / empty / skipped."""
    c = {"new": 0, "marked": 0, "labeled": 0, "empty": 0, "skipped": 0}
    names = {None: "new", "todo": "marked", "pos": "labeled", "neg": "empty", "skip": "skipped"}
    for f in index:
        c[names.get(f.status, "new")] += 1
    return c


__all__ = ["GROUPS", "SORTS", "STAGE_ORDER", "ViewDef", "VIEWS", "all_views", "find_view", "next_stage", "stage_label",
           "status_counts", "view_counts", "view_frames", "view_progress"]
