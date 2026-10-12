"""The views of a session: named sets of frames shown in the left panel, grouped by what you do with them.

A *view* answers "which frames am I looking at?" (see queues.py for the rules); the *mode* (grid or editor) answers
"how am I looking at them?" and is independent: every view can be shown in both modes, each one has a default.

Vocabulary used everywhere in the UI: a frame is *new* (no decision yet), *marked* ("has objects" ticked in the grid,
boxes still to do), *labeled*, *empty* or *skipped*. The views of the "To do" and "Check" groups only ever contain
frames without a decision (new or marked), so what is left to do is always visible; "Done" shows the decisions.
"""
from __future__ import annotations

from dataclasses import dataclass

from .queues import FrameInfo, build_queue

GROUPS = (("todo", "To do"), ("check", "Check"), ("done", "Done"), ("browse", "Browse"))


@dataclass(frozen=True)
class ViewDef:
    key: str
    label: str
    group: str
    mode: str      # default mode: "grid" or "edit"
    hint: str      # what these frames are and what to do with them (one or two sentences)


VIEWS: tuple[ViewDef, ...] = (
    ViewDef("triage", "Triage: model found nothing", "todo", "grid",
            "New frames where the model found nothing at the shown confidence. Click the tiles that have objects; "
            "when you commit the page the unmarked tiles are saved as empty."),
    ViewDef("edit", "Label: proposals and marked", "todo", "edit",
            "New frames with model proposals, and frames you marked 'has objects' in the grid. Fix the boxes and accept."),
    ViewDef("todo", "Marked, boxes still to do", "todo", "edit",
            "Frames you marked 'has objects' in the grid that have no boxes yet."),
    ViewDef("uncertain", "Unsure proposals first", "todo", "edit",
            "New frames whose best proposal is unsure (confidence 0.15-0.6), the most unsure first."),
    ViewDef("small", "Small (far) objects first", "todo", "edit",
            "New frames with small or far objects, the smallest first: easy to miss, worth a careful look."),
    ViewDef("unreviewed", "All new frames", "todo", "grid",
            "Every frame without a decision, whatever the model said."),
    ViewDef("ui", "Looks like an interface screen", "check", "grid",
            "New frames that look like a menu, map or inventory (no HUD). They are pre-marked SKIP when you commit."),
    ViewDef("zone", "Seen only in an ignore zone", "check", "grid",
            "New frames where the model saw something only inside an ignore zone (e.g. a HUD icon): they look empty. "
            "The hidden boxes are drawn in cyan."),
    ViewDef("weak", "Weak proposals only", "check", "grid",
            "New frames whose best proposal is below the shown confidence (dotted boxes): maybe a far object."),
    ViewDef("disagree", "Model disagrees with my labels", "check", "edit",
            "Decided frames where a confident proposal has no label, or a label has no proposal: finds missed boxes."),
    ViewDef("pos", "Labeled", "done", "edit", "Frames saved with boxes."),
    ViewDef("neg", "Empty", "done", "grid", "Frames saved as empty (a negative example)."),
    ViewDef("skip", "Skipped", "done", "grid", "Frames set aside as unusable (not exported)."),
    ViewDef("all", "All frames", "browse", "edit", "Every frame of the session in time order, whatever its status."),
)
BY_KEY = {v.key: v for v in VIEWS}


def class_view(i: int, name: str) -> ViewDef:
    return ViewDef(f"class:{i}", f"Has class {name}", "browse", "edit", f"Decided frames that have a box of class {name}.")


def all_views(class_names: list[str]) -> list[ViewDef]:
    """Every view of a project in menu order: the fixed ones, with one 'has class' view per class in its group."""
    out: list[ViewDef] = []
    for group, _ in GROUPS:
        out += [v for v in VIEWS if v.group == group]
        if group == "browse":
            out += [class_view(i, n) for i, n in enumerate(class_names)]
    return out


def find_view(key: str, class_names: list[str]) -> ViewDef:
    if key.startswith("class:"):
        i = int(key.split(":", 1)[1])
        return class_view(i, class_names[i] if 0 <= i < len(class_names) else str(i))
    return BY_KEY[key]


def view_frames(index: list[FrameInfo], key: str, conf: float, ui_threshold: float) -> list[FrameInfo]:
    """The frames of a view. Views that are about work still to do never contain decided frames."""
    return build_queue(index, key, conf=conf, ui_threshold=ui_threshold, hide_reviewed=True)


def view_counts(index: list[FrameInfo], class_names: list[str], conf: float, ui_threshold: float) -> dict[str, int]:
    """Frame count of every view (shown next to its name)."""
    return {v.key: len(view_frames(index, v.key, conf, ui_threshold)) for v in all_views(class_names)}


def status_counts(index: list[FrameInfo]) -> dict[str, int]:
    """Frames per decision: new / marked / labeled / empty / skipped."""
    c = {"new": 0, "marked": 0, "labeled": 0, "empty": 0, "skipped": 0}
    names = {None: "new", "todo": "marked", "pos": "labeled", "neg": "empty", "skip": "skipped"}
    for f in index:
        c[names.get(f.status, "new")] += 1
    return c
