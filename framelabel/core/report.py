"""A report of everything that is filtered out of the normal views of a session, and where to look at it.

Frames can disappear from a labeling session in several places: the recorder drops screens before they are saved
(capture.json), interface-screen detection pre-marks frames as skipped, ignore zones hide model proposals (a frame
then looks empty), the shown confidence hides weak proposals, and model classes without a project class are
dropped by prelabel. This module turns each of them into a counted row with the queue that shows those frames.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .queues import FrameInfo, build_queue
from .views import find_view, stage_label, view_frames
from .session import Session

CAPTURE_REASONS = {
    "ui": "interface screen (menu, inventory, map...) dropped by the recorder's UI filter",
    "duplicate": "almost identical to the previous saved frame",
    "not_game": "the game window was not in front",
    "waiting": "sampling interval not elapsed yet (normal pacing, not a decision about the frame)",
    "no_new_frame": "the screen had not changed since the last capture",
}
SAVED_REASONS = {"first": "first frame", "change": "the picture changed", "burst": "burst key (saved around the press)"}


@dataclass(frozen=True)
class ReportRow:
    label: str
    count: int | None
    where: str      # how to open these frames in the app, or what to do about them
    note: str = ""
    mode: str | None = None    # "grid" / "edit" when the frames can be opened
    queue: str | None = None   # the queue key (see queues.QUEUES)


def capture_info(session: Session) -> dict | None:
    """The recorder's capture.json of the session (counts of saved and skipped screens), None if there is none."""
    path = session.root / "capture.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def capture_rows(info: dict | None) -> list[ReportRow]:
    if not info:
        return [ReportRow("No capture.json in this session", None,
                          "older recordings did not keep what the recorder dropped")]
    where = "not on disk: cannot be reviewed here"
    rows = [ReportRow(f"Saved by the recorder ({SAVED_REASONS.get(k, k)})", int(n), "these are the frames of this session")
            for k, n in (info.get("saved_by_reason") or {}).items()]
    skipped = info.get("skipped_by_reason") or {}
    for key in sorted(skipped, key=lambda k: -int(skipped[k])):
        n = int(skipped[key])
        pace = key in ("waiting", "no_new_frame")
        rows.append(ReportRow(f"Recorder skipped: {key}", n, "(counts checks, not frames)" if pace else where,
                              CAPTURE_REASONS.get(key, "")))
    settings = info.get("settings") or {}
    policy = settings.get("policy") or {}
    if "ui_filter" in settings:
        rows.append(ReportRow("Recorder setting: ui_filter", None,
                              "on: interface screens are not saved" if settings["ui_filter"]
                              else "off: interface screens are saved (framelabel can find them)",
                              f"threshold {policy.get('ui_threshold', '?')}" if settings["ui_filter"] else ""))
    return rows


QUEUE_ROWS = [  # (view, label, note): the label says what the frames are, the view name says where they are
    ("triage", "New frames the model found nothing in (no proposal at the shown confidence)",
     "confirm as empty a page at a time; the three lines below are parts of it"),
    ("weak", "...with a weak proposal (below the shown confidence)",
     "dotted boxes in the grid; press W in the editor to see them"),
    ("zone", "...with a proposal only inside an ignore zone",
     "e.g. the HUD icon: the frame looks empty; the hidden boxes are cyan"),
    ("ui", "...that look like interface screens (framelabel's own detection)",
     "pre-marked SKIP when a page of the grid is saved unless you change the mark"),
    ("edit", "New frames with a proposal, plus the frames you marked in Triage", "the frames to label in the editor"),
    ("todo", "...marked 'has objects' in Triage", "no boxes yet"),
    ("review", "Saved frames that deserve a second look", "the three lines below are parts of it (checked frames left out)"),
    ("missed", "...where the model is confident about a box that was not drawn", "possibly missed objects"),
    ("extra", "...with a box the model has no proposal for", "possibly wrong, or hard objects"),
    ("smalllab", "...labeled with a small (far) object", "check that the box is tight"),
    ("neg", "Saved as empty", ""),
    ("skip", "Skipped", "set by you in the grid / editor, or accepted from the interface suggestion"),
    ("unreviewed", "New: no decision yet", ""),
]


def frame_rows(index: list[FrameInfo], conf: float, ui_threshold: float, class_names: list[str] | None = None,
               checked=frozenset()) -> list[ReportRow]:
    rows = [ReportRow("Frames in the session", len(index), "all frames on disk", mode="grid", queue="all")]
    for key, label, note in QUEUE_ROWS:
        view = find_view(key, class_names or [])
        n = len(view_frames(index, key, conf, ui_threshold, checked=checked))
        rows.append(ReportRow(label, n, stage_label(key), note, view.mode, key))
    return rows


def prelabel_rows(session: Session) -> list[ReportRow]:
    meta = session.auto_meta()
    if not meta:
        return [ReportRow("No model proposals yet", None, "Auto-label > Prelabel")]
    w = meta.get("weights")
    names = [Path(x).parent.parent.name or Path(x).stem for x in (w if isinstance(w, list) else [w] if w else [])]
    rows = [ReportRow("Prelabel models", None, ", ".join(names) or "?", f"imgsz {meta.get('imgsz')}, tta {meta.get('tta')}, "
                                                                          f"stored from conf {meta.get('conf')}, {meta.get('time', '')}")]
    run = meta.get("run")
    if isinstance(run, dict):
        rows.append(ReportRow("Last prelabel run: frames without any detection", run.get("without_proposals"), "(whole run)"))
        rows.append(ReportRow("Last prelabel run: frames whose detections are all inside an ignore zone",
                              run.get("only_in_ignore_zone"), "(whole run)", "see the zone view"))
        if run.get("unreadable"):
            rows.append(ReportRow("Last prelabel run: frames that could not be read", run["unreadable"], "(whole run)",
                                  "no proposals; the next prelabel run tries them again"))
        dropped = run.get("dropped_classes") or {}
        rows.append(ReportRow("Last prelabel run: detections of classes without a project class (dropped)",
                              sum(dropped.values()) if dropped else 0,
                              ", ".join(f"{k} {v}" for k, v in list(dropped.items())[:8]) or "none",
                              "set class_map in the project to keep a class"))
    else:
        rows.append(ReportRow("Dropped classes / zone-hidden frames of the last run", None,
                              "not recorded: this run predates the report (run prelabel again with 'redo' to record it)"))
    return rows


def hidden_line(index: list[FrameInfo], capture: dict | None, conf: float, ui_threshold: float) -> str:
    """One line for the main window: what hides in the Triage views, and what the recorder dropped."""
    n = {k: len(view_frames(index, k, conf, ui_threshold)) for k in ("ui", "zone", "weak")}
    parts = [f"interface screens {n['ui']}", f"only in ignore zone {n['zone']}", f"weak proposals only {n['weak']}"]
    if capture:
        skipped = capture.get("skipped_by_reason") or {}
        parts.append(f"recorder dropped: {skipped.get('ui', 0)} ui, {skipped.get('duplicate', 0)} duplicate, "
                     f"{skipped.get('not_game', 0)} not game")
    return "Triage: " + "; ".join(parts) + "  (Report for details)"
