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


QUEUE_ROWS = [  # (mode, queue, label, note)
    ("grid", "triage", "Model found nothing (no proposal at the shown confidence)",
     "confirm as empty a page at a time; frames with weak or zone-hidden proposals are in here too"),
    ("grid", "weak", "...of which unreviewed with a weak proposal (below the shown confidence)",
     "dotted boxes in the grid; press W in the editor to see them"),
    ("grid", "zone", "Proposals hidden by an ignore zone",
     "the model saw something inside the ignore zone only (e.g. the HUD icon): the frame looks empty; shown as cyan boxes"),
    ("grid", "ui", "Suspected interface screens (framelabel's own detection)",
     "pre-marked SKIP when a page of the grid is committed unless you change the mark"),
    ("edit", "edit", "To review: has proposals at the shown confidence, or marked / labeled / skipped", ""),
    ("edit", "uncertain", "Uncertain proposals", "confidence 0.15-0.6, unreviewed"),
    ("edit", "small", "Small (far) objects", ""),
    ("edit", "disagree", "Model disagrees with the labels", "confident proposals without a label, labels without a proposal"),
    ("grid", "neg", "Saved as empty", ""),
    ("grid", "skip", "Skipped", "set by you in the grid / editor, or accepted from the interface suggestion"),
    ("grid", "unreviewed", "Not reviewed yet", ""),
]


def frame_rows(index: list[FrameInfo], conf: float, ui_threshold: float) -> list[ReportRow]:
    rows = [ReportRow("Frames in the session", len(index), "all frames on disk", mode="edit", queue="all")]
    for mode, queue, label, note in QUEUE_ROWS:
        n = len(build_queue(index, queue, conf=conf, ui_threshold=ui_threshold))
        where = f"{'Grid' if mode == 'grid' else 'Editor'} view \"{queue}\""
        rows.append(ReportRow(label, n, where, note, mode, queue))
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
    """One line for the main window: what is filtered out of the normal views."""
    n_ui = len(build_queue(index, "ui", conf=conf, ui_threshold=ui_threshold))
    n_zone = len(build_queue(index, "zone", conf=conf, ui_threshold=ui_threshold))
    n_weak = len(build_queue(index, "weak", conf=conf, ui_threshold=ui_threshold))
    parts = [f"suspected interface screens {n_ui}", f"zone-hidden proposals {n_zone}", f"weak only {n_weak}"]
    if capture:
        skipped = capture.get("skipped_by_reason") or {}
        parts.append(f"recorder dropped: {skipped.get('ui', 0)} ui, {skipped.get('duplicate', 0)} duplicate, "
                     f"{skipped.get('not_game', 0)} not game")
    return "Filtered: " + "; ".join(parts) + "  (View > Session report)"
