"""Frames the user marked as checked in the Review stage ("this one is fine, stop showing it").

The mark is a personal note about a session, not part of the dataset: it is kept in the per-user app folder, never in
the session (the session format stays what the capture tool and screendet read), and one file per session.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from framelabel.perf import app_data_dir
from .session import Session


def checked_path(session: Session) -> Path:
    key = hashlib.sha1(str(session.root.resolve()).lower().encode()).hexdigest()[:20]
    return app_data_dir() / "checked" / f"{key}.json"


def load_checked(session: Session) -> set[str]:
    try:
        data = json.loads(checked_path(session).read_text(encoding="utf-8"))
        return {str(s) for s in data.get("stems", [])}
    except (OSError, ValueError, AttributeError, TypeError):
        return set()


def save_checked(session: Session, stems: set[str]) -> None:
    path = checked_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": 1, "session": session.name, "stems": sorted(stems)}), encoding="utf-8")
    os.replace(tmp, path)
