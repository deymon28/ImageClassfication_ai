"""A session = a folder of frames plus its reviewed labels, model proposals and review statuses.

Layouts (detected by `Session.open`):
  session   <s>/frames/*.jpg, <s>/labels/*.txt, <s>/labels_auto/*.txt, <s>/review.json   (screendet collector)
  yolo      <d>/images/<split>/*.jpg -> <d>/labels/<split>, <d>/labels_auto/<split>, <d>/review/<split>.json
  folder    <f>/*.jpg -> <f>/labels, <f>/labels_auto, <f>/review.json
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STATUSES = ("pos", "neg", "skip", "todo")  # todo = marked "has objects" in the grid, not labeled yet


class Review:
    """Per-frame status in review.json ({"version": 1, "classes": [...], "status": {stem: status}}), written
    atomically after every change. Unknown top-level keys are preserved."""

    def __init__(self, path: Path, classes: list[str] | None = None):
        self.path = path
        self.classes = list(classes) if classes else None
        self.stored_classes: list[str] | None = None  # the class list the file was written with
        self.extra: dict = {}
        self.status: dict[str, str] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or not isinstance(data.get("status", {}), dict):
                    raise ValueError("not a review file")
            except ValueError as e:  # never treated as empty: the next save would overwrite every decision
                raise ValueError(f"{path} is damaged ({e}); fix or restore it before labeling this session") from e
            self.status = dict(data.get("status", {}))
            self.extra = {k: v for k, v in data.items() if k not in ("version", "classes", "status")}
            stored = data.get("classes")
            self.stored_classes = list(stored) if stored else None
            if self.classes is None:
                self.classes = stored

    @property
    def classes_conflict(self) -> bool:
        """The project's class list is not the one this session was labeled with (or an extension of it). Labels store the
        class *number*, so a reordered, renamed or shortened list would silently change what existing boxes mean."""
        s, c = self.stored_classes, self.classes
        return bool(s and c and list(c[:len(s)]) != list(s))

    def adopt_classes(self) -> None:
        """Accept the project's class list for this session (after the user confirmed that the numbers still mean the same)."""
        self.stored_classes = list(self.classes or [])
        self.save()

    def get(self, stem: str) -> str | None:
        return self.status.get(stem)

    def set(self, stem: str, status: str | None) -> None:
        self.set_many({stem: status})

    def set_many(self, changes: dict[str, str | None]) -> None:
        for stem, status in changes.items():
            if status is None:
                self.status.pop(stem, None)
            else:
                if status not in STATUSES:
                    raise ValueError(f"unknown status {status!r}")
                self.status[stem] = status
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # a conflicting project list never overwrites the one the labels were made with; an extension of it does
        classes = (self.stored_classes if self.classes_conflict else self.classes) or []
        self.stored_classes = list(classes) or None
        data = {"version": 1, "classes": classes, "status": self.status, **self.extra}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=0), encoding="utf-8")
        os.replace(tmp, self.path)

    def counts(self) -> dict[str, int]:
        c = {s: 0 for s in STATUSES}
        for s in self.status.values():
            c[s] = c.get(s, 0) + 1
        return c


@dataclass
class Session:
    name: str
    images_dir: Path
    labels_dir: Path
    auto_dir: Path
    review_path: Path
    layout: str

    @classmethod
    def open(cls, path: Path) -> Session:
        path = Path(path)
        if (path / "frames").is_dir():
            return cls(path.name, path / "frames", path / "labels", path / "labels_auto", path / "review.json", "session")
        parts = path.parts
        if "images" in parts:
            i = len(parts) - 1 - parts[::-1].index("images")
            root, rel = Path(*parts[:i]), Path(*parts[i + 1:]) if i + 1 < len(parts) else Path()
            split = rel.as_posix().replace("/", "_") or "all"
            return cls(f"{root.name}/{rel.as_posix() or 'images'}", path, root / "labels" / rel, root / "labels_auto" / rel,
                       root / "review" / f"{split}.json", "yolo")
        return cls(path.name, path, path / "labels", path / "labels_auto", path / "review.json", "folder")

    @property
    def root(self) -> Path:
        return self.images_dir.parent if self.layout == "session" else self.images_dir

    @property
    def cache_dir(self) -> Path:
        """Where derived per-session data (interface-screen scores) is cached."""
        return self.root if self.layout != "yolo" else self.review_path.parent

    def frames(self) -> list[Path]:
        if not self.images_dir.is_dir():
            return []
        return sorted(p for p in self.images_dir.iterdir() if p.suffix.lower() in IMG_EXT)

    def label_path(self, stem: str) -> Path:
        return self.labels_dir / f"{stem}.txt"

    def auto_path(self, stem: str) -> Path:
        return self.auto_dir / f"{stem}.txt"

    def review(self, classes: list[str] | None = None) -> Review:
        return Review(self.review_path, classes)

    def auto_meta(self) -> dict:
        p = self.auto_dir / "_meta.json"
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except (OSError, ValueError):
            return {}


def session_overview(session: Session) -> dict[str, int]:
    """Frame count and review counts (pos / neg / skip / todo) of a session, without building its index; `damaged` is 1
    when its review.json cannot be read (the session then fails to open with the reason)."""
    try:
        return {"frames": len(session.frames()), "damaged": 0, **session.review().counts()}
    except ValueError:
        return {"frames": len(session.frames()), "damaged": 1, **{s: 0 for s in STATUSES}}


def is_session_dir(path: Path) -> bool:
    return (path / "frames").is_dir() or (path.is_dir() and any(p.suffix.lower() in IMG_EXT for p in path.iterdir()))


def discover_sessions(path: Path) -> list[Session]:
    """A session folder itself, or every session folder directly inside a root (e.g. datasets/collected), or every
    split of a YOLO dataset (images/<split>)."""
    path = Path(path)
    if not path.is_dir():
        return []
    if (path / "frames").is_dir():
        return [Session.open(path)]
    if (path / "images").is_dir():
        subs = sorted(p for p in (path / "images").iterdir() if p.is_dir())
        return [Session.open(p) for p in subs] if subs else [Session.open(path / "images")]
    subs = sorted(p for p in path.iterdir() if p.is_dir() and (p / "frames").is_dir())
    if subs:
        return [Session.open(p) for p in subs]
    if any(p.suffix.lower() in IMG_EXT for p in path.iterdir()):
        return [Session.open(path)]
    return []
