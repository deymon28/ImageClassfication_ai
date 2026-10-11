"""Project file (YAML): classes, where the sessions are, models offered for auto-labeling, export settings.

Relative paths are resolved against the project file's folder. A minimal project is just a class list and one source:

    name: my project
    classes: [cat, dog]
    sources: [data/frames_2024]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .session import Session, discover_sessions

PALETTE = ("#ff2828", "#00beff", "#38d94a", "#ffd000", "#c040ff", "#ff8c00", "#00e0c0", "#ff60b0", "#a0a0ff", "#b0ff60")


@dataclass
class ClassDef:
    name: str
    color: str


@dataclass
class Project:
    name: str
    classes: list[ClassDef]
    sources: list[Path] = field(default_factory=list)
    ignore_zones: list[tuple[float, float, float, float]] = field(default_factory=list)
    ui_zone: tuple[float, float, float, float] | None = None
    ui_threshold: float = 0.3
    game_processes: list[str] = field(default_factory=list)
    model_dirs: list[Path] = field(default_factory=list)
    default_models: list[Path] = field(default_factory=list)
    class_map: dict[str, str] = field(default_factory=dict)  # model class name -> project class name
    export_out: Path | None = None
    export: dict = field(default_factory=dict)
    prelabel: dict = field(default_factory=dict)
    path: Path | None = None  # the project file

    @property
    def class_names(self) -> list[str]:
        return [c.name for c in self.classes]

    def color(self, cls: int) -> str:
        return self.classes[cls].color if 0 <= cls < len(self.classes) else "#ffffff"

    def sessions(self) -> list[Session]:
        out, seen = [], set()
        for src in self.sources:
            for s in discover_sessions(src):
                key = s.images_dir.resolve()
                if key not in seen:
                    seen.add(key)
                    out.append(s)
        return out

    def map_model_class(self, model_name: str, model_index: int, n_model_classes: int) -> int | None:
        """Project class for a model class: explicit class_map, then same name, then (one-class model) class 0."""
        target = self.class_map.get(model_name)
        if target is not None:
            return self.class_names.index(target) if target in self.class_names else None
        if model_name in self.class_names:
            return self.class_names.index(model_name)
        if n_model_classes == 1:
            return 0
        return None

    # ---------------- (de)serialization
    @classmethod
    def load(cls, path: Path) -> Project:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(data, path.parent, path)

    @classmethod
    def from_dict(cls, data: dict, base: Path, path: Path | None = None) -> Project:
        def p(v) -> Path:
            q = Path(str(v)).expanduser()
            return q if q.is_absolute() else (base / q).resolve()

        classes = []
        for i, c in enumerate(data.get("classes") or []):
            if isinstance(c, str):
                classes.append(ClassDef(c, PALETTE[i % len(PALETTE)]))
            else:
                classes.append(ClassDef(str(c["name"]), str(c.get("color", PALETTE[i % len(PALETTE)]))))
        if not classes:
            raise ValueError("project has no classes")
        ui = data.get("ui_detect") or {}
        exp = dict(data.get("export") or {})
        out = exp.pop("out", None)
        return cls(
            name=str(data.get("name") or (path.stem if path else "project")),
            classes=classes,
            sources=[p(s) for s in data.get("sources") or []],
            ignore_zones=[tuple(map(float, z)) for z in data.get("ignore_zones") or []],
            ui_zone=tuple(map(float, ui["zone"])) if ui.get("zone") else None,
            ui_threshold=float(ui.get("threshold", 0.3)),
            game_processes=[str(x) for x in data.get("game_processes") or []],
            model_dirs=[p(m) for m in data.get("model_dirs") or []],
            default_models=[p(m) for m in data.get("default_models") or []],
            class_map={str(k): str(v) for k, v in (data.get("class_map") or {}).items()},
            export_out=p(out) if out else None,
            export=exp,
            prelabel=dict(data.get("prelabel") or {}),
            path=path,
        )

    def to_dict(self) -> dict:
        base = self.path.parent if self.path else None

        def rel(q: Path) -> str:
            if base is not None:
                try:
                    return Path(q).resolve().relative_to(base.resolve()).as_posix()
                except ValueError:
                    import os

                    try:
                        return Path(os.path.relpath(q, base)).as_posix()
                    except ValueError:  # another drive
                        pass
            return Path(q).as_posix()

        d: dict = {"name": self.name, "classes": [{"name": c.name, "color": c.color} for c in self.classes],
                   "sources": [rel(s) for s in self.sources]}
        if self.ignore_zones:
            d["ignore_zones"] = [list(z) for z in self.ignore_zones]
        if self.ui_zone:
            d["ui_detect"] = {"zone": list(self.ui_zone), "threshold": self.ui_threshold}
        if self.game_processes:
            d["game_processes"] = list(self.game_processes)
        if self.model_dirs:
            d["model_dirs"] = [rel(m) for m in self.model_dirs]
        if self.default_models:
            d["default_models"] = [rel(m) for m in self.default_models]
        if self.class_map:
            d["class_map"] = dict(self.class_map)
        exp = dict(self.export)
        if self.export_out:
            exp["out"] = rel(self.export_out)
        if exp:
            d["export"] = exp
        if self.prelabel:
            d["prelabel"] = dict(self.prelabel)
        return d

    def save(self, path: Path | None = None) -> None:
        if path is not None:
            self.path = Path(path)
        if self.path is None:
            raise ValueError("no project path")
        self.path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def folder_project(folder: Path, classes: list[str] | None = None) -> Project:
    """An ad-hoc project for a folder opened without a project file; classes come from its review.json if any."""
    from .session import Review

    sessions = discover_sessions(folder)
    names = classes
    if not names:
        for s in sessions:
            if s.review_path.exists() and (r := Review(s.review_path)).classes:
                names = r.classes
                break
    names = names or ["object"]
    return Project(name=Path(folder).name, classes=[ClassDef(n, PALETTE[i % len(PALETTE)]) for i, n in enumerate(names)],
                   sources=[Path(folder).resolve()])
