"""framelabel: dataset labeling app.

  python -m framelabel [PROJECT.yaml | FOLDER]          open the labeling window
  python -m framelabel prelabel TARGET [--models M ...]  model proposals into labels_auto/ (GPU, game closed)
  python -m framelabel export PROJECT [--out DIR]        YOLO dataset from reviewed frames
  python -m framelabel eval --data DATA.yaml --setup NAME=MODEL[+MODEL][@IMGSZ][:tta] ...
  python -m framelabel train ...                         see python -m framelabel.ml.train -h

TARGET is a project file or a session folder (or a folder of sessions).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_target(target: Path):
    from framelabel.core.project import Project, folder_project

    return Project.load(target) if target.suffix.lower() in (".yaml", ".yml") else folder_project(target)


def cmd_prelabel(a) -> None:
    from framelabel.ml.guard import GuardError, check_gpu_allowed
    from framelabel.ml.models import model_class_names
    from framelabel.ml.predict import PredictConfig, prelabel_sessions

    machine = a.progress_json  # machine-readable lines for the labeling window: @@progress / @@result / @@error

    def fail(msg: str) -> None:
        if machine:
            print(f"@@error {msg}", flush=True)
        sys.exit(msg if not machine else 1)

    try:
        proj = load_target(a.target)
        try:
            check_gpu_allowed(proj.game_processes or ["UAGame.exe"], a.force)
        except GuardError as e:
            fail(str(e))
        models = a.models or proj.default_models
        if not models:
            fail("no model: pass --models or set default_models in the project")
        sessions = proj.sessions()
        if a.session:
            sessions = [s for s in sessions if s.name in a.session]
        pre = proj.prelabel
        tta = a.tta if a.tta is not None else bool(pre.get("tta", False))
        cfg = PredictConfig(models=[Path(m) for m in models], imgsz=a.imgsz or int(pre.get("imgsz", 1280)),
                            conf=a.conf if a.conf is not None else float(pre.get("conf", 0.05)),
                            tta=tta, max_vram_gb=a.max_vram_gb, batch=a.batch)
        for m in cfg.models:
            names = model_class_names(str(m))
            cmap = {i: proj.map_model_class(n, i, len(names)) for i, n in enumerate(names)}
            cfg.class_maps.append(cmap)
            print(f"{m.name}: " + ", ".join(f"{n} -> {proj.class_names[cmap[i]] if cmap[i] is not None else 'dropped'}"
                                            for i, n in enumerate(names)), flush=True)
        print(f"{len(sessions)} sessions, imgsz {cfg.imgsz}, conf >= {cfg.conf}, tta {cfg.tta}", flush=True)
        if machine:
            def progress(i, n, msg):
                print("@@progress " + json.dumps({"done": i, "total": n, "msg": msg}), flush=True)
        else:
            def progress(i, n, msg):
                if i % 200 < cfg.batch or i == n:
                    print(f"  {i}/{n} {msg}", flush=True)
        cancel_file = a.cancel_file
        stats = prelabel_sessions(sessions, cfg, a.redo, progress,
                                  (lambda: cancel_file.exists()) if cancel_file else None, proj.ignore_zones)
    except SystemExit:
        raise
    except Exception as e:
        if not machine:
            raise
        import traceback

        traceback.print_exc()
        fail(f"{type(e).__name__}: {e}")
    if machine:
        print("@@result " + json.dumps(stats), flush=True)
    print(f"done: {stats}")


def cmd_export(a) -> None:
    from framelabel.core.export import export_dataset

    proj = load_target(a.target)
    sessions = proj.sessions()
    if a.session:
        sessions = [s for s in sessions if s.name in a.session]
    exp = proj.export
    out = a.out or proj.export_out
    if out is None:
        sys.exit("no output folder: pass --out or set export.out in the project")
    try:
        res = export_dataset(sessions, proj.class_names, out,
                             a.neg_ratio if a.neg_ratio is not None else exp.get("neg_ratio", 0.15),
                             a.val_fraction if a.val_fraction is not None else exp.get("val_fraction", 0.15),
                             a.seed if a.seed is not None else exp.get("seed", 0), a.split or exp.get("split", "blocks"),
                             a.block_s or exp.get("block_s", 120.0), proj.ignore_zones)
    except (ValueError, FileExistsError) as e:  # nothing labeled yet, or a folder that is not an exported dataset
        sys.exit(f"export: {e}")
    print(res.summary(proj.class_names))


def parse_setup(text: str) -> tuple[str, list[Path], int | None, bool]:
    """'name=a.pt+b.pt@1536:tta' -> (name, models, imgsz, tta)."""
    name, _, spec = text.partition("=")
    if not spec:
        name, spec = Path(text).stem, text
    tta = spec.endswith(":tta")
    spec = spec[:-4] if tta else spec
    imgsz = None
    if "@" in spec:
        spec, _, sz = spec.rpartition("@")
        imgsz = int(sz)
    return name, [Path(m) for m in spec.split("+")], imgsz, tta


def cmd_eval(a) -> None:
    from framelabel.core.project import Project
    from framelabel.ml import evaluate as ev
    from framelabel.ml.guard import GuardError, check_gpu_allowed

    proj = Project.load(a.project) if a.project else None
    try:
        check_gpu_allowed((proj.game_processes if proj else None) or ["UAGame.exe"], a.force)
    except GuardError as e:
        sys.exit(str(e))
    zones = proj.ignore_zones if proj else []
    split = ev.load_split(a.data, a.split)
    report = {"data": str(a.data), "split": a.split, "ignore_zones": zones, "setups": {}}
    for text in a.setup:
        name, models, imgsz, tta = parse_setup(text)
        imgsz = imgsz or a.imgsz
        print(f"{name}: {' + '.join(m.name for m in models)} @ {imgsz}{' tta' if tta else ''}", flush=True)
        preds = ev.predict_split(split, models, imgsz, tta, a.batch, a.max_vram_gb)
        res = ev.evaluate_predictions(split, preds, zones)
        res.update({"models": [str(m) for m in models], "imgsz": imgsz, "tta": tta})
        report["setups"][name] = res
        print(ev.format_report(name, res), flush=True)
    out = a.out or Path("runs") / f"eval_{ev.timestamp()}.json"
    ev.save_report(out, report)
    print(f"-> {out}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("prelabel", "export", "eval", "train", "-h", "--help"):
        from framelabel.gui.app import run_app

        sys.exit(run_app(Path(argv[0]) if argv else None))
    if argv[0] == "train":
        from framelabel.ml.train import main as train_main

        return train_main(argv[1:])
    from framelabel.perf import setup_logging

    setup_logging(to_file=False)  # the window may be running and owns framelabel.log
    a = build_parser().parse_args(argv)
    {"prelabel": cmd_prelabel, "export": cmd_export, "eval": cmd_eval}[a.cmd](a)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="framelabel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prelabel", help="model proposals for every frame (GPU, game closed)")
    p.add_argument("target", type=Path)
    p.add_argument("--models", type=Path, nargs="+", help="one model, or several for an ensemble")
    p.add_argument("--session", nargs="*", help="only these session names")
    p.add_argument("--imgsz", type=int)
    p.add_argument("--conf", type=float, help="lowest confidence stored (default 0.05)")
    p.add_argument("--tta", action=argparse.BooleanOptionalAction, default=None,
                   help="test-time augmentation (default: the project's prelabel.tta)")
    p.add_argument("--redo", action="store_true", help="re-run frames that already have proposals")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-vram-gb", type=float, default=8.0)
    p.add_argument("--force", action="store_true", help="run even if the game is running")
    p.add_argument("--progress-json", action="store_true", help="print @@progress / @@result / @@error lines")
    p.add_argument("--cancel-file", type=Path, help="stop after the current batch when this file exists")
    e = sub.add_parser("export", help="YOLO dataset from reviewed frames")
    e.add_argument("target", type=Path)
    e.add_argument("--session", nargs="*")
    e.add_argument("--out", type=Path)
    e.add_argument("--neg-ratio", type=float)
    e.add_argument("--val-fraction", type=float)
    e.add_argument("--split", choices=("blocks", "time"))
    e.add_argument("--block-s", type=float)
    e.add_argument("--seed", type=int)
    v = sub.add_parser("eval", help="compare labeler setups on a labeled split")
    v.add_argument("--data", type=Path, required=True)
    v.add_argument("--split", default="val")
    v.add_argument("--setup", nargs="+", required=True, help="NAME=MODEL[+MODEL][@IMGSZ][:tta]")
    v.add_argument("--imgsz", type=int, default=1280)
    v.add_argument("--project", type=Path, help="use this project's ignore zones (drop proposals there)")
    v.add_argument("--batch", type=int, default=4)
    v.add_argument("--max-vram-gb", type=float, default=8.0)
    v.add_argument("--out", type=Path)
    v.add_argument("--force", action="store_true")
    return ap


if __name__ == "__main__":
    main()
