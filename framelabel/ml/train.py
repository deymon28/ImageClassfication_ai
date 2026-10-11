"""Train or fine-tune a labeler model (ultralytics). Run with the game closed: it uses the GPU fully.

  python -m framelabel.ml.train --preset abi-person        # stage 1: YOLO26x, one class 'person', public ABI + own train
  python -m framelabel.ml.train --preset abi-own           # stage 2: enemy/corpse fine-tune on own frames
  python -m framelabel.ml.train --data D --model M --name N [--imgsz ...]   # any project (the app's "Retrain")
  python -m framelabel.ml.train --resume --name N

Refuses to start while the game is running or another training of this app is active, caps PyTorch VRAM
(--max-vram-gb) and keeps dataloader workers low (on Windows every worker is a full process of ~1.5 GB).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PRESETS = {
    "abi-person": dict(model=ROOT / "models/yolo26x.pt", data=ROOT / "datasets/abi_person/data.yaml", name="abi_person_y26x",
                       args=dict(epochs=60, patience=15, imgsz=640, batch=16, cos_lr=True, close_mosaic=10)),
    "abi-own": dict(model=ROOT / "train_runs/abi_person_y26x/weights/best.pt", data=ROOT / "datasets/abi_own/data.yaml",
                    name="abi_own_y26x",
                    args=dict(epochs=80, patience=25, imgsz=1280, batch=4, optimizer="AdamW", lr0=0.0005,
                              warmup_epochs=2, cos_lr=True, close_mosaic=15, freeze=10)),
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--data", type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--runs", type=Path, default=ROOT / "train_runs", help="output folder (default train_runs)")
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--freeze", type=int)
    ap.add_argument("--lr0", type=float)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true", help="continue the run --name from its last.pt")
    ap.add_argument("--max-vram-gb", type=float, default=13.0, help="PyTorch VRAM cap (default 13 of 16)")
    ap.add_argument("--force", action="store_true", help="train even if the game is running")
    ap.add_argument("--game", nargs="*", default=["UAGame.exe"], help="game process names that block training")
    args = ap.parse_args(argv)

    from .guard import GuardError, cap_vram, check_gpu_allowed

    try:
        check_gpu_allowed(args.game, args.force)
    except GuardError as e:
        sys.exit(str(e))
    cfg = dict(PRESETS[args.preset]) if args.preset else {"args": {}}
    cfg["args"] = dict(cfg["args"])
    name = args.name or cfg.get("name")
    if not name:
        sys.exit("--name is required without --preset")
    run = args.runs / name
    lock = args.runs / f".{name}.lock"
    args.runs.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            import psutil

            pid = int(lock.read_text())
            if psutil.pid_exists(pid):
                sys.exit(f"run {name} is already training (pid {pid})")
        except (ValueError, OSError):
            pass
    import os

    lock.write_text(str(os.getpid()))
    try:
        cap_vram(args.max_vram_gb)
        from ultralytics import YOLO

        if args.resume:
            YOLO(str(run / "weights/last.pt")).train(resume=True, workers=args.workers)
            return
        model = args.model or cfg.get("model")
        data = args.data or cfg.get("data")
        if not model or not data:
            sys.exit("--model and --data are required without --preset")
        over = {k: v for k, v in (("imgsz", args.imgsz), ("epochs", args.epochs), ("batch", args.batch),
                                  ("freeze", args.freeze), ("lr0", args.lr0), ("patience", args.patience)) if v is not None}
        train_args = {"epochs": 50, "patience": 15, "imgsz": 1280, "batch": 4, "cos_lr": True, **cfg["args"], **over}
        print(f"training {model} on {data} -> {run}  {train_args}", flush=True)
        YOLO(str(model)).train(data=str(data), workers=args.workers, device=0, project=str(args.runs), name=name,
                               exist_ok=True, plots=False, **train_args)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
