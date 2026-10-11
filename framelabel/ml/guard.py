"""Safety checks before GPU work: the game must be closed (it crashed once from resource overuse), PyTorch VRAM is
capped, and other GPU compute processes are reported so heavy jobs are not run in parallel."""
from __future__ import annotations

import os
import subprocess

DEFAULT_GAME_PROCESSES = ("UAGame.exe",)


def running_processes(names) -> list[str]:
    import psutil

    want = {n.lower() for n in names}
    found = set()
    for p in psutil.process_iter(["name"]):
        n = (p.info["name"] or "").lower()
        if n in want:
            found.add(p.info["name"])
    return sorted(found)


def game_running(names=DEFAULT_GAME_PROCESSES) -> bool:
    return bool(running_processes(names))


def gpu_compute_processes(exclude_self: bool = True) -> list[tuple[int, str, str]]:
    """(pid, name, used memory) of processes doing CUDA compute, from nvidia-smi ([] if it is unavailable)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            pid = int(parts[0])
            if exclude_self and pid == os.getpid():
                continue
            rows.append((pid, os.path.basename(parts[1]), parts[2] if len(parts) > 2 else "?"))
    return rows


class GuardError(RuntimeError):
    pass


def check_gpu_allowed(game_processes=DEFAULT_GAME_PROCESSES, force: bool = False) -> None:
    """Raise GuardError when the game is running (unless force)."""
    found = running_processes(game_processes or DEFAULT_GAME_PROCESSES)
    if found and not force:
        raise GuardError(f"{', '.join(found)} is running: close the game first (GPU work can crash it).")


def cap_vram(max_gb: float, device: int = 0) -> None:
    import torch

    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_gb * 2**30 / total), device)
