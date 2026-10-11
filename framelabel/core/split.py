"""Train/val splitting for video-like frame sequences, and negative selection.

Neighbouring frames are near-duplicates, so a random split leaks them into val. Frames are grouped into time
blocks instead, and whole blocks go to val. Frame names are a time of day (HHMMSS_mmm) inside a session named
session_YYYYmmdd_HHMMSS; keys are '<session>_<stem>'. Keys without a parsable time keep their sorted order and
are cut into blocks by position.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict

KEY_RE = re.compile(r"(.+)_(\d{2})(\d{2})(\d{2})_(\d{3})$")  # "<session>_<HHMMSS>_<mmm>"
SESSION_START_RE = re.compile(r"_(\d{2})(\d{2})(\d{2})$")  # session_YYYYmmdd_HHMMSS


def frame_time(key: str) -> tuple[str, float]:
    """Export key -> (session, seconds since the session start). Frame names are only a time of day, so a frame
    earlier in the day than the session start was taken after midnight. Keys without a time keep their own order."""
    m = KEY_RE.match(key)
    if not m:
        return key, 0.0
    t = int(m[2]) * 3600 + int(m[3]) * 60 + int(m[4]) + int(m[5]) / 1000
    if s := SESSION_START_RE.search(m[1]):
        t -= int(s[1]) * 3600 + int(s[2]) * 60 + int(s[3])
        if t < -60:
            t += 86400
    return m[1], t


def split_by_time(keys: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    """Time-sorted keys -> (train, val) with val the last contiguous chunk."""
    keys = sorted(keys, key=frame_time)
    n_val = round(len(keys) * val_fraction)
    return (keys[: len(keys) - n_val], keys[len(keys) - n_val:]) if n_val else (keys, [])


def time_blocks(keys: list[str], block_s: float = 120.0, gap_s: float = 30.0, untimed_block: int = 20) -> list[list[str]]:
    """Keys -> blocks of consecutive frames of one session (at most block_s long; a gap over gap_s starts a new
    one). Keys without a time form blocks of untimed_block consecutive names."""
    sessions: dict[str, list[tuple[float, str]]] = defaultdict(list)
    untimed: list[str] = []
    for k in keys:
        if KEY_RE.match(k):
            sess, t = frame_time(k)
            sessions[sess].append((t, k))
        else:
            untimed.append(k)
    blocks: list[list[str]] = []
    for sess in sorted(sessions):
        start = prev = None
        first = True
        for t, k in sorted(sessions[sess]):
            if first or t - start > block_s or t - prev > gap_s:
                blocks.append([])
                start, first = t, False
            blocks[-1].append(k)
            prev = t
    untimed.sort()
    blocks += [untimed[i:i + untimed_block] for i in range(0, len(untimed), untimed_block)]
    return blocks


def _session_of(block: list[str]) -> str:
    return frame_time(block[0])[0] if KEY_RE.match(block[0]) else "_untimed"


def split_by_blocks(keys: list[str], val_fraction: float, seed: int = 0, block_s: float = 120.0,
                    gap_s: float = 30.0) -> tuple[list[str], list[str]]:
    """Send a random share of each session's blocks, about val_fraction of its frames, to val. Whole blocks move
    together, so near-duplicate neighbouring frames stay on one side, while val still covers every part of every
    session instead of only the end of the last one."""
    rng = random.Random(seed)
    by_session: dict[str, list[list[str]]] = defaultdict(list)
    for b in time_blocks(keys, block_s, gap_s):
        by_session[_session_of(b)].append(b)
    train, val = [], []
    for sess in sorted(by_session):
        blocks = by_session[sess]
        target = sum(map(len, blocks)) * val_fraction
        order = list(range(len(blocks)))
        rng.shuffle(order)
        chosen, n = set(), 0
        for i in order:
            size = len(blocks[i])
            if n + size <= target or (target - n > n + size - target):  # take it unless it overshoots more than it fills
                chosen.add(i)
                n += size
        for i, b in enumerate(blocks):
            (val if i in chosen else train).extend(b)
    return train, val


def balanced_block_split(keys: list[str], box_classes: dict[str, list[int]], val_fraction: float, seed: int = 0,
                         block_s: float = 120.0, tries: int = 50, n_classes: int | None = None) -> tuple[list[str], list[str]]:
    """split_by_blocks with the block draw (out of `tries`) whose val share of every class's boxes is closest to
    val_fraction: objects like corpses stay in view for many frames, so one draw can put most of a class in val."""
    if n_classes is None:
        n_classes = 1 + max((c for cs in box_classes.values() for c in cs), default=-1)
    totals = [sum(c == cls for cs in box_classes.values() for c in cs) for cls in range(n_classes)]

    def score(val: list[str]) -> float:
        got = [sum(c == cls for k in val for c in box_classes.get(k, ())) for cls in range(n_classes)]
        return max(abs(g / t - val_fraction) for g, t in zip(got, totals) if t)

    if not any(totals):
        return split_by_blocks(keys, val_fraction, seed, block_s)
    return min((split_by_blocks(keys, val_fraction, seed * tries + i, block_s) for i in range(tries)),
               key=lambda tv: score(tv[1]))


def choose_negatives(negs: list[tuple[str, float]], n: int, seed: int = 0) -> list[str]:
    """Pick n negatives: half are the 'hardest' (highest model confidence = likely false positives), half random."""
    if n >= len(negs):
        return [k for k, _ in negs]
    ranked = sorted(negs, key=lambda kv: -kv[1])
    hard = [k for k, s in ranked[: n // 2] if s > 0]
    hard_set = set(hard)
    rest = [k for k, _ in ranked if k not in hard_set]
    random.Random(seed).shuffle(rest)
    return hard + rest[: n - len(hard)]
