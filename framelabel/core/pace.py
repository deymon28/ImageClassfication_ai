"""How fast the decisions come, and how long what is left would take at that speed (shown under the progress bar)."""
from __future__ import annotations

import time
from collections import deque

WINDOW_S = 600.0   # the pace is measured over the last ten minutes
MIN_EVENTS = 10    # ... and only once there is something to measure
MIN_SPAN_S = 45.0


class Pace:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.events: deque[tuple[float, int]] = deque()

    def add(self, n: int = 1) -> None:
        if n != 0:  # a negative n takes back decisions that were undone
            self.events.append((self.clock(), n))
            self.trim()

    def reset(self) -> None:
        self.events.clear()

    def trim(self) -> None:
        now = self.clock()
        while self.events and now - self.events[0][0] > WINDOW_S:
            self.events.popleft()

    def per_minute(self) -> float | None:
        """Decisions per minute over the recent window, None while there is too little to tell."""
        self.trim()
        total = max(0, sum(n for _t, n in self.events))
        if total < MIN_EVENTS or not self.events:
            return None
        span = max(self.clock() - self.events[0][0], MIN_SPAN_S)
        return total / span * 60.0

    def minutes_left(self, left: int) -> float | None:
        rate = self.per_minute()
        return None if not rate or left <= 0 else left / rate


def format_minutes(m: float) -> str:
    if m < 1:
        return "under a minute"
    if m < 60:
        return f"{round(m)} min"
    h, rest = divmod(round(m), 60)
    return f"{h} h {rest:02d} min" if h < 24 else f"{h // 24} d {h % 24} h"


def pace_line(pace: Pace, left: int) -> str:
    """'42 per min, about 1 h 20 min left' (empty while the pace is unknown or nothing is left)."""
    rate, eta = pace.per_minute(), pace.minutes_left(left)
    if rate is None or eta is None:
        return ""
    return f"{rate:.0f} per min, about {format_minutes(eta)} left"
