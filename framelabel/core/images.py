"""Reading frames from disk. cv2.imread cannot open a path with non-ASCII characters on Windows (a folder named in
Cyrillic, say): it returns None as if the file were broken. The bytes are read by Python and decoded by OpenCV
instead, which takes the same flags (the IMREAD_REDUCED_* ones included) and also releases the GIL."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: Path | str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """cv2.imread for any path: the decoded image, or None when the file is missing, unreadable or not an image."""
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    if not data.size:
        return None
    return cv2.imdecode(data, flags)
