"""Headless smoke tests of the window: open a session, triage a grid page, draw / edit / save boxes with the keyboard
and mouse handlers."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from framelabel.core.project import Project  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def close_windows(app):
    """Destroy the test's windows deterministically (and collect the garbage) before the next test: a window that is
    only freed by a later garbage collection can be deleted at an arbitrary moment and crash Qt."""
    yield
    import gc

    for w in QApplication.topLevelWidgets():
        w.close()
        w.deleteLater()
    app.processEvents()
    QApplication.sendPostedEvents(None, 0)  # QEvent.DeferredDelete
    gc.collect()


def wait_until(app, cond, timeout=15.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        app.processEvents()
        time.sleep(0.01)


def open_and_wait(app, w, path):
    w.open_target(path)
    wait_until(app, lambda: w.ws is not None and w.loader is None)


@pytest.fixture
def project(tmp_path, monkeypatch):
    # keep the settings (recent projects, geometry, keymap) away from the user's real ones (the registry)
    monkeypatch.setenv("FRAMELABEL_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    s = tmp_path / "session_20261008_100000"
    (s / "frames").mkdir(parents=True)
    (s / "labels_auto").mkdir()
    rng = np.random.default_rng(0)
    for i in range(8):
        cv2.imwrite(str(s / "frames" / f"1000{i:02d}_000.jpg"), rng.integers(0, 255, (180, 320, 3), dtype=np.uint8))
        (s / "labels_auto" / f"1000{i:02d}_000.txt").write_text("0 0.5 0.5 0.1 0.3 0.80" if i >= 6 else "")
    path = tmp_path / "p.yaml"
    Project.from_dict({"name": "t", "classes": ["enemy", "corpse"], "sources": [str(tmp_path)]}, tmp_path, path).save()
    return path, s


def test_grid_then_editor_flow(app, project):
    from framelabel.gui.app import MainWindow

    path, s = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    w.resize(1200, 800)
    w.show()
    app.processEvents()
    assert w.mode == "grid" and len(w.queue) == 6  # six frames without proposals go to the grid
    w.grid.cols, w.grid.rows = 3, 2
    w.grid.set_page(0)
    first = w.queue[0].stem
    w.grid.toggle(first, "people")
    w.grid.toggle(w.queue[1].stem, "skip")
    assert w.handle_key("Space")  # commit the only page -> the editor opens
    review = w.ws.review
    assert review.get(first) == "todo" and review.get(w.queue[1].stem) == "skip" if w.mode == "grid" else True
    assert w.mode == "edit"
    st = {k: v for k, v in w.ws.review.status.items()}
    assert sorted(st.values()).count("neg") == 4 and list(st.values()).count("todo") == 1
    # the edit queue: the marked frame and the two frames with proposals (skipped frames are re-checkable too)
    stems = [f.stem for f in w.queue]
    assert stems[0] == first and "100006_000" in stems and "100007_000" in stems
    c = w.canvas
    c.resize(640, 360)
    app.processEvents()
    QTest.mousePress(c, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(100, 100))
    QTest.mouseMove(c, QPoint(200, 220))
    QTest.mouseRelease(c, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(200, 220))
    assert len(c.boxes) == 1 and c.dirty
    assert w.handle_key("2")  # class corpse for the selected box
    assert c.boxes[0].cls == 1
    assert w.handle_key("Ctrl+Z") and c.boxes[0].cls == 0 and w.handle_key("Ctrl+Y") and c.boxes[0].cls == 1
    assert w.handle_key("Space")
    lab = (s / "labels" / f"{first}.txt").read_text().split()
    assert w.ws.review.get(first) == "pos" and lab[0] == "1"
    assert w.queue[w.idx].stem == w.queue[1].stem and c.boxes == []  # the skipped frame comes next in time order
    assert w.handle_key("N")  # next unreviewed: a frame with a model proposal
    cur = w.queue[w.idx].stem
    assert cur == "100006_000" and len(c.boxes) == 1 and c.boxes[0].conf == 0.8  # the next frame shows the model proposal
    assert w.handle_key("E")  # empty
    assert w.ws.review.get(cur) == "neg" and (s / "labels" / f"{cur}.txt").read_text() == ""
    assert w.handle_key("A") and w.queue[w.idx].stem == cur
    assert c.boxes == []  # reviewed as empty: the proposal is no longer shown
    assert w.handle_key("R") and len(c.boxes) == 1  # back to the model proposal
    w.handle_key("X")
    assert w.ws.review.get(cur) == "skip"
    w.close()


def test_russian_layout_keys_map_to_the_same_actions():
    from framelabel.gui.keys import key_name

    # Cyrillic U+0443 on the physical E key: Qt reports the Cyrillic key code, the virtual key is still VK_E (0x45)
    assert key_name(0x0443, 0x45, Qt.KeyboardModifier.NoModifier) == "E"
    assert key_name(Qt.Key.Key_Z, 0x5A, Qt.KeyboardModifier.ControlModifier) == "Ctrl+Z"
    assert key_name(Qt.Key.Key_Space, 0x20, Qt.KeyboardModifier.NoModifier) == "Space"
    assert key_name(Qt.Key.Key_2, 0x32, Qt.KeyboardModifier.NoModifier) == "2"
    assert key_name(Qt.Key.Key_Equal, 0xBB, Qt.KeyboardModifier.NoModifier) == "="
    assert key_name(Qt.Key.Key_Minus, 0xBD, Qt.KeyboardModifier.NoModifier) == "-"


def test_loupe_magnifies_the_view_smoothly():
    from framelabel.gui.canvas import LOUPE_MAX, loupe_window

    # whole 1920 px frame fitted into 1040 px: the loupe shows 3x what is on screen, not a fixed huge zoom
    x1, y1, x2, y2, k = loupe_window(960, 540, 1040 / 1920, 200, 3.0)
    assert np.isclose(k, 3 * 1040 / 1920) and np.isclose(x2 - x1, 200 / k) and np.isclose((x1 + x2) / 2, 960)
    # zooming the view 2x halves the image window (smooth, proportional)
    k2 = loupe_window(960, 540, 2 * 1040 / 1920, 200, 3.0)[4]
    assert np.isclose(k2, 2 * k)
    # capped when the view is already zoomed far in
    assert loupe_window(10, 10, 30.0, 200, 3.0)[4] == 40.0


def test_loupe_zoom_is_adjustable_and_bounded(app):
    from framelabel.gui.canvas import LOUPE_MAX, LOUPE_MIN, Canvas

    c = Canvas()
    seen = []
    c.loupe_changed.connect(seen.append)
    c.change_loupe_zoom(1.25)
    assert np.isclose(c.loupe_zoom, 3.75) and seen == [c.loupe_zoom]
    for _ in range(50):
        c.change_loupe_zoom(1.25)
    assert c.loupe_zoom == LOUPE_MAX
    for _ in range(50):
        c.change_loupe_zoom(0.8)
    assert c.loupe_zoom == LOUPE_MIN


def test_switching_sessions_quickly_loads_the_last_one(app, project):
    from framelabel.gui.app import MainWindow

    path, s = project
    s2 = s.parent / "session_20261008_110000"
    (s2 / "frames").mkdir(parents=True)
    cv2.imwrite(str(s2 / "frames" / "110000_000.jpg"), np.zeros((180, 320, 3), np.uint8))
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    w.on_session_changed(0)
    w.on_session_changed(1)
    wait_until(app, lambda: w.loader is None and w.ws is not None)
    assert w.ws.session.name == w.sessions[1].name == s2.name
    w.close()


def test_keys_during_loading_are_ignored(app, project):
    from framelabel.gui.app import MainWindow

    path, _ = project
    w = MainWindow()
    w.settings.clear()
    w.open_target(path)
    assert w.ws is None and w.loader is not None
    assert w.handle_key("Space") is False
    wait_until(app, lambda: w.loader is None and w.ws is not None)
    w.close()
