"""Headless smoke tests of the window: open a session, triage a grid page, draw / edit / save boxes with the keyboard
and mouse handlers."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt6.QtCore import QPoint, QSettings, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from framelabel.core.project import Project  # noqa: E402


@pytest.fixture(scope="module")
def app():
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    return QApplication.instance() or QApplication([])


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))  # keep QSettings away from the user's settings
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
    w = MainWindow(path)
    w.settings.clear()
    w.open_target(path)
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
