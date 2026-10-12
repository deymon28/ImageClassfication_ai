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
    # the edit queue is work still to do: the marked frame and the two frames with proposals (no skipped / empty ones)
    stems = [f.stem for f in w.queue]
    assert stems == [first, "100006_000", "100007_000"]
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
    cur = w.queue[w.idx].stem  # the skipped frame is not work to do: the next one is the first with a proposal
    assert cur == "100006_000" and len(c.boxes) == 1 and c.boxes[0].conf == 0.8
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


def test_reload_picks_up_new_proposals_and_keeps_the_frame(app, project):
    from framelabel.gui.app import MainWindow

    path, s = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    w.set_mode("edit", queue="all")
    w.go(3)
    assert w.cur_path.stem == "100003_000" and w.canvas.boxes == []
    (s / "labels_auto" / "100003_000.txt").write_text("0 0.5 0.5 0.1 0.3 0.90")
    w.reload_session()
    wait_until(app, lambda: w.reloader is None)
    assert w.cur_path.stem == "100003_000" and w.idx == 3
    assert len(w.canvas.boxes) == 1 and w.canvas.boxes[0].conf == 0.9
    w.close()


class FakeFitter:
    delay = 0.0

    def __init__(self, weights):
        pass

    def load(self):
        pass

    def fit(self, img, box=None, point=None):
        from framelabel.core.boxes import Box

        time.sleep(FakeFitter.delay)
        return Box(10, 10, 50, 90, 0)


@pytest.fixture
def fake_sam(monkeypatch, tmp_path):
    import framelabel.ml.guard as guard
    import framelabel.ml.models as models
    import framelabel.ml.sam as sam

    monkeypatch.setattr(guard, "check_gpu_allowed", lambda *a, **k: None)
    monkeypatch.setattr(models, "find_sam", lambda dirs: tmp_path / "sam_fake.pt")
    monkeypatch.setattr(sam, "BoxFitter", FakeFitter)
    FakeFitter.delay = 0.0
    yield FakeFitter
    FakeFitter.delay = 0.0


def editor_with_box(app, project):
    from framelabel.core.boxes import Box
    from framelabel.gui.app import MainWindow

    path, _ = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    w.set_mode("edit", queue="all")
    w.canvas.resize(640, 360)
    w.canvas.add_box(Box(100, 100, 200, 160, 1))
    return w


def test_sam_worker_result_applies_to_the_same_box(app, project, fake_sam):
    w = editor_with_box(app, project)
    w.sam_fit(w.canvas.boxes[0], index=0)
    wait_until(app, lambda: not w.sam_pending)
    b = w.canvas.boxes[0]
    assert (b.x1, b.y1, b.x2, b.y2) == (10, 10, 50, 90) and b.cls == 1  # geometry from SAM, class of the old box
    assert w.canvas.history.can_undo
    w.close()


def test_sam_click_adds_a_box_and_a_dropped_result_changes_nothing(app, project, fake_sam):
    w = editor_with_box(app, project)
    w.sam_fit((120.0, 130.0))
    wait_until(app, lambda: not w.sam_pending)
    assert len(w.canvas.boxes) == 2
    fake_sam.delay = 0.3
    w.sam_fit(w.canvas.boxes[0], index=0)
    w.go(w.idx + 1)  # the frame changes before SAM answers
    wait_until(app, lambda: not w.sam_pending)
    assert w.canvas.boxes == [] and "dropped" in w.status_label.text()
    w.close()


def run_process_job(app, code):
    import sys

    from framelabel.gui.process_job import ProcessJob

    job = ProcessJob(["-c", code], sys.executable)
    seen = {"progress": [], "finished": [], "failed": []}
    job.progress.connect(lambda i, n, m: seen["progress"].append((i, n, m)))
    job.finished.connect(seen["finished"].append)
    job.failed.connect(seen["failed"].append)
    job.start()
    wait_until(app, lambda: seen["finished"] or seen["failed"])
    return seen


def test_process_job_parses_the_protocol(app):
    code = ("print('noise', flush=True);"
            "print('@@progress {\"done\": 4, \"total\": 10, \"msg\": \"a\"}', flush=True);"
            "print('@@progress {\"done\": 10, \"total\": 10, \"msg\": \"b\"}', flush=True);"
            "print('@@result {\"done\": 10, \"cancelled\": false}', flush=True)")
    seen = run_process_job(app, code)
    assert seen["progress"] == [(4, 10, "a"), (10, 10, "b")]
    assert seen["finished"] == [{"done": 10, "cancelled": False}] and not seen["failed"]


def test_process_job_reports_errors(app):
    seen = run_process_job(app, "import sys; print('@@error boom', flush=True); sys.exit(1)")
    assert seen["failed"] == ["boom"] and not seen["finished"]
    seen = run_process_job(app, "print('some text'); raise SystemExit(3)")
    assert "code 3" in seen["failed"][0] and "some text" in seen["failed"][0]


def test_process_job_cancel_file_stops_the_child(app):
    import sys

    from framelabel.gui.process_job import ProcessJob

    code = ("import os, time, sys\nf = sys.argv[1]\nwhile not os.path.exists(f): time.sleep(0.05)\n"
            "print('@@result {\"cancelled\": true}', flush=True)")
    job = ProcessJob(["-c", code, "{cancel_file}"], sys.executable)
    got = []
    job.finished.connect(got.append)
    job.start()
    app.processEvents()
    job.cancel()
    wait_until(app, lambda: got)
    assert got == [{"cancelled": True}] and not job.cancel_file.exists()


def test_export_dialog_runs_as_a_background_job(app, project, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from framelabel.gui.app import MainWindow
    from framelabel.gui.dialogs import ExportDialog

    path, s = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    for i in (6, 7):  # two labeled frames
        w.ws.save(s / "frames" / f"1000{i:02d}_000.jpg", [__import__("framelabel.core.boxes", fromlist=["Box"]).Box(5, 5, 60, 90)],
                  320, 180)
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    dlg = ExportDialog(w.project, w.sessions, w)
    out = path.parent / "exported"
    dlg.out.setText(str(out))
    dlg.accept()
    assert dlg.out_path == out and not out.exists()  # accept() only validates
    w.run_job(dlg.make_job(), "Exporting", lambda res: shown.append(res.summary(w.project.class_names)))
    wait_until(app, lambda: shown)
    assert w.job is None and (out / "data.yaml").exists() and "positives 2" in shown[0]
    w.close()


def test_frame_list_model_follows_the_queue(app, project):
    from PyQt6.QtCore import Qt

    from framelabel.gui.app import MainWindow

    path, _ = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    m = w.frame_model
    assert m.rowCount() == len(w.queue) == 6
    assert m.data(m.index(0)).endswith("[new]")
    w.set_mode("edit", queue="all")
    assert m.rowCount() == 8
    w.go(2)
    w.do("empty_next")  # the frame becomes empty, the list row follows
    assert "[empty]" in m.data(m.index(2)) and m.data(m.index(2), Qt.ItemDataRole.ForegroundRole).name() == "#8b93a1"
    assert w.frame_list.currentIndex().row() == 3  # the editor moved on
    w.close()


def test_to_qimage_keeps_colours():
    from framelabel.gui.canvas import to_qimage

    img = np.zeros((1, 2, 3), np.uint8)
    img[0, 0] = (255, 0, 0)  # BGR blue
    img[0, 1] = (0, 0, 255)  # BGR red
    q = to_qimage(img)
    assert q.pixelColor(0, 0).getRgb()[:3] == (0, 0, 255) and q.pixelColor(1, 0).getRgb()[:3] == (255, 0, 0)
    g = to_qimage(np.full((2, 2), 90, np.uint8))
    assert g.pixelColor(1, 1).getRgb()[:3] == (90, 90, 90)


def test_editor_prefetches_around_the_current_frame(app, project):
    from framelabel.gui.app import MainWindow

    path, s = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    w.set_mode("edit", queue="all")
    stems = lambda: sorted(p.stem for p in w.prefetch)  # noqa: E731
    assert w.idx == 0 and stems() == ["100001_000", "100002_000"]
    w.go(3)
    assert stems() == ["100002_000", "100004_000", "100005_000"]  # behind (kept), and two ahead
    w.go(2)
    img, qimg = w.canvas.img, w.canvas.qimg
    assert img.shape[:2] == (180, 320) and qimg.width() == 320 and w.cur_path.stem == "100002_000"
    w.close()


def test_grid_loads_tiles_in_the_background_and_debounces_resize(app, tmp_path):
    from framelabel.core.queues import FrameInfo
    from framelabel.gui.grid import Grid

    items = []
    for i in range(12):
        p = tmp_path / f"{i:06d}.jpg"
        cv2.imwrite(str(p), np.full((90, 160, 3), 20 * i, np.uint8))
        items.append(FrameInfo(p))
    g = Grid()
    g.cols, g.rows = 2, 2
    g.resize(400, 300)
    g.show()
    app.processEvents()
    wait_until(app, lambda: not g.resizing)
    g.set_items(items, 0)
    submits = []
    real = g.pool.submit
    g.pool.submit = lambda *a, **k: (submits.append(a[1:]), real(*a, **k))[1]
    g.repaint()  # asks for the page and its neighbours
    first = len(submits)
    assert 1 <= first <= 2  # page 0 and page 1 (there is no page -1)
    wait_until(app, lambda: g.tiles(*g.layout_now()[2:]) is not None)
    assert len(g.tiles(*g.layout_now()[2:])) == 4
    n = len(submits)
    for width in range(400, 500, 10):  # a resize drag: nothing is reloaded until it settles
        g.resize(width, 300)
        g.repaint()
    assert len(submits) == n and g.resizing
    wait_until(app, lambda: not g.resizing)
    g.repaint()
    assert n < len(submits) <= n + 3
    g.page = 1
    g.set_page(1)
    g.repaint()
    assert {k[0] for k in g.cache} <= {0, 1, 2}
    g.shutdown()


def test_process_job_through_run_job_cleans_up(app, project):
    import sys

    from framelabel.gui.app import MainWindow
    from framelabel.gui.process_job import ProcessJob

    path, _ = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    got = []
    code = "print('@@progress {\"done\": 1, \"total\": 2, \"msg\": \"x\"}', flush=True); print('@@result {\"ok\": 1}', flush=True)"
    w.run_job(ProcessJob(["-c", code], sys.executable), "Fake", got.append)
    assert w.job is not None and w.cancel_btn.isVisibleTo(w)
    wait_until(app, lambda: got)
    assert got == [{"ok": 1}] and w.job is None and not w.cancel_btn.isVisibleTo(w)
    for _ in range(20):  # the deferred delete of the finished job
        app.processEvents()
    w.close()


def test_closing_with_a_running_job_stops_it(app, project, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from framelabel.core.index_cache import Cancelled
    from framelabel.gui.app import MainWindow
    from framelabel.gui.workers import Job

    path, _ = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    started = []

    def long_job(progress, cancelled):
        started.append(1)
        while True:
            if cancelled():
                raise Cancelled()
            time.sleep(0.01)

    answers = iter([QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes])
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: next(answers))
    w.run_job(Job(long_job), "Long", lambda r: None)
    wait_until(app, lambda: started)
    w.show()
    assert w.close() is False and w.job is not None  # "No": keeps running
    assert w.close() is True  # "Yes": cancels and waits for the thread
    assert not w._reaper.jobs or all(not th.isRunning() for th, _ in w._reaper.jobs)


def test_resume_goto_and_reload_keys(app, project, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog

    from framelabel.gui.app import MainWindow

    path, s = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    w.set_mode("edit", queue="all")
    w.go(4)
    w.close()  # the position is remembered
    w2 = MainWindow()
    open_and_wait(app, w2, path)
    w2.set_mode("edit", queue="all")
    assert w2.cur_path.stem == "100004_000"  # not the first unreviewed frame
    answers = iter([("2", True), ("100006", True), ("999", True)])
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: next(answers))
    assert w2.handle_key("Ctrl+G") and w2.idx == 1
    assert w2.handle_key("Ctrl+G") and w2.cur_path.stem == "100006_000"
    assert w2.handle_key("Ctrl+G") and w2.cur_path.stem == "100006_000"  # unknown: stays
    (s / "labels_auto" / "100006_000.txt").write_text("0 0.5 0.5 0.1 0.3 0.40")
    assert w2.handle_key("F5")
    wait_until(app, lambda: w2.reloader is None)
    assert w2.cur_path.stem == "100006_000" and w2.canvas.boxes[0].conf == 0.4
    w2.close()


def test_frame_list_filter_maps_rows_back_to_the_queue(app, project):
    from framelabel.gui.app import MainWindow

    path, _ = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    w.set_mode("edit", queue="all")
    w.frame_filter.setText("100005")
    assert w.frame_proxy.rowCount() == 1 and w.frame_model.rowCount() == 8
    w.on_frame_clicked(w.frame_proxy.index(0, 0))
    assert w.cur_path.stem == "100005_000"  # the proxy row 0 is queue row 5
    assert w.frame_list.currentIndex().row() == 0
    w.go(1)  # a frame hidden by the filter: nothing to highlight, no error
    assert w.cur_path.stem == "100001_000"
    w.frame_filter.setText("")
    assert w.frame_proxy.rowCount() == 8
    w.close()


def test_session_box_shows_counts(app, project):
    from framelabel.gui.app import MainWindow

    path, s = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    wait_until(app, lambda: "frames" in w.session_box.itemText(0))
    assert w.session_box.itemText(0).startswith(s.name) and "(8 frames, 0 decided, 0 labeled)" in w.session_box.itemText(0)
    w.close()


def test_sam_fit_all_boxes(app, project, fake_sam):
    from framelabel.core.boxes import Box

    w = editor_with_box(app, project)
    w.canvas.add_box(Box(200, 20, 260, 90, 0))
    w.do("sam_fit_all")
    wait_until(app, lambda: not w.sam_pending)
    assert [(b.x1, b.y1, b.x2, b.y2, b.cls) for b in w.canvas.boxes] == [(10, 10, 50, 90, 1), (10, 10, 50, 90, 0)]
    w.close()


def test_performance_settings_are_applied(app, project):
    from framelabel.core import index_cache
    from framelabel.gui.app import MainWindow
    from framelabel.gui.dialogs import PerformanceDialog

    path, _ = project
    w = MainWindow()
    w.settings.clear()
    dlg = PerformanceDialog(w.settings, w)
    assert (dlg.io.value(), dlg.prefetch.value(), dlg.watchdog.isChecked(), dlg.preload.isChecked()) == (8, 2, True, False)
    dlg.io.setValue(3)
    dlg.prefetch.setValue(0)
    dlg.watchdog.setChecked(False)
    dlg.apply()
    w.apply_perf_settings()
    try:
        assert index_cache.IO_WORKERS == 3 and w.prefetch_ahead == 0 and not w.setting_bool("perf/watchdog", True)
        open_and_wait(app, w, path)
        w.set_mode("edit", queue="all")
        w.go(2)
        assert sorted(p.stem for p in w.prefetch) == ["100001_000"]  # only the frame behind
    finally:
        index_cache.IO_WORKERS = 8
    w.close()


def test_sam_preload_loads_the_model_after_the_session_opens(app, project, fake_sam):
    from framelabel.gui.app import MainWindow

    path, _ = project
    w = MainWindow()
    w.settings.clear()
    w.settings.setValue("perf/sam_preload", True)
    open_and_wait(app, w, path)
    wait_until(app, lambda: w.sam_worker is not None and w.sam_worker.fitter is not None)
    assert "SAM ready" in w.status_label.text() or w.sam_worker.fitter is not None
    w.close()


def test_sam_requests_are_served_by_one_python_thread(app, project, fake_sam):
    """Every request must run in the same long-lived Python thread: in a Qt slot thread the second torch call crashed
    the process (pybind11 caches the thread state of the calling thread)."""
    import threading

    seen = []
    real = FakeFitter.fit
    FakeFitter.fit = lambda self, *a, **k: (seen.append(threading.get_ident()), real(self, *a, **k))[1]
    try:
        w = editor_with_box(app, project)
        for _ in range(3):
            w.sam_fit(w.canvas.boxes[0], index=0)
            wait_until(app, lambda: not w.sam_pending)
    finally:
        FakeFitter.fit = real
    assert len(seen) == 3 and len(set(seen)) == 1 and seen[0] != threading.get_ident()
    assert isinstance(w.sam_worker.thread, threading.Thread) and w.sam_worker.thread.is_alive()
    w.close()
    assert w.sam_worker is None


def test_session_report_window_opens_the_view_of_a_row(app, project, monkeypatch):
    from PyQt6.QtWidgets import QDialog

    from framelabel.gui.app import MainWindow
    from framelabel.gui.dialogs import ReportDialog

    path, s = project
    (s / "capture.json").write_text('{"saved_by_reason": {"change": 8}, "skipped_by_reason": {"ui": 50}}')
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    assert "50 ui" in w.status_label.text() and "<b>8</b> new" in w.counts_label.text()
    seen = {}

    def fake_exec(self):  # the user double-clicks the "weak proposals" row, the report closes, the view opens
        seen["dlg"] = self
        row = next(r for r, v in self.views.items() if v == ("grid", "zone"))
        self.on_double_click(row, 0)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ReportDialog, "exec", fake_exec)
    w.session_report()
    text = seen["dlg"].as_text()
    assert "Recorder skipped: ui: 50" in text and "Proposals hidden by an ignore zone" in text
    assert w.mode == "grid" and w.view_key == "zone" and w.grid.show_ignored
    w.set_mode("grid", queue="triage")
    assert not w.grid.show_ignored
    w.close()


def make_grid(app, tmp_path, cols, rows, n=6):
    from framelabel.core.queues import FrameInfo
    from framelabel.gui.grid import Grid

    items = []
    for i in range(n):
        f = tmp_path / f"{i:06d}.jpg"
        cv2.imwrite(str(f), np.full((90, 160, 3), 30 * i, np.uint8))
        items.append(FrameInfo(f))
    g = Grid()
    g.cols, g.rows = cols, rows
    g.resize(640, 360)
    g.show()
    app.processEvents()
    wait_until(app, lambda: not g.resizing)
    g.set_items(items, 0)
    return g


def tile_center(g, i):
    from PyQt6.QtCore import QPoint

    x0, y0, tw, th = g.layout_now()
    from framelabel.gui.grid import GAP

    return QPoint(int(x0 + (i % g.cols) * (tw + GAP) + tw / 2), int(y0 + (i // g.cols) * (th + GAP) + th / 2))


def test_right_button_hold_peeks_and_click_toggles(app, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    from framelabel.gui.grid import PEEK_HOLD_S

    g = make_grid(app, tmp_path, 3, 2)
    RMB, NO = Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier
    pos = tile_center(g, 1)
    # hold: opens on press, closes on release
    QTest.mousePress(g, RMB, NO, pos)
    assert g.preview is not None and g.preview.stem == "000001"
    time.sleep(PEEK_HOLD_S + 0.1)
    QTest.mouseRelease(g, RMB, NO, pos)
    assert g.preview is None
    # quick click: stays open; the next right click closes it
    QTest.mouseClick(g, RMB, NO, pos)
    assert g.preview is not None and g.preview.stem == "000001"
    QTest.mousePress(g, RMB, NO, pos)
    assert g.preview is None
    QTest.mouseRelease(g, RMB, NO, pos)  # the release of the closing click must not reopen or break anything
    assert g.preview is None and g.peek_since is None
    # a left click while peeking marks the frame (the preview closes as before), the right release is harmless
    QTest.mousePress(g, RMB, NO, pos)
    QTest.mouseClick(g, Qt.MouseButton.LeftButton, NO, pos)
    assert g.marks.get("000001") == "people" and g.preview is None
    QTest.mouseRelease(g, RMB, NO, pos)
    assert g.preview is None
    g.shutdown()


def test_grid_sizes_1x1_2x1_1x2(app, project):
    from framelabel.gui.app import MainWindow

    path, _ = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    w.resize(1400, 900)
    w.show()
    from PyQt6.QtGui import QAction

    labels = [a.text() for a in w.findChildren(QAction)]
    for cols, rows in ((1, 1), (2, 1), (1, 2)):
        w.set_grid_size(cols, rows)
        g = w.grid
        assert (g.cols, g.rows, g.per_page) == (cols, rows, cols * rows)
        assert g.n_pages == -(-len(w.queue) // (cols * rows))
        g.repaint()
        wait_until(app, lambda: g.tiles(*g.layout_now()[2:]) is not None)
        x0, y0, tw, th = g.layout_now()
        assert tw > 100 and th > 50 and x0 >= 0 and y0 >= 0  # one big tile fits the area
        last = tile_center(g, min(cols * rows, len(w.queue)) - 1)  # every tile of the page can be hit
        assert g.tile_at(tile_center(g, 0).x(), tile_center(g, 0).y()) == 0
        assert g.tile_at(last.x(), last.y()) == cols * rows - 1
    assert any("1 × 1" in t for t in labels) and any("2 × 1" in t for t in labels) and any("1 × 2" in t for t in labels)
    assert w.settings.value("grid") == "1x2"
    w.close()


def test_todo_views_only_list_frames_without_a_decision(app, project):
    from framelabel.gui.app import MainWindow

    path, s = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    first, second = w.queue[0].stem, w.queue[1].stem
    w.grid.toggle(first, "people")
    w.grid.toggle(second, "skip")
    assert w.handle_key("Space")  # commit the page: one marked, one skipped, the rest empty
    assert w.mode == "edit" and w.view_key == "edit"  # triage is done: the editor opens with the work that is left
    assert [f.stem for f in w.queue] == [first, "100006_000", "100007_000"]  # marked + new proposals; no empty / skipped
    # coming back later (reopening, switching views): the decided frames are gone from the to-do views, nothing is lost
    w.set_mode("grid", queue="triage")
    assert w.queue == [] and "Nothing in" in w.grid.message and "decision" in w.grid.message
    w.set_mode("grid", queue="skip")
    assert [f.stem for f in w.queue] == [second]  # the Done views show the decisions
    w.set_mode("grid", queue="all")
    assert len(w.queue) == 8
    w.set_mode("edit", queue="edit")
    assert [f.stem for f in w.queue] == [first, "100006_000", "100007_000"]
    w.close()


def test_views_have_counts_and_the_current_one_is_highlighted(app, project):
    from framelabel.gui.app import MainWindow

    path, s = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    wait_until(app, lambda: bool(w.counts_by_view))
    c = w.counts_by_view
    assert c["triage"] == 6 and c["edit"] == 2 and c["all"] == 8 and c["pos"] == 0 and c["unreviewed"] == 8
    tree = w.view_tree
    assert tree.items["triage"].text(1) == "6" and tree.current_key() == "triage"
    # headings are not views; classes get a view each
    assert all(not (g.flags() & Qt.ItemFlag.ItemIsSelectable) for g in tree.groups.values())
    assert "class:0" in tree.items and "class:1" in tree.items
    # the context bar says where you are and what to do
    bar = w.context_bar
    assert bar.title.text() == "Triage: model found nothing" and "page 1 of 1" in bar.where.text()
    assert "saved as empty" in bar.hint.text()
    # a decision updates the counts (debounced) and the progress
    w.grid.toggle(w.queue[0].stem, "skip")
    w.handle_key("Space")
    wait_until(app, lambda: w.counts_by_view.get("skip") == 1)
    assert w.done_bar.value() == 6 and "6 of 8 decided" in w.done_bar.text()
    w.close()


def test_mode_and_view_are_independent_and_the_mode_is_remembered_per_view(app, project):
    from framelabel.gui.app import MainWindow

    path, s = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    w.on_view_chosen("skip")  # a Done view opens the way it suits best: a grid of tiles
    assert w.mode == "grid" and w.view_key == "skip" and w.view_tree.current_key() == "skip"
    w.on_mode_chosen("edit")  # the switch changes how it is shown, not what
    assert w.mode == "edit" and w.view_key == "skip" and w.mode_switch.buttons["edit"].isChecked()
    assert w.stack.currentWidget() is w.canvas and w.action_pages.currentIndex() == 1
    w.on_view_chosen("neg")
    assert w.mode == "grid" and w.view_key == "neg" and w.mode_switch.buttons["grid"].isChecked()
    w.on_view_chosen("skip")
    assert w.mode == "edit"  # last time you looked at 'skipped' in the editor
    w.on_view_chosen("pos")
    w.handle_key("V")  # next view that has frames: the empty ones are passed over
    assert w.view_key in w.view_tree.items and w.view_key != "pos"
    w.close()


def test_one_by_one_grid_is_the_photo_itself(app, tmp_path):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest

    NO, LMB, RMB = Qt.KeyboardModifier.NoModifier, Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton
    g = make_grid(app, tmp_path, 1, 1)
    assert g.sticky and g.preview is not None and not g.modal_preview  # no right click needed
    pos = QPoint(g.width() // 2, g.height() // 2)
    stem = g.preview.stem
    QTest.mouseClick(g, LMB, NO, pos)
    assert g.marks.get(stem) == "people" and g.preview is not None  # the click marks and the photo stays
    QTest.mouseClick(g, LMB, Qt.KeyboardModifier.ControlModifier, pos)
    assert g.marks.get(stem) == "skip" and g.preview is not None
    QTest.mouseClick(g, RMB, NO, pos)
    assert g.preview is not None  # the right button does not close it
    z = g.preview_view.zoom
    g.preview_view.zoom_at(pos.x(), pos.y(), 2.0)
    assert g.preview_view.zoom > z
    g.set_page(1)  # the next page shows the next photo, fitted again
    assert g.preview is not None and g.preview.stem != stem and abs(g.preview_view.zoom - 1.0) < 1e-6
    g.cols, g.rows = 2, 2  # with several tiles the preview is a peek again
    g.set_page(0)
    assert not g.sticky and g.preview is None
    g.shutdown()


def test_actions_confirm_themselves_and_the_confidence_applies_at_once(app, project):
    from framelabel.gui.app import MainWindow

    path, s = project
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    w.resize(1300, 800)
    w.show()
    w.set_mode("edit", queue="all")
    w.do("empty_next")
    assert "empty" in w.central.toast.text().lower() and w.central.toast.isVisible()
    w.do("skip_next")
    assert "skipped" in w.central.toast.text().lower()
    # the shown confidence rebuilds the view: 0.80 proposals are below 0.9, so they join the 'found nothing' views
    w.set_mode("grid", queue="triage")
    assert len(w.queue) == 4  # six frames found nothing, two of them are decided now
    w.conf_spin.setValue(0.9)
    w.apply_conf()
    assert len(w.queue) == 6  # the two frames with a 0.80 proposal come in
    w.close()


def test_every_action_has_a_readable_name_and_the_help_lists_the_keys(app):
    from framelabel.gui.dialogs import HelpDialog, KeymapDialog, ordered_actions
    from framelabel.gui.keys import ACTION_INFO, DEFAULT_KEYMAP, Keymap

    assert set(DEFAULT_KEYMAP) <= set(ACTION_INFO)  # no raw identifiers like accept_next in the UI
    assert sorted(ordered_actions()) == sorted(DEFAULT_KEYMAP)
    html = HelpDialog.html(Keymap({"empty_next": ["F9"]}))
    assert "F9" in html and "Space" in html and "Photo" not in html and "1 × 1" in html
    d = KeymapDialog(Keymap({"empty_next": ["F9"]}))
    assert d.overrides() == {"empty_next": ["F9"]}


def test_the_shown_confidence_you_chose_survives_a_restart(app, project):
    """The threshold decides which frames are in the grid ("model found nothing") and which in the editor list. It used
    to jump back to the project's show_conf on every start, which moved reviewed-away frames back into the grid."""
    from framelabel.core.project import Project
    from framelabel.gui.app import MainWindow

    path, _ = project
    proj = Project.load(path)
    proj.prelabel = {"show_conf": 0.2}
    proj.save()
    w = MainWindow()
    w.settings.clear()
    open_and_wait(app, w, path)
    assert abs(w.conf_spin.value() - 0.2) < 1e-9  # first start: the project's recommendation
    w.conf_spin.setValue(0.1)  # you choose 0.10 and review with it
    w.settings.sync()
    w.close()
    w2 = MainWindow()  # next start: opens the last project by itself
    wait_until(app, lambda: w2.ws is not None and w2.loader is None)
    assert abs(w2.conf_spin.value() - 0.1) < 1e-9 and abs(w2.ws.conf - 0.1) < 1e-9  # not back to 0.2
    assert w2.mode == "grid"
    w2.update_status()
    assert "shown conf 0.10" in w2.status_label.text()  # the threshold is visible next to the grid view
    w2.close()


def test_theme_builds_a_style_sheet_with_its_icons(app, tmp_path):
    from framelabel.gui.theme import T, apply_theme, stylesheet, write_icons

    icons = write_icons()  # FRAMELABEL_HOME points into tmp_path (conftest)
    assert set(icons) == {"check", "down", "up", "right", "branch_down"}
    assert all(__import__("pathlib").Path(p).read_text(encoding="utf-8").startswith("<svg") for p in icons.values())
    css = stylesheet(icons)
    assert T["accent"] in css and "QPushButton" in css and "{{" not in css and "}}" not in css  # no unformatted braces
    apply_theme(app)
    assert app.styleSheet() == css
    app.setStyleSheet("")  # the other tests run with the plain style


def test_grid_and_context_bar_paint_with_the_theme(app, tmp_path):
    from framelabel.gui.panels import ContextBar

    g = make_grid(app, tmp_path, 2, 2)
    g.toggle(g.items[0].stem, "people")
    g.toggle(g.items[1].stem, "skip")
    g.hover = 2
    img = g.grab().toImage()  # marked, skipped, hovered and plain tiles all paint without errors
    assert not img.isNull() and img.width() > 100
    bar = ContextBar()
    bar.set_text("A <b> view", "Grid  |  page 1 of 3  |  9 frames", "hint")
    assert "page 1 of 3" in bar.where.text() and bar.title.text() == "A <b> view"
    g.shutdown()
