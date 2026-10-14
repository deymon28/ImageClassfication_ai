import json

import cv2
import numpy as np
import pytest

from framelabel.core.boxes import Box, box_at, corner_at, edge_at, parse_rows, read_yolo, yolo_lines
from framelabel.core.export import export_dataset, max_conf
from framelabel.core.history import History
from framelabel.core.project import Project, folder_project
from framelabel.core.propagate import propagate, snap_to_proposals, track_box
from framelabel.core.queues import build_index, build_queue, disagreement, uncertainty
from framelabel.core.session import Review, Session, discover_sessions
from framelabel.core.split import balanced_block_split, choose_negatives, frame_time, split_by_blocks, split_by_time
from framelabel.core.uiscreen import ui_scores
from framelabel.core.view import View
from framelabel.core.workspace import Workspace

HUD = [(0.0, 0.75, 0.07, 0.92)]


def test_yolo_roundtrip_clips_and_drops_degenerate(tmp_path):
    boxes = [Box(110, 50, 10, 250, cls=1), Box(-20, 900, 40, 1200), Box(5, 5, 5.5, 50)]
    lines = yolo_lines(boxes, 1920, 1080)
    assert len(lines) == 2  # the 0.5 px wide box is dropped
    f = tmp_path / "a.txt"
    f.write_text("\n".join(lines))
    back = read_yolo(f, 1920, 1080)
    assert back[0].cls == 1 and np.allclose(back[0].xyxy(), [10, 50, 110, 250], atol=0.01)
    assert np.allclose([back[1].x1, back[1].y2], [0, 1080], atol=0.01)  # clipped to the image


def test_read_proposals_filters_by_conf_and_zone(tmp_path):
    f = tmp_path / "auto.txt"
    f.write_text("0 0.5 0.5 0.1 0.2 0.80\n0 0.2 0.2 0.05 0.1 0.12")
    assert [b.conf for b in read_yolo(f, 100, 100, 0.3)] == [0.8]
    assert max_conf(f) == 0.8 and max_conf(tmp_path / "missing.txt") == 0.0
    hud = tmp_path / "hud.txt"  # a proposal on the body-status icon is ignored
    hud.write_text("0 0.037 0.84 0.05 0.18 0.90")
    assert read_yolo(hud, 100, 100, 0.3, HUD) == [] and max_conf(hud, HUD) == 0.0
    hud.write_text("0 0.037 0.84 0.05 0.18")  # but a reviewed label there is kept
    assert len(read_yolo(hud, 100, 100, 0.3, HUD)) == 1


def test_polygon_rows_become_boxes():
    (row,) = parse_rows("2 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6")
    assert row[0] == 2 and np.allclose(row[1:5], [0.2, 0.4, 0.2, 0.4]) and row[5] is None


def test_bad_rows_cost_only_themselves(tmp_path):
    rows = parse_rows("class cx cy w h\n0 0.5 0.5 0.1 0.1\n1 0.2 nan 0.1 0.1\n1 0.3 0.3 0.1\n0 0.5 0.5 0.1 0.1 0.9")
    assert [r[0] for r in rows] == [0, 0] and rows[1][5] == 0.9
    (frames := tmp_path / "s" / "frames").mkdir(parents=True)
    cv2.imwrite(str(frames / "a.jpg"), np.zeros((20, 20, 3), np.uint8))
    s = Session.open(tmp_path / "s")
    s.auto_dir.mkdir()
    s.auto_path("a").write_bytes(b"\xff\xfe garbage\n0 0.5 0.5 0.2 0.2 0.8")
    assert build_index(s, s.review())[0].confs == [0.8]  # the session opens, the good row is kept
    assert len(read_yolo(s.auto_path("a"), 20, 20)) == 1


def test_frames_in_a_non_ascii_folder_are_read(tmp_path):
    from framelabel.core.images import imread

    (frames := tmp_path / "кадры" / "frames").mkdir(parents=True)
    ok, buf = cv2.imencode(".jpg", np.full((40, 60, 3), 200, np.uint8))
    (frames / "a.jpg").write_bytes(buf.tobytes())
    assert imread(frames / "a.jpg").shape == (40, 60, 3)
    assert imread(frames / "a.jpg", cv2.IMREAD_REDUCED_GRAYSCALE_2).shape == (20, 30)
    (frames / "b.jpg").write_bytes(b"not a jpeg")
    assert imread(frames / "b.jpg") is None and imread(frames / "missing.jpg") is None


def test_a_damaged_review_file_is_never_taken_for_an_empty_one(tmp_path):
    from framelabel.core.session import session_overview

    (frames := tmp_path / "s" / "frames").mkdir(parents=True)
    cv2.imwrite(str(frames / "a.jpg"), np.zeros((20, 20, 3), np.uint8))
    s = Session.open(tmp_path / "s")
    s.review_path.write_text('{"version": 1, "status": {"a": "pos"', encoding="utf-8")  # cut off mid-write
    with pytest.raises(ValueError, match="damaged"):
        s.review()
    assert session_overview(s)["damaged"] == 1 and folder_project(tmp_path / "s").class_names == ["object"]
    assert s.review_path.read_text(encoding="utf-8").endswith('"pos"')  # left as it was, for the user to restore


def test_box_corner_edge_hit_testing():
    big, small = Box(0, 0, 100, 100), Box(40, 40, 60, 60)
    assert box_at([big, small], 50, 50) == 1  # the inner box wins
    assert box_at([big, small], 10, 10) == 0 and box_at([big, small], 200, 200) == -1
    assert corner_at(small, 61, 39, tol=3) == 1 and corner_at(small, 50, 50, tol=3) == -1
    assert edge_at(small, 40.5, 50, tol=2) == "l" and edge_at(small, 50, 61, tol=2) == "b" and edge_at(small, 50, 50, 2) == ""


def test_choose_negatives_prefers_hard_then_random():
    negs = [(f"k{i}", 0.0) for i in range(20)] + [("hard1", 0.25), ("hard2", 0.2)]
    picked = choose_negatives(negs, 6)
    assert len(picked) == 6 and len(set(picked)) == 6 and {"hard1", "hard2"} <= set(picked)
    assert set(choose_negatives(negs[:3], 10)) == {"k0", "k1", "k2"}


def test_frame_time_crosses_midnight():
    assert frame_time("session_20261008_223625_224433_188") == ("session_20261008_223625", pytest.approx(488.188))
    assert frame_time("session_20261008_223625_000936_072")[1] == pytest.approx(86400 - 81385 + 576.072)
    assert frame_time("session_20261008_223625_223620_000")[1] == -5.0  # clock jitter around the start is not a day
    assert split_by_time(["b", "a", "d", "c", "e", "f", "g", "h", "i", "j"], 0.2)[1] == ["i", "j"]


def _timed_keys():
    return [f"s{s}_{(b * 70 + i) // 60 % 60 + 10 * 100:04d}{(b * 70 + i) % 60:02d}_000"
            for s in (1, 2) for b in range(10) for i in range(10)]


def test_split_by_blocks_keeps_blocks_whole_and_covers_every_session():
    keys = _timed_keys()  # two sessions, each 10 blocks of 10 frames, blocks 60 s apart
    train, val = split_by_blocks(keys, 0.2, seed=3)
    assert sorted(train + val) == sorted(keys) and not set(train) & set(val) and len(val) == 40
    for sess in ("s1", "s2"):
        assert len([k for k in val if k.startswith(sess)]) == 20
    blocks = {}
    for k in keys:
        sess, t = frame_time(k)
        blocks.setdefault((sess, int(t) // 70), set()).add(k)
    assert all(b <= set(val) or not b & set(val) for b in blocks.values())
    assert split_by_blocks(keys, 0.2, seed=3) == (train, val)


def test_split_by_blocks_matches_the_screendet_tool():
    """Same keys and seed -> the same val split as fin_snip_helper's label_frames.py (reference values computed
    with that implementation), so re-exporting the user's sessions keeps datasets/own's val split."""
    train, val = split_by_blocks(_timed_keys(), 0.2, seed=3)
    assert val[:3] == ["s1_100110_000", "s1_100111_000", "s1_100112_000"] and len(val) == 40


def test_untimed_keys_are_split_by_position_blocks():
    keys = [f"img_{i:04d}" for i in range(100)]
    train, val = split_by_blocks(keys, 0.2, seed=0)
    assert len(val) == 20 and sorted(train + val) == keys


def test_balanced_block_split_spreads_a_clustered_class():
    keys = [f"s1_10{(b * 65 + i) // 60:02d}{(b * 65 + i) % 60:02d}_000" for b in range(20) for i in range(5)]
    classes = {k: [0] + ([1] if n < 20 else []) for n, k in enumerate(keys)}
    _, val = balanced_block_split(keys, classes, 0.25, tries=50)
    assert sum(1 in classes[k] for k in val) == 5 and len(val) == 25


def test_review_persists_and_keeps_format(tmp_path):
    r = Review(tmp_path / "review.json", ["enemy", "corpse"])
    r.set("f1", "pos")
    r.set_many({"f2": "neg", "f3": "skip"})
    r.set("f2", None)
    data = json.loads((tmp_path / "review.json").read_text())
    assert data == {"version": 1, "classes": ["enemy", "corpse"], "status": {"f1": "pos", "f3": "skip"}}
    assert Review(tmp_path / "review.json").classes == ["enemy", "corpse"]
    with pytest.raises(ValueError):
        r.set("f4", "weird")


def _write_session(root, stems_status, size=(64, 36)):
    s = root / "session_20261008_100000"
    (s / "frames").mkdir(parents=True)
    (s / "labels").mkdir()
    r = Review(s / "review.json", ["enemy", "corpse"])
    for stem, st in stems_status.items():
        cv2.imwrite(str(s / "frames" / f"{stem}.jpg"), np.zeros((size[1], size[0], 3), np.uint8))
        if st:
            r.set(stem, st)
    return s


def test_export_builds_two_class_dataset(tmp_path):
    s = _write_session(tmp_path, {"100000_000": "pos", "100001_000": "neg", "100002_000": "skip", "100003_000": "pos",
                                  "100004_000": None})
    (s / "labels/100000_000.txt").write_text("0 0.5 0.5 0.1 0.2\n1 0.2 0.2 0.1 0.1")
    (s / "labels/100001_000.txt").write_text("")
    (s / "labels/100003_000.txt").write_text("0 0.5 0.5 0.1 0.2")
    out = tmp_path / "own"
    res = export_dataset([Session.open(s)], ["enemy", "corpse"], out, neg_ratio=1.0, val_fraction=0.34, split="time")
    train = sorted(p.name for p in (out / "images/train").iterdir())
    val = sorted(p.name for p in (out / "images/val").iterdir())
    pre = "session_20261008_100000_"
    assert train == [pre + "100000_000.jpg", pre + "100001_000.jpg"] and val == [pre + "100003_000.jpg"]
    assert (out / "labels/train" / f"{pre}100001_000.txt").read_text() == ""
    assert "\n  1: corpse\n" in (out / "data.yaml").read_text() and res.boxes["train"] == [1, 1]
    export_dataset([Session.open(s)], ["no: helmet", "#2"], out, split="time")  # names YAML would misread
    import yaml

    assert yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))["names"] == {0: "no: helmet", 1: "#2"}
    (tmp_path / "notadataset").mkdir()
    with pytest.raises(FileExistsError):
        export_dataset([Session.open(s)], ["enemy", "corpse"], tmp_path / "notadataset")


def test_export_cancel_keeps_the_previous_dataset(tmp_path):
    from framelabel.core.index_cache import Cancelled

    s = _write_session(tmp_path, {"100000_000": "pos", "100001_000": "neg", "100003_000": "pos"})
    (s / "labels/100000_000.txt").write_text("0 0.5 0.5 0.1 0.2")
    (s / "labels/100001_000.txt").write_text("")
    (s / "labels/100003_000.txt").write_text("0 0.5 0.5 0.1 0.2")
    out, seen = tmp_path / "own", []
    export_dataset([Session.open(s)], ["enemy", "corpse"], out, neg_ratio=1.0, split="time", progress=lambda *a: seen.append(a))
    assert seen and seen[-1][:2] == (3, 3)
    before = sorted(str(p.relative_to(out)) for p in out.rglob("*"))
    with pytest.raises(Cancelled):
        export_dataset([Session.open(s)], ["enemy", "corpse"], out, neg_ratio=1.0, split="time", cancelled=lambda: True)
    assert sorted(str(p.relative_to(out)) for p in out.rglob("*")) == before
    assert not (tmp_path / "own.partial").exists()


def test_session_layouts(tmp_path):
    s = _write_session(tmp_path, {"100000_000": None})
    assert Session.open(s).layout == "session" and [x.name for x in discover_sessions(tmp_path)] == [s.name]
    y = tmp_path / "ds" / "images" / "train"
    y.mkdir(parents=True)
    cv2.imwrite(str(y / "a.jpg"), np.zeros((8, 8, 3), np.uint8))
    ys = Session.open(y)
    assert ys.layout == "yolo" and ys.labels_dir == tmp_path / "ds/labels/train"
    assert ys.review_path == tmp_path / "ds/review/train.json"
    assert [x.layout for x in discover_sessions(tmp_path / "ds")] == ["yolo"]
    f = tmp_path / "plain"
    f.mkdir()
    cv2.imwrite(str(f / "b.png"), np.zeros((8, 8, 3), np.uint8))
    assert Session.open(f).layout == "folder" and folder_project(f).class_names == ["object"]


def test_project_roundtrip_and_class_mapping(tmp_path):
    d = {"name": "p", "classes": ["enemy", {"name": "corpse", "color": "#00beff"}], "sources": ["data"],
         "class_map": {"person": "enemy"}, "ignore_zones": [[0, 0.75, 0.07, 0.92]], "export": {"out": "out", "neg_ratio": 0.2}}
    p = Project.from_dict(d, tmp_path, tmp_path / "p.yaml")
    assert p.sources == [(tmp_path / "data").resolve()] and p.classes[1].color == "#00beff"
    assert p.map_model_class("person", 0, 1) == 0 and p.map_model_class("corpse", 1, 2) == 1
    assert p.map_model_class("car", 0, 1) == 0 and p.map_model_class("car", 2, 3) is None
    p.save()
    q = Project.load(tmp_path / "p.yaml")
    assert q.to_dict()["sources"] == ["data"] and q.export_out == (tmp_path / "out").resolve() and q.export["neg_ratio"] == 0.2


def test_view_fits_centred_and_zooms_around_cursor():
    v = View(1920, 1080)
    v.fit(2560, 1394)
    x0, y0 = v.to_widget(0, 0)
    x1, y1 = v.to_widget(1920, 1080)
    assert abs(y0) < 1e-6 and abs(y1 - 1394) < 1e-6 and abs(x0 - (2560 - x1)) < 1e-6
    before = v.to_img(700, 300)
    v.zoom_at(700, 300, 4.0)
    assert np.allclose(v.to_img(700, 300), before)
    v.zoom_at(700, 300, 0.01)
    assert v.zoom == 1.0 and np.allclose(v.to_widget(960, 540), (1280, 697))
    v.zoom_to(100, 100, 110, 120)
    assert v.zoom > 1 and np.allclose((v.cx, v.cy), (105, 110)) or v.cy > 110


def test_history_undo_redo():
    h, boxes = History(), [Box(0, 0, 1, 1)]
    h.push(boxes)
    boxes = boxes + [Box(2, 2, 3, 3)]
    prev = h.undo(boxes)
    assert len(prev) == 1 and len(h.redo(prev)) == 2 and h.redo(prev) is None


def test_track_box_follows_a_shifted_object():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 60, (240, 320, 3), dtype=np.uint8)
    a[100:160, 120:150] = rng.integers(150, 255, (60, 30, 3), dtype=np.uint8)  # textured object
    b = np.roll(a, (6, 11), axis=(0, 1))
    moved, score = track_box(a, b, Box(120, 100, 150, 160))
    assert score > 0.8 and np.allclose(moved.xyxy(), (131, 106, 161, 166), atol=1.0)
    snapped = snap_to_proposals([moved], [Box(132, 105, 160, 167, conf=0.7)])
    assert snapped[0].conf is None and np.allclose(snapped[0].xyxy(), (132, 105, 160, 167))
    assert len(propagate(a, b, [Box(120, 100, 150, 160, cls=1)])) == 1


def test_queue_scores():
    assert uncertainty([0.9, 0.38], 0.15, 0.6) > 0.9 and uncertainty([0.9], 0.15, 0.6) == 0.0


def _ui_session(tmp_path, n_game=12, n_ui=3):
    s = tmp_path / "session_20261008_100000"
    (s / "frames").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n_game + n_ui):
        img = rng.integers(60, 120, (360, 640, 3), dtype=np.uint8)
        if i < n_game:
            cv2.rectangle(img, (14, 285), (30, 320), (230, 230, 230), -1)  # stand-in for the body-status icon
            cv2.circle(img, (22, 278), 6, (230, 230, 230), -1)
        else:
            img[:] = 15
            cv2.rectangle(img, (100, 50), (500, 300), (80, 80, 80), 2)
        cv2.imwrite(str(s / "frames" / f"10{i:04d}_000.jpg"), img)
    return s


def test_ui_scores_flag_interface_frames(tmp_path):
    s = Session.open(_ui_session(tmp_path))
    frames = s.frames()
    sc = ui_scores(s, frames, HUD[0])
    assert all(sc[p.stem] > 0.6 for p in frames[:12]) and all(sc[p.stem] < 0.3 for p in frames[12:])
    cached = json.loads((s.root / "ui_scores.json").read_text())
    assert cached["_zone"] == list(HUD[0]) and ui_scores(s, frames, HUD[0]) == sc


def _project(root):
    return Project.from_dict({"name": "t", "classes": ["enemy", "corpse"], "sources": [str(root)],
                              "ignore_zones": [list(HUD[0])], "ui_detect": {"zone": list(HUD[0]), "threshold": 0.3}},
                             root)


def test_workspace_grid_skip_marks_and_commit(tmp_path):
    root = _ui_session(tmp_path)
    proj = _project(tmp_path)
    s = proj.sessions()[0]
    frames = [p.stem for p in s.frames()]
    Review(s.review_path).set(frames[13], "neg")  # an interface frame wrongly saved as empty earlier
    ws = Workspace(proj, s)
    triage = ws.queue("triage")
    assert len(triage) == 15
    marks = {fi.stem: m for fi in triage if (m := ws.initial_mark(fi, "triage"))}
    assert marks == {frames[12]: "skip", frames[14]: "skip"}  # neg is not overridden
    ui = ws.queue("ui")
    marks = {fi.stem: m for fi in ui if (m := ws.initial_mark(fi, "ui"))}
    assert marks == {f: "skip" for f in frames[12:]}
    del marks[frames[14]]  # user: this one is fine as empty
    ws.commit_page([fi.stem for fi in ui], marks)
    st = ws.review.status
    assert st[frames[12]] == st[frames[13]] == "skip" and st[frames[14]] == "neg"
    ws.commit_page(frames[:2], {frames[0]: "people"})
    assert ws.review.get(frames[0]) == "todo" and ws.review.get(frames[1]) == "neg"
    assert (s.labels_dir / f"{frames[1]}.txt").read_text() == "" and not (s.labels_dir / f"{frames[12]}.txt").exists()
    assert root.name == s.name


def test_workspace_edit_save_and_queues(tmp_path):
    root = _ui_session(tmp_path, n_game=6, n_ui=0)
    proj = _project(tmp_path)
    s = proj.sessions()[0]
    stems = [p.stem for p in s.frames()]
    s.auto_dir.mkdir()
    (s.auto_path(stems[0])).write_text("0 0.5 0.5 0.1 0.2 0.85\n0 0.3 0.3 0.05 0.02 0.35\n0 0.037 0.84 0.05 0.18 0.9")
    (s.auto_path(stems[1])).write_text("0 0.5 0.5 0.1 0.2 0.12")
    ws = Workspace(proj, s, conf=0.3)
    assert [f.stem for f in ws.queue("edit")] == [stems[0]]
    assert [f.stem for f in ws.queue("uncertain")] == [stems[0]]  # 0.35 is in the uncertain band
    assert [f.stem for f in ws.queue("small")] == [stems[0]]  # a 0.02-high proposal; frame 1 has only a big one
    boxes, weak = ws.load_boxes(s.frames()[0], 640, 360)
    assert [b.conf for b in boxes] == [0.85, 0.35] and weak == []  # the HUD proposal is ignored
    assert ws.save(s.frames()[0], boxes[:1], 640, 360) == "pos"
    assert ws.review.get(stems[0]) == "pos" and ws.index[0].label_cls == [0]
    boxes, _ = ws.load_boxes(s.frames()[0], 640, 360)
    assert len(boxes) == 1 and boxes[0].conf is None  # reviewed labels are shown now
    assert disagreement(ws.index[0], 0.5) == 0.0
    ws.save(s.frames()[0], [], 640, 360)  # labeled empty although the model is confident -> disagreement
    assert ws.review.get(stems[0]) == "neg" and [f.stem for f in ws.queue("disagree")] == [stems[0]]
    assert ws.save(s.frames()[2], [], 640, 360, "skip") == "skip" and not s.label_path(stems[2]).exists()
    assert root.exists()


def test_build_index_reads_every_frame(tmp_path):
    _ui_session(tmp_path, n_game=3, n_ui=0)
    s = _project(tmp_path).sessions()[0]
    idx = build_index(s, s.review())
    assert len(idx) == 3 and build_queue(idx, "all") == idx and build_queue(idx, "class:0") == []


# ---------------------------------------------------------------- index cache
def _index_session(tmp_path):
    from framelabel.core.boxes import write_yolo

    s = tmp_path / "session_20261008_100000"
    (s / "frames").mkdir(parents=True)
    (s / "labels").mkdir()
    (s / "labels_auto").mkdir()
    stems = [f"10{i:04d}_000" for i in range(7)]
    for st in stems:
        cv2.imwrite(str(s / "frames" / f"{st}.jpg"), np.zeros((36, 64, 3), np.uint8))
    auto = {1: "", 2: "0 0.5 0.5 0.1 0.3 0.80\n0 0.03 0.8 0.02 0.05 0.60",  # second row is inside the HUD zone
            3: "1 0.4 0.4 0.2 0.2 0.20", 4: "0 0.5 0.5 0.1 0.2 0.40", 5: "0 0.2 0.2 0.1 0.1 0.9"}
    for i, txt in auto.items():
        (s / "labels_auto" / f"{stems[i]}.txt").write_text(txt)
    (s / "labels_auto" / f"{stems[6]}.txt").write_text("0 0.1 0.1 0.2 0.2 0.5 0.1 0.1\n")  # malformed: ignored
    review = Review(s / "review.json", ["enemy", "corpse"])
    review.set_many({stems[3]: "pos", stems[4]: "neg", stems[5]: "todo"})
    write_yolo(s / "labels" / f"{stems[3]}.txt", [Box(10, 5, 30, 25, 1)], 64, 36)
    (s / "labels" / f"{stems[4]}.txt").write_text("")
    write_yolo(s / "labels" / f"{stems[5]}.txt", [Box(10, 5, 30, 25, 0)], 64, 36)  # todo: labels are ignored
    return Session.open(s), stems


def _same_info(a, b):
    assert a.path == b.path and a.status == b.status and a.confs == b.confs and a.prop_h == b.prop_h
    assert a.label_cls == b.label_cls and a.label_h == b.label_h and a.ui == b.ui
    for x, y in ((a.prop_xyxy, b.prop_xyxy), (a.label_xyxy, b.label_xyxy)):
        assert (x is None) == (y is None) and (x is None or np.array_equal(x, y))


def test_cached_index_equals_direct_read(tmp_path):
    from framelabel.core.queues import frame_info

    s, _ = _index_session(tmp_path)
    review = s.review()
    direct = [frame_info(s, p, review, HUD) for p in s.frames()]
    for _ in range(2):  # the second run comes from the cache
        for got, want in zip(build_index(s, review, HUD), direct):
            _same_info(got, want)


def test_index_cache_sees_changes(tmp_path):
    import os

    s, stems = _index_session(tmp_path)
    review = s.review()
    build_index(s, review)
    f = s.auto_path(stems[3])
    f.write_text("0 0.5 0.5 0.1 0.2 0.95")
    t = f.stat().st_mtime_ns + 2_000_000_000
    os.utime(f, ns=(t, t))
    s.auto_path(stems[5]).unlink()
    s.auto_path(stems[0]).write_text("0 0.3 0.3 0.1 0.1 0.7")
    idx = {fi.stem: fi for fi in build_index(s, review)}
    assert idx[stems[3]].confs == [0.95] and idx[stems[5]].prop_xyxy is None and idx[stems[0]].confs == [0.7]


def test_index_cache_survives_corrupt_cache(tmp_path):
    from framelabel.core.index_cache import cache_dir

    s, stems = _index_session(tmp_path)
    build_index(s, s.review())
    for f in cache_dir().glob("rows_*.json"):
        f.write_text("{not json")
    assert [fi.confs for fi in build_index(s, s.review())][2] == [0.8, 0.6]


def test_build_index_cancel(tmp_path):
    from framelabel.core.index_cache import Cancelled

    s, _ = _index_session(tmp_path)
    for i in range(250):
        s.auto_path(f"extra{i}").write_text("0 0.5 0.5 0.1 0.1 0.5")
    with pytest.raises(Cancelled):
        build_index(s, s.review(), cancelled=lambda: True)


def test_ui_scores_incremental(tmp_path, monkeypatch):
    s = Session.open(_ui_session(tmp_path, n_game=12, n_ui=2))
    first = ui_scores(s, s.frames(), HUD[0])
    rng = np.random.default_rng(1)
    game = rng.integers(60, 120, (360, 640, 3), dtype=np.uint8)
    cv2.rectangle(game, (14, 285), (30, 320), (230, 230, 230), -1)
    cv2.circle(game, (22, 278), 6, (230, 230, 230), -1)
    menu = np.full((360, 640, 3), 15, np.uint8)
    cv2.imwrite(str(s.images_dir / "109998_000.jpg"), game)
    cv2.imwrite(str(s.images_dir / "109999_000.jpg"), menu)
    import framelabel.core.uiscreen as uiscreen

    read = []
    real = uiscreen.imread
    monkeypatch.setattr(uiscreen, "imread", lambda path, *a: (read.append(str(path)), real(path, *a))[1])
    again = ui_scores(s, s.frames(), HUD[0])
    assert sorted(p.split("\\")[-1].split("/")[-1] for p in read) == ["109998_000.jpg", "109999_000.jpg"]
    assert again["109998_000"] > 0.6 and again["109999_000"] < 0.3
    assert all(again[k] == v for k, v in first.items())  # old scores are kept as they were


def test_ui_scores_cancel(tmp_path):
    from framelabel.core.index_cache import Cancelled

    s = Session.open(_ui_session(tmp_path))
    with pytest.raises(Cancelled):
        ui_scores(s, s.frames(), HUD[0], cancelled=lambda: True)
    assert not (s.root / "ui_scores.json").exists()


def test_apply_reload_keeps_statuses_and_takes_new_proposals(tmp_path):
    s, stems = _index_session(tmp_path)
    ws = Workspace(Project.from_dict({"name": "t", "classes": ["enemy", "corpse"], "sources": [str(tmp_path)]}, tmp_path), s)
    fresh = ws.build_fresh()  # the snapshot is taken ...
    ws.save(s.frames()[0], [], 64, 36, "skip")  # ... a decision follows, newer than the snapshot
    s.auto_path(stems[0]).write_text("0 0.5 0.5 0.2 0.2 0.9")
    ws.apply_reload(ws.build_fresh())
    assert ws.index[0].status == "skip" and ws.index[0].confs == [0.9]
    ws.apply_reload(fresh)  # an older snapshot must not undo the status either
    assert ws.index[0].status == "skip" and ws.review.get(stems[0]) == "skip"
    assert ws.index[3].status == "pos" and ws.index[3].label_cls == [1]  # labels survive


# ---------------------------------------------------------------- report: what is filtered out
def _report_session(tmp_path):
    from framelabel.core.boxes import write_yolo

    s = tmp_path / "session_20261008_100000"
    (s / "frames").mkdir(parents=True)
    (s / "labels_auto").mkdir()
    stems = [f"10{i:04d}_000" for i in range(6)]
    for st in stems:
        cv2.imwrite(str(s / "frames" / f"{st}.jpg"), np.zeros((36, 64, 3), np.uint8))
    auto = {0: "",                                                      # nothing detected
            1: "0 0.03 0.8 0.02 0.05 0.60",                              # only inside the HUD zone: looks empty
            2: "0 0.5 0.5 0.1 0.3 0.10",                                 # weak (below the shown 0.2)
            3: "0 0.5 0.5 0.1 0.3 0.70\n0 0.03 0.8 0.02 0.05 0.50",    # a real one + one hidden by the zone
            4: "0 0.5 0.5 0.1 0.3 0.30"}
    for i, txt in auto.items():
        (s / "labels_auto" / f"{stems[i]}.txt").write_text(txt)
    return Session.open(s), stems


def test_zone_and_weak_queues_show_what_is_hidden(tmp_path):
    s, stems = _report_session(tmp_path)
    idx = build_index(s, s.review(), HUD)
    by = {f.stem: f for f in idx}
    assert by[stems[1]].ignored_conf == 0.6 and by[stems[1]].max_conf == 0.0 and by[stems[1]].ignored_xyxy.shape == (1, 4)
    assert by[stems[0]].ignored_xyxy is None and by[stems[5]].prop_xyxy is None
    q = lambda name: [f.stem for f in build_queue(idx, name, conf=0.2)]  # noqa: E731
    assert q("zone") == [stems[1], stems[3]]                    # the zone proposals, whatever else the frame has
    assert q("weak") == [stems[2]]                              # unreviewed, best proposal 0.10 < 0.2
    assert stems[1] in q("triage") and stems[2] in q("triage")  # they look empty in the grid: now explained


def test_report_rows_count_what_each_view_shows(tmp_path):
    import json

    from framelabel.core.report import capture_info, capture_rows, frame_rows, hidden_line, prelabel_rows

    s, stems = _report_session(tmp_path)
    idx = build_index(s, s.review(), HUD)
    from framelabel.core.views import view_frames

    rows = {r.queue: r for r in frame_rows(idx, 0.2, 0.3) if r.queue}
    for name, row in rows.items():  # every count equals the size of the view it links to
        assert row.count == len(view_frames(idx, name, 0.2, 0.3)) and row.mode in ("grid", "edit")
    # a lens is a part of its stage: stems[3] has a real proposal (so it is a frame to label) and one hidden by the zone
    assert rows["zone"].count == 1 and rows["weak"].count == 1 and rows["all"].count == 6
    assert rows["zone"].where == "Triage > Only in ignore zone" and rows["review"].where == "Review"
    assert capture_info(s) is None and "No capture.json" in capture_rows(None)[0].label
    (s.root / "capture.json").write_text(json.dumps({
        "saved_by_reason": {"change": 5, "burst": 1},
        "skipped_by_reason": {"ui": 40, "duplicate": 3, "waiting": 99},
        "settings": {"ui_filter": True, "policy": {"ui_threshold": 0.3}}}))
    info = capture_info(s)
    got = {r.label: r for r in capture_rows(info)}
    assert got["Recorder skipped: ui"].count == 40 and "cannot be reviewed" in got["Recorder skipped: ui"].where
    assert got["Recorder setting: ui_filter"].where.startswith("on:")
    assert [r.count for r in capture_rows(info) if r.label == "Recorder skipped: ui"] == [40]
    line = hidden_line(idx, info, 0.2, 0.3)
    assert "40 ui" in line and "only in ignore zone 1" in line and "weak proposals only 1" in line
    assert prelabel_rows(s)[0].label == "No model proposals yet"  # no _meta.json written yet
    (s.auto_dir / "_meta.json").write_text(json.dumps({"weights": ["x/run_a/weights/best.pt"], "imgsz": 1280, "tta": True,
                                                       "conf": 0.05, "run": {"without_proposals": 4, "only_in_ignore_zone": 2,
                                                                             "dropped_classes": {"car": 7}}}))
    pr = {r.label: r for r in prelabel_rows(s)}
    assert pr["Prelabel models"].where == "run_a"
    assert pr["Last prelabel run: detections of classes without a project class (dropped)"].count == 7


def test_workspace_reads_capture_info_and_summarizes(tmp_path):
    import json

    s, _ = _report_session(tmp_path)
    (s.root / "capture.json").write_text(json.dumps({"skipped_by_reason": {"ui": 12}}))
    proj = Project.from_dict({"name": "t", "classes": ["enemy"], "sources": [str(tmp_path)],
                              "ignore_zones": [list(HUD[0])]}, tmp_path)
    ws = Workspace(proj, s, conf=0.2)
    assert ws.capture == {"skipped_by_reason": {"ui": 12}} and "12 ui" in ws.hidden_summary()


def test_hide_reviewed_frames_only_touches_the_work_views():
    from framelabel.core.queues import FrameInfo, build_queue

    def fi(i, status, conf):
        return FrameInfo(Path(f"{i}.jpg"), status, confs=[conf] if conf else [], prop_h=[0.02] if conf else [],
                         prop_xyxy=None)

    from pathlib import Path

    idx = [fi(0, None, 0.0), fi(1, "neg", 0.0), fi(2, "skip", 0.0), fi(3, "todo", 0.0), fi(4, None, 0.9), fi(5, "pos", 0.9)]
    q = lambda name, hide: [f.path.stem for f in build_queue(idx, name, conf=0.3, hide_reviewed=hide)]  # noqa: E731
    assert q("triage", False) == ["0", "1", "2", "3"] and q("triage", True) == ["0"]  # grid: any decision hides
    assert q("edit", False) == ["2", "3", "4", "5"] and q("edit", True) == ["3", "4"]  # editor: 'todo' still waits
    for name in ("all", "neg", "skip", "pos", "todo"):  # views of a status and 'all' are never filtered
        assert q(name, True) == q(name, False)


def _frames_for_views():
    """A small index that has every kind of frame the stages and their lenses care about."""
    from pathlib import Path

    import numpy as np

    from framelabel.core.queues import FrameInfo

    def fi(name, status=None, props=(), labels=(), ui=1.0, ignored=0.0):
        """props: (conf, x1, y1, x2, y2) rows, labels: (x1, y1, x2, y2) rows, both normalized."""
        confs = [p[0] for p in props]
        xy = np.array([p[1:] for p in props], float).reshape(-1, 4)
        lab = np.array(labels, float).reshape(-1, 4)
        return FrameInfo(Path(f"{name}.jpg"), status, confs=confs, prop_h=list(xy[:, 3] - xy[:, 1]), prop_xyxy=xy,
                         label_cls=[0] * len(lab), label_h=list(lab[:, 3] - lab[:, 1]), label_xyxy=lab, ui=ui,
                         ignored_conf=ignored)

    box = (0.4, 0.2, 0.5, 0.6)       # a normal person
    far = (0.4, 0.2, 0.41, 0.22)     # a far one: 0.02 high
    return [
        fi("n1"),                                               # new, nothing found
        fi("n2", props=[(0.1, *box)]),                          # new, weak proposal
        fi("n3", props=[(0.8, *box)]),                          # new, confident proposal
        fi("n4", props=[(0.35, *far)]),                         # new, unsure + far
        fi("n5", props=[(0.55, *box)]),                         # new, a little unsure
        fi("m1", "todo"),                                       # marked in Triage, no proposal
        fi("u1", ui=0.1),                                       # looks like an interface screen
        fi("u2", ui=0.1, props=[(0.9, *box)]),                  # ... but the model sees something
        fi("z1", ignored=0.6),                                  # only inside an ignore zone
        fi("z2", props=[(0.7, *box)], ignored=0.6),             # a real proposal and a hidden one
        fi("A", "pos", props=[(0.9, *box)], labels=[box]),      # labeled, the model agrees
        fi("B", "neg", props=[(0.8, *box)]),                    # empty although the model is confident: possibly missed
        fi("C", "pos", labels=[box]),                           # a label the model has no proposal for: possibly wrong
        fi("D", "pos", props=[(0.9, *far)], labels=[far]),      # agrees, but a far object
        fi("E", "skip"),
    ]


def test_triage_and_label_partition_the_frames_without_a_decision():
    from framelabel.core.views import view_frames

    idx = _frames_for_views()
    stems = lambda key, **kw: [f.stem for f in view_frames(idx, key, 0.3, 0.3, **kw)]  # noqa: E731
    triage, label = stems("triage"), stems("edit")
    assert triage == ["n1", "n2", "u1", "z1"] and label == ["n3", "n4", "n5", "m1", "u2", "z2"]
    undecided = [f.stem for f in idx if f.status in (None, "todo")]
    assert sorted(triage + label) == sorted(undecided) and not set(triage) & set(label)  # every frame is in one stage
    # a lens is a part of its stage, in the stage's order
    assert stems("weak") == ["n2"] and stems("ui") == ["u1"] and stems("zone") == ["z1"]  # z2 / u2 are Label frames
    assert stems("proposals") == ["n3", "n4", "n5", "u2", "z2"] and stems("todo") == ["m1"]
    for lens, stage in (("weak", "triage"), ("ui", "triage"), ("zone", "triage"), ("proposals", "edit"), ("todo", "edit"),
                        ("missed", "review"), ("extra", "review"), ("smalllab", "review")):
        assert set(stems(lens)) <= set(stems(stage))


def test_label_order_is_a_sort_not_a_view():
    from framelabel.core.views import view_frames

    idx = _frames_for_views()
    stems = lambda key, sort: [f.stem for f in view_frames(idx, key, 0.3, 0.3, sort)]  # noqa: E731
    assert stems("edit", "time") == ["n3", "n4", "n5", "m1", "u2", "z2"]
    assert stems("edit", "unsure")[:2] == ["n4", "n5"]           # 0.35 is the most unsure, then 0.55; the rest keep their order
    assert stems("edit", "unsure")[2:] == ["n3", "m1", "u2", "z2"]
    assert stems("edit", "small")[:2] == ["n4", "n3"]            # the 0.02 high proposal first
    assert stems("proposals", "small")[0] == "n4"                # a lens keeps its stage's order
    assert stems("triage", "small") == stems("triage", "time")   # only the Label stage sorts


def test_review_flags_decided_frames_and_checked_ones_drop_out():
    from framelabel.core.views import view_counts, view_frames

    idx = _frames_for_views()
    stems = lambda key, **kw: [f.stem for f in view_frames(idx, key, 0.3, 0.3, **kw)]  # noqa: E731
    assert stems("missed") == ["B"] and stems("extra") == ["C"] and stems("smalllab") == ["D"]
    assert stems("review") == ["B", "C", "D"]  # the biggest disagreement first, then the frames that only have a far object
    assert stems("review", checked={"B", "D"}) == ["C"] and stems("smalllab", checked={"D"}) == []
    counts = view_counts(idx, ["enemy"], 0.3, 0.3, checked={"B"})
    assert counts["review"] == 2 and counts["missed"] == 0 and counts["triage"] == 4 and counts["edit"] == 6
    assert counts["pos"] == 3 and counts["neg"] == 1 and counts["skip"] == 1 and counts["all"] == 15


def test_library_views_and_stage_order():
    from framelabel.core.views import GROUPS, STAGE_ORDER, all_views, find_view, next_stage, stage_label, status_counts, view_progress

    views = all_views(["enemy", "corpse"])
    assert [g for g, _ in GROUPS] == ["workflow", "library"] and STAGE_ORDER == ("triage", "edit", "review")
    assert [v.key for v in views if v.is_stage] == ["triage", "edit", "review"]
    assert [v.step for v in views if v.is_stage] == [1, 2, 3]
    assert [v.key for v in views if v.group == "library"] == ["all", "unreviewed", "pos", "neg", "skip", "checked", "class:0", "class:1"]
    assert len({v.key for v in views}) == len(views)
    assert all(v.parent in STAGE_ORDER for v in views if v.parent)  # a lens always refines a stage
    assert find_view("class:1", ["enemy", "corpse"]).label == "Class: corpse"
    assert stage_label("weak") == "Triage > Weak proposals" and stage_label("edit") == "Label" and stage_label("pos") == "Labeled"
    # the way each stage opens: the point of the design
    assert [find_view(k).mode for k in ("triage", "edit", "review", "pos")] == ["grid", "edit", "grid", "grid"]
    idx = _frames_for_views()
    assert status_counts(idx) == {"new": 9, "marked": 1, "labeled": 3, "empty": 1, "skipped": 1}
    assert next_stage("triage", {"triage": 0, "edit": 5, "review": 0}) == "edit"
    assert next_stage("edit", {"triage": 0, "edit": 5, "review": 0}) is None
    assert next_stage("edit", {"triage": 2, "edit": 0, "review": 3}) == "review"
    assert next_stage("review", {"triage": 2, "edit": 0, "review": 3}) == "triage"
    assert next_stage(None, {"triage": 0, "edit": 0, "review": 0}) is None
    assert view_progress(idx, 0.3, {"triage": 4, "edit": 6}) == {"triage": (4, 7), "edit": (6, 10)}


def test_a_page_save_never_changes_a_labeled_frame_and_can_be_undone(tmp_path):
    _ui_session(tmp_path, n_game=5, n_ui=0)
    proj = _project(tmp_path)
    s = proj.sessions()[0]
    stems = [p.stem for p in s.frames()]
    ws = Workspace(proj, s)
    ws.save(s.frames()[0], [Box(10, 10, 100, 200, 0)], 640, 360)  # labeled with one box
    labels = s.label_path(stems[0]).read_text()
    ws.save(s.frames()[4], [], 640, 360, "skip")
    # a labeled frame is left alone whatever its mark says (a stale SKIP mark must not cost it its boxes)
    undo = ws.commit_page(stems, {stems[1]: "people", stems[4]: "people", stems[0]: "skip"})
    assert ws.review.get(stems[0]) == "pos" and s.label_path(stems[0]).read_text() == labels
    assert [ws.review.get(x) for x in stems[1:]] == ["todo", "neg", "neg", "todo"]
    assert set(undo) == set(stems[1:])  # the labeled frame was not touched, so it is not in the record
    assert undo[stems[1]] == (None, None, "todo") and undo[stems[2]] == (None, None, "neg") and undo[stems[4]][2] == "todo"
    # undo puts back statuses and label files exactly (the skipped frame stays skipped, the new ones are new again)
    restored, left = ws.undo_page(undo)
    assert sorted(restored) == sorted(stems[1:]) and left == []
    assert [ws.review.get(x) for x in stems] == ["pos", None, None, None, "skip"]
    assert s.label_path(stems[0]).read_text() == labels and not s.label_path(stems[2]).exists()
    assert not s.label_path(stems[4]).exists() and ws.index[ws.by_stem[stems[2]]].status is None


def test_undoing_a_page_never_undoes_later_work(tmp_path):
    _ui_session(tmp_path, n_game=4, n_ui=0)
    proj = _project(tmp_path)
    s = proj.sessions()[0]
    stems = [p.stem for p in s.frames()]
    ws = Workspace(proj, s)
    undo = ws.commit_page(stems, {stems[0]: "people", stems[1]: "people"})  # both marked in Triage; the rest empty
    assert [ws.review.get(x) for x in stems] == ["todo", "todo", "neg", "neg"]
    ws.save(s.frames()[0], [Box(10, 10, 100, 200, 0)], 640, 360)  # later, in Label: the first one gets its box
    ws.save(s.frames()[2], [Box(20, 20, 80, 160, 1)], 640, 360)  # and a frame that was saved as empty turns out to have one
    restored, left = ws.undo_page(undo)
    assert sorted(left) == sorted([stems[0], stems[2]]) and sorted(restored) == sorted([stems[1], stems[3]])
    assert ws.review.get(stems[0]) == "pos" and s.label_path(stems[0]).read_text().strip() != ""  # the boxes are still there
    assert ws.review.get(stems[2]) == "pos" and s.label_path(stems[2]).read_text().strip() != ""
    assert ws.review.get(stems[1]) is None and ws.review.get(stems[3]) is None  # the rest is undone
    assert ws.undo_page({}) == ([], [])


def test_a_failing_note_file_never_blocks_a_decision(tmp_path, monkeypatch):
    _ui_session(tmp_path, n_game=2, n_ui=0)
    proj = _project(tmp_path)
    s = proj.sessions()[0]
    stems = [p.stem for p in s.frames()]
    ws = Workspace(proj, s)
    ws.check([stems[0]])

    def broken(*_a, **_k):
        raise PermissionError("the app folder is read-only")

    monkeypatch.setattr("framelabel.core.workspace.save_checked", broken)
    assert ws.save(s.frames()[0], [Box(10, 10, 100, 200, 0)], 640, 360) == "pos"  # the label and the status are saved
    assert s.label_path(stems[0]).read_text().strip() and Workspace(proj, s).review.get(stems[0]) == "pos"
    assert stems[0] not in ws.checked  # and the in-memory note is right
    ws.check([stems[1]])  # no exception either
    assert stems[1] in ws.checked


def test_checked_frames_are_a_personal_note_kept_outside_the_session(tmp_path):
    from framelabel.core.checked import checked_path, load_checked

    _ui_session(tmp_path, n_game=3, n_ui=0)
    proj = _project(tmp_path)
    s = proj.sessions()[0]
    stems = [p.stem for p in s.frames()]
    ws = Workspace(proj, s)
    ws.check([stems[0], stems[1]])
    assert load_checked(s) == {stems[0], stems[1]} and s.root not in checked_path(s).parents  # app data, not the session
    assert not any(p.name.startswith("checked") for p in s.root.iterdir())
    ws.save(s.frames()[0], [Box(10, 10, 100, 200, 0)], 640, 360)  # a decision makes the frame look-at-able again
    assert load_checked(s) == {stems[1]} and Workspace(proj, s).checked == {stems[1]}
    ws.uncheck([stems[1]])
    assert load_checked(s) == set()


def test_ghosts_are_the_proposals_a_decided_frame_does_not_cover(tmp_path):
    _ui_session(tmp_path, n_game=3, n_ui=0)
    proj = _project(tmp_path)
    s = proj.sessions()[0]
    stems = [p.stem for p in s.frames()]
    s.auto_dir.mkdir()
    s.auto_path(stems[0]).write_text("0 0.5 0.5 0.1 0.3 0.90\n0 0.2 0.5 0.1 0.3 0.70\n0 0.8 0.5 0.1 0.3 0.10")
    ws = Workspace(proj, s, conf=0.3)
    assert ws.ghosts(s.frames()[0], 640, 360) == []  # a frame without a decision shows its proposals as boxes
    ws.save(s.frames()[0], [Box(0.45 * 640, 0.35 * 360, 0.55 * 640, 0.65 * 360, 0)], 640, 360)  # covers the 0.90 one
    ghosts = ws.ghosts(s.frames()[0], 640, 360)
    assert [round(g.conf, 2) for g in ghosts] == [0.7]  # the 0.10 one is below the shown confidence
    ws.save(s.frames()[0], [], 640, 360)
    assert [round(g.conf, 2) for g in ws.ghosts(s.frames()[0], 640, 360)] == [0.9, 0.7]


def test_pace_needs_enough_decisions_and_forgets_old_ones():
    from framelabel.core.pace import Pace, format_minutes, pace_line

    now = [1000.0]
    pace = Pace(clock=lambda: now[0])
    assert pace.per_minute() is None and pace_line(pace, 100) == ""
    pace.add(5)
    now[0] += 60
    assert pace.per_minute() is None  # five decisions say nothing yet
    pace.add(25)
    rate = pace.per_minute()
    assert rate == 30.0  # 30 decisions in the 60 s since the first one
    assert pace.minutes_left(90) == 3.0 and pace_line(pace, 90) == "30 per min, about 3 min left"
    assert pace_line(pace, 0) == ""
    now[0] += 700  # everything is older than ten minutes now
    assert pace.per_minute() is None
    assert format_minutes(0.3) == "under a minute" and format_minutes(80) == "1 h 20 min" and format_minutes(3000) == "2 d 2 h"


def test_pace_takes_back_undone_decisions():
    from framelabel.core.pace import Pace

    t = [0.0]
    p = Pace(clock=lambda: t[0])
    p.add(30)
    t[0] = 60.0
    assert p.per_minute() is not None
    p.add(-30)
    assert p.per_minute() is None  # nothing left to measure


def test_grow_box_moves_every_side_inside_the_image():
    from framelabel.core.boxes import Box, grow_box

    b = grow_box(Box(100, 50, 150, 90, 1), 2, 320, 180)
    assert (b.x1, b.y1, b.x2, b.y2, b.cls) == (98, 48, 152, 92, 1)
    b = grow_box(Box(1, 1, 318, 179), 3, 320, 180)
    assert (b.x1, b.y1, b.x2, b.y2) == (0, 0, 320, 180)  # kept inside the photo
    b = grow_box(Box(100, 50, 104, 54), -5, 320, 180)
    assert b.w >= 2 and b.h >= 2  # never turned inside out
    b = grow_box(Box(150, 90, 100, 50), -1, 320, 180)  # a box drawn backwards is normalised first
    assert (b.x1, b.y1, b.x2, b.y2) == (101, 51, 149, 89)


def test_review_keeps_the_class_list_the_labels_were_made_with(tmp_path):
    import json

    from framelabel.core.session import Review

    p = tmp_path / "review.json"
    p.write_text(json.dumps({"version": 1, "classes": ["a", "b"], "status": {"x": "pos"}}))
    r = Review(p, ["b", "a"])  # reordered: the numbers in the labels would change meaning
    assert r.classes_conflict
    r.set("y", "neg")
    assert json.loads(p.read_text())["classes"] == ["a", "b"]  # never overwritten by a conflicting list
    r = Review(p, ["a", "b", "c"])  # an extension is safe
    assert not r.classes_conflict
    r.set("z", "neg")
    assert json.loads(p.read_text())["classes"] == ["a", "b", "c"]
    r = Review(p, ["c", "a"])
    assert r.classes_conflict
    r.adopt_classes()  # the user said the numbers still mean the same
    assert json.loads(p.read_text())["classes"] == ["c", "a"] and not Review(p, ["c", "a"]).classes_conflict
    assert not Review(p).classes_conflict  # no project list: nothing to compare