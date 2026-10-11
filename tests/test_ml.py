import numpy as np

from framelabel.__main__ import parse_setup
from framelabel.ml.evaluate import Split, average_precision, class_map_by_name, evaluate_predictions, match
from framelabel.ml.fusion import weighted_boxes_fusion
from framelabel.ml.models import scan_models
from framelabel.ml.predict import rows_for, tta_sizes
from framelabel.ml.sam import crop_window, mask_box


def test_wbf_merges_overlaps_and_downweights_single_votes():
    a = (np.array([[10, 10, 50, 90], [200, 200, 220, 240]]), np.array([0.9, 0.6]), np.array([0, 0]))
    b = (np.array([[12, 10, 52, 92]]), np.array([0.7]), np.array([0]))
    xyxy, conf, cls = weighted_boxes_fusion([a, b], iou_thr=0.5)
    assert len(xyxy) == 2 and cls.tolist() == [0, 0]
    assert np.allclose(xyxy[0], [(10 * 0.9 + 12 * 0.7) / 1.6, 10, (50 * 0.9 + 52 * 0.7) / 1.6, (90 * 0.9 + 92 * 0.7) / 1.6])
    assert np.isclose(conf[0], 0.8) and np.isclose(conf[1], 0.3)  # found by one of two passes: halved
    c = (np.array([[10, 10, 50, 90]]), np.array([0.8]), np.array([1]))  # another class is never merged
    assert len(weighted_boxes_fusion([a, c])[0]) == 3
    assert len(weighted_boxes_fusion([])[0]) == 0


def test_rows_for_clips_and_filters():
    rows = rows_for(np.array([[-5, 10, 50, 90], [0, 0, 0.5, 10]]), np.array([0.9, 0.9]), np.array([1, 0]), 100, 100, 0.1)
    assert rows == ["1 0.250000 0.500000 0.500000 0.800000 0.900"]
    assert rows_for(np.array([[0, 0, 50, 50]]), np.array([0.05]), np.array([0]), 100, 100, 0.1) == []
    assert tta_sizes(1280, True) == [1280, 1664] and tta_sizes(1280, False) == [1280]


def test_match_and_ap():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    pred = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [21, 20, 30, 30]], float)
    tp, hit = match(pred, np.array([0.9, 0.8, 0.7]), gt)
    assert tp.tolist() == [True, False, True] and hit.all()
    assert 0.6 < average_precision(np.array([0.9, 0.8, 0.7]), tp, 2) < 0.9


def test_evaluate_counts_fp_per_frame_and_recall_by_height(tmp_path):
    import cv2

    img_dir, lab_dir = tmp_path / "images", tmp_path / "labels"
    img_dir.mkdir()
    lab_dir.mkdir()
    for k in ("a", "b"):
        cv2.imwrite(str(img_dir / f"{k}.jpg"), np.zeros((1080, 1920, 3), np.uint8))
    (lab_dir / "a.txt").write_text("0 0.5 0.5 0.01 0.0148\n1 0.2 0.5 0.05 0.1")  # 16 px and 108 px tall
    (lab_dir / "b.txt").write_text("")
    split = Split(sorted(img_dir.iterdir()), lab_dir, ["enemy", "corpse"])
    big = [0.2 * 1920 - 48, 540 - 54, 0.2 * 1920 + 48, 540 + 54]
    preds = {"a": (np.array([big, [10, 900, 60, 1000]]), np.array([0.9, 0.95]), np.array([1, 0])),
             "b": (np.array([[100, 100, 150, 200]]), np.array([0.25]), np.array([0]))}
    res = evaluate_predictions(split, preds, ignore_zones=[(0.0, 0.75, 0.07, 0.92)])  # the 0.95 box is in the zone
    person = res["categories"]["person"]["at_conf"]
    assert person["0.2"]["fp"] == 1 and person["0.2"]["fp_per_frame"] == 0.5 and person["0.3"]["fp"] == 0
    rb = person["0.3"]["recall_by_height"]
    assert rb["0-20px"] == (0.0, 1) and rb["50-px"] == (1.0, 1)
    assert res["categories"]["corpse"]["at_conf"]["0.5"]["recall"] == 1.0


def test_class_map_by_name():
    assert class_map_by_name(["enemy"], ["enemy", "corpse"]) == {0: 0}
    assert class_map_by_name(["person"], ["enemy", "corpse"]) == {0: 0}
    assert class_map_by_name(["corpse", "car"], ["enemy", "corpse"]) == {0: 1, 1: None}


def test_parse_setup():
    name, models, imgsz, tta = parse_setup("ens=a.pt+b.pt@1536:tta")
    assert name == "ens" and [m.name for m in models] == ["a.pt", "b.pt"] and imgsz == 1536 and tta
    assert parse_setup("x/best.pt")[0] == "best" and parse_setup("x/best.pt")[2] is None


def test_scan_models_finds_runs(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "yolo.pt").write_bytes(b"0")
    (tmp_path / "m" / "sam2.1_b.pt").write_bytes(b"0")
    run = tmp_path / "runs" / "abi" / "weights"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"0")
    (run.parent / "results.csv").write_text("epoch,  metrics/mAP50(B)\n1,0.5\n2,0.7\n")
    found = scan_models([tmp_path / "m", tmp_path / "runs"])
    assert [m.label for m in found] == ["yolo", "abi (best)"] and found[1].info == "2 epochs, best mAP50 0.700"


def test_sam_crop_and_mask_box():
    assert crop_window(100, 100, 110, 130, 1920, 1080) == (9, 19, 201, 211)
    assert crop_window(0, 0, 10, 10, 50, 50) == (0, 0, 50, 50)
    m = np.zeros((50, 50), np.uint8)
    m[10:20, 5:15] = 1
    m[30:45, 30:48] = 1
    assert mask_box(m, near=(8, 12)) == (5.0, 10.0, 15.0, 20.0)
    assert mask_box(m) == (30.0, 30.0, 48.0, 45.0)
    assert mask_box(np.zeros((5, 5))) is None
