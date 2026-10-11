# Labeler model for Arena Breakout Infinite

Goal: proposals good enough that labeling is mostly accepting boxes: few false positives per frame (each one is a box
to delete) and high recall on medium and far (small) people, which matter most. Speed is not a concern.

## Data

* Public sets: 18 Roboflow sets of PC Arena Breakout Infinite footage, 7348 images with people + 734 empty
  (`tools/build_labeler_data.py`, sources and exclusions in `tools/abi_sources.yaml`). Excluded after looking at
  random labeled samples of every set: Escape from Tarkov (9 sets, some of them were in screendet's merged set),
  Arma 3, Arena Breakout mobile, Modern Warfare II frames inside a mixed set, a multi-game "humanoid" set.
  Public labels mark corpses (and web-page thumbnails) as enemies.
* Own frames (`fin_snip_helper/datasets/own`): train split 1193 images, cut by time blocks into 1043 train + 150 dev
  (`tools/split_own_dev.py`); the dev split picks the epoch. The val split (211 images, 151 enemy + 64 corpse boxes)
  is used only for the final comparison below.
* Negatives test (`tools/build_neg_eval.py`): 500 random frames reviewed as empty that are not in the export and not
  within 10 s of an exported frame. Most frames met while labeling are empty, and the val split is 85 % frames with
  people, so false positives per frame are reported on both.

## Models

| name | how |
| --- | --- |
| A `train_runs/abi_own_y26x` | YOLO26x: stage 1 one class `person` on public + own train at 640 (40 epochs, mAP50 0.94 on public val), stage 2 enemy/corpse on own train at 1280 (freeze 10, AdamW 5e-4) |
| B `train_runs/abi_own_coco_y26x` | YOLO26x from COCO weights, stage 2 only (same settings) |
| old `enemy_y26s` | screendet's previous prelabeler (YOLO26s on the merged public sets) |
| `enemy26s_own` | enemy_y26s fine-tuned on own frames (screendet `own_y26s_f16`) |
| `coco_y26m_own` | screendet: YOLO26m from COCO on own frames. Its best epoch was picked on the own val split, so its val numbers below are slightly optimistic |

The public "person" stage (A) hurts the corpse class (public labels call corpses enemies), as screendet found for
its detectors; starting from COCO (B) is better.

## Results (`python -m framelabel eval`, inference 1280, HUD-icon zone ignored)

Own val, 211 frames. AP50 per category; then at the confidence where proposals are shown (0.3 / 0.2):
false positives per frame (class-agnostic) and person recall by box height at 1080p.

| setup | person AP50 | enemy AP50 | corpse AP50 | FP/frame @0.3 | recall <20 px @0.3 | 20-50 px @0.3 | 50+ px @0.3 | FP/frame @0.2 | <20 / 20-50 px @0.2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old enemy_y26s | 0.526 | 0.735 | - | 0.152 | 0.00 | 0.32 | 0.73 | 0.194 | 0.00 / 0.34 |
| old enemy_y26s @1920 | 0.586 | 0.802 | - | 0.190 | 0.04 | 0.39 | 0.72 | 0.299 | 0.04 / 0.41 |
| enemy26s_own | 0.644 | 0.802 | 0.179 | 0.081 | 0.04 | 0.32 | 0.75 | 0.175 | 0.07 / 0.37 |
| coco_y26m_own | 0.750 | 0.834 | 0.512 | 0.104 | 0.29 | 0.54 | 0.82 | 0.152 | 0.29 / 0.54 |
| A | 0.698 | 0.846 | 0.295 | 0.142 | 0.04 | 0.41 | 0.84 | 0.256 | 0.04 / 0.49 |
| B | 0.731 | 0.875 | 0.370 | 0.081 | 0.14 | 0.54 | 0.83 | 0.100 | 0.14 / 0.56 |
| B + TTA | 0.780 | 0.903 | 0.448 | 0.038 | 0.14 | 0.61 | 0.82 | 0.047 | 0.18 / 0.66 |
| A + B (WBF) | 0.765 | 0.902 | 0.364 | 0.066 | 0.11 | 0.49 | 0.81 | 0.095 | 0.14 / 0.54 |
| B + coco_y26m_own | 0.778 | 0.880 | 0.496 | 0.052 | 0.14 | 0.51 | 0.81 | 0.095 | 0.21 / 0.56 |
| **B + coco_y26m_own + TTA** | **0.831** | **0.902** | **0.612** | **0.028** | 0.14 | 0.54 | 0.79 | **0.047** | 0.25 / 0.61 |

Negatives test, 500 empty frames: false positives per frame at 0.2 / 0.3 / 0.4

| setup | @0.2 | @0.3 | @0.4 |
| --- | --- | --- | --- |
| old enemy_y26s | 0.236 | 0.136 | 0.076 |
| enemy26s_own | 0.108 | 0.038 | 0.022 |
| coco_y26m_own | 0.036 | 0.016 | 0.010 |
| A | 0.120 | 0.062 | 0.034 |
| B | 0.030 | 0.018 | 0.010 |
| B + coco_y26m_own | 0.020 | 0.010 | 0.000 |

## Default

`projects/abi.yaml` (and `examples/abi.yaml`) prelabels with the ensemble **B + coco_y26m_own with TTA**, stores
proposals from 0.05 and shows boxes from 0.2. Against the old prelabeler at its 0.3 threshold that is about 3x fewer
false positives on labeled frames (0.047 vs 0.152) and 7x fewer on empty frames (0.020 vs 0.136 at 0.2), recall
for 20-50 px people 0.61 vs 0.32 and for people under 20 px 0.25 vs 0.00, and corpse proposals (AP50 0.61), which
the old model did not have. About 0.5 s per frame on the RTX 5070 Ti (2 models x 4 passes), 1.2 GB VRAM.

Without the screendet model, B + TTA alone is close (enemy AP50 0.903, FP/frame 0.047 at 0.2) but weaker on corpses.

Limits: people under 20 px are still mostly missed (8 such boxes in val for enemies, 28 for all classes; the numbers
are noisy); "empty" frames in the negatives test may contain a few tiny people the user did not label, which would
count against the models that find them. Retraining on more labeled frames (Auto-label > Retrain) is the way to
improve corpses and far people.
