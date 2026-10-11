# framelabel

A desktop labeling app for object-detection datasets in YOLO format, built for frames recorded from games and video:
fast grid triage of empty frames, a box editor with model proposals, review queues, SAM box fitting, box tracking
between neighbouring frames, batch auto-labeling with a selectable model or an ensemble, export with a leak-free
validation split, and retraining a labeler model on your own labels.

It started as `tools/label_frames.py` of the screendet project (screen capture -> TensorRT YOLO detector for Arena
Breakout Infinite) and is now developed as one of three sibling projects of that system:

| project | role |
| --- | --- |
| `ImageHook_ai` | frame capture: records gameplay sessions (`session_YYYYmmdd_HHMMSS/frames/HHMMSS_mmm.jpg`) |
| `ImageClassfication_ai` (this) | labeling: proposals, review, export to a YOLO dataset; labeler model training |
| `fin_snip_helper` (screendet) | real-time detection, detector training (`tools/train.py`) and model comparison |

Sessions written by the capture tool and by screendet open here without conversion, and `export` writes the same
`datasets/own` layout (and the same validation split) that screendet's `tools/train.py --own` trains on.

## Setup

Windows, Python 3.13, an NVIDIA GPU for auto-labeling/SAM/training (labeling itself needs no GPU).

```
py -3.13 -m venv --system-site-packages .venv      # reuses the global CUDA build of torch (and PyQt6)
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest tests
```

Install torch with CUDA separately (https://pytorch.org) if it is not installed globally. Model weights go to
`models/` (ignored by git): the labeler base `yolo26x.pt`, `yolo11x.pt` and `sam2.1_b.pt` are official Ultralytics
releases (`from ultralytics.utils.downloads import attempt_download_asset`).

## Running

```
.venv\Scripts\python -m framelabel projects\abi.yaml      # a project file
.venv\Scripts\python -m framelabel D:\frames\session_x    # or any folder (session, folder of sessions, images, YOLO set)
```

Without a project file the class list comes from the folder's `review.json` (or a single class `object`);
*File > New project* writes one. A project file (YAML, see `examples/abi.yaml`) holds:

* `classes` - names and colours; the order is the class id. Digits 1-9 select a class.
* `sources` - session folders, folders of sessions (`datasets/collected`), plain image folders or YOLO datasets
  (`images/<split>` -> `labels/<split>`). Paths are relative to the project file.
* `ignore_zones` - normalized areas where model proposals are ignored (a HUD icon detected as a person).
* `ui_detect` - a HUD element present in every gameplay frame; frames where it is missing (menus, inventory, map,
  death screens) are pre-marked "skip".
* `model_dirs`, `default_models`, `class_map` - models offered for auto-labeling and how their classes map to the
  project's (by name by default; a one-class model labels class 1).
* `export`, `prelabel` - defaults for the export and prelabel dialogs, `game_processes` - processes that block GPU work.

### Data layout

| layout | images | labels | model proposals | review statuses |
| --- | --- | --- | --- | --- |
| session | `<s>/frames/` | `<s>/labels/` | `<s>/labels_auto/` | `<s>/review.json` |
| YOLO set | `<d>/images/<split>/` | `<d>/labels/<split>/` | `<d>/labels_auto/<split>/` | `<d>/review/<split>.json` |
| folder | `<f>/` | `<f>/labels/` | `<f>/labels_auto/` | `<f>/review.json` |

Labels are YOLO rows `cls cx cy w h` (normalized); proposals add a confidence column. `review.json` keeps the
per-frame status: `pos` (labeled with objects), `neg` (empty), `skip` (unusable), `todo` (marked in the grid, not
labeled yet). Every decision is written immediately (atomically), so labeling can stop and resume at any time.

## Workflow

1. **Auto-label** (*Auto-label > Prelabel*, or `python -m framelabel prelabel projects\abi.yaml`): pick one model, or
   check several for an ensemble (boxes merged with weighted box fusion), the inference size, and optional
   test-time augmentation (horizontal flip + a 1.3x larger scale, fused). Proposals from confidence 0.05 are stored,
   so the shown threshold can be changed later without re-running. GPU: refuses while the game runs, caps VRAM.
2. **Grid triage**: frames where the model found nothing come as a page of tiles (weak proposals drawn dotted).
   LMB = has objects, MMB/Ctrl+LMB = skip, RMB = full-size zoomable preview, Space = commit the page (unmarked
   frames become empty). Grid views: model found nothing, suspected interface screens, saved as empty, skipped, all.
3. **Editor**: proposals are dashed boxes with their confidence; drag on empty space = new box, drag a box = move,
   drag a corner or an edge = resize, RMB = delete, wheel = zoom at cursor, MMB drag = pan. Space = accept and next,
   E = empty, X = skip, A/D = previous/next, N = next unreviewed, Ctrl+Z / Ctrl+Y = undo/redo, R = back to the model's
   proposals, Shift+Space = accept all proposals as labels.
4. **Faster boxes**: W shows weak proposals below the threshold (click one to accept it); P copies the boxes of the
   previous labeled frame, T tracks them onto this frame (template matching, then snapped to overlapping proposals);
   "Carry boxes" does that automatically when moving to an unreviewed frame without proposals. F fits the selected
   box with SAM, Alt+click creates a box from a point with SAM, "Fit new boxes with SAM" refines every drawn box.
   SAM runs on a crop around the object, so far (small) people are fitted too.
5. **Review queues** (left panel): to review, uncertain proposals first (confidence 0.15-0.6), small/far objects
   first, "model disagrees with labels" (confident proposals without a label, labels without a proposal: catches
   missed boxes), by status, by class.
6. **Export** (*File > Export*, `python -m framelabel export projects\abi.yaml`): all labeled frames plus a share of
   empty ones (half of them the model's hardest false positives); validation = random 2-minute time blocks of every
   session, chosen so every class's val share is close to the target (neighbouring frames are near-duplicates, so a
   random split would leak). The same seed gives the same split as screendet's `label_frames.py export`.
7. **Retrain** (*Auto-label > Retrain*): exports, then fine-tunes a chosen model on your labels in a background
   process (log in the dialog); the result appears in the prelabel model list.

All keys can be changed in *File > Keyboard shortcuts*; letters work in any keyboard layout (a Russian layout
triggers the same actions).

## Labeler model for Arena Breakout Infinite

The default prelabeler `train_runs/abi_own_y26x` is a YOLO26x trained in two stages
(`python -m framelabel train --preset abi-person`, then `--preset abi-own`):

1. one class `person` at 640 px on public Arena Breakout Infinite sets plus the train split of the own labeled frames
   (`tools/build_labeler_data.py`). The public sets were inspected visually; sets from other games (Escape from
   Tarkov, Arma 3, Modern Warfare II, Arena Breakout mobile, mixed "humanoid" sets) are excluded - the list and
   reasons are in `tools/abi_sources.yaml`. Public labels mark corpses as enemies, so they only teach "a human body".
2. two classes `enemy`/`corpse` at 1280 px on the own frames (`tools/split_own_dev.py`: the own train split cut into
   train and a dev split for picking the epoch), so the own val split stays unseen until the final comparison.

Compare labelers on the own val split (`docs/labeler-model.md` has the measured results):

```
python -m framelabel eval --data ..\fin_snip_helper\datasets\own\data.yaml --project projects\abi.yaml ^
    --setup old=..\fin_snip_helper\train_runs\enemy_y26s\weights\best.pt new=train_runs\abi_own_y26x\weights\best.pt
```

It reports AP50 and, per confidence, precision, recall, false positives per frame and recall by box height
(<20, 20-50, 50+ px at 1080p). `NAME=A.pt+B.pt@1536:tta` evaluates an ensemble / another size / TTA.

## Safety

GPU work (prelabel, SAM, training, eval) refuses to start while the game process runs (`game_processes`, default
`UAGame.exe`; the game crashed once from resource overuse), caps PyTorch VRAM, and the UI warns when another Python
process is using the GPU. Training keeps dataloader workers at 2 (each is a full process on Windows).

## Layout

* `framelabel/core` - pure logic: boxes and YOLO files, sessions and `review.json`, project files, queues,
  splitting and export, interface-screen detection, box tracking, undo history, view math.
* `framelabel/ml` - GPU parts: guard, model registry, prediction + WBF, SAM fitting, evaluation, training.
* `framelabel/gui` - PyQt6 window, canvas, grid, dialogs, keymap, background jobs.
* `tools/` - labeler data preparation; `tests/` - pytest (GUI tests run headless).
