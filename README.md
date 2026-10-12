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
*File > New project* writes one (*Settings > Project* edits it). A project file (YAML, see `examples/abi.yaml`) holds:

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
2. **The way through the work** (left panel; the design and its reasons: `docs/ui-design.md`). Three stages, each opening
   the way that suits it, then the library:
   * **1 Triage** (grid): the frames the model found nothing in come as a page of tiles. Click = has objects, MMB /
     Ctrl+click = skip, hold the right button = peek at the full-size photo, Space = save the page (unmarked tiles become
     empty, ticked ones go to Label). The toast has **Undo** (Ctrl+Z in the grid takes back the last saved pages; frames
     decided again since stay as they are). With
     *Tiles* 1 x 1 the tile is the photo itself (wheel zoom, drag pan, Space saves and shows the next). Lenses: *Weak
     proposals*, *Only in ignore zone*, *Interface screens* (pre-marked SKIP).
   * **2 Label** (editor): the frames with a proposal at the shown confidence, plus the ones you marked in Triage. Fix the
     boxes, Space saves and goes on. The order is a sort in the top bar (time, unsure proposals first, small objects
     first). Lenses: *From the model*, *Marked in Triage*.
   * **3 Review** (grid): saved frames that deserve a second look: *Possibly missed* (the model is confident about a box
     that was not drawn, or the frame is empty), *Possibly wrong* (a box with no matching proposal), *Small objects*.
     Tiles and the editor show the labels in the class colours and the proposals they do not cover as dashed amber boxes
     (in the editor a click adds one). Space saves the page as **checked** (K marks one frame in the editor); checked
     frames leave Review.
   * **Library**: all frames, new, labeled, empty, skipped, by class (grid; double-click a tile = open it in the editor,
     Tab comes back to its page; labeled tiles are locked in the grid so a click cannot cost them their boxes).

   Every stage row shows what is left (a green check when done, a bar for how far Triage and Label are), the icon of the
   way it opens, and its key (Ctrl+1 / 2 / 3); V goes to the next stage that has frames. A frame is *new* (no
   decision), *marked* ('has objects' ticked in Triage, boxes still to do), *labeled*, *empty* or *skipped*. The **shown
   confidence** (right panel, with presets) decides which frames are in Triage and which in Label; it is applied at once
   and remembered per project (the project's `show_conf` is only the first default). *Tiles* (top bar, or *View >
   Grid tiles per page*): 1 x 1, 2 x 1, 1 x 2, 2 x 2 ... 5 x 4. *Ctrl+B* hides the side panels, **Ctrl+K** searches
   views, actions and frames, F1 lists every key.
3. **Editor**: proposals are dashed boxes with their confidence; drag on empty space = new box, drag a box = move,
   drag a corner or an edge = resize, RMB = delete, wheel = zoom at cursor, MMB drag = pan. Space = accept and next,
   E = empty, X = skip, A/D = previous/next, N = next unreviewed, Ctrl+Z / Ctrl+Y = undo/redo, R = back to the model's
   proposals, Shift+Space = accept all proposals as labels. M toggles the loupe: it magnifies what is on screen
   (3x by default, following the view zoom smoothly); Ctrl+wheel or +/- change it (1.5-10x, remembered).
4. **Faster boxes**: W shows weak proposals below the threshold (click one to accept it); P copies the boxes of the
   previous labeled frame, T tracks them onto this frame (template matching, then snapped to overlapping proposals);
   "Carry boxes" does that automatically when moving to an unreviewed frame without proposals. F fits the selected
   box with SAM, Alt+click creates a box from a point with SAM, "Fit new boxes with SAM" refines every drawn box.
   SAM runs on a crop around the object, so far (small) people are fitted too.
5. **Where you are and what happened**: the thin bar above the work area says which stage (and lens) and where in it, in
   the grid what saving the page will change; the hint card under the list says what the current view is for; the keys
   of the mode are in the status bar; actions confirm themselves with a short message over the image (*Saved: 2 boxes*,
   *Page saved: ...*), and at the end of a view the message has a button to the next stage. Under the progress bar the
   pace appears once there is enough to measure ("42 per min, about 1 h 20 min left").
6. **Export** (*File > Export*, `python -m framelabel export projects\abi.yaml`): all labeled frames plus a share of
   empty ones (half of them the model's hardest false positives); validation = random 2-minute time blocks of every
   session, chosen so every class's val share is close to the target (neighbouring frames are near-duplicates, so a
   random split would leak). The same seed gives the same split as screendet's `label_frames.py export`.
7. **Retrain** (*Auto-label > Retrain*): exports, then fine-tunes a chosen model on your labels in a background
   process (log in the dialog); the result appears in the prelabel model list.

What counts as an object, how tight a box is, and when to mark a frame empty or skipped: `docs/labeling-guide.md`.

The frame list on the left has a filter box (part of a name or of a tag such as `new`, `empty`, `marked`); in Review
it also says what was found (`missed 1, small`). F5 re-reads
proposals and new frames without leaving the current frame, Ctrl+G jumps to a frame by number or name, and the
editor reopens where you stopped in each queue. *Edit > Fit all boxes on this frame with SAM* refines every box.

All keys can be changed in *Settings > Keyboard shortcuts* (F1 shows what every key and mouse button does); letters work in any keyboard layout (a Russian layout
triggers the same actions).

## Labeler model for Arena Breakout Infinite

The default prelabeler for the ABI project is an ensemble of `train_runs/abi_own_coco_y26x` (YOLO26x from COCO
weights fine-tuned on the own labeled frames) and screendet's `coco_y26m_own`, with test-time augmentation. Measured
against screendet's previous prelabeler `enemy_y26s` on the own val split it has about 3x fewer false positives per
frame (7x fewer on empty frames), twice the recall on 20-50 px people and corpse proposals; numbers and the
alternatives that were tried are in `docs/labeler-model.md`. Training (`python -m framelabel train`):

* `--preset abi-own --model models/yolo26x.pt --name abi_own_coco_y26x` - the model above. Data:
  `tools/split_own_dev.py` cuts the own train split into train and a dev split for picking the epoch, so the own
  val split stays unseen until the final comparison.
* `--preset abi-person`, then `--preset abi-own` - a variant that first learns one class `person` at 640 px from
  public Arena Breakout Infinite sets (`tools/build_labeler_data.py`). The public sets were inspected visually;
  sets from other games (Escape from Tarkov, Arma 3, Modern Warfare II, Arena Breakout mobile, mixed "humanoid"
  sets) are excluded - the list and reasons are in `tools/abi_sources.yaml`. Public labels mark corpses as enemies,
  which made this variant worse on corpses.

Compare labelers on the own val split, and on empty frames (`tools/build_neg_eval.py`):

```
python -m framelabel eval --data ..\fin_snip_helper\datasets\own\data.yaml --project projects\abi.yaml ^
    --setup old=..\fin_snip_helper\train_runs\enemy_y26s\weights\best.pt new=train_runs\abi_own_coco_y26x\weights\best.pt
```

It reports AP50 and, per confidence, precision, recall, false positives per frame and recall by box height
(<20, 20-50, 50+ px at 1080p). `NAME=A.pt+B.pt@1536:tta` evaluates an ensemble / another size / TTA.

## What is filtered out, and where to see it

Frames can be hidden from the normal views in several places; *View > Session report* (or the *Report* button next to
the session) counts every one of them and double-clicking a row opens the view that shows those frames. The
lenses of Triage and the status bar carry the same numbers.

* **The recorder** (ImageHook, `ui_filter`) drops interface screens, duplicates and non-game frames *before* saving;
  its counts are read from the session's `capture.json`. These frames are not on disk and cannot be reviewed here
  (record with `ui_filter` off to keep them).
* **Suspected interface screens** (framelabel's own detection, lens *Triage > Interface screens*): pre-marked SKIP; the status line says how
  many frames of the page are pre-marked, a click on a tile undoes the mark.
* **Proposals hidden by ignore zones** (lens *Triage > Only in ignore zone*): the model detected something only inside an ignore zone
  (the HUD icon), so the frame looks empty; the hidden boxes are drawn in cyan.
* **Weak proposals** (lens *Triage > Weak proposals*): below the shown confidence; dotted boxes, `W` in the editor shows them.
* **Dropped classes**: model classes without a project class (`class_map`) are not written; a prelabel run records the
  counts (frames without any detection, frames only inside a zone, dropped classes) in `labels_auto/_meta.json` under
  `run` and shows them when it finishes.

## Responsiveness

Nothing that can take longer than ~50 ms runs on the window's thread:

* Opening a session (listing, parsed proposal/label files, interface-screen scores) runs in a background thread with
  a progress bar. Parsed label rows are cached in `%LOCALAPPDATA%\framelabel\cache` (keyed by file time and size, so
  only changed files are re-read), and interface-screen scores of newly captured frames are computed on their own.
  A 7000-frame session opens in about half a second with a warm cache.
* Prelabel runs as a child process (`python -m framelabel prelabel ... --progress-json`), so torch is not loaded into
  the window and a CUDA failure cannot close it. SAM runs in its own thread. Export (and the export before a retrain)
  runs in the background and is swapped in atomically: a cancelled export keeps the previous dataset.
* The editor reads the next frames ahead, the grid reloads tiles only after a window resize settles.
* *Settings > Performance* sets the I/O threads, the read-ahead, an optional SAM preload and the freeze watchdog (all apply at once).

Logs and caches live in `%LOCALAPPDATA%\framelabel` (override with `FRAMELABEL_HOME` / `FRAMELABEL_CACHE_DIR`):
`logs/framelabel.log` (rotating), `logs/freeze.log` (the stacks of all threads when the window was blocked for 2 s),
`cache/` (safe to delete, it is rebuilt). `python tools/profile_session.py projects\abi.yaml` prints open/queue
timings of the biggest session without writing into it.

## Safety

GPU work (prelabel, SAM, training, eval) refuses to start while the game process runs (`game_processes`, default
`UAGame.exe`; the game crashed once from resource overuse), caps PyTorch VRAM, and the UI warns when another Python
process is using the GPU. Training keeps dataloader workers at 2 (each is a full process on Windows). Prelabel in
the window is a separate process: *Cancel* stops it after the current batch, and quitting asks before stopping it.

## Layout

* `framelabel/core` - pure logic: boxes and YOLO files, sessions and `review.json`, project files, queues and views,
  splitting and export, interface-screen detection, box tracking, undo history, view math, `index_cache.py`
  (cached label rows, `Cancelled`), `report.py` (what is filtered out), `views.py` (stages, lenses, counts),
  `checked.py` (frames checked in Review), `pace.py`.
* `framelabel/ml` - GPU parts: guard, model registry, prediction + WBF, SAM fitting, evaluation, training.
* `framelabel/gui` - PyQt6 window, canvas, grid, dialogs, keymap; background work: `workers.py` (thread jobs),
  `process_job.py` (child processes), `sam_worker.py`, `frame_list.py` (model/view), `workflow.py` (stages, modes, counts), `panels.py` (workflow list, mode switch,
  context bar, hint card, toast), `palette.py` (Ctrl+K), `theme.py`, `icons.py`, `watchdog.py`.
* `framelabel/perf.py` - logging setup, timing helper, per-user app folder.
* `tools/` - labeler data preparation, `profile_session.py`; `tests/` - pytest (GUI tests run headless).
