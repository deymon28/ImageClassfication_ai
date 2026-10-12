# Window design: views, modes and where you are

The window answers three questions at every moment, each in its own place and with its own control:

| question | answer | where |
| --- | --- | --- |
| **Which frames?** | a *view*: a named set of frames, with its count | left, the grouped list |
| **How do I look at them?** | a *mode*: **Grid** (many tiles) or **Editor** (one frame) | left, the Grid / Editor switch |
| **Where am I, what can I do?** | view, mode, position, a one-line description and the keys | the bar above the work area |

A view and a mode are independent. Clicking a view shows it the way you looked at it last time (else the way it suits
best: triage is a grid, labeling is the editor); the switch changes only the mode and keeps the view.

## Vocabulary (used in every label, list tag and message)

A frame is **new** (no decision), **marked** ("has objects" ticked in the grid, boxes still to do), **labeled**,
**empty** or **skipped**. *Decided* means labeled, empty or skipped. (`review.json` calls them `None`, `todo`, `pos`,
`neg`, `skip`; the file format did not change.)

## Views

Defined in `framelabel/core/views.py` (one place: label, group, default mode, description; counts are computed in a
background thread, debounced after each decision). The rules of each view are in `core/queues.py`.

* **To do** - only frames without a decision, so what is left is always visible: *Triage: nothing found*,
  *To label*, *Marked in grid*, *Unsure first*, *Small objects first*,
  *All new frames*.
* **Check** - things that need a second look: *Interface screens*, *Only in ignore zone*,
  *Weak proposals only* (all three: new frames) and *Disagrees with labels* (decided frames).
* **Done** - the decisions: *Labeled*, *Empty*, *Skipped*.
* **Browse** - *All frames*, *Has class X*.

There is no "hide reviewed" switch any more: the To do and Check views never contain decided frames, the Done and
Browse views always show everything. A to-do view that is finished says so ("Nothing in ... Every frame here has a
decision").

## Panels

The photos are the product, so the chrome is small: the side panels are narrow (about 224 and 236 px, 71% of a
1600 px window is photo), have no title bars, and *Ctrl+B* hides them (100% photo).

* **Top row** (one line): the view, where in it (`Grid / page 3 of 70 / 630 frames`, `Editor / frame 5 of 80 /
  name / status`), in the grid what the page will do (`3 has objects  9 skip  0 empty`) and the tiles per page. The
  description of the view is the tooltip of its title.
* **Left**: session (+ *Report*), a thin progress bar with the counts of decided frames, the Grid | Editor switch, the
  views, the frames of the current view with a filter.
* **Right, top** (editor only): class for new boxes, the boxes of the frame.
* **Right, bottom**: buttons for the actions of the current mode (they show their key; the main one is coloured),
  the shown confidence with presets (applied at once, remembered per project), weak proposals, editor helpers.
* **Status bar**: the technical line on the left, the keys of the current mode on the right.
* **Menus**: File (projects, export), Edit, View, Auto-label, **Settings** (project, keyboard shortcuts, performance:
  all take effect immediately), Help (F1: keys and mouse for each mode, generated from the current keymap).
* **Feedback**: every action that changes data confirms itself with a toast over the work area. Grid with 1 x 1
  tiles is the photo itself (zoom and pan without a right click).

## Where the principles come from

* Nielsen Norman Group usability heuristics (secondary summaries): *visibility of system status* (context bar, counts,
  progress, toasts), *recognition rather than recall* (the list of views with counts instead of a drop-down that
  changes with the mode; buttons that show their keys; readable names for every shortcut).
* Apple Human Interface Guidelines, segmented controls: use them to switch between views of one context (Grid |
  Editor), not to trigger actions; one option is always selected and visible.
* Microsoft Windows app-settings guidelines: group related settings under headings, apply a change immediately,
  keep frequent commands out of settings (hence the actions and the confidence presets next to the work, not in a
  dialog).
* Annotation-tool practice (Supervisely labeling queues, annotation-project guidelines): "take the next item" with one
  action, keyboard shortcuts for speed with a visible, customizable list, a way to flag unusable items (skip).

## Look

One dark theme in `framelabel/gui/theme.py`: design tokens (`T`: near-black surfaces in four steps, three text levels,
one blue accent, status colours), the palette and a single style sheet; the painted canvas and grid use the same
tokens. Widgets are styled by class, by object name (`#contextBar`, `#segTrack`, `#toast`) or by a property (`role`,
`primary`, `chip`, `quiet`), never inline. Rules:

* Quiet chrome: hairline borders, no frames around lists, no shadows, 9 pt type, controls about 24 px high.
* The accent colour means "you are here / you chose this" (the selected view, the active mode, the main action, tiles
  marked *has objects*); status colours are used only for statuses (labeled green, marked amber, skipped orange,
  empty grey).
* Nothing explains what the user already knows: descriptions are tooltips, keys are in the status bar and in F1.
* Icons the style sheet needs (checkmark, chevrons) are small SVGs written to the per-user data folder at start.

The first version of the theme spent the centre on bigger type, a three-line header and 300 px panels; measuring
the share of the window the photo gets (63% then, 71% now) is the check to repeat after any layout change.
