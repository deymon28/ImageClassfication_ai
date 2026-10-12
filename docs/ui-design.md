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

* **To do** - only frames without a decision, so what is left is always visible: *Triage: model found nothing*,
  *Label: proposals and marked*, *Marked, boxes still to do*, *Unsure proposals first*, *Small (far) objects first*,
  *All new frames*.
* **Check** - things that need a second look: *Looks like an interface screen*, *Seen only in an ignore zone*,
  *Weak proposals only* (all three: new frames) and *Model disagrees with my labels* (decided frames).
* **Done** - the decisions: *Labeled*, *Empty*, *Skipped*.
* **Browse** - *All frames*, *Has class X*.

There is no "hide reviewed" switch any more: the To do and Check views never contain decided frames, the Done and
Browse views always show everything. A to-do view that is finished says so ("Nothing in ... Every frame here has a
decision").

## Panels

* **Left**: session (+ *Report*), a progress bar of decided frames with the per-status counts, the Grid | Editor switch,
  the views, the frames of the current view (with a filter).
* **Right, top** ("This frame" / "This page"): editor: class for new boxes, the boxes of the frame; grid: what
  committing the page will do (n objects -> editor, n skip, n empty).
* **Right, bottom** ("Actions and settings"): buttons for the actions of the current mode (they show their key),
  the shown confidence with presets (applied at once, remembered per project), weak proposals, editor helpers, help.
* **Menus**: File (projects, export), Edit, View, Auto-label, **Settings** (project, keyboard shortcuts, performance:
  all of them take effect immediately), Help (F1: keys and mouse for each mode, generated from the current keymap).
* **Feedback**: every action that changes data confirms itself with a toast over the work area; the status bar keeps
  the technical line. Grid with 1 x 1 tiles is the photo itself (zoom and pan without a right click).

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

One dark theme in `framelabel/gui/theme.py`: design tokens (`T`: three background levels, text levels, one blue accent,
status colours), the palette and a single style sheet; the painted canvas and grid use the same tokens. Widgets are
styled by class, by object name (`#contextBar`, `#segTrack`, `#toast`) or by a `role` / `primary` property, never
inline. Rules: the accent colour means "this is where you are / what you chose" (the selected view, the active mode,
the main action, tiles marked *has objects*); status colours are used only for statuses (labeled green, marked amber,
skipped orange, empty grey); one radius family (6-10 px), thin borders, no shadows. Icons the style sheet needs
(checkmark, chevrons) are small SVGs written to the per-user data folder at start.
