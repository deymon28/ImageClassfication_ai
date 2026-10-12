# Window design: stages, lenses, the library and the way of looking

The work has an order, so the window shows the order. Three **stages**, each with the way of looking that suits it,
and a **library** for everything else:

| | stage | opens as | what you do | frames in it |
| --- | --- | --- | --- | --- |
| 1 | **Triage** | grid | click the tiles that show an object; Space saves the page, the unmarked tiles become empty | new frames the model found nothing in |
| 2 | **Label** | editor | fix the model's boxes or draw your own; Space saves and goes on | new frames with a proposal, plus the frames you marked in Triage |
| 3 | **Review** | grid | a second look at saved frames: fix them, or mark them checked | decided frames with a possibly missed / wrong box or a small object |
| | **Library** | grid | find anything: all, new, labeled, empty, skipped, checked, by class | every frame, by decision, class or "checked" |

The stage rows of the list carry a **number** (the order), an **icon** (the way it opens: a grid of tiles, or a box with
handles; the same two icons are on the Grid | Editor switch) and the **count** of what is left (a green check when it is
done, a thin bar for how far Triage and Label are). The stage that should be worked on next has its count highlighted.

A stage can be narrowed with a **lens**: Triage > *Weak proposals*, *Only in ignore zone*, *Interface screens*; Label >
*From the model*, *Marked in Triage*; Review > *Possibly missed*, *Possibly wrong*, *Small objects*. A lens is always a
subset of its stage (so its count never exceeds the stage's), and only the current stage shows its lenses.

The **order** of the Label stage (time, unsure proposals first, small objects first) is a sort in the top bar, not a
view: it does not change which frames there are.

**Without the mouse** (grid): the arrow keys move a cursor over the tiles (a dashed ring), Enter ticks the tile like a click,
X (or S) marks it SKIP like Ctrl+click, O opens it in the editor; A / D / PageUp / PageDown flip pages. *Settings > Arrow
keys in the grid* chooses what the arrows do, and only one of the two is on: **move between tiles** (they stay on the
page) or **flip pages** (Left / Up back, Right / Down forward, no cursor). The arrow keys and Enter belong to a list or
the tree while one has the keyboard focus.

**Checked** (Library): the frames checked in Review, so a mistaken check can be taken back: click the tiles to look at
again (the pill says RETURN, labeled tiles are not locked here) and press Space; with none ticked the whole page returns.
The toast has Undo.

**Mode is independent**: every view can be shown as a grid or in the editor (Tab, or the switch at the top; each view
remembers how you last looked at it). Double-click a tile (or O) opens it in the editor, Tab comes back to its page.

## Invariants (tested in `tests/test_core.py`)

* Triage and Label **partition** the frames without a decision (new and marked): every such frame is in exactly one.
* Triage and Label never contain decided frames; Review and the Library show decisions.
* A lens is a subset of its stage, in the stage's order.
* **Saving a grid page never changes a labeled frame**: labeled tiles are locked in the grid, and the workspace ignores
  them even for a stale mark; a labeled frame is changed in the editor. Marks that were not saved yet survive a
  rebuild (a switch to the editor and back) only for frames whose decision did not change meanwhile.
* Ticks not saved yet stay when you page back and forth; saving the page turns them into statuses.
* **Undoing a page never undoes later work**: only the frames that are still as the save left them are restored; a
  frame that was labeled or decided again since stays as it is (the toast says how many).

## Vocabulary (used in every label, list tag and message)

A frame is **new** (no decision), **marked** ("has objects" ticked in Triage, boxes still to do), **labeled**,
**empty** or **skipped**; *decided* means labeled, empty or skipped. (`review.json` calls them `None`, `todo`, `pos`,
`neg`, `skip`; the file format did not change.)

## The window

* **Top row**: the stage (and lens), where in it (`page 3 of 70`, `frame 5 of 80 / name / status`), in the grid what
  saving the page will change (`2 has objects  1 skip  6 empty`), the order (Label only), the tiles per page and the
  Grid | Editor switch. The description of the view is the tooltip of its title.
* **Left**: session (+ *Report*), a progress bar of decided frames with the pace ("42 per min, about 1 h 20 min left"
  once there is enough to measure), the workflow and library list, a hint card for the current view (what it is for
  and what to do), the frames of the view with a filter (Review adds what it found: `missed 1, small`).
* **Right, top** (editor only): class for new boxes, the boxes of the frame.
* **Right, bottom**: buttons for the actions of the current mode (they show their key; the main one is coloured), the
  shown confidence with presets (applied at once, remembered per project), weak proposals, editor helpers, Search and Help.
* **Status bar**: the technical line on the left, the keys of the current mode on the right.
* The side panels have no title bars and **Ctrl+B** hides them; their widths are remembered.

## Seeing what the model sees

Tiles and the editor draw the same things: the boxes of a labeled frame in the class colours, the model's proposals the
labels do not cover as **dashed amber** boxes ("model sees more": in the editor a click adds one), weak proposals dotted,
boxes hidden by an ignore zone cyan. Review is therefore a visual comparison, not a list of numbers. A tile shows the
frame's decision in its caption (labeled 2, empty, skipped, marked) and a pill only for a **pending change** (HAS
OBJECTS, SKIP, EMPTY); hovering a tile gives the details (boxes, proposals, interface-screen suspicion).

## Feedback and safety

* Every action that changes data confirms itself with a toast over the work area. At the end of a view the toast leads
  with what was just done and offers a button to the next stage that has work ("Go to Label").
* **Undo instead of confirmation**: saving a grid page shows a toast with *Undo* (only when the page changed
  something); Ctrl+Z in the grid, the *Undo page* button and *Edit > Undo the last saved grid page* take back the last
  saved pages (statuses and label files exactly as they were).
* At the end of a view the message says what is really left: frames passed over in this view ("Back to the start"),
  the rest of the stage after a lens, or the next stage with work; an empty lens never claims that every frame has a
  decision while its stage still has frames.
* **Checked** (Review): K marks a frame as checked, saving it in Review does too, and a saved page in Review is checked;
  checked frames leave Review. It is a personal note kept in the app data folder, never in the session. Saving a frame
  again makes it look-at-able again.
* **Ctrl+K** opens a palette that finds views, actions (with their keys, which teaches them) and frames by typing.

## Look

One dark theme in `framelabel/gui/theme.py`: design tokens (`T`: near-black surfaces in four steps, three text levels, one
blue accent, status colours), the palette and a single style sheet; the painted canvas and grid use the same tokens.
Widgets are styled by class, by object name (`#contextBar`, `#segTrack`, `#toast`, `#coach`) or by a property (`role`,
`primary`, `chip`, `quiet`), never inline. Rules:

* Quiet chrome: hairline borders, no frames around lists, no shadows, 9 pt type, controls about 24 px high.
* The accent colour means "you are here / you chose this" (the selected row, the active mode, the main action, tiles
  marked *has objects*); status colours are used only for statuses (labeled green, marked amber, skipped orange,
  empty grey); amber dashed is the model.
* Nothing explains what the user already knows: descriptions are tooltips and the hint card, keys are in the status bar,
  F1 and the palette.
* The photos are the product: the share of the window they get (71% of the width at 1600 px) is the number to check
  after any layout change.

## Code map

`core/views.py` (stages, lenses, library, counts, progress, next stage), `core/queues.py` (the rules of every queue,
sorts, review flags), `core/checked.py`, `core/pace.py`; `gui/workflow.py` (navigation: modes, stages, counts, empty
states; a mixin of the window), `gui/panels.py` (tree, mode switch, context bar, hint card, toast), `gui/palette.py`,
`gui/icons.py`, `gui/theme.py`, `gui/grid.py`, `gui/canvas.py`, `gui/app.py`.

## Where the principles come from

* Staged work with a fast first pass in a grid and the hard decisions later, filters instead of folders (photo culling in
  Lightroom: [Library module, grid and loupe, flags and the filter bar](https://thelenslounge.com/how-to-cull-photos-in-lightroom/)).
* Annotate, then review, with an explicit state per item and accept / fix-and-accept as separate outcomes
  ([Label Studio review](https://docs.humansignal.com/guide/quality)).
* Steppers: a current step highlighted, completed and upcoming steps with their own markers, for a process of three or more
  steps ([Carbon progress indicator](https://carbondesignsystem.com/components/progress-indicator/usage/)); here the steps are
  also jumpable, as in a non-linear flow.
* Undo instead of "Are you sure?" for reversible actions
  ([saasui: destructive action patterns](https://www.saasui.design/blog/saas-destructive-actions-confirmation-ux-patterns)).
* A command palette lists places, actions and objects, shows shortcuts and needs a visible hint
  ([Command palette: past, present and future](https://command.ai/blog/command-palette-past-present-and-future)).
* Nielsen's heuristics: visibility of system status (counts, progress, toasts, the hint card) and recognition rather than
  recall (stage icons that match the mode switch, readable names for every key); Apple HIG: a segmented control switches
  views within one context and does not trigger actions; Microsoft's app-settings guidance: group settings, apply at once.
* The UX copy rules used for names and messages: clear, concise, consistent, useful; an empty state says what this is, why
  it is empty and where to go on; confirmations name the consequence.
