# Labeling guide (Arena Breakout Infinite)

How to label frames so that the dataset is consistent. The detector learns exactly what the labels show: every
inconsistency (a corpse sometimes `enemy`, a far person sometimes missed, boxes sometimes tight and sometimes loose)
becomes noise in training and in the evaluation numbers. When two rules below seem to conflict, the order of priority
is: **1. never label something that is not there, 2. label every clearly visible object, 3. be consistent with how
you labeled similar cases before.**

Rules marked **(project)** are fixed by the project (classes, export, evaluation). Rules marked **(default)** are the
recommended convention; change them only deliberately and for the whole dataset (relabeling old frames), not from
some day on.

## 1. What the detector is for

The labels train a real-time detector that must find **people the player can see on screen** (enemies and teammates)
and **corpses**. The labeler model (see [labeler-model.md](labeler-model.md)) proposes boxes; you correct them. A
missed object and a wrong box are both errors, but they are not equal:

* A **false box** (box on a non-object) teaches the detector to fire on that thing in the live game.
* A **missing box** (object not labeled) teaches it that the object is background, and it also counts as a false
  positive when the detector later finds it correctly.
* An **unlabeled frame that is wrongly marked empty** is the worst case: it is exported as a negative and used to
  measure false positives.

## 2. Classes (project)

| id | key | class | what it is |
| --- | --- | --- | --- |
| 0 | `1` | `enemy` | a living person: an enemy player, a bot (AI scav) **and a teammate** |
| 1 | `2` | `corpse` | a dead person lying on the ground (a body, a ragdoll) |

* **Teammates are `enemy`.** The detector must not tell friend from foe; it detects "a person". Do not invent a
  third class and do not skip teammates.
* **Bots / AI** are `enemy` like players.
* **Not a class, never labeled:** the player's own hands and weapon (first person view), the player's own shadow,
  mannequins and statues, posters, dropped loot, backpacks and gear lying on the ground without a body, vehicles,
  animals.

### Living or dead?

| situation | class |
| --- | --- |
| standing, walking, running, crouching, prone, climbing, sitting, aiming | `enemy` |
| lying and still, **in a death pose** (limbs loose, body flat, ragdoll, loot-able body) | `corpse` |
| lying prone on purpose (aiming, crawling) | `enemy` |
| knocked down / downed but not dead | `enemy` (the person is still alive) |
| a body being dragged or reviving | `enemy` |
| cannot tell whether it is a prone player or a corpse (far away, dark) | follow the "unsure" rule, section 8 |

A person who dies on the frame sequence is `enemy` until the frame where he is clearly lying dead; from then on
`corpse`. Do not relabel the earlier frames. Corpses stay in view for a long time, so the same corpse appears on
many neighbouring frames: label it on **every** frame where it is visible.

## 3. What gets a box

Label an object when **you can recognise it as a person or a corpse from the frame alone** (without comparing with the
neighbouring frames).

* **Fully visible** person: box.
* **Partly hidden** (behind a wall, door frame, bush, car, another person, or cut by the screen edge): box **if at
  least ~30-40 % of the body or the head and upper body are visible**. Box only the visible part (see 4).
* **Only a small piece is visible** (a hand, a foot, the barrel and a sliver of an arm, a shadow, the top of a
  helmet over cover): **no box** unless it is clearly a person's head/torso. A head alone above cover is labeled.
* **Through a scope, a window, a smoke, glass, at night, in a dark room:** label what you can recognise. If the
  recognition needs guessing, see section 8.
* **Far away (small) people:** label them down to the point where you can still tell it is a person-shaped figure
  (**default: about 8-10 px tall at 1080p**). Far people are the weakest part of the detector and the most valuable
  to teach. Smaller specks are not labeled (section 8).
* **Several people close together:** one box per person, even if the boxes overlap heavily. Never one box for a group.
* **Person in a mirror, a monitor, a reflection in glass or water, on a screen inside the game:** no box.
* **Person on a loading screen, killcam, spectator view, menu preview, inventory model:** these frames are `skip`
  (section 7), not labeled.

## 4. How to draw the box

The box is the **tightest rectangle around the visible part of the object**, aligned to the screen axes.

1. **Tight, not loose.** The edges touch the outermost visible pixels of the body. No empty margin, but do not cut
   into the body either. If you hesitate between slightly looser and slightly tighter, be **a pixel or two looser**:
   cutting off a head or a foot is worse than a thin margin.
2. **Include** everything that is physically part of the visible person: head and helmet, headset, hair, a raised
   arm, a backpack worn on the body, a vest, boots, a **weapon held in the hands when it overlaps the body**.
3. **A long weapon sticking out** (a rifle barrel, a sniper rifle, a held shield) far from the body: **exclude the
   protruding part**. The box covers the person, not the whole length of the gun. (default)
4. **Do not include** the shadow on the ground, a muzzle flash, tracers, smoke, dust, a name tag or icon above the
   head, a health bar, or an outline/highlight drawn by the game around teammates.
5. **Occlusion:** box only what is **visible**. Do not extend the box to where you think the hidden legs or arm must
   be. A person behind a low wall with only the head and shoulders showing gets a box around the head and shoulders.
6. **Screen edge:** the box stops at the image border. A person cut by the edge gets a box up to the edge; the
   exporter clips boxes to the image.
7. **Motion blur / ghosting:** box the main body, not the whole blur trail.
8. **Prone / lying** people (enemy or corpse): a wide, low box around the whole body, again visible part only.
9. **Do not draw boxes smaller than 1 pixel;** such boxes are dropped on export. Anything that small should not be
   labeled anyway.

### Consistency for the same person across frames

A person who moves across consecutive frames should be boxed with the same logic every time (same inclusion of the
weapon, same tightness). If you use T (track) or P (copy previous), **check every carried-over box**: tracking drifts
and the person's pose changes.

## 5. Working with model proposals

Proposals are dashed boxes with a confidence number. They are **suggestions, not truth**.

* **Accept a proposal only after you looked at it.** Space (accept and next) is only safe when every box on the frame
  is correct. A hurried Space on a bad frame silently turns model mistakes into labels and then into the next
  model's mistakes.
* **Fix the geometry.** The model often draws boxes that are too loose, include the weapon barrel, or cover only the
  torso. Drag the edges until they follow section 4.
* **Fix the class.** Corpses are often proposed as `enemy` and the other way round (C toggles the class of the
  selected box).
* **Delete wrong boxes:** HUD icons (the body-status silhouette), reflections, posters, loot, duplicates (two boxes on
  the same person) with RMB.
* **Look for what the model missed.** The views *Model disagrees with my labels*, *Small (far) objects first* and weak proposals (W) exist for
  this. Far people and people in shadow are the typical misses.
* **A high confidence is not a reason to skip checking.** A 0.9 box on a corpse labeled `enemy` is still wrong.
* **A low confidence is not a reason to delete.** A 0.2 box on a real far person is correct.
* **Ignore zone:** boxes whose centre lies in the ignore zone (the HUD silhouette icon bottom-left) are hidden by
  design. Never label that icon.
* **SAM (F / Alt+click)** only snaps a box to the object outline. Check the result: it can fit the weapon, the cover
  or the shadow. The tight-box rules above still apply.

## 6. Frame decisions (what to press at the end of a frame)

Every frame ends in exactly one status; every decision is saved immediately.

| status | meaning | when |
| --- | --- | --- |
| `pos` (labeled) | the frame has one or more objects, all of them labeled | Space / accept |
| `neg` (empty) | the frame was looked at and **contains no enemy and no corpse at all** | E |
| `skip` | the frame is unusable for training | X |
| `todo` | marked in the grid as "has objects", boxes not drawn yet | grid LMB |

### `neg` (empty) is a strong claim

A frame marked empty is exported as a **negative**: "nothing here". Mark it empty only when you actually looked at the
whole picture. Check the corners, the edges, windows, dark areas and far distance. If the frame might contain a person
you cannot decide on, do **not** press E; see section 8.

The grid triage works in pages: unmarked tiles of a committed page become empty. Before committing a page, scan each
tile once; a tile with even a possible person gets LMB (has objects) or MMB (skip).

### `pos` requires completeness

A labeled frame must have **all** visible objects boxed, not only the main one. A frame with two people where one is
boxed teaches the detector that the second is background. If you do not want to label a hard object, the frame is
`skip`, not `pos`.

## 7. Frames to skip (`X`)

Mark `skip` when the frame cannot be a clean positive or a clean negative:

* Menus, inventory, stash, map, loading screens, death/result screens, killcam, spectator view, main lobby. (The
  interface-screen detector pre-marks many of them: a tile pre-marked `skip` is correct unless you see a real game
  view in it.)
* The picture is frozen, black, overexposed, or severely corrupted (capture artifacts, tearing).
* A cut-scene or a game-view transition where objects are not real (warp, fade).
* The frame is a **near duplicate** of one you just handled and adds nothing (neighbouring frames of a static scene).
  You do not have to label every frame of a long static scene; skip some, but never skip only the hard ones.
* The frame contains a person you **cannot classify or localize** and cannot honestly call empty (section 8).
* The frame shows the player's own character in a non-game view (equipment screen model).

`skip` frames are not exported at all. Skipping is always safer than guessing, but over-skipping hard frames (far,
dark, partly hidden) makes the dataset easier than the real game; use it for genuinely unusable frames.

## 8. Unsure cases

The principle: **an uncertain label is worse than a missing frame.**

| case | action |
| --- | --- |
| A clear person, but too far/small to box precisely | box it as well as you can; a slightly loose small box is fine |
| A speck that **might** be a person (a few pixels, no shape) | do not box it; if the rest of the frame is clean, **skip the frame** rather than marking it empty or labeled |
| Person vs corpse cannot be decided (far prone figure) | if it has not moved on the neighbouring frames and has a dead pose: `corpse`; if it moves or aims: `enemy`; otherwise skip the frame |
| Partly hidden, you are not sure it is a person | no box; skip if it would matter |
| Very dark / smoke / flashbang white-out | label only clearly visible objects; skip the frame if you doubt there is nobody else |
| Something looks like a person on another monitor / texture / statue | no box |

When in doubt, look at the previous and next frames (A / D): the movement or the pose usually answers the question.
A consistent decision for the same object across frames matters more than the "right" decision on one frame.

## 9. Common mistakes (checklist)

* Labeling the HUD body-status silhouette (bottom-left) as an enemy.
* Corpse labeled `enemy`, or a living prone player labeled `corpse`.
* Teammate left unlabeled or skipped.
* Box too loose around a long weapon, or cut through the head/feet.
* Box extended to the hidden part of a person behind cover.
* One box around two close people.
* Frame marked `pos` while another visible person is not boxed.
* Frame marked `neg` with a far person still in it.
* Space pressed through a page of proposals without looking.
* Carried-over (P / T) boxes left in the wrong place after the person moved.
* Duplicate boxes on the same person.
* Different convention on different days (always include the weapon one day, never the next).

## 10. Self-check before export and the split

* Open the view *Model disagrees* (confident proposals without a label, and labels without a proposal): it finds most
  missed boxes and wrong labels.
* Open *Small (far) objects* and verify that far people are labeled the same way throughout.
* Sort the frame list by class and look at the `corpse` frames: this class is the least reliable.
* Look at a random sample of your `neg` frames, especially the ones the model flagged (they are the hard negatives
  that go into the export).
* Open *Session report* and check the skipped and filtered counts: a large `skip` share of hard frames is a sign of
  the over-skipping described in section 7.
* Never edit the validation set differently from the training set; the split is made at export, and the same rules
  apply to every labeled frame.

## 11. What this guide does not cover

* The capture settings of ImageHook (which frames are saved at all) are described in that project.
* Which model proposes boxes and how it was trained: [labeler-model.md](labeler-model.md).
* Keyboard shortcuts and the view modes: README, section Workflow; shortcuts can be changed in
  *Settings > Keyboard shortcuts*.
