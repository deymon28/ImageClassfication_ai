"""Keyboard shortcuts: layout-independent key names (a Russian layout gives the same actions as QWERTY) and the
default keymap, which can be overridden in settings (action -> key string like 'Ctrl+Z', 'Space', 'E')."""
from __future__ import annotations

from PyQt6.QtCore import Qt

# action -> default keys (first is shown in menus)
DEFAULT_KEYMAP: dict[str, list[str]] = {
    "accept_next": ["Space", "Return", "Enter"],
    "empty_next": ["E"],
    "skip_next": ["X"],
    "prev": ["A", "Left"],
    "next": ["D", "Right"],
    "next_unreviewed": ["N"],
    "undo": ["Ctrl+Z", "U"],
    "redo": ["Ctrl+Y", "Ctrl+Shift+Z"],
    "reset_proposals": ["R"],
    "toggle_boxes": ["H"],
    "toggle_loupe": ["M"],
    "loupe_in": ["=", "+", "Shift++"],
    "loupe_out": ["-"],
    "toggle_class": ["C"],
    "delete_box": ["Delete", "Backspace"],
    "fit_view": ["Z"],
    "copy_prev": ["P"],
    "propagate_prev": ["T"],
    "sam_fit": ["F"],
    "accept_proposals": ["Shift+Space"],
    "switch_mode": ["Tab", "G"],
    "grid_view": ["V"],
    "fullscreen": ["F11"],
    "toggle_panels": ["Ctrl+B"],
    "go_triage": ["Ctrl+1"],
    "go_label": ["Ctrl+2"],
    "go_review": ["Ctrl+3"],
    "command_palette": ["Ctrl+K"],
    "mark_checked": ["K"],
    "open_frame": ["O"],
    "save": ["Ctrl+S"],
    "reload": ["F5"],
    "goto_frame": ["Ctrl+G"],
    "prev_page": ["A", "PageUp", "Backspace"],
    "next_page": ["D", "PageDown"],
    "commit_page": ["Space"],
    "arrow_left": ["Left"],
    "arrow_right": ["Right"],
    "arrow_up": ["Up"],
    "arrow_down": ["Down"],
    "tile_mark": ["Return", "Enter"],
    "tile_skip": ["X", "S"],
}

# action -> (group, what the user reads in the shortcut dialog and the cheat sheet)
ACTION_INFO: dict[str, tuple[str, str]] = {
    "accept_next": ("Editor", "Save the frame (labeled, or empty without boxes) and go to the next"),
    "empty_next": ("Editor", "Save as empty and go to the next"),
    "skip_next": ("Editor", "Skip the frame (unusable) and go to the next"),
    "prev": ("Editor", "Previous frame"),
    "next": ("Editor", "Next frame"),
    "next_unreviewed": ("Editor", "Next new frame (no decision yet)"),
    "undo": ("Everywhere", "Undo (boxes in the editor, the last saved page in the grid)"),
    "redo": ("Editor", "Redo"),
    "reset_proposals": ("Editor", "Back to the model's proposals"),
    "toggle_boxes": ("Editor", "Show / hide boxes"),
    "toggle_loupe": ("Editor", "Loupe on / off"),
    "loupe_in": ("Editor", "Loupe: stronger"),
    "loupe_out": ("Editor", "Loupe: weaker"),
    "toggle_class": ("Editor", "Switch the selected box to the next class"),
    "delete_box": ("Editor", "Delete the selected box"),
    "fit_view": ("Editor", "Fit the image to the window"),
    "copy_prev": ("Editor", "Copy the boxes of the previous labeled frame"),
    "propagate_prev": ("Editor", "Track the previous frame's boxes onto this frame"),
    "sam_fit": ("Editor", "Fit the selected box with SAM"),
    "accept_proposals": ("Editor", "Accept all proposals as labels"),
    "save": ("Editor", "Save now"),
    "prev_page": ("Grid", "Previous page"),
    "next_page": ("Grid", "Next page"),
    "commit_page": ("Grid", "Commit the page: unmarked tiles become empty (in Checked: return the ticked tiles to Review)"),
    "arrow_left": ("Grid", "Left arrow: tile to the left, or previous page (Settings > Arrow keys in the grid)"),
    "arrow_right": ("Grid", "Right arrow: tile to the right, or next page"),
    "arrow_up": ("Grid", "Up arrow: tile above, or previous page"),
    "arrow_down": ("Grid", "Down arrow: tile below, or next page"),
    "tile_mark": ("Grid", "Tick the selected tile 'has objects' (again to undo); Checked: tick it to return"),
    "tile_skip": ("Grid", "Mark the selected tile SKIP (again to undo)"),
    "switch_mode": ("Everywhere", "Switch between grid and editor"),
    "grid_view": ("Everywhere", "Next stage that still has frames"),
    "go_triage": ("Everywhere", "Go to Triage (1: frames the model found nothing in)"),
    "go_label": ("Everywhere", "Go to Label (2: fix the boxes)"),
    "go_review": ("Everywhere", "Go to Review (3: a second look at saved frames)"),
    "command_palette": ("Everywhere", "Search views, actions and frames"),
    "mark_checked": ("Editor", "Mark the frame as checked (hides it from Review) and go on"),
    "open_frame": ("Grid", "Open the tile under the mouse in the editor"),
    "fullscreen": ("Everywhere", "Fullscreen on / off"),
    "toggle_panels": ("Everywhere", "Hide / show the side panels (more room for the photos)"),
    "reload": ("Everywhere", "Re-read proposals and new frames"),
    "goto_frame": ("Everywhere", "Go to a frame by number or name"),
}
ACTION_GROUPS = ("Everywhere", "Grid", "Editor")

_NAMED = {
    Qt.Key.Key_Space: "Space", Qt.Key.Key_Return: "Return", Qt.Key.Key_Enter: "Enter", Qt.Key.Key_Tab: "Tab",
    Qt.Key.Key_Backspace: "Backspace", Qt.Key.Key_Delete: "Delete", Qt.Key.Key_Escape: "Esc", Qt.Key.Key_Left: "Left",
    Qt.Key.Key_Right: "Right", Qt.Key.Key_Up: "Up", Qt.Key.Key_Down: "Down", Qt.Key.Key_Home: "Home",
    Qt.Key.Key_End: "End", Qt.Key.Key_PageUp: "PageUp", Qt.Key.Key_PageDown: "PageDown",
    **{getattr(Qt.Key, f"Key_F{i}"): f"F{i}" for i in range(1, 13)},
}


def key_name(key: int, native_vk: int, modifiers) -> str:
    """'Ctrl+Shift+Z'-style name. Letters and digits come from the Windows virtual-key code, which does not depend
    on the keyboard layout; other keys from Qt's key code."""
    if 0x41 <= native_vk <= 0x5A or 0x30 <= native_vk <= 0x39:
        base = chr(native_vk)
    elif 0x60 <= native_vk <= 0x69:  # numpad digits
        base = chr(native_vk - 0x30)
    elif key in _NAMED:
        base = _NAMED[key]
    elif 0x20 < key < 0x7F:
        base = chr(key).upper()
    else:
        return ""
    mods = []
    if modifiers & Qt.KeyboardModifier.ControlModifier:
        mods.append("Ctrl")
    if modifiers & Qt.KeyboardModifier.AltModifier:
        mods.append("Alt")
    if modifiers & Qt.KeyboardModifier.ShiftModifier and base not in ("Backspace",):
        mods.append("Shift")
    return "+".join(mods + [base])


class Keymap:
    def __init__(self, overrides: dict[str, list[str]] | None = None):
        self.map = {a: list(k) for a, k in DEFAULT_KEYMAP.items()}
        for a, keys in (overrides or {}).items():
            if a in self.map:
                self.map[a] = [keys] if isinstance(keys, str) else list(keys)

    def actions_for(self, name: str) -> list[str]:
        return [a for a, keys in self.map.items() if name in keys]

    def first(self, action: str) -> str:
        keys = self.map.get(action) or [""]
        return keys[0]
