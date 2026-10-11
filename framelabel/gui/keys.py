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
    "save": ["Ctrl+S"],
    "prev_page": ["A", "Left", "Backspace"],
    "next_page": ["D", "Right"],
    "commit_page": ["Space", "Return", "Enter"],
}

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
