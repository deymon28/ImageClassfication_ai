"""The dark theme: design tokens, the palette and one style sheet for the whole application.

Everything colourful in the widgets comes from here (the painted canvas and grid import the same tokens), so the look
can be changed in one place. Widgets pick their style by class, by an object name (`#contextBar`) or by a `role`
property (`QLabel[role="muted"]`); nothing sets colours inline.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QColor, QFont, QPalette

from framelabel.perf import app_data_dir

T = {
    "bg0": "#121317",        # work area behind the photos
    "bg1": "#1a1b20",        # panels, menus, bars
    "bg2": "#23252b",        # controls, inset lists
    "bg3": "#2c2f36",        # hover
    "line": "#2f323a",       # borders and separators
    "line2": "#3d414b",      # control borders, hover borders
    "text": "#e8eaed",
    "text2": "#a3a9b4",
    "text3": "#6e7480",
    "accent": "#5b8cff",
    "accent_hi": "#7aa0ff",
    "accent_lo": "#4672e0",
    "accent_soft": "rgba(91, 140, 255, 46)",
    "ok": "#4ade80",
    "warn": "#fbbf24",
    "danger": "#f87171",
    "skip": "#e0a458",
    "neutral": "#8b93a1",
}
RADIUS = 6


def qc(name: str, alpha: int | None = None) -> QColor:
    c = QColor(T[name])
    if alpha is not None:
        c.setAlpha(alpha)
    return c


# ---------------------------------------------------------------- icons used by the style sheet (SVG: sharp at any DPI)
def _svg(body: str, color: str, size: int = 16) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 16 16" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


ICONS = {
    "check": _svg('<path d="M3.5 8.5l3 3 6-7"/>', "#ffffff"),
    "down": _svg('<path d="M4 6l4 4 4-4"/>', T["text2"]),
    "up": _svg('<path d="M4 10l4-4 4 4"/>', T["text2"]),
    "right": _svg('<path d="M6 4l4 4-4 4"/>', T["text3"]),
    "branch_down": _svg('<path d="M4 6l4 4 4-4"/>', T["text3"]),
}


def write_icons() -> dict[str, str]:
    """Write the icons next to the other per-user data and return name -> url() path for the style sheet."""
    folder = app_data_dir() / "theme"
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, svg in ICONS.items():
        path = folder / f"{name}.svg"
        if not path.exists() or path.read_text(encoding="utf-8") != svg:
            path.write_text(svg, encoding="utf-8")
        out[name] = Path(path).as_posix()
    return out


def stylesheet(icons: dict[str, str]) -> str:
    t = T
    return f"""
* {{ outline: 0; }}
QToolTip {{ background: {t['bg2']}; color: {t['text']}; border: 1px solid {t['line2']}; border-radius: 6px; padding: 6px 8px; }}

QMainWindow, QDialog {{ background: {t['bg1']}; }}
QMainWindow::separator {{ background: {t['line']}; width: 1px; height: 1px; }}

/* menus */
QMenuBar {{ background: {t['bg1']}; border-bottom: 1px solid {t['line']}; padding: 2px 6px; }}
QMenuBar::item {{ background: transparent; padding: 6px 10px; border-radius: 6px; }}
QMenuBar::item:selected {{ background: {t['bg3']}; }}
QMenu {{ background: {t['bg2']}; border: 1px solid {t['line2']}; border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 7px 28px 7px 14px; border-radius: 6px; }}
QMenu::item:selected {{ background: {t['accent_soft']}; color: #ffffff; }}
QMenu::item:disabled {{ color: {t['text3']}; }}
QMenu::separator {{ height: 1px; background: {t['line']}; margin: 5px 8px; }}

/* docks and bars */
QDockWidget {{ titlebar-close-icon: none; color: {t['text2']}; font-weight: 600; }}
QDockWidget::title {{ background: {t['bg1']}; padding: 9px 12px 7px 12px; text-align: left; border-bottom: 1px solid {t['line']}; }}
QStatusBar {{ background: {t['bg1']}; border-top: 1px solid {t['line']}; color: {t['text2']}; }}
QStatusBar::item {{ border: none; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:vertical {{ height: 8px; }}
QSplitter::handle:hover {{ background: {t['line']}; }}

/* buttons */
QPushButton, QToolButton {{ background: {t['bg2']}; border: 1px solid {t['line2']}; border-radius: {RADIUS}px;
    padding: 6px 12px; color: {t['text']}; }}
QPushButton:hover, QToolButton:hover {{ background: {t['bg3']}; border-color: #4a4f5b; }}
QPushButton:pressed, QToolButton:pressed {{ background: {t['line']}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {t['text3']}; background: {t['bg1']}; border-color: {t['line']}; }}
QPushButton:default, QPushButton[primary="true"] {{ background: {t['accent']}; border-color: {t['accent']}; color: #ffffff; font-weight: 600; }}
QPushButton:default:hover, QPushButton[primary="true"]:hover {{ background: {t['accent_hi']}; border-color: {t['accent_hi']}; }}
QPushButton:default:pressed, QPushButton[primary="true"]:pressed {{ background: {t['accent_lo']}; border-color: {t['accent_lo']}; }}
QPushButton[chip="true"] {{ padding: 4px 6px; color: {t['text2']}; }}
QPushButton[chip="true"]:hover {{ color: {t['text']}; }}

/* segmented switch: a track with one raised, coloured segment */
#segTrack {{ background: {t['bg0']}; border: 1px solid {t['line']}; border-radius: 9px; }}
#segTrack QPushButton {{ background: transparent; border: none; border-radius: 6px; padding: 7px 14px; color: {t['text2']}; font-weight: 600; }}
#segTrack QPushButton:hover {{ color: {t['text']}; background: {t['bg2']}; }}
#segTrack QPushButton:checked {{ background: {t['accent']}; color: #ffffff; }}

/* inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextBrowser, QComboBox {{ background: {t['bg2']}; border: 1px solid {t['line2']};
    border-radius: {RADIUS}px; padding: 5px 8px; selection-background-color: {t['accent']}; selection-color: #ffffff; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{ border-color: {t['accent']}; }}
QComboBox {{ padding-right: 26px; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url({icons['down']}); width: 14px; height: 14px; }}
QComboBox QAbstractItemView {{ background: {t['bg2']}; border: 1px solid {t['line2']}; border-radius: 8px; padding: 4px;
    selection-background-color: {t['accent_soft']}; selection-color: #ffffff; outline: 0; }}
QAbstractSpinBox {{ padding-right: 4px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 18px; border: none; background: transparent; }}
QAbstractSpinBox::up-arrow {{ image: url({icons['up']}); width: 12px; height: 12px; }}
QAbstractSpinBox::down-arrow {{ image: url({icons['down']}); width: 12px; height: 12px; }}

QCheckBox {{ spacing: 9px; padding: 3px 0; }}
QCheckBox::indicator {{ width: 17px; height: 17px; border-radius: 5px; border: 1px solid #4a4f5b; background: {t['bg2']}; }}
QCheckBox::indicator:hover {{ border-color: {t['accent']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']}; image: url({icons['check']}); }}

QProgressBar {{ background: {t['bg0']}; border: 1px solid {t['line']}; border-radius: 9px; color: {t['text']}; text-align: center; min-height: 18px; max-height: 18px; }}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 8px; margin: 0px; }}

/* lists, trees, tables */
QListView, QListWidget, QTreeWidget, QTableWidget, QTableView {{ background: {t['bg0']}; border: 1px solid {t['line']};
    border-radius: 8px; padding: 4px; alternate-background-color: {t['bg1']}; }}
QListView::item, QListWidget::item {{ padding: 4px 8px; border-radius: 5px; }}
QListView::item:hover, QListWidget::item:hover {{ background: {t['bg2']}; }}
QListView::item:selected, QListWidget::item:selected {{ background: {t['accent_soft']}; }}
QTreeWidget {{ background: transparent; border: none; padding: 0; }}
QTreeWidget::item {{ padding: 3px 6px; border-radius: 6px; margin: 1px 0; }}
QTreeWidget::item:hover {{ background: {t['bg2']}; }}
QTreeWidget::item:selected {{ background: {t['accent']}; color: #ffffff; }}
QTreeView::branch {{ background: transparent; }}
QTreeView::branch:has-children:closed {{ image: url({icons['right']}); }}
QTreeView::branch:has-children:open {{ image: url({icons['branch_down']}); }}
QHeaderView::section {{ background: {t['bg1']}; color: {t['text2']}; padding: 6px 8px; border: none; border-bottom: 1px solid {t['line']}; font-weight: 600; }}
QTableWidget {{ gridline-color: {t['line']}; }}
QTableWidget::item {{ padding: 3px 6px; }}
QTableWidget::item:selected {{ background: {t['accent_soft']}; color: #ffffff; }}
QTableCornerButton::section {{ background: {t['bg1']}; border: none; }}

/* scroll bars */
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle {{ background: #3a3e47; border-radius: 4px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: #4d535f; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* roles */
QLabel[role="muted"] {{ color: {t['text2']}; }}
QLabel[role="section"] {{ color: {t['text3']}; font-weight: 700; font-size: 8pt; padding-top: 10px; padding-bottom: 2px; }}
QLabel[role="note"] {{ color: {t['text3']}; font-size: 8.5pt; }}
QLabel[role="title"] {{ color: {t['text']}; font-size: 15pt; font-weight: 700; }}
#contextBar {{ background: {t['bg1']}; border-bottom: 1px solid {t['line']}; }}
#toast {{ background: rgba(26, 27, 32, 238); color: {t['text']}; border: 1px solid {t['line2']}; border-radius: 10px;
    padding: 10px 20px; font-size: 11pt; }}
"""


def apply_theme(app) -> None:
    """Fusion + palette + the style sheet + the font; call once after the QApplication exists."""
    app.setStyle("Fusion")
    f = QFont()
    f.setFamilies(["Segoe UI Variable Text", "Segoe UI", "Inter", "Arial"])
    f.setPointSizeF(9.5)
    app.setFont(f)
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: "bg1", QPalette.ColorRole.WindowText: "text", QPalette.ColorRole.Base: "bg0",
        QPalette.ColorRole.AlternateBase: "bg1", QPalette.ColorRole.Text: "text", QPalette.ColorRole.Button: "bg2",
        QPalette.ColorRole.ButtonText: "text", QPalette.ColorRole.Highlight: "accent",
        QPalette.ColorRole.HighlightedText: "text", QPalette.ColorRole.ToolTipBase: "bg2",
        QPalette.ColorRole.ToolTipText: "text", QPalette.ColorRole.PlaceholderText: "text3", QPalette.ColorRole.Link: "accent_hi",
    }
    for role, name in roles.items():
        pal.setColor(role, QColor(T[name]))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(T["text3"]))
    app.setPalette(pal)
    try:
        app.setStyleSheet(stylesheet(write_icons()))
    except OSError:  # no writable data folder: the style without the icons would look broken, so keep the plain palette
        pass
