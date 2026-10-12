"""The dark theme: design tokens, the palette and one style sheet for the whole application.

Design rules (see docs/ui-design.md): the photos are the product, so the chrome is quiet - near-black surfaces,
hairline borders, small type, one accent colour that only means "you are here / you chose this". Widgets pick their
style by class, by an object name (`#contextBar`) or by a property (`role`, `primary`, `chip`); nothing sets colours
inline, and the painted canvas and grid import the same tokens.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QColor, QFont, QPalette

from framelabel.perf import app_data_dir

T = {
    "bg0": "#0d0e11",        # behind the photos
    "bg1": "#141519",        # side panels, bars
    "bg2": "#1b1d22",        # controls
    "bg3": "#24272d",        # hover
    "line": "#23252b",       # separators
    "line2": "#2e3139",      # control borders
    "text": "#d8dbe0",
    "text2": "#8b919c",
    "text3": "#5a606b",
    "accent": "#6c8cff",
    "accent_hi": "#8ea6ff",
    "accent_lo": "#5373e8",
    "accent_soft": "rgba(108, 140, 255, 42)",
    "ok": "#4ade80",
    "warn": "#fbbf24",
    "danger": "#f87171",
    "skip": "#e0a458",
    "neutral": "#8b93a1",
    "select": "#232941",     # accent_soft over bg1 as a solid colour, for what the palette paints itself
}
RADIUS = 5


def qc(name: str, alpha: int | None = None) -> QColor:
    c = QColor(T[name])
    if alpha is not None:
        c.setAlpha(alpha)
    return c


# ---------------------------------------------------------------- icons used by the style sheet (SVG: sharp at any DPI)
def _svg(body: str, color: str, size: int = 16, width: float = 2) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 16 16" fill="none" '
            f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


ICONS = {
    "check": _svg('<path d="M3.5 8.5l3 3 6-7"/>', "#ffffff"),
    "down": _svg('<path d="M4 6l4 4 4-4"/>', T["text2"], width=1.6),
    "up": _svg('<path d="M4 10l4-4 4 4"/>', T["text2"], width=1.6),
    "right": _svg('<path d="M6 4l4 4-4 4"/>', T["text3"], width=1.6),
    "branch_down": _svg('<path d="M4 6l4 4 4-4"/>', T["text3"], width=1.6),
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
QToolTip {{ background: {t['bg2']}; color: {t['text']}; border: 1px solid {t['line2']}; padding: 5px 8px; }}

QMainWindow, QDialog {{ background: {t['bg1']}; }}
QMainWindow::separator {{ background: {t['line']}; width: 1px; height: 1px; }}

/* menus */
QMenuBar {{ background: {t['bg1']}; padding: 0 4px; }}
QMenuBar::item {{ background: transparent; padding: 5px 9px; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {t['bg3']}; }}
QMenu {{ background: {t['bg2']}; border: 1px solid {t['line2']}; border-radius: 8px; padding: 5px; }}
QMenu::item {{ padding: 6px 26px 6px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {t['accent_soft']}; color: #ffffff; }}
QMenu::item:disabled {{ color: {t['text3']}; }}
QMenu::separator {{ height: 1px; background: {t['line']}; margin: 4px 6px; }}

/* bars */
QStatusBar {{ background: {t['bg1']}; border-top: 1px solid {t['line']}; color: {t['text2']}; }}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ color: {t['text2']}; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:vertical {{ height: 6px; }}
QSplitter::handle:hover {{ background: {t['line']}; }}

/* buttons */
QPushButton, QToolButton {{ background: {t['bg2']}; border: 1px solid {t['line2']}; border-radius: {RADIUS}px;
    padding: 4px 10px; color: {t['text']}; }}
QPushButton:hover, QToolButton:hover {{ background: {t['bg3']}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {t['line']}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {t['text3']}; background: {t['bg1']}; border-color: {t['line']}; }}
QPushButton:default, QPushButton[primary="true"] {{ background: {t['accent']}; border-color: {t['accent']}; color: #ffffff; }}
QPushButton:default:hover, QPushButton[primary="true"]:hover {{ background: {t['accent_hi']}; border-color: {t['accent_hi']}; }}
QPushButton:default:pressed, QPushButton[primary="true"]:pressed {{ background: {t['accent_lo']}; border-color: {t['accent_lo']}; }}
QPushButton[chip="true"] {{ padding: 3px 0; color: {t['text2']}; }}
QPushButton[chip="true"]:hover {{ color: {t['text']}; }}
QToolButton[quiet="true"] {{ background: transparent; border-color: transparent; color: {t['text2']}; }}
QToolButton[quiet="true"]:hover {{ background: {t['bg3']}; color: {t['text']}; }}

/* segmented switch: a track with one raised, coloured segment */
#segTrack {{ background: {t['bg0']}; border: 1px solid {t['line']}; border-radius: 7px; }}
#segTrack QPushButton {{ background: transparent; border: none; border-radius: 5px; padding: 4px 10px; color: {t['text2']}; }}
#segTrack QPushButton:hover {{ color: {t['text']}; background: {t['bg2']}; }}
#segTrack QPushButton:checked {{ background: {t['accent']}; color: #ffffff; font-weight: 600; }}

/* inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextBrowser, QComboBox {{ background: {t['bg2']}; border: 1px solid {t['line2']};
    border-radius: {RADIUS}px; padding: 3px 7px; selection-background-color: {t['accent']}; selection-color: #ffffff; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{ border-color: {t['accent']}; }}
QComboBox {{ padding-right: 22px; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{ image: url({icons['down']}); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: {t['bg2']}; border: 1px solid {t['line2']}; padding: 3px;
    selection-background-color: {t['accent_soft']}; selection-color: #ffffff; outline: 0; }}
QAbstractSpinBox {{ padding-right: 2px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 16px; border: none; background: transparent; }}
QAbstractSpinBox::up-arrow {{ image: url({icons['up']}); width: 10px; height: 10px; }}
QAbstractSpinBox::down-arrow {{ image: url({icons['down']}); width: 10px; height: 10px; }}

QCheckBox {{ spacing: 8px; padding: 2px 0; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border-radius: 4px; border: 1px solid #454a55; background: {t['bg2']}; }}
QCheckBox::indicator:hover {{ border-color: {t['accent']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']}; image: url({icons['check']}); }}

QProgressBar {{ background: {t['bg3']}; border: none; border-radius: 2px; min-height: 4px; max-height: 4px; }}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 2px; }}

/* lists, trees, tables: no frames, hairlines only */
QListView, QListWidget, QTreeWidget, QTableWidget, QTableView {{ background: transparent; border: none; alternate-background-color: {t['bg1']}; }}
QListView::item, QListWidget::item {{ padding: 3px 8px; border-radius: 4px; }}
QListView::item:hover, QListWidget::item:hover {{ background: {t['bg2']}; }}
QListView::item:selected, QListWidget::item:selected {{ background: {t['accent_soft']}; }}
QTreeWidget::item {{ padding: 2px 4px; border-radius: 4px; }}
QTreeWidget::item:hover {{ background: {t['bg2']}; }}
QTreeWidget::item:selected {{ background: {t['accent_soft']}; color: #ffffff; }}
QTreeView::branch {{ background: transparent; }}
QTreeView::branch:selected {{ background: {t['accent_soft']}; }}
QTreeView::branch:has-children:closed {{ image: url({icons['right']}); }}
QTreeView::branch:has-children:open {{ image: url({icons['branch_down']}); }}
QHeaderView::section {{ background: transparent; color: {t['text3']}; padding: 4px 6px; border: none; border-bottom: 1px solid {t['line']}; }}
QTableWidget {{ gridline-color: {t['line']}; }}
QTableWidget::item {{ padding: 2px 6px; }}
QTableWidget::item:selected {{ background: {t['accent_soft']}; color: #ffffff; }}
QTableCornerButton::section {{ background: transparent; border: none; }}
QTextBrowser, QPlainTextEdit {{ padding: 6px; }}

/* scroll bars */
QScrollBar:vertical {{ background: transparent; width: 9px; margin: 1px; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; margin: 1px; }}
QScrollBar::handle {{ background: #30333b; border-radius: 3px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: #454a55; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* roles */
QLabel[role="muted"] {{ color: {t['text2']}; }}
QLabel[role="section"] {{ color: {t['text3']}; font-weight: 600; font-size: 7.5pt; padding-top: 8px; padding-bottom: 1px; }}
QLabel[role="note"] {{ color: {t['text3']}; font-size: 8pt; }}
QLabel[role="title"] {{ color: {t['text']}; font-size: 11pt; font-weight: 600; }}
#contextBar {{ background: {t['bg1']}; border-bottom: 1px solid {t['line']}; }}
QLabel[role="lens"] {{ color: {t['text2']}; font-size: 10pt; }}
#palette {{ background: {t['bg2']}; border: 1px solid {t['line2']}; border-radius: 10px; }}
#palette QLineEdit {{ background: {t['bg0']}; padding: 7px 10px; font-size: 10.5pt; }}
#palette QListWidget {{ background: transparent; }}
#coach {{ background: {t['bg2']}; color: {t['text2']}; border: none; border-left: 2px solid {t['accent']}; border-radius: 4px;
    padding: 6px 8px; font-size: 8.5pt; }}
#toast {{ background: rgba(20, 21, 25, 240); border: 1px solid {t['line2']}; border-radius: 8px; }}
#toast QLabel {{ background: transparent; color: {t['text']}; font-size: 10pt; }}
#toast QPushButton {{ padding: 3px 12px; }}
"""


def apply_theme(app) -> None:
    """Fusion + palette + the style sheet + the font; call once after the QApplication exists."""
    app.setStyle("Fusion")
    f = QFont()
    f.setFamilies(["Segoe UI Variable Text", "Segoe UI", "Inter", "Arial"])
    f.setPointSizeF(9)
    app.setFont(f)
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: "bg1", QPalette.ColorRole.WindowText: "text", QPalette.ColorRole.Base: "bg0",
        QPalette.ColorRole.AlternateBase: "bg1", QPalette.ColorRole.Text: "text", QPalette.ColorRole.Button: "bg2",
        QPalette.ColorRole.ButtonText: "text", QPalette.ColorRole.Highlight: "select",
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
