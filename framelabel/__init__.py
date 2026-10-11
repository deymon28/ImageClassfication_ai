"""framelabel: dataset labeling app and labeler model training."""
import os
import sys


def _preload_system_cpp_runtime() -> None:
    """Load the system C++ runtime before PyQt6 can load its own, older copy.

    PyQt6 bundles msvcp140*.dll 14.26 (2020). If Qt is imported first, that copy stays in the process and torch's
    c10.dll later fails with "WinError 1114". A module already loaded under the same file name is reused by later
    LoadLibrary calls, so loading the newer System32 copies first makes Qt and torch share them.
    """
    if sys.platform != "win32":
        return
    import ctypes

    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for name in ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "concrt140.dll"):
        try:
            ctypes.WinDLL(os.path.join(system32, name))
        except OSError:
            pass  # missing runtime: leave the default DLL search in charge


_preload_system_cpp_runtime()
