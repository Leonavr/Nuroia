# app/utils.py
import sys
import os

def resource_path(relative_path: str) -> str:
    """
    Get absolute path to a resource, works for development and for PyInstaller.
    This function ensures that files (like DLLs, images, data files) can be found
    both when running the source code and when running the packaged .exe file.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.dirname(sys.argv[0]))

    return os.path.join(base_path, relative_path)