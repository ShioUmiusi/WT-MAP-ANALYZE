"""Local paths that are machine specific: the game install and the Oodle kernel.

Nothing here is redistributed with the repository, so both paths come from the
environment:

    WT_GAME_DIR    War Thunder install root, the folder that contains ``levels/``
    WT_OODLE_DLL   ``daKernel-dev.dll`` (the Oodle kernel exported by the Dagor
                   Asset Explorer build; the game ships the same decoder)

Functions raise ``SystemExit`` with a readable message when a required path is
missing, so command line tools fail with an instruction instead of a traceback.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OODLE_DLL = os.path.join(HERE, "vendor", "daKernel-dev.dll")


def game_dir(required=True):
    """Return the game install root from ``WT_GAME_DIR``."""
    value = (os.environ.get("WT_GAME_DIR") or "").strip().strip('"')
    if value:
        if not os.path.isdir(value):
            raise SystemExit(
                "WT_GAME_DIR does not exist: %s" % value)
        return os.path.abspath(value)
    if required:
        raise SystemExit(
            "War Thunder install not found. Set WT_GAME_DIR to the folder that "
            "contains levels/ (for example D:\\Games\\War Thunder).")
    return None


def levels_dir(required=True):
    """Return ``<game>/levels``."""
    root = game_dir(required=required)
    return os.path.join(root, "levels") if root else None


def oodle_dll(required=True):
    """Return the Oodle kernel DLL path from ``WT_OODLE_DLL`` or ``vendor/``."""
    value = (os.environ.get("WT_OODLE_DLL") or "").strip().strip('"')
    path = os.path.abspath(value) if value else DEFAULT_OODLE_DLL
    if os.path.isfile(path):
        return path
    if required:
        raise SystemExit(
            "Oodle kernel DLL not found at %s. Set WT_OODLE_DLL to daKernel-dev.dll "
            "(shipped with Dagor Asset Explorer builds; it is not redistributed here)."
            % path)
    return None
