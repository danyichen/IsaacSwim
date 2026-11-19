"""Launch settings for Isaac Sim; call prepare_launch before AppLauncher."""

from __future__ import annotations

import os
import sys
import threading

__all__ = ["D3D12_KIT_ARG", "prepare_launch", "render_kit_args", "update_app"]

D3D12_KIT_ARG = "--/app/vulkan=false"


def render_kit_args(kit_args: str | None = "") -> str:
    kit_args = (kit_args or "").strip()
    if (sys.platform != "win32" or os.environ.get("ISAACSWIM_VULKAN") == "1"
            or "/app/vulkan" in kit_args):
        return kit_args
    return f"{kit_args} {D3D12_KIT_ARG}".strip()


def update_app() -> None:
    import carb.settings
    import omni.kit.app

    settings = carb.settings.get_settings()
    key = "/app/player/playSimulations"
    previous = settings.get(key)
    settings.set_bool(key, False)
    try:
        omni.kit.app.get_app().update()
    finally:
        if previous is None:
            settings.destroy_item(key)
        else:
            settings.set_bool(key, previous)


def _exit_nonzero_on_error():
    previous = sys.excepthook
    if getattr(previous, "_isaacswim", False):
        return

    def hook(kind, value, tb):
        handled = (sys.exc_info()[1] is not None
                   or threading.current_thread() is not threading.main_thread())
        caller = sys._getframe().f_back
        while caller is not None:
            handled |= caller.f_globals.get("__name__") == "__main__"
            caller = caller.f_back
        previous(kind, value, tb)
        if handled:
            return
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    hook._isaacswim = True
    sys.excepthook = hook


def prepare_launch(args, *, oceansim: bool = False, ros2: bool = False):
    _exit_nonzero_on_error()
    get = args.get if isinstance(args, dict) else lambda k, d=None: getattr(args, k, d)
    renders = bool(get("enable_cameras", False)) or not get("headless", False)
    value = get("kit_args", "") or ""
    if oceansim:
        from isaacswim.oceansim import KIT_ARGS
        value = f"{value} {KIT_ARGS}".strip()
    if ros2:
        from isaacswim.ros2 import BRIDGE_EXTENSION, setup_environment
        setup_environment()
        value = f"{value} --enable {BRIDGE_EXTENSION}".strip()
    if renders:
        value = render_kit_args(value)
    if value != (get("kit_args", "") or ""):
        if isinstance(args, dict):
            args["kit_args"] = value
        else:
            args.kit_args = value
    return args
