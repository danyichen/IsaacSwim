from __future__ import annotations

import argparse
import subprocess
import sys

import pytest

from isaacswim import launch


def test_caught_extension_error_does_not_terminate_the_application():
    code = """
import sys
from isaacswim.launch import prepare_launch
prepare_launch({'headless': True})
try:
    raise ValueError('optional extension failed')
except ValueError:
    sys.excepthook(*sys.exc_info())
print('application continues')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stderr
    assert "application continues" in result.stdout


def test_deferred_exception_logging_is_not_an_uncaught_error():
    code = """
import sys
from isaacswim.launch import prepare_launch
prepare_launch({'headless': True})
try:
    raise ValueError('deferred extension log')
except ValueError:
    captured = sys.exc_info()
sys.excepthook(*captured)
print('application continues')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stderr
    assert "application continues" in result.stdout


def test_uncaught_error_overrides_successful_kit_shutdown():
    code = """
import atexit, os
from isaacswim.launch import prepare_launch
atexit.register(lambda: os._exit(0))
prepare_launch({'headless': True})
raise ValueError('uncaught script failure')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 1
    assert "uncaught script failure" in result.stderr


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(launch.sys, "platform", "win32")
    monkeypatch.delenv("ISAACSWIM_VULKAN", raising=False)


def test_rendering_launch_on_windows_goes_to_d3d12(windows):
    args = argparse.Namespace(headless=True, enable_cameras=True, kit_args="--foo")
    assert launch.prepare_launch(args) is args
    assert args.kit_args == "--foo --/app/vulkan=false"
    kw = launch.prepare_launch(dict(headless=True, enable_cameras=True))
    assert kw["kit_args"] == "--/app/vulkan=false"


def test_gui_counts_as_rendering(windows):
    args = launch.prepare_launch(argparse.Namespace(headless=False, enable_cameras=False))
    assert args.kit_args == "--/app/vulkan=false"


def test_physics_only_launch_is_left_on_vulkan(windows):
    args = argparse.Namespace(headless=True, enable_cameras=False, kit_args="")
    launch.prepare_launch(args)
    assert args.kit_args == ""


def test_opt_outs(windows, monkeypatch):
    assert launch.render_kit_args("--/app/vulkan=true") == "--/app/vulkan=true"
    monkeypatch.setenv("ISAACSWIM_VULKAN", "1")
    assert launch.render_kit_args(None) == ""


def test_linux_untouched(monkeypatch):
    monkeypatch.setattr(launch.sys, "platform", "linux")
    assert launch.render_kit_args("--foo") == "--foo"


def test_ros2_enables_the_bridge_on_the_bundled_libraries(windows, monkeypatch, tmp_path):
    from isaacswim import ros2
    lib = tmp_path / "humble" / "lib"
    lib.mkdir(parents=True)
    monkeypatch.setattr(ros2, "_bundled_lib", lambda distro: lib)
    monkeypatch.setattr(ros2.sys, "platform", "win32")
    monkeypatch.delenv("ROS_DISTRO", raising=False)
    monkeypatch.delenv("RMW_IMPLEMENTATION", raising=False)
    monkeypatch.setenv("PATH", "first")
    args = launch.prepare_launch(argparse.Namespace(headless=True, kit_args=""),
                                 ros2=True)
    assert args.kit_args == "--enable isaacsim.ros2.bridge"
    assert ros2.os.environ["ROS_DISTRO"] == "humble"
    assert ros2.os.environ["RMW_IMPLEMENTATION"] == "rmw_fastrtps_cpp"
    assert ros2.os.environ["PATH"].split(ros2.os.pathsep) == ["first", str(lib)]


def test_a_sourced_ros2_is_left_alone(monkeypatch):
    from isaacswim import ros2
    monkeypatch.setenv("ROS_DISTRO", "jazzy")
    monkeypatch.setenv("PATH", "mine")
    assert ros2.setup_environment() is False
    assert ros2.os.environ["PATH"] == "mine"


def test_windows_registers_the_ros_dll_directory(windows, monkeypatch, tmp_path):
    from isaacswim import ros2

    lib = tmp_path / "humble/lib"
    lib.mkdir(parents=True)
    handle = object()
    seen = []

    def add_directory(path):
        seen.append(path)
        return handle

    monkeypatch.setattr(ros2, "_bundled_lib", lambda _: lib)
    monkeypatch.setattr(ros2.os, "add_dll_directory", add_directory, raising=False)
    monkeypatch.delenv("ROS_DISTRO", raising=False)
    assert ros2.setup_environment()
    assert seen == [str(lib)]
    assert handle in ros2._DLL_DIR_HANDLES


def test_bundled_ros_libraries_can_live_in_the_sim6_core_extension(monkeypatch, tmp_path):
    import types

    from isaacswim import ros2

    lib = tmp_path / "exts/isaacsim.ros2.core/humble/lib"
    lib.mkdir(parents=True)
    monkeypatch.setattr(ros2.importlib.util, "find_spec", lambda _: types.SimpleNamespace(
        submodule_search_locations=[str(tmp_path)]))
    assert ros2._bundled_lib("humble") == lib


def test_windows_preloads_the_bundled_middleware_by_absolute_path(windows, monkeypatch, tmp_path):
    import ctypes

    from isaacswim import ros2

    lib = tmp_path / "humble/lib"
    lib.mkdir(parents=True)
    middleware = lib / "rmw_fastrtps_cpp.dll"
    middleware.touch()
    handle = object()
    seen = []

    def load(path):
        seen.append(path)
        return handle

    monkeypatch.setattr(ros2, "_bundled_lib", lambda _: lib)
    monkeypatch.setattr(ctypes, "WinDLL", load, raising=False)
    monkeypatch.delenv("ROS_DISTRO", raising=False)
    monkeypatch.delenv("RMW_IMPLEMENTATION", raising=False)
    assert ros2.setup_environment()
    assert seen == [str(middleware)]
    assert handle in ros2._RMW_DLL_HANDLES


@pytest.mark.parametrize("crash", [False, True])
def test_pumping_app_events_does_not_advance_physics_and_restores_state(monkeypatch, crash):
    import types

    key = "/app/player/playSimulations"
    values = {key: True}
    physics_steps = []
    calls = []
    settings = types.SimpleNamespace(get=values.get, set_bool=values.__setitem__)

    def update():
        calls.append("event")
        if values[key]:
            physics_steps.append("uncontrolled step")
        if crash:
            raise RuntimeError("event callback failed")

    carb = types.ModuleType("carb")
    carb.settings = types.SimpleNamespace(get_settings=lambda: settings)
    omni = types.ModuleType("omni")
    omni.kit = types.ModuleType("omni.kit")
    omni.kit.app = types.SimpleNamespace(get_app=lambda: types.SimpleNamespace(update=update))
    for name, module in [("carb", carb), ("carb.settings", carb.settings), ("omni", omni),
                         ("omni.kit", omni.kit), ("omni.kit.app", omni.kit.app)]:
        monkeypatch.setitem(sys.modules, name, module)
    if crash:
        with pytest.raises(RuntimeError, match="event callback failed"):
            launch.update_app()
    else:
        launch.update_app()
    assert calls == ["event"]
    assert physics_steps == []
    assert values[key] is True


def test_deferred_worker_extension_log_does_not_terminate_the_process():
    code = r"""
import threading, types
from isaacswim.launch import prepare_launch
prepare_launch({'headless': True})
module = types.ModuleType('extension_worker')
exec(
    "import sys\n"
    "def log():\n"
    " try:raise ValueError('worker log')\n"
    " except ValueError:captured=sys.exc_info()\n"
    " sys.excepthook(*captured)", module.__dict__)
worker = threading.Thread(target=module.log)
worker.start()
worker.join()
print('application continues')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stderr
    assert "application continues" in result.stdout
