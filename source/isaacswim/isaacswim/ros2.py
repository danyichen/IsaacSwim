"""ROS 2 topics for vehicles and sensors through Isaac Sim's bridge."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

__all__ = ["BRIDGE_EXTENSION", "setup_environment", "publish_vehicle", "CmdVel",
           "articulation_root", "update"]

BRIDGE_EXTENSION = "isaacsim.ros2.bridge"
_DLL_DIR_HANDLES = []
_RMW_DLL_HANDLES = []


def _bundled_lib(distro: str) -> Path | None:
    spec = importlib.util.find_spec("isaacsim")
    for root in (spec.submodule_search_locations or []) if spec else []:
        for extension in (BRIDGE_EXTENSION, "isaacsim.ros2.core"):
            lib = Path(root) / "exts" / extension / distro / "lib"
            if lib.is_dir():
                return lib
    return None


def setup_environment(distro: str = "humble", rmw: str = "rmw_fastrtps_cpp") -> bool:
    if os.environ.get("ROS_DISTRO"):
        return False
    lib = _bundled_lib(distro)
    if lib is None:
        return False
    os.environ["ROS_DISTRO"] = distro
    os.environ.setdefault("RMW_IMPLEMENTATION", rmw)
    if sys.platform == "win32":
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + str(lib)
        if hasattr(os, "add_dll_directory"):
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(lib)))
        middleware = lib / f"{os.environ['RMW_IMPLEMENTATION']}.dll"
        if middleware.is_file():
            import ctypes
            _RMW_DLL_HANDLES.append(ctypes.WinDLL(str(middleware)))
    elif str(lib) not in os.environ.get("LD_LIBRARY_PATH", ""):
        print(f"[isaacswim.ros2] for Isaac Sim's bundled ROS 2 {distro}, start with\n"
              f"  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{lib}", flush=True)
    return True


def update() -> None:
    from isaacswim.isaaclab._lab import lab_major

    if lab_major() >= 3:
        from isaacswim.launch import update_app
        update_app()


def _require_bridge() -> None:
    import omni.graph.core as og

    if f"{BRIDGE_EXTENSION}.ROS2PublishClock" in og.get_registered_nodes():
        return
    hint = "launch with isaacswim.launch.prepare_launch(args, ros2=True)"
    lib = _bundled_lib(os.environ.get("ROS_DISTRO") or "humble")
    if (sys.platform != "win32" and lib is not None
            and str(lib) not in os.environ.get("LD_LIBRARY_PATH", "")):
        hint += (", and either source a ROS 2 installation or, for Isaac Sim's bundled "
                 f"ROS 2, start with\n  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{lib}")
    raise RuntimeError(f"Isaac Sim's {BRIDGE_EXTENSION} extension is not running: {hint}")


def _root(path: str) -> str:
    return articulation_root(path)


def _keys():
    import omni.graph.core as og
    return og, og.Controller.Keys


def _set_targets(node_path: str, name: str, prims) -> None:
    from isaacsim.core.nodes.scripts.utils import set_target_prims
    set_target_prims(primPath=node_path, inputName=f"inputs:{name}",
                     targetPrimPaths=list(prims))


def articulation_root(path: str) -> str:
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    raise ValueError(f"no articulation root at or under {path!r}")


def publish_vehicle(articulation_root: str, base_link: str, *, namespace: str = "",
                    graph_path: str = "/isaacswim/ROS2", clock: bool = True,
                    tf: bool = True, joint_states: bool = True, odometry: bool = True,
                    odom_frame: str = "odom", base_frame: str = "base_link"):
    _require_bridge()
    og, keys = _keys()
    nodes = [("tick", "omni.graph.action.OnPlaybackTick"),
             ("time", "isaacsim.core.nodes.IsaacReadSimulationTime")]
    connect, values, targets = [], [], []

    def add(name, kind, **inputs):
        nodes.append((name, f"{BRIDGE_EXTENSION}.{kind}"))
        connect.append(("tick.outputs:tick", f"{name}.inputs:execIn"))
        connect.append(("time.outputs:simulationTime", f"{name}.inputs:timeStamp"))
        values.extend((f"{name}.inputs:{k}", v) for k, v in
                      {"nodeNamespace": namespace, **inputs}.items())

    if clock:
        add("clock", "ROS2PublishClock")
    if tf:
        add("tf", "ROS2PublishTransformTree")
        targets.append(("tf", "targetPrims", [_root(articulation_root)]))
    if joint_states:
        add("joints", "ROS2PublishJointState")
        targets.append(("joints", "targetPrim", [_root(articulation_root)]))
    if odometry:
        nodes.append(("odom_compute", "isaacsim.core.nodes.IsaacComputeOdometry"))
        connect.append(("tick.outputs:tick", "odom_compute.inputs:execIn"))
        add("odom", "ROS2PublishOdometry", odomFrameId=odom_frame,
            chassisFrameId=base_frame)
        for out, inp in (("position", "position"), ("orientation", "orientation"),
                         ("linearVelocity", "linearVelocity"),
                         ("angularVelocity", "angularVelocity")):
            connect.append((f"odom_compute.outputs:{out}", f"odom.inputs:{inp}"))
        targets.append(("odom_compute", "chassisPrim", [base_link]))

    graph, *_ = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {keys.CREATE_NODES: nodes, keys.CONNECT: connect, keys.SET_VALUES: values})
    for node, name, prims in targets:
        _set_targets(f"{graph_path}/{node}", name, prims)
    return graph


class CmdVel:
    def __init__(self, topic: str = "/cmd_vel", *, namespace: str = "",
                 graph_path: str = "/isaacswim/CmdVel"):
        _require_bridge()
        og, keys = _keys()
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {keys.CREATE_NODES: [("tick", "omni.graph.action.OnPlaybackTick"),
                                 ("twist", f"{BRIDGE_EXTENSION}.ROS2SubscribeTwist")],
             keys.CONNECT: [("tick.outputs:tick", "twist.inputs:execIn")],
             keys.SET_VALUES: [("twist.inputs:topicName", topic),
                               ("twist.inputs:nodeNamespace", namespace)]})
        self._lin = og.Controller.attribute(f"{graph_path}/twist.outputs:linearVelocity")
        self._ang = og.Controller.attribute(f"{graph_path}/twist.outputs:angularVelocity")

    def read(self):
        import numpy as np
        return (np.asarray(self._lin.get(), dtype=np.float64),
                np.asarray(self._ang.get(), dtype=np.float64))
