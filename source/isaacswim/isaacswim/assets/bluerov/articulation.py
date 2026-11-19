from __future__ import annotations

from pathlib import Path

__all__ = ["robot_cfg", "gripper_shapes"]


def gripper_shapes(asset_dir: Path) -> list:
    import json

    from isaacswim import FluidShape

    path = Path(asset_dir) / "robotiq_hydro.json"
    if not path.is_file():
        return []
    rows = json.loads(path.read_text())
    return [FluidShape(body=b, semiaxes=r["semiaxes"], pos=r["pos"], volume=r["volume"])
            for b, r in rows.items()]


def robot_cfg(usd: Path, *, pos, quat, joint_pos: dict):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    from isaacswim.isaaclab._lab import init_rot

    return ArticulationCfg(
        prim_path="/World/rov",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(v) for v in pos), rot=init_rot(quat),
            joint_pos={k: float(v) for k, v in joint_pos.items()}),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["alpha_axis_.*", "camera_mount_joint"],
                                       stiffness=200.0, damping=30.0),
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"], effort_limit_sim=10.0,
                velocity_limit_sim=1.0, stiffness=11.25, damping=0.1),
            "gripper_finger": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_joint"], effort_limit_sim=1.0,
                velocity_limit_sim=1.0, stiffness=0.2, damping=0.001),
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
                effort_limit_sim=1.0, velocity_limit_sim=1.0, stiffness=0.0, damping=0.0),
        },
    )
