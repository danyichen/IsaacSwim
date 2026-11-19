from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import RigidObject, RigidObjectCfg

from isaacswim import FluidShape, Medium
from isaacswim.isaaclab import Hydrodynamics, HydrodynamicsCfg, _lab
from isaacswim.isaaclab._lab import init_rot

from ..core import Joint, Prop, SceneSpec

__all__ = [
    "spawn_props",
    "add_joints",
    "prop_hydrodynamics",
    "write_pose",
    "set_joint_friction",
    "REFERENCE_SPEED",
    "PROP_ROOT",
]

PROP_ROOT = "/World/props"
REFERENCE_SPEED = 0.5


def _shape_cfg(p: Prop, **kw):
    if p.shape == "box":
        return sim_utils.CuboidCfg(size=p.size, **kw)
    if p.shape == "sphere":
        return sim_utils.SphereCfg(radius=p.size[0], **kw)
    return sim_utils.CylinderCfg(radius=p.size[0], height=p.size[1], axis=p.axis, **kw)


def spawn_props(
    spec: SceneSpec, *, root="/World", origin=(0.0, 0.0, 0.0)
) -> dict[str, RigidObject]:
    out = {}
    for p in spec.props:
        dynamic = p.kind == "dynamic"
        material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(p.color),
            roughness=0.6,
            **({"emissive_color": tuple(p.emissive)} if p.emissive else {}),
        )
        kw = {
            "visual_material": material,
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=not dynamic,
                solver_position_iteration_count=16 if dynamic else 4,
                solver_velocity_iteration_count=4 if dynamic else 1,
                max_depenetration_velocity=1.0,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
        }
        if p.kind != "visual":
            kw["collision_props"] = sim_utils.CollisionPropertiesCfg(
                contact_offset=0.004, rest_offset=0.003 if dynamic else 0.0
            )
            kw["physics_material"] = sim_utils.RigidBodyMaterialCfg(
                static_friction=p.friction[0], dynamic_friction=p.friction[1], restitution=0.0
            )
        if dynamic:
            kw["mass_props"] = sim_utils.MassPropertiesCfg(density=p.density)
        pos, quat = spec.pose(p.name)
        out[p.name] = RigidObject(
            RigidObjectCfg(
                prim_path=f"{root}/props/{p.name}",
                spawn=_shape_cfg(p, **kw),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(float(v) for v in np.asarray(pos) + origin), rot=init_rot(quat)
                ),
            )
        )
        if not dynamic:
            _pin(f"{root}/props/{p.name}")
        if p.kind == "visual":
            _visual_inertia(f"{root}/props/{p.name}")
    return out


def _visual_inertia(path: str) -> None:
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdPhysics

    prim = get_current_stage().GetPrimAtPath(path)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(1.0)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0, 1.0, 1.0))


def _pin(path: str) -> None:
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import PhysxSchema

    prim = get_current_stage().GetPrimAtPath(path)
    api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    api.CreateLockedPosAxisAttr(7)
    api.CreateLockedRotAxisAttr(7)


def add_joints(spec: SceneSpec, *, root="/World", origin=(0.0, 0.0, 0.0)) -> None:
    import omni.usd
    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    for j in spec.joints:
        body = f"{root}/props/{j.body}"
        pos, _ = spec.pose(j.body)
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/joints/{j.name}")
        joint.CreateBody1Rel().SetTargets([body])
        joint.CreateAxisAttr(j.axis.upper())
        joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in np.asarray(j.anchor) + origin]))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        local1 = np.asarray(j.anchor, dtype=np.float64) - pos
        joint.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in local1]))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateDampingAttr(_damping_usd(j))


def _damping_usd(j: Joint) -> float:
    per_rad = j.friction_torque / REFERENCE_SPEED + j.damping
    return float(per_rad * math.pi / 180.0)


def set_joint_friction(spec: SceneSpec, *, root="/World") -> None:
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    for j in spec.joints:
        drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"{root}/joints/{j.name}"), "angular")
        drive.GetDampingAttr().Set(_damping_usd(j))


def prop_hydrodynamics(spec: SceneSpec, objects: dict) -> dict[str, Hydrodynamics]:
    jointed = {j.body for j in spec.joints}
    out = {}
    for p in spec.props:
        if p.kind != "dynamic" or p.name in jointed:
            continue
        obj = objects[p.name]
        body = obj.body_names[0]
        he = p.half_extents
        if p.shape == "box":
            shape = FluidShape.box(body=body, half_extents=list(he), volume=p.volume)
        elif p.shape == "sphere":
            shape = FluidShape.sphere(body=body, radius=p.size[0])
        else:
            shape = FluidShape.cylinder(
                body=body, radius=p.size[0], half_length=p.size[1] / 2, axis=p.axis.lower()
            )
        out[p.name] = Hydrodynamics(
            HydrodynamicsCfg(shapes=[shape], medium=Medium.sea_water(), buoyancy=True), obj
        )
    return out


def write_pose(obj, pos, quat, device) -> None:
    pose = torch.tensor([[*pos, *quat]], dtype=torch.float32, device=device)
    vel = torch.zeros(1, 6, dtype=torch.float32, device=device)
    if hasattr(obj, "write_root_pose_to_sim"):
        _lab.write_root_pose(obj, pose)
        _lab.write_root_velocity(obj, vel)
    else:  # pragma: no cover
        obj.write_root_state_to_sim(torch.cat([pose, vel], dim=1))
    obj.reset()
