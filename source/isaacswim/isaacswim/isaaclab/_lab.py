from __future__ import annotations

import torch

__all__ = ["LabData", "data", "is_lab3", "to_torch", "quat_from_lab", "quat_to_lab",
           "physx_view", "view_masses", "view_inertias", "view_coms", "init_rot",
           "write_root_pose", "write_root_velocity", "write_root_state", "write_joint_state",
           "lab_major", "set_joint_position_target", "set_joint_effort_target", "write_wrench"]


def lab_major() -> int:
    try:
        from importlib.metadata import version
        return int(version("isaaclab").split(".")[0])
    except Exception:
        return 2


def to_torch(x):
    if isinstance(x, torch.Tensor):
        return x
    t = getattr(x, "torch", None)
    if t is not None:
        return t
    if type(x).__module__.startswith("warp"):
        import warp as wp
        return wp.to_torch(x)
    return torch.as_tensor(x)


def is_lab3(asset) -> bool:
    cached = getattr(asset, "_isaacswim_lab3", None)
    if cached is not None:
        return cached
    d = asset.data
    probe = None
    for name in ("root_link_pos_w", "root_pos_w", "body_link_pos_w", "body_pos_w"):
        try:
            probe = getattr(d, name)
            break
        except AttributeError:
            continue
    lab3 = probe is not None and not isinstance(probe, torch.Tensor) and hasattr(probe, "torch")
    try:
        asset._isaacswim_lab3 = lab3
    except Exception:  # pragma: no cover
        pass
    return lab3


def quat_from_lab(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 3:4], q[..., 0:3]], -1)


def quat_to_lab(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 1:4], q[..., 0:1]], -1)


def init_rot(quat_wxyz, lab3: bool | None = None) -> tuple:
    q = tuple(float(v) for v in quat_wxyz)
    if lab3 if lab3 is not None else lab_major() >= 3:
        return (q[1], q[2], q[3], q[0])
    return q


def _has_quat(name: str) -> int | None:
    if "quat" in name:
        return 0
    if "pose" in name or "state" in name:
        return 3
    return None


class LabData:
    __slots__ = ("_asset", "_lab3")

    def __init__(self, asset):
        object.__setattr__(self, "_asset", asset)
        object.__setattr__(self, "_lab3", is_lab3(asset))

    def __getattr__(self, name):
        raw = getattr(self._asset.data, name)
        if not self._lab3 or not hasattr(raw, "torch"):
            return raw
        t = raw.torch
        k = _has_quat(name)
        if k is None or t.shape[-1] < k + 4:
            return t
        return torch.cat([t[..., :k], quat_from_lab(t[..., k:k + 4]), t[..., k + 4:]], -1)

    def get(self, name, default=None):
        try:
            return getattr(self, name)
        except AttributeError:
            return default


def data(asset) -> LabData:
    return LabData(asset)


def physx_view(asset):
    view = getattr(type(asset), "root_view", None) and asset.root_view
    if view is None:
        view = getattr(asset, "root_physx_view", None)
    return view


def view_masses(asset) -> torch.Tensor:
    return to_torch(physx_view(asset).get_masses()).detach().cpu()


def view_inertias(asset) -> torch.Tensor:
    v = to_torch(physx_view(asset).get_inertias()).detach().cpu()
    return v.reshape(v.shape[0], -1, 9)


def view_coms(asset) -> torch.Tensor:
    v = to_torch(physx_view(asset).get_coms()).detach().cpu()
    return v.reshape(v.shape[0], -1, 7)


def _write(asset, base: str, value, env_ids, kw: str):
    if is_lab3(asset):
        fn = getattr(asset, f"{base}_index")
        return fn(**{kw: value, "env_ids": env_ids})
    fn = getattr(asset, base)
    return fn(value) if env_ids is None else fn(value, env_ids)


def write_root_pose(asset, pose_wxyz: torch.Tensor, env_ids=None, link: bool = False):
    pose = pose_wxyz
    if is_lab3(asset):
        pose = torch.cat([pose_wxyz[..., :3], quat_to_lab(pose_wxyz[..., 3:7])], -1)
    base = "write_root_link_pose_to_sim" if link else "write_root_pose_to_sim"
    return _write(asset, base, pose, env_ids, "root_pose")


def write_root_velocity(asset, vel: torch.Tensor, env_ids=None, link: bool = False):
    base = "write_root_link_velocity_to_sim" if link else "write_root_velocity_to_sim"
    return _write(asset, base, vel, env_ids, "root_velocity")


def write_root_state(asset, state_wxyz: torch.Tensor, env_ids=None):
    write_root_pose(asset, state_wxyz[..., :7], env_ids)
    write_root_velocity(asset, state_wxyz[..., 7:13], env_ids)


def write_joint_state(asset, pos: torch.Tensor, vel: torch.Tensor, env_ids=None):
    if is_lab3(asset):
        asset.write_joint_position_to_sim_index(position=pos, env_ids=env_ids)
        asset.write_joint_velocity_to_sim_index(velocity=vel, env_ids=env_ids)
        return
    if env_ids is None:
        asset.write_joint_state_to_sim(pos, vel)
    else:
        asset.write_joint_state_to_sim(pos, vel, None, env_ids)


def set_joint_position_target(asset, target: torch.Tensor, joint_ids=None):
    if is_lab3(asset):
        return asset.set_joint_position_target_index(target=target, joint_ids=joint_ids)
    return asset.set_joint_position_target(target, joint_ids=joint_ids)


def set_joint_effort_target(asset, target: torch.Tensor, joint_ids=None):
    if is_lab3(asset):
        return asset.set_joint_effort_target_index(target=target, joint_ids=joint_ids)
    return asset.set_joint_effort_target(target, joint_ids=joint_ids)


def write_wrench(asset, F: torch.Tensor, T: torch.Tensor, is_global: bool = False):
    composer = getattr(asset, "permanent_wrench_composer", None)
    if composer is not None:
        if hasattr(composer, "set_forces_and_torques_index"):
            return composer.set_forces_and_torques_index(forces=F, torques=T, is_global=is_global)
        if hasattr(composer, "set_forces_and_torques"):
            return composer.set_forces_and_torques(forces=F, torques=T)
    return asset.set_external_force_and_torque(F, T)
