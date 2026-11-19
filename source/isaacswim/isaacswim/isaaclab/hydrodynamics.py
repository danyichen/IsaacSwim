"""Applies the water's force and torque to an Isaac Lab asset every physics step."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from isaacswim.added_mass import AddedMass
from isaacswim.ellipsoid import ellipsoid_volume
from isaacswim.isaaclab import _lab
from isaacswim.medium import Medium
from isaacswim.shapes import FluidShape
from isaacswim.torch_math import quat_apply
from isaacswim.wrench import HydroWrench

__all__ = ["Hydrodynamics", "HydrodynamicsCfg", "shapes_from_articulation"]

try:
    from isaaclab.utils import configclass as _configclass
except Exception:  # pragma: no cover
    _configclass = dataclass


@_configclass
class HydrodynamicsCfg:
    shapes: list = field(default_factory=list)
    medium: Medium = field(default_factory=Medium)
    auto_shapes: bool | str = False
    auto_body_pattern: str = ".*"
    body_density: float | None = None
    buoyancy: bool = True
    mesh_buoyancy: bool = True
    drag_scales_with_submersion: bool = True
    current_fn: Callable | None = None
    surface_fn: Callable | None = None
    added_mass: bool = False
    physics_dt: float | None = None
    is_global: bool = False
    debug_check_nan: bool = False


def shapes_from_articulation(asset, *, pattern: str = ".*",
                             body_density: float | None = None,
                             min_mass: float = 1e-3,
                             min_implied_density: float = 50.0,
                             coef=None, verbose: bool = True) -> list[FluidShape]:
    import re

    names = list(asset.body_names)
    sel = [i for i, n in enumerate(names) if re.fullmatch(pattern, n)]
    if not sel:
        raise ValueError(f"pattern {pattern!r} matched none of {names}")

    masses = _lab.view_masses(asset).numpy()[0]
    inertias = _lab.view_inertias(asset).numpy()[0]
    coms = _lab.view_coms(asset).numpy()[0]

    shapes: list[FluidShape] = []
    skipped: list[tuple[str, str]] = []
    for i in sel:
        name = names[i]
        mass = float(masses[i])
        if mass < min_mass:
            skipped.append((name, f"mass {mass:.2e} kg < {min_mass:g}"))
            continue
        I = inertias[i].reshape(3, 3)
        diag = np.array([I[0, 0], I[1, 1], I[2, 2]])
        volume = None if body_density is None else mass / body_density
        kw = {} if coef is None else {"coef": coef}
        try:
            shape = FluidShape.from_inertia(name, mass, diag, volume=volume,
                                            pos=coms[i, :3], **kw)
        except ValueError as exc:
            skipped.append((name, str(exc).split(":", 1)[-1].strip()[:60]))
            continue

        fitted = float(ellipsoid_volume(shape.semiaxes))
        implied = mass / max(fitted, 1e-30)
        if implied < min_implied_density:
            skipped.append((name, f"fit implies {implied:.1f} kg/m^3 "
                                  f"({fitted * 1e3:.1f} L for {mass:.3f} kg)"))
            continue
        shapes.append(shape)

    if not shapes:
        raise ValueError(
            f"every body was filtered out: {skipped}.  Lower min_mass / "
            "min_implied_density, or give shapes explicitly.")
    if verbose and skipped:
        print(f"[IsaacSwim] fitted {len(shapes)} drag ellipsoids, skipped {len(skipped)}:")
        for name, why in skipped:
            print(f"[IsaacSwim]   {name}: {why}")
    return shapes


def _auto_shapes(asset, cfg: HydrodynamicsCfg) -> list[FluidShape]:
    import re

    mode = "geometry" if cfg.auto_shapes is True else cfg.auto_shapes
    if mode == "inertia":
        return shapes_from_articulation(asset, pattern=cfg.auto_body_pattern,
                                        body_density=cfg.body_density)
    if mode != "geometry":
        raise ValueError(f"auto_shapes must be True, 'geometry' or 'inertia', not {mode!r}")
    from isaacswim.usd import shapes_from_articulation_usd

    names = [n for n in asset.body_names if re.fullmatch(cfg.auto_body_pattern, n)]
    if not names:
        raise ValueError(f"auto_body_pattern {cfg.auto_body_pattern!r} matches no body")
    return shapes_from_articulation_usd(asset, body_names=names,
                                        mesh_buoyancy=cfg.mesh_buoyancy)


class Hydrodynamics:
    def __init__(self, cfg: HydrodynamicsCfg, asset, device: str | None = None):
        self.cfg = cfg
        self.asset = asset
        self.device = device or str(asset.device)

        try:
            body_names = list(asset.body_names)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "asset.body_names is not available yet.  Build Hydrodynamics after "
                "the simulation has been reset -- in a DirectRLEnv that means after "
                "super().__init__(), not inside _setup_scene()."
            ) from exc
        if not body_names:
            raise RuntimeError("the asset reports no bodies; it is not initialised yet")

        shapes = list(cfg.shapes)
        if not shapes:
            if not cfg.auto_shapes:
                raise ValueError("HydrodynamicsCfg has no shapes and auto_shapes is False")
            shapes = _auto_shapes(asset, cfg)

        self.wrench = HydroWrench(
            shapes, body_names, cfg.medium,
            body_com=self._body_com(asset),
            buoyancy=cfg.buoyancy,
            mesh_buoyancy=cfg.mesh_buoyancy,
            drag_scales_with_submersion=cfg.drag_scales_with_submersion,
            device=self.device,
            dtype=self._asset_dtype(asset),
        )
        self._needs_pos = self._pos_needed()
        self._writer = self._resolve_writer(asset)
        self.added_mass: AddedMass | None = None
        if cfg.added_mass and cfg.physics_dt is not None:
            self._build_added_mass(cfg.physics_dt)

    @staticmethod
    def _asset_dtype(asset) -> torch.dtype:
        for attr in ("body_link_quat_w", "body_quat_w"):
            buf = getattr(_lab.data(asset), attr, None)
            if buf is not None:
                return buf.dtype
        return torch.float32

    @staticmethod
    def _body_com(asset):
        data = _lab.data(asset)
        for attr in ("com_pos_b", "body_com_pos_b"):
            com = getattr(data, attr, None)
            if com is not None:
                return com
        return None

    def _link_state(self):
        data = _lab.data(self.asset)
        quat = getattr(data, "body_link_quat_w", None)
        if quat is None:
            return data.body_quat_w, data.body_pos_w, data.body_lin_vel_w, data.body_ang_vel_w, None
        vel = getattr(data, "body_com_vel_w", None)
        if vel is not None:
            vel = vel.clone()
            return quat, data.body_link_pos_w, vel[..., 0:3], vel[..., 3:6], vel
        return quat, data.body_link_pos_w, data.body_link_lin_vel_w, data.body_link_ang_vel_w, None

    def _resolve_writer(self, asset):
        composer = getattr(asset, "permanent_wrench_composer", None)
        if composer is not None and hasattr(composer, "set_forces_and_torques_index"):
            is_global = self.cfg.is_global

            def write(F, T):
                composer.set_forces_and_torques_index(forces=F, torques=T, is_global=is_global)
            return write
        if composer is not None and hasattr(composer, "set_forces_and_torques"):
            params = inspect.signature(composer.set_forces_and_torques).parameters
            if self.cfg.is_global and "is_global" not in params:
                raise ValueError("this IsaacLab composer has no is_global external wrench")
            def write(F, T):
                kw = {"is_global": self.cfg.is_global} if "is_global" in params else {}
                composer.set_forces_and_torques(forces=F, torques=T, **kw)
            return write

        fn = asset.set_external_force_and_torque
        params = inspect.signature(fn).parameters
        kw = {}
        if "is_global" in params:
            kw["is_global"] = self.cfg.is_global
        elif self.cfg.is_global:
            raise ValueError("this IsaacLab version has no is_global external wrench")

        def write(F, T):
            fn(F, T, **kw)
        return write

    def compute(self):
        quat, pos, lin, ang, com_vel = self._link_state()
        self._state = quat, pos, com_vel
        kw = {} if self.cfg.surface_fn is None else {"surface": self.cfg.surface_fn}
        F, T = self.wrench.compute(quat, lin, ang, pos_w=pos if self._needs_pos else None,
                                   current=self.cfg.current_fn, vel_at_com=com_vel is not None,
                                   **kw)
        if self.cfg.debug_check_nan:
            bad = ~(torch.isfinite(F).all() & torch.isfinite(T).all())
            if bool(bad):
                raise FloatingPointError(
                    "hydrodynamic wrench went non-finite.  Usual causes: a body "
                    "already had a non-finite velocity, or an ellipsoid with a "
                    "near-zero semi-axis.")
        return F, T

    def apply(self, add: tuple[torch.Tensor, torch.Tensor] | None = None,
              dt: float | None = None):
        F, T = self.compute()
        if add is not None:
            F = F + add[0]
            T = T + add[1]
        if self.cfg.added_mass:
            if self.added_mass is None:
                self._build_added_mass(dt)
            quat, pos, com_vel = self._state
            fa, ta = self.added_mass.force(quat, self._com_vel() if com_vel is None else com_vel,
                                           F, T, pos)
            F, T = F + fa, T + ta
        if self.cfg.is_global:
            quat = self._state[0]
            F, T = quat_apply(quat, F), quat_apply(quat, T)
        self._writer(F, T)
        return F, T

    def reset(self, env_ids=None):
        if self.added_mass is not None:
            self.added_mass.reset(env_ids)

    def _com_vel(self):
        data = _lab.data(self.asset)
        vel = getattr(data, "body_com_vel_w", None)
        if vel is not None:
            return vel
        lin = getattr(data, "body_com_lin_vel_w", None)
        if lin is None:
            lin = data.body_lin_vel_w
            return torch.cat([lin, data.body_ang_vel_w], -1)
        return torch.cat([lin, data.body_com_ang_vel_w], -1)

    def _build_added_mass(self, dt=None):
        dt = dt or self.cfg.physics_dt
        if dt is None:
            try:
                from isaaclab.sim import SimulationContext
                dt = SimulationContext.instance().get_physics_dt()
            except Exception as exc:
                raise ValueError("added_mass needs the physics step: set "
                                 "HydrodynamicsCfg.physics_dt or pass apply(dt=...)") from exc
        masses = _lab.view_masses(self.asset)
        inertias = _lab.view_inertias(self.asset)
        E = masses.shape[0]
        self.added_mass = AddedMass(
            self.wrench.added_mass_matrix(), masses, inertias.reshape(E, -1, 3, 3),
            float(dt), num_envs=E, device=self.device, dtype=self.wrench.dtype,
            body_com=self._body_com(self.asset))

    def set_medium(self, medium: Medium):
        self.cfg.medium = medium
        self.wrench.set_medium(medium)
        self._needs_pos = self._pos_needed()
        if self.added_mass is not None:
            self._build_added_mass(self.added_mass.dt)

    def _pos_needed(self) -> bool:
        free_surface = self.wrench.free_surface or self.cfg.surface_fn is not None
        return (self.wrench.buoyancy and free_surface) or self.cfg.current_fn is not None

    def refresh_body_com(self):
        self.wrench.set_body_com(self._body_com(self.asset))
        if self.added_mass is not None:
            self._build_added_mass(self.added_mass.dt)

    def __repr__(self) -> str:
        return f"Hydrodynamics({self.wrench!r})"
