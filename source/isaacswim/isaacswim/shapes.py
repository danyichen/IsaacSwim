"""FluidShape: an ellipsoid on a body that carries its drag, added mass and buoyancy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from isaacswim.ellipsoid import added_mass, ellipsoid_volume
from isaacswim.meshvolume import (
    TriMesh,
    mat_to_quat,
    principal_obb,
    quat_apply,
    quat_mul,
    triangulate,
)

__all__ = ["FluidShape", "DEFAULT_COEF"]

DEFAULT_COEF = (0.5, 0.25, 1.5, 1.0, 1.0)

_AXIS = {"x": 0, "y": 1, "z": 2}
_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def _as_vec(v, n: int, name: str) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size != n:
        raise ValueError(f"{name} must have {n} entries, got {a.size}")
    return a


@dataclass
class FluidShape:
    body: str
    semiaxes: np.ndarray
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array(_IDENTITY_QUAT))
    coef: np.ndarray = field(default_factory=lambda: np.array(DEFAULT_COEF))
    virtual_mass: np.ndarray | None = None
    virtual_inertia: np.ndarray | None = None
    mesh: TriMesh | None = None
    volume: float | None = None
    enabled: bool = True

    def __post_init__(self):
        self.semiaxes = _as_vec(self.semiaxes, 3, "semiaxes")
        if np.any(self.semiaxes <= 0.0):
            raise ValueError(f"{self.body}: semiaxes must be positive, got {self.semiaxes}")
        self.pos = _as_vec(self.pos, 3, "pos")
        self.quat = _as_vec(self.quat, 4, "quat")
        n = np.linalg.norm(self.quat)
        if n < 1e-9:
            raise ValueError(f"{self.body}: quat is not a rotation")
        self.quat = self.quat / n
        self.coef = _as_vec(self.coef, 5, "coef")

        if self.virtual_mass is None or self.virtual_inertia is None:
            vm, vi = added_mass(self.semiaxes)
            if self.virtual_mass is None:
                self.virtual_mass = vm
            if self.virtual_inertia is None:
                self.virtual_inertia = vi
        self.virtual_mass = _as_vec(self.virtual_mass, 3, "virtual_mass")
        self.virtual_inertia = _as_vec(self.virtual_inertia, 3, "virtual_inertia")

        if self.mesh is not None and not isinstance(self.mesh, TriMesh):
            m = self.mesh
            self.mesh = TriMesh.from_dict(m) if isinstance(m, dict) else TriMesh(*m)

        if self.volume is None:
            self.volume = float(self.mesh.volume if self.mesh is not None
                                else ellipsoid_volume(self.semiaxes))
        self.volume = float(self.volume)

    @classmethod
    def sphere(cls, body: str, radius: float, mesh: bool = False, **kw) -> FluidShape:
        return cls(body=body, semiaxes=(radius, radius, radius), **kw)

    @classmethod
    def capsule(cls, body: str, radius: float, half_length: float,
                axis: str = "z", *, mesh: bool = False, segments: int = 16,
                **kw) -> FluidShape:
        s = np.full(3, float(radius))
        s[_AXIS[axis]] = float(half_length) + float(radius)
        return cls(body=body, semiaxes=s,
                   **cls._primitive_mesh(mesh, kw, TriMesh.capsule,
                                         (radius, half_length, axis, segments,
                                          max(4, segments // 4)),
                                         2.0 * np.pi * radius ** 2 * half_length
                                         + 4.0 / 3.0 * np.pi * radius ** 3),
                   **kw)

    @classmethod
    def cylinder(cls, body: str, radius: float, half_length: float,
                 axis: str = "z", *, mesh: bool = False, segments: int = 16,
                 **kw) -> FluidShape:
        s = np.full(3, float(radius))
        s[_AXIS[axis]] = float(half_length)
        return cls(body=body, semiaxes=s,
                   **cls._primitive_mesh(mesh, kw, TriMesh.cylinder,
                                         (radius, half_length, axis, segments),
                                         2.0 * np.pi * radius ** 2 * half_length),
                   **kw)

    @classmethod
    def box(cls, body: str, half_extents, *, mesh: bool = False,
            **kw) -> FluidShape:
        h = _as_vec(half_extents, 3, "half_extents")
        return cls(body=body, semiaxes=h,
                   **cls._primitive_mesh(mesh, kw, TriMesh.box, (h,),
                                         8.0 * float(h.prod())),
                   **kw)

    @staticmethod
    def _primitive_mesh(want: bool, kw: dict, factory, args, exact_volume: float):
        if not want:
            return {}
        out = {"mesh": factory(*args, check_watertight=False)}
        if "volume" not in kw:
            out["volume"] = exact_volume
        return out

    @classmethod
    def from_mesh(cls, body: str, points, faces=None, *, counts=None, indices=None,
                  pos=None, quat=None, scale=None, volume: float | None = None,
                  check_watertight: bool = True, **kw) -> FluidShape:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if faces is None:
            if counts is None or indices is None:
                raise ValueError("give either `faces` or both `counts` and `indices`")
            faces = triangulate(counts, indices)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if scale is not None:
            pts = pts * _as_vec(scale, 3, "scale")

        obb = principal_obb(pts, faces)
        if obb is None:
            raise ValueError(
                f"{body}: the mesh is degenerate or encloses no volume, so neither a "
                "drag ellipsoid nor a displaced volume can be derived from it.")
        semi, centroid, R = obb

        pos = np.zeros(3) if pos is None else _as_vec(pos, 3, "pos")
        quat = np.array(_IDENTITY_QUAT) if quat is None else _as_vec(quat, 4, "quat")
        quat = quat / max(float(np.linalg.norm(quat)), 1e-30)
        shape_pos = pos + quat_apply(quat, centroid)
        shape_quat = quat_mul(quat, mat_to_quat(R))

        mesh = TriMesh.from_shape_frame(pts, faces, centre=centroid, rotation=R,
                                        check_watertight=check_watertight, name=body)
        return cls(body=body, semiaxes=semi, pos=shape_pos, quat=shape_quat,
                   mesh=mesh, volume=volume, **kw)

    @classmethod
    def from_inertia(cls, body: str, mass: float, inertia_diag,
                     *, volume: float | None = None, **kw) -> FluidShape:
        I = _as_vec(inertia_diag, 3, "inertia_diag")
        m = float(mass)
        if m <= 0.0:
            raise ValueError(f"{body}: mass must be positive")
        sq = 1.5 / m * np.array([I[1] + I[2] - I[0],
                                 I[2] + I[0] - I[1],
                                 I[0] + I[1] - I[2]])
        if np.any(sq <= 0.0):
            raise ValueError(
                f"{body}: inertia {I.tolist()} does not admit a box fit "
                "(it violates the triangle inequality, or the body is a shell). "
                "Give semiaxes explicitly.")
        return cls(body=body, semiaxes=np.sqrt(sq), volume=volume, **kw)

    @property
    def displaced_mass(self) -> float:
        return self.volume

    @property
    def ellipsoid_volume(self) -> float:
        return float(ellipsoid_volume(self.semiaxes))

    def neutral_density(self) -> float:
        return float("inf") if self.volume == 0.0 else 1.0 / self.volume

    def to_dict(self) -> dict:
        return {
            "body": self.body,
            "semiaxes": self.semiaxes.tolist(),
            "pos": self.pos.tolist(),
            "quat": self.quat.tolist(),
            "coef": self.coef.tolist(),
            "virtual_mass": self.virtual_mass.tolist(),
            "virtual_inertia": self.virtual_inertia.tolist(),
            "volume": self.volume,
            "enabled": self.enabled,
            **({} if self.mesh is None else {"mesh": self.mesh.to_dict()}),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FluidShape:
        return cls(**d)
