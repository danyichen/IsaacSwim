from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "TriMesh",
    "MeshDisplacement",
    "triangulate",
    "mesh_volume_centroid",
    "principal_obb",
    "quat_mul",
    "quat_apply",
    "mat_to_quat",
]

_TINY = 1e-300


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw - ax*bx - ay*by - az*bz,
                     aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw])


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, xyz = q[0], q[1:]
    t = np.cross(xyz, v)
    return v + 2.0 * (w * t + np.cross(xyz, t))


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0.0:
        w = math.sqrt(1.0 + tr) * 0.5
        k = 0.25 / w
        q = np.array([w, (R[2, 1] - R[1, 2]) * k,
                      (R[0, 2] - R[2, 0]) * k, (R[1, 0] - R[0, 1]) * k])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k_ = (i + 1) % 3, (i + 2) % 3
        d = math.sqrt(max(1e-30, 1.0 + R[i, i] - R[j, j] - R[k_, k_])) * 2.0
        q = np.empty(4)
        q[0] = (R[k_, j] - R[j, k_]) / d
        q[1 + i] = 0.25 * d
        q[1 + j] = (R[j, i] + R[i, j]) / d
        q[1 + k_] = (R[k_, i] + R[i, k_]) / d
    n = np.linalg.norm(q)
    return q / n if n > 1e-30 else np.array([1.0, 0.0, 0.0, 0.0])


def triangulate(counts, indices) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if int(counts.sum()) != indices.size:
        raise ValueError(f"face stream is {indices.size} long but counts sum to "
                         f"{int(counts.sum())}")
    tris: list[tuple[int, int, int]] = []
    off = 0
    for c in counts:
        c = int(c)
        if c >= 3:
            fan = indices[off:off + c]
            tris.extend((int(fan[0]), int(fan[i]), int(fan[i + 1])) for i in range(1, c - 1))
        off += c
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def mesh_volume_centroid(points: np.ndarray, faces: np.ndarray):
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = det.sum() / 6.0
    if abs(vol) < _TINY:
        return 0.0, np.zeros(3)
    centroid = np.einsum("i,ij->j", det, a + b + c) / (24.0 * vol)
    return float(vol), centroid


def _open_edge_count(faces: np.ndarray) -> int:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    direction = np.where(e[:, 0] < e[:, 1], 1, -1)
    balance = np.bincount(inverse, weights=direction, minlength=len(counts))
    return int(((counts != 2) | (balance != 0)).sum())


def principal_obb(points: np.ndarray, faces: np.ndarray):
    if faces.size == 0:
        return None
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = det.sum() / 6.0
    if abs(vol) < 1e-30:
        return None
    centroid = (np.einsum("i,ij->j", det, a + b + c) / 4.0) / (6.0 * vol)

    P = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            vi = np.stack([a[:, i], b[:, i], c[:, i]], axis=1)
            vj = np.stack([a[:, j], b[:, j], c[:, j]], axis=1)
            P[i, j] = P[j, i] = (det * (vi.sum(axis=1) * vj.sum(axis=1)
                                        + np.einsum("ij,ij->i", vi, vj))).sum() / 120.0
    I = np.eye(3) * np.trace(P) - P
    I = I - vol * (np.eye(3) * float(centroid @ centroid) - np.outer(centroid, centroid))
    I = 0.5 * (I + I.T)

    scale = max(float(np.max(np.abs(np.diag(I)))), 1e-300)
    I[np.abs(I) < 1e-9 * scale] = 0.0
    _, R = np.linalg.eigh(I)
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]

    local = (points - centroid) @ R
    half = np.max(np.abs(local), axis=0)
    if np.any(half <= 0.0):
        return None
    return half, centroid, R


@dataclass
class TriMesh:
    points: np.ndarray
    faces: np.ndarray
    name: str = ""

    def __init__(self, points, faces=None, *, counts=None, indices=None,
                 check_watertight: bool = True, name: str = ""):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if faces is None:
            if counts is None or indices is None:
                raise ValueError("give either `faces` or both `counts` and `indices`")
            f = triangulate(counts, indices)
        else:
            f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if f.size == 0:
            raise ValueError(f"{name or 'mesh'}: no triangles")
        if f.max() >= len(pts) or f.min() < 0:
            raise ValueError(f"{name or 'mesh'}: face index out of range")

        a, b, c = pts[f[:, 0]], pts[f[:, 1]], pts[f[:, 2]]
        area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        f = f[area2 > 1e-24 * max(float(area2.max()), 1e-300)]
        if f.size == 0:
            raise ValueError(f"{name or 'mesh'}: every triangle is degenerate")

        vol, _ = mesh_volume_centroid(pts, f)
        if abs(vol) < 1e-30:
            raise ValueError(
                f"{name or 'mesh'}: encloses no volume ({vol:.3e} m^3).  A flat "
                "sheet or an open surface displaces nothing; give the shape an "
                "explicit `volume` instead, or close the mesh.")
        if vol < 0.0:
            f = f[:, [0, 2, 1]]

        self.points = pts
        self.faces = f
        self.name = name

        self.volume, self.centroid = mesh_volume_centroid(pts, f)
        self.radius = float(np.linalg.norm(pts, axis=1).max())
        self.open_edges = _open_edge_count(f)
        if check_watertight and self.open_edges:
            warnings.warn(
                f"{name or 'mesh'}: {self.open_edges} boundary edges or "
                "inconsistently wound edges; "
                f"displaced volume ({self.volume * 1e3:.2f} L) and centroid may be invalid. "
                "Repair the mesh or use .convex_hull() when a convex approximation is suitable.",
                stacklevel=2)

    @classmethod
    def from_shape_frame(cls, points, faces=None, *, counts=None, indices=None,
                         centre=None, rotation=None, scale=None, **kw) -> TriMesh:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if scale is not None:
            pts = pts * np.asarray(scale, dtype=np.float64).reshape(3)
        if centre is not None:
            cen = np.asarray(centre, dtype=np.float64).reshape(3)
            if scale is not None:
                cen = cen * np.asarray(scale, dtype=np.float64).reshape(3)
            pts = pts - cen
        if rotation is not None:
            pts = pts @ np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        return cls(pts, faces, counts=counts, indices=indices, **kw)

    @classmethod
    def box(cls, half_extents, **kw) -> TriMesh:
        h = np.asarray(half_extents, dtype=np.float64).reshape(3)
        s = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
                     dtype=np.float64) * h
        f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                      [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int64)
        return cls(s, f, name="box", **kw)

    @classmethod
    def sphere(cls, radius: float, subdivisions: int = 3, **kw) -> TriMesh:
        t = (1.0 + 5.0 ** 0.5) / 2.0
        v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                      [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                      [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=np.float64)
        f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]],
                     dtype=np.int64)
        for _ in range(int(subdivisions)):
            mid: dict[tuple[int, int], int] = {}
            verts = list(v)
            new = []
            for tri in f:
                m = []
                for i in range(3):
                    a, b = int(tri[i]), int(tri[(i + 1) % 3])
                    k = (min(a, b), max(a, b))
                    if k not in mid:
                        mid[k] = len(verts)
                        verts.append((verts[a] + verts[b]) * 0.5)
                    m.append(mid[k])
                a, b, c = (int(x) for x in tri)
                new += [[a, m[0], m[2]], [b, m[1], m[0]], [c, m[2], m[1]], m]
            v = np.asarray(verts, dtype=np.float64)
            f = np.asarray(new, dtype=np.int64)
        v = v / np.linalg.norm(v, axis=1, keepdims=True) * float(radius)
        return cls(v, f, name="sphere", **kw)

    @classmethod
    def cylinder(cls, radius: float, half_length: float, axis: str = "z",
                 segments: int = 32, **kw) -> TriMesh:
        k = {"x": 0, "y": 1, "z": 2}[axis]
        th = np.linspace(0.0, 2.0 * np.pi, int(segments), endpoint=False)
        ring = np.zeros((segments, 3))
        u, w = (k + 1) % 3, (k + 2) % 3
        ring[:, u], ring[:, w] = radius * np.cos(th), radius * np.sin(th)
        top, bot = ring.copy(), ring.copy()
        top[:, k], bot[:, k] = half_length, -half_length
        cen_t, cen_b = np.zeros(3), np.zeros(3)
        cen_t[k], cen_b[k] = half_length, -half_length
        pts = np.concatenate([bot, top, cen_b[None], cen_t[None]])
        n = segments
        f = []
        for i in range(n):
            j = (i + 1) % n
            f += [[i, j, n + j], [i, n + j, n + i]]
            f += [[2 * n, j, i]]
            f += [[2 * n + 1, n + i, n + j]]
        return cls(pts, np.asarray(f, dtype=np.int64), name="cylinder", **kw)

    @classmethod
    def capsule(cls, radius: float, half_length: float, axis: str = "z",
                segments: int = 32, rings: int = 8, **kw) -> TriMesh:
        k = {"x": 0, "y": 1, "z": 2}[axis]
        u, w = (k + 1) % 3, (k + 2) % 3
        n, rings = int(segments), int(rings)
        th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        phi = np.concatenate([np.linspace(-np.pi / 2, 0.0, rings + 1)[1:],
                              np.linspace(0.0, np.pi / 2, rings + 1)[:-1]])
        offs = np.concatenate([np.full(rings, -half_length), np.full(rings, half_length)])
        pts = []
        for ph, off in zip(phi, offs, strict=True):
            ring = np.zeros((n, 3))
            ring[:, u] = radius * np.cos(ph) * np.cos(th)
            ring[:, w] = radius * np.cos(ph) * np.sin(th)
            ring[:, k] = radius * np.sin(ph) + off
            pts.append(ring)
        pts = np.concatenate(pts)
        nb = len(phi)
        f = []
        for r in range(nb - 1):
            for i in range(n):
                j = (i + 1) % n
                a, b = r * n + i, r * n + j
                c, d = (r + 1) * n + i, (r + 1) * n + j
                f += [[a, b, d], [a, d, c]]
        cap_lo, cap_hi = np.zeros(3), np.zeros(3)
        cap_lo[k], cap_hi[k] = -half_length - radius, half_length + radius
        lo_i, hi_i = len(pts), len(pts) + 1
        pts = np.concatenate([pts, cap_lo[None], cap_hi[None]])
        for i in range(n):
            j = (i + 1) % n
            f += [[lo_i, j, i]]
            f += [[hi_i, (nb - 1) * n + i, (nb - 1) * n + j]]
        return cls(pts, np.asarray(f, dtype=np.int64), name="capsule", **kw)

    def convex_hull(self) -> TriMesh:
        try:
            from scipy.spatial import ConvexHull
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TriMesh.convex_hull needs scipy (`pip install scipy`).  Without it, "
                "repair the mesh upstream or set the shape's `volume` explicitly."
            ) from exc
        hull = ConvexHull(self.points)
        faces = np.asarray(hull.simplices, dtype=np.int64).copy()
        a, b, c = (self.points[faces[:, k]] for k in range(3))
        inward = (np.cross(b - a, c - a) * hull.equations[:, :3]).sum(axis=1) < 0
        faces[inward] = faces[inward][:, [0, 2, 1]]
        return TriMesh(self.points, faces,
                       name=f"{self.name}:hull" if self.name else "hull")

    def scaled_to_volume(self, volume: float) -> TriMesh:
        k = (float(volume) / self.volume) ** (1.0 / 3.0)
        return TriMesh(self.points * k, self.faces, check_watertight=False,
                       name=self.name)

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def is_watertight(self) -> bool:
        return self.open_edges == 0

    def to_dict(self) -> dict:
        return {"points": self.points.tolist(), "faces": self.faces.tolist(),
                "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> TriMesh:
        return cls(d["points"], d["faces"], check_watertight=False,
                   name=d.get("name", ""))

    def __repr__(self) -> str:
        return (f"TriMesh({self.name or 'mesh'}: {len(self.points)} verts, "
                f"{self.num_faces} tris, {self.volume * 1e3:.3f} L"
                + ("" if self.is_watertight else f", {self.open_edges} open edges") + ")")


class MeshDisplacement:
    def __init__(self, meshes, *, device="cpu", dtype=torch.float32,
                 max_clip_triangles: int = 4_000_000,
                 warn_faces: int = 200_000):
        self.device = torch.device(device)
        self.dtype = dtype
        self.n_slots = len(meshes)

        pts = []
        counts = np.zeros(self.n_slots, dtype=np.int64)
        volume = np.zeros(self.n_slots)
        centroid = np.zeros((self.n_slots, 3))
        radius = np.zeros(self.n_slots)
        has = np.zeros(self.n_slots, dtype=bool)
        for s, m in enumerate(meshes):
            if m is None:
                continue
            tri = m.points[m.faces]
            pts.append(tri)
            counts[s] = len(tri)
            volume[s] = m.volume
            centroid[s] = m.centroid
            radius[s] = m.radius
            has[s] = True

        self.has_mesh_np = has
        self.any_mesh = bool(has.any())
        t = lambda a, d=dtype: torch.as_tensor(np.asarray(a), dtype=d, device=self.device)  # noqa: E731

        self.tri = (t(np.concatenate(pts)) if pts
                    else torch.zeros((0, 3, 3), dtype=dtype, device=self.device))
        self.face_count = t(counts, torch.long)
        self.face_start = t(np.concatenate([[0], np.cumsum(counts)[:-1]]), torch.long)
        self.volume = t(volume)
        self.centroid = t(centroid)
        self.radius = t(radius)
        self.has_mesh = t(has, torch.bool)
        self.num_faces = int(counts.sum())
        self.max_clip_triangles = max(1, int(max_clip_triangles))

        if self.num_faces > warn_faces:
            warnings.warn(
                f"mesh buoyancy is carrying {self.num_faces} triangles.  Only the "
                "shapes straddling the free surface are clipped each step, so this "
                "is usually fine, but if many links are at the surface at once "
                "consider TriMesh.convex_hull() or a decimated proxy mesh.",
                stacklevel=2)

    def submerged(self, depth: torch.Tensor, up_shape: torch.Tensor):
        if depth.dim() != 2 or depth.shape[1] != self.n_slots:
            raise ValueError(f"depth must be (E, {self.n_slots}), got {tuple(depth.shape)}")
        if up_shape.shape != depth.shape + (3,):
            raise ValueError(f"up_shape must be {depth.shape + (3,)}, "
                             f"got {tuple(up_shape.shape)}")
        if depth.device != self.device:
            raise ValueError(
                f"the packed meshes are on {self.device} but the state is on "
                f"{depth.device}; build MeshDisplacement with device={depth.device}")
        E, S = depth.shape
        frac = torch.zeros((E, S), dtype=self.dtype, device=self.device)
        cob = torch.zeros((E, S, 3), dtype=self.dtype, device=self.device)
        if not self.any_mesh:
            return frac, cob

        n = up_shape / torch.linalg.vector_norm(up_shape, dim=-1, keepdim=True).clamp_min(1e-30)

        R = self.radius.view(1, S)
        wet = (depth >= R) & self.has_mesh.view(1, S)
        cut = (depth > -R) & (depth < R) & self.has_mesh.view(1, S)

        frac = torch.where(wet, torch.ones_like(frac), frac)
        cob = torch.where(wet.unsqueeze(-1), self.centroid.view(1, S, 3).expand(E, S, 3), cob)

        pair = cut.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if pair.numel() == 0:
            return frac, cob

        slot = pair % S
        n_p = n.reshape(-1, 3)[pair]
        d_p = depth.reshape(-1)[pair]

        P = pair.numel()
        cnt = self.face_count[slot]
        vol_p = torch.zeros(P, dtype=self.dtype, device=self.device)
        mom_p = torch.zeros((P, 3), dtype=self.dtype, device=self.device)

        csum = torch.cumsum(cnt, 0)
        n_tri = int(csum[-1])
        lo = 0
        while lo < P:
            if lo == 0 and n_tri <= self.max_clip_triangles:
                hi, total = P, n_tri
            else:
                base = 0 if lo == 0 else int(csum[lo - 1])
                hi = int(torch.searchsorted(csum, base + self.max_clip_triangles,
                                            right=True))
                hi = min(max(hi, lo + 1), P)
                total = int(csum[hi - 1]) - (0 if lo == 0 else int(csum[lo - 1]))
            c = cnt[lo:hi]

            flat = torch.repeat_interleave(
                torch.arange(hi - lo, device=self.device), c)
            offs = torch.cumsum(c, 0) - c
            local = torch.arange(total, device=self.device) - offs[flat]
            tri = self.tri[self.face_start[slot[lo:hi]][flat] + local]

            vol_t, mom_t = _clip_integrate(tri, n_p[lo:hi][flat], d_p[lo:hi][flat])
            vol_p[lo:hi].index_add_(0, flat, vol_t)
            mom_p[lo:hi].index_add_(0, flat, mom_t)
            lo = hi

        total_v = self.volume[slot]
        vol_p = vol_p.clamp(min=0.0)
        frac_p = (vol_p / total_v.clamp_min(1e-30)).clamp(0.0, 1.0)
        cob_p = torch.where(vol_p.unsqueeze(-1) > 1e-12 * total_v.unsqueeze(-1),
                            mom_p / vol_p.unsqueeze(-1).clamp_min(1e-30),
                            torch.zeros_like(mom_p))

        frac.view(-1)[pair] = frac_p.to(frac.dtype)
        cob.view(-1, 3)[pair] = cob_p.to(cob.dtype)
        return frac, cob

    def __repr__(self) -> str:
        n = int(self.has_mesh_np.sum())
        return (f"MeshDisplacement({n}/{self.n_slots} slots meshed, "
                f"{self.num_faces} triangles, "
                f"{float(self.volume.sum()) * 1e3:.3f} L)")


def _fan_integrate(poly: torch.Tensor, O: torch.Tensor):
    q = poly - O.unsqueeze(1)
    a = q[:, 0:1]
    b, c = q[:, 1:-1], q[:, 2:]
    det = (a * torch.cross(b, c, dim=-1)).sum(-1)
    vol = det.sum(1) / 6.0
    return vol, O * vol.unsqueeze(-1) + (det.unsqueeze(-1) * (a + b + c)).sum(1) / 24.0


def _clip_integrate(tri: torch.Tensor, n: torch.Tensor, d: torch.Tensor):
    O = d.unsqueeze(-1) * n
    s = (tri * n.unsqueeze(1)).sum(-1) - d.unsqueeze(-1)
    inside = s <= 0.0
    n_in = inside.sum(-1)

    vol = torch.zeros(tri.shape[0], dtype=tri.dtype, device=tri.device)
    mom = torch.zeros((tri.shape[0], 3), dtype=tri.dtype, device=tri.device)

    wet = (n_in == 3).nonzero(as_tuple=False).squeeze(-1)
    if wet.numel():
        vol[wet], mom[wet] = _fan_integrate(tri[wet], O[wet])

    cut = ((n_in > 0) & (n_in < 3)).nonzero(as_tuple=False).squeeze(-1)
    if cut.numel():
        vol[cut], mom[cut] = _fan_integrate(
            _clip_to_halfspace(tri[cut], inside[cut], s[cut]), O[cut])
    return vol, mom


def _clip_to_halfspace(tri: torch.Tensor, inside: torch.Tensor, s: torch.Tensor):
    i, j = [0, 1, 2], [1, 2, 0]
    si, sj = s[:, i], s[:, j]
    den = si - sj
    safe = torch.where(den.abs() > 1e-30, den, torch.ones_like(den))
    tpar = torch.where(den.abs() > 1e-30, si / safe, torch.zeros_like(den)).clamp(0.0, 1.0)
    X = tri[:, i] + tpar.unsqueeze(-1) * (tri[:, j] - tri[:, i])
    crossing = inside[:, i] ^ inside[:, j]

    prev = tri[:, 0]
    for _ in range(2):
        slots = []
        for e in range(3):
            prev = torch.where(inside[:, e:e + 1], tri[:, e], prev)
            slots.append(prev)
            prev = torch.where(crossing[:, e:e + 1], X[:, e], prev)
            slots.append(prev)
    return torch.stack(slots, dim=1)
