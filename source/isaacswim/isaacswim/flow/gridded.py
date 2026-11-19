from __future__ import annotations

import math

import numpy as np
import torch

__all__ = ["Gridded"]

_EARTH_RADIUS = 6371008.8


class Gridded:
    def __init__(self, velocity, x, y, z, *, times=None, t0: float = 0.0,
                 outside: str = "zero", loop: bool = False, device: str = "cpu"):
        v = np.asarray(velocity, dtype=np.float32)
        if v.ndim == 4:
            v = v[None]
        if v.ndim != 5 or v.shape[-1] != 3:
            raise ValueError(f"velocity must be (Nx, Ny, Nz, 3) or (Nt, Nx, Ny, Nz, 3), "
                             f"not {v.shape}")
        axes = [np.array(a, dtype=np.float64).reshape(-1) for a in (x, y, z)]
        for name, a, k in zip("xyz", axes, v.shape[1:4]):
            if not len(a) or not np.isfinite(a).all():
                raise ValueError(f"{name} coordinates must be nonempty and finite")
            if len(a) != k:
                raise ValueError(f"{name} has {len(a)} samples but velocity has {k}")
            if len(a) > 1 and not np.all(np.diff(a) > 0):
                raise ValueError(f"{name} coordinates must be strictly increasing")
        if times is None:
            if v.shape[0] != 1:
                raise ValueError("a time series needs `times`")
            times = np.zeros(1)
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if not len(times) or not np.isfinite(times).all():
            raise ValueError("times must be nonempty and finite")
        if not math.isfinite(t0):
            raise ValueError("t0 must be finite")
        if len(times) != v.shape[0]:
            raise ValueError(f"{len(times)} times for {v.shape[0]} frames")
        if len(times) > 1 and not np.all(np.diff(times) > 0):
            raise ValueError("times must be strictly increasing")
        if outside not in ("zero", "clamp"):
            raise ValueError(f"outside must be 'zero' or 'clamp', not {outside!r}")

        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        self._grid = torch.as_tensor(np.ascontiguousarray(v.transpose(0, 4, 3, 2, 1)),
                                     device=device)
        self._axes = [torch.as_tensor(a, device=device) for a in axes]
        self.axes = axes
        self.times = times
        self.time = float(t0)
        self.outside = outside
        self.loop = bool(loop)

    def step(self, dt: float) -> None:
        self.time += float(dt)

    def _frames(self) -> tuple[int, int, float]:
        ts = self.times
        if len(ts) == 1:
            return 0, 0, 0.0
        t = self.time
        if self.loop:
            period = ts[-1] - ts[0] + (ts[-1] - ts[-2])
            t = ts[0] + (t - ts[0]) % period
            if t > ts[-1]:
                return len(ts) - 1, 0, (t - ts[-1]) / (period - (ts[-1] - ts[0]))
        t = min(max(t, ts[0]), ts[-1])
        i = int(np.searchsorted(ts, t, side="right")) - 1
        i = min(i, len(ts) - 2)
        return i, i + 1, (t - ts[i]) / (ts[i + 1] - ts[i])

    @property
    def stats(self) -> dict:
        speed = self._grid.norm(dim=1)
        return {"grid": [len(a) for a in self.axes], "frames": len(self.times),
                "time": self.time, "max_speed": float(speed.max())}

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        shape = pos.shape
        p = pos.reshape(-1, 3).to(torch.float64)
        dev = pos.device
        if self._grid.device != dev:
            self._grid = self._grid.to(dev)
            self._axes = [a.to(dev) for a in self._axes]
        norm, inside = [], torch.ones(p.shape[0], dtype=torch.bool, device=dev)
        for k, a in enumerate(self._axes):
            if len(a) == 1:
                norm.append(torch.zeros_like(p[:, k]))
                continue
            inside &= (p[:, k] >= a[0]) & (p[:, k] <= a[-1])
            idx = _index(p[:, k], a)
            norm.append(2.0 * idx / (len(a) - 1) - 1.0)
        g = torch.stack(norm, -1).to(self._grid.dtype).view(1, -1, 1, 1, 3)
        i0, i1, w = self._frames()

        def sample(i):
            out = torch.nn.functional.grid_sample(
                self._grid[i:i + 1], g, mode="bilinear", padding_mode="border",
                align_corners=True)
            return out.view(3, -1).T

        v = sample(i0)
        if w > 0.0:
            v = (1.0 - w) * v + w * sample(i1)
        if self.outside == "zero":
            v = torch.where(inside.unsqueeze(-1), v, torch.zeros_like(v))
        return v.to(pos.dtype).reshape(shape)

    @classmethod
    def from_dataset(cls, ds, *, origin_lonlat=None, u: str = "uo", v: str = "vo",
                     w: str | None = None, surface_z: float = 0.0,
                     time_origin=None, **kw) -> Gridded:
        def coord(*names):
            for n in names:
                if n in ds.coords or n in ds.variables:
                    return n
            return None

        lon_n, lat_n = coord("longitude", "lon"), coord("latitude", "lat")
        dep_n, t_n = coord("depth", "deptht", "z"), coord("time")
        if lon_n is None or lat_n is None:
            raise ValueError("dataset has no longitude/latitude coordinates")
        lon = np.asarray(ds[lon_n].values, dtype=np.float64)
        lat = np.asarray(ds[lat_n].values, dtype=np.float64)
        lon0, lat0 = origin_lonlat if origin_lonlat is not None else (
            0.5 * (lon[0] + lon[-1]), 0.5 * (lat[0] + lat[-1]))
        x = np.radians(lon - lon0) * _EARTH_RADIUS * math.cos(math.radians(lat0))
        y = np.radians(lat - lat0) * _EARTH_RADIUS
        if dep_n is not None:
            depth = np.asarray(ds[dep_n].values, dtype=np.float64)
            z = surface_z - depth
        else:
            z = np.array([surface_z])

        order = [d for d in (t_n, dep_n, lat_n, lon_n) if d is not None]

        def comp(name):
            if name is None:
                return None
            return np.asarray(ds[name].transpose(*order).values, dtype=np.float32)

        cu, cv, cw = comp(u), comp(v), comp(w)
        if cw is None:
            cw = np.zeros_like(cu)
        vel = np.stack([cu, cv, cw], -1)
        if t_n is None:
            vel = vel[None]
        if dep_n is None:
            vel = vel[:, None]
        vel = vel.transpose(0, 3, 2, 1, 4)
        coords = [x, y, z]
        for k in range(3):
            if len(coords[k]) > 1 and coords[k][0] > coords[k][-1]:
                coords[k] = coords[k][::-1]
                vel = np.flip(vel, axis=k + 1)
        x, y, z = (np.ascontiguousarray(c) for c in coords)
        if dep_n is not None and z[-1] < surface_z:
            z = np.append(z, surface_z)
            vel = np.concatenate([vel, vel[:, :, :, -1:]], axis=3)
        vel = np.ascontiguousarray(vel)

        times = None
        if t_n is not None:
            t = ds[t_n].values
            if np.issubdtype(np.asarray(t).dtype, np.datetime64):
                ref = np.datetime64(time_origin) if time_origin is not None else t[0]
                times = (t - ref) / np.timedelta64(1, "s")
            else:
                times = np.asarray(t, dtype=np.float64)
                times = times - (times[0] if time_origin is None else time_origin)
            times = np.asarray(times, dtype=np.float64)
        return cls(vel, x, y, z, times=times, **kw)

    @classmethod
    def from_netcdf(cls, path, **kw) -> Gridded:
        try:
            import xarray as xr
        except ImportError as exc:                       # pragma: no cover
            raise ImportError("reading NetCDF needs xarray: "
                              "pip install isaacswim[netcdf]") from exc
        with xr.open_dataset(path) as ds:
            return cls.from_dataset(ds.load(), **kw)

    @classmethod
    def from_scattered(cls, points, velocity, *, spacing: float, bounds=None,
                       **kw) -> Gridded:
        try:
            from scipy.interpolate import LinearNDInterpolator
        except ImportError as e:
            raise ImportError("Gridded.from_scattered needs scipy: "
                              "pip install 'isaacswim[cfd]'") from e

        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be finite and positive")

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        vel = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        lo, hi = (pts.min(0), pts.max(0)) if bounds is None else map(np.asarray, bounds)
        axes = [np.arange(lo[k], hi[k] + 0.5 * spacing, spacing) for k in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        f = LinearNDInterpolator(pts, vel, fill_value=0.0)
        grid = f(np.stack([X, Y, Z], -1).reshape(-1, 3)).reshape(*X.shape, 3)
        return cls(grid, *axes, **kw)

    def __repr__(self) -> str:
        n = "x".join(str(len(a)) for a in self.axes)
        span = ", ".join(f"{a[0]:.3g}..{a[-1]:.3g}" for a in self.axes)
        return (f"Gridded({n} grid over [{span}] m, {len(self.times)} frame(s), "
                f"outside={self.outside})")


def _index(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    x = x.clamp(a[0], a[-1])
    i = torch.searchsorted(a, x.contiguous()).clamp(1, len(a) - 1)
    a0, a1 = a[i - 1], a[i]
    return (i - 1).to(x.dtype) + (x - a0) / (a1 - a0)
