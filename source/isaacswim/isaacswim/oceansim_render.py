from __future__ import annotations

import os

import numpy as np

__all__ = [
    "OceanSimWater", "OCEANSIM_DEFAULT", "VEILS", "JERLOV_ATTEN",
    "JERLOV_BACKSCATTER", "CAUSTICS_DEFAULT", "apply_underwater", "render_frame", "kernels",
]


OCEANSIM_DEFAULT = dict(
    veil=(0.0, 0.31, 0.24),
    backscatter=(0.05, 0.05, 0.2),
    attenuation=(0.05, 0.05, 0.05),
)

VEILS = {
    "deep_sea": (0.0, 0.0, 0.28),
    "shallow_water": (0.05, 0.11, 0.7),
    "akdeniz": (0.14, 0.3, 0.5),
    "river": (0.294, 0.4, 0.263),
    "mud": (0.259, 0.259, 0.024),
    "mhl": (0.0, 0.3021, 0.239),
    "murky": (0.275, 0.212, 0.071),
    "seaclear": (0.08, 0.42, 0.52),
    "oceansim_default": OCEANSIM_DEFAULT["veil"],
}

JERLOV_BACKSCATTER = {
    "I": (0.905, 0.961, 0.982), "IA": (0.804, 0.954, 0.975),
    "IB": (0.830, 0.940, 0.968), "II": (0.800, 0.925, 0.940),
    "III": (0.750, 0.885, 0.890), "1": (0.750, 0.885, 0.875),
    "3": (0.710, 0.820, 0.800), "5": (0.670, 0.730, 0.670),
    "7": (0.620, 0.610, 0.590), "9": (0.550, 0.460, 0.290),
}
JERLOV_ATTEN = {k: tuple(1.5 * c for c in v) for k, v in JERLOV_BACKSCATTER.items()}

CAUSTICS_DEFAULT = dict(blend_weight=0.5, depth_min=0.0, depth_max=100.0,
                        uv_scale=(1.0, 1.0), time_speed=2.0)


class OceanSimWater:
    def __init__(self, veil=OCEANSIM_DEFAULT["veil"],
                 attenuation=OCEANSIM_DEFAULT["attenuation"],
                 backscatter=OCEANSIM_DEFAULT["backscatter"],
                 scale: float = 1.0):
        if isinstance(veil, str):
            if veil not in VEILS:
                raise KeyError(f"unknown veil {veil!r}; have {sorted(VEILS)}")
            veil = VEILS[veil]
        self.veil = tuple(float(c) for c in veil)
        self.attenuation = tuple(float(c) for c in attenuation)
        self.backscatter = tuple(float(c) for c in backscatter)
        self.scale = float(scale)

    @classmethod
    def jerlov(cls, water_type: str = "II", veil: str = "seaclear",
               scale: float = 0.5) -> OceanSimWater:
        wt = water_type.replace("Type ", "").strip()
        if wt not in JERLOV_BACKSCATTER:
            raise KeyError(f"unknown Jerlov type {water_type!r}; "
                           f"have {sorted(JERLOV_BACKSCATTER)}")
        return cls(veil=veil, attenuation=JERLOV_BACKSCATTER[wt],
                   backscatter=JERLOV_ATTEN[wt], scale=scale)

    def visibility_range(self) -> float:
        b = np.max(np.asarray(self.attenuation) * self.scale)
        return float(np.log(50.0) / max(b, 1e-9))

    def to_uw_param(self) -> np.ndarray:
        return np.concatenate([self.veil, np.asarray(self.attenuation) * self.scale,
                               np.asarray(self.backscatter) * self.scale])

    @classmethod
    def from_uw_param(cls, uw_param) -> OceanSimWater:
        p = np.asarray(uw_param, dtype=np.float64).reshape(-1)
        if p.size != 9:
            raise ValueError(f"UW_param must have 9 entries [veil(3), atten(3), "
                             f"backscatter(3)], got {p.size}")
        return cls(veil=p[0:3], attenuation=p[3:6], backscatter=p[6:9])

    def __repr__(self):
        return (f"OceanSimWater(veil={self.veil}, atten={self.attenuation}, "
                f"backscatter={self.backscatter}, scale={self.scale}, "
                f"visibility={self.visibility_range():.1f} m)")


def apply_underwater(rgb, depth, water: OceanSimWater, *, max_range: float | None = 1000.0):
    import torch

    u8 = rgb.dtype == torch.uint8
    x = rgb.float() / 255.0 if u8 else rgb
    z = torch.nan_to_num(depth.to(x.dtype), nan=0.0, posinf=max_range or 1e4)
    if z.dim() == x.dim() - 1:
        z = z.unsqueeze(-1)
    if max_range is not None:
        z = z.clamp(0.0, max_range)
    s = water.scale
    vec = lambda v: torch.tensor(v, dtype=x.dtype, device=x.device)  # noqa: E731
    out = (x * torch.exp(-z * vec(water.attenuation) * s)
           + vec(water.veil) * (1.0 - torch.exp(-z * vec(water.backscatter) * s)))
    return (out.clamp(0.0, 1.0) * 255.0).to(torch.uint8) if u8 else out


_KERNELS = None


def kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    U = None
    try:
        from isaacsim.oceansim.utils import UWrenderer_utils as U  # noqa: F811
    except Exception:
        pass
    if U is None:
        try:
            from isaacswim.oceansim import enable
            if enable():
                from isaacsim.oceansim.utils import UWrenderer_utils as U  # noqa: F811
        except Exception:
            U = None
    if U is None:
        U = _load_renderer_utils_directly()
    _KERNELS = {"UW_render_2": U.UW_render_2,
                "water_caustics": U.water_caustics,
                "blend_caustics_PyTX": U.blend_caustics_PyTX}
    return _KERNELS


def _oceansim_roots():
    import glob
    roots = []
    env = os.environ.get("OCEANSIM_ROOT")
    if env:
        roots.append(env)
    try:
        import isaacsim
        for base in getattr(isaacsim, "__path__", []):
            roots.append(os.path.join(base, "extsUser", "OceanSim"))
            roots.extend(glob.glob(os.path.join(base, "extsUser", "*OceanSim*")))
            roots.extend(glob.glob(os.path.join(base, "exts", "*oceansim*")))
    except Exception:
        pass
    return [r for r in roots if os.path.isdir(r)]


def _load_renderer_utils_directly():
    import importlib.util
    tried = []
    for root in _oceansim_roots():
        path = os.path.join(root, "isaacsim", "oceansim", "utils",
                            "UWrenderer_utils.py")
        tried.append(path)
        if not os.path.isfile(path):
            continue
        spec = importlib.util.spec_from_file_location(
            "isaacswim._oceansim_UWrenderer_utils", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    from isaacswim.oceansim import OCEANSIM_INSTALL_HINT
    raise ImportError(
        f"{OCEANSIM_INSTALL_HINT}\n\nDirect load also failed; looked in: "
        f"{tried or '(no candidate roots found)'}.  Set OCEANSIM_ROOT to the "
        "clone if it lives somewhere else.")


def _rgba_u8(rgb):
    import torch
    if rgb.dtype != torch.uint8:
        rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8)
    if rgb.shape[-1] == 4:
        return rgb.contiguous()
    alpha = torch.full(rgb.shape[:-1] + (1,), 255, dtype=torch.uint8,
                       device=rgb.device)
    return torch.cat([rgb[..., :3], alpha], dim=-1).contiguous()


def render_frame(rgb, depth, normals, water: OceanSimWater, *,
                 frame: int = 0, caustics: bool = True,
                 caustics_params: dict | None = None,
                 sun_dir=(0.0, 0.0, 1.0)):
    import torch
    import warp as wp

    K = kernels()
    cp = dict(CAUSTICS_DEFAULT)
    cp.update(caustics_params or {})

    src = _rgba_u8(rgb)
    H, W = src.shape[0], src.shape[1]
    depth = torch.nan_to_num(depth.float(), nan=cp["depth_max"],
                             posinf=cp["depth_max"]).contiguous()

    wp_src = wp.from_torch(src, dtype=wp.uint8)
    wp_depth = wp.from_torch(depth, dtype=wp.float32)
    out = wp.empty(shape=(H, W, 4), dtype=wp.uint8, device=wp_src.device)

    stage_in = wp_src
    if caustics:
        if normals is None:
            raise ValueError(
                "caustics need the `normals` AOV; add \"normals\" to the camera's "
                "data_types, or pass caustics=False")
        nml = normals.float().contiguous()
        tex = wp.empty(shape=(H, W, 4), dtype=wp.uint8, device=wp_src.device)
        wp.launch(kernel=K["water_caustics"], dim=(W, H),
                  inputs=[tex, W, H, float(frame), float(cp["time_speed"])])
        blended = wp.empty(shape=(H, W, 4), dtype=wp.uint8, device=wp_src.device)
        wp.launch(kernel=K["blend_caustics_PyTX"], dim=(W, H),
                  inputs=[wp_src, wp_depth, wp.from_torch(nml, dtype=wp.float32),
                          tex, wp.vec3f(*sun_dir), float(cp["blend_weight"]),
                          float(cp["depth_min"]), float(cp["depth_max"])],
                  outputs=[blended])
        stage_in = blended

    wp.launch(kernel=K["UW_render_2"], dim=(H, W),
              inputs=[stage_in, wp_depth, float(water.scale),
                      wp.vec3f(*water.veil), wp.vec3f(*water.attenuation),
                      wp.vec3f(*water.backscatter)],
              outputs=[out])
    return wp.to_torch(out)
