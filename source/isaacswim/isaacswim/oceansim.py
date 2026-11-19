"""Adapters for OceanSim's camera, imaging sonar, DVL, barometer and waves."""

from __future__ import annotations

import contextlib
import inspect
import os

import numpy as np

from isaacswim.oceansim_render import OceanSimWater

__all__ = [
    "make_uw_camera",
    "make_water_surface",
    "Waves",
    "WAVE_DEFAULTS",
    "ros_publishers",
    "enable",
    "available",
    "OCEANSIM_INSTALL_HINT",
    "Sonar",
    "sonar_fan",
    "DVL",
    "Barometer",
    "Camera",
    "set_sonar_reflectivity",
]

OCEANSIM_INSTALL_HINT = (
    "OceanSim is not importable.  It ships as an Isaac Sim *extension*, and cloning "
    "it is not enough -- the extension has to be enabled before its Python exists:\n"
    "  1. git clone https://github.com/umfieldrobotics/OceanSim "
    "<isaacsim>/extsUser/OceanSim\n"
    "  2. call isaacswim.oceansim.enable() after launching the app, or enable "
    "OCEANSIM from Window -> Extensions in the GUI\n"
    "Verified working on Isaac Sim 5.1 and 6.2 with extension id 'OceanSim-1.0.0'."
)


KIT_EXTENSIONS = ("isaacsim.sensors.physx", "isaacsim.sensors.camera",
                  "isaacsim.sensors.physics", "isaacsim.replicator.writers",
                  "isaacsim.storage.native")


def _deprecated_ext_folder() -> str | None:
    import importlib.util
    import os
    spec = importlib.util.find_spec("isaacsim")
    roots = list(spec.submodule_search_locations or []) if spec else []
    for env in ("ISAAC_PATH", "ISAACSIM_PATH"):
        if os.environ.get(env):
            roots.append(os.environ[env])
    for root in roots:
        path = os.path.join(root, "extsDeprecated")
        if os.path.isdir(os.path.join(path, "isaacsim.sensors.camera")):
            return path
    return None


KIT_ARGS = " ".join(f"--enable {e}" for e in KIT_EXTENSIONS)
# The sonar's annotators make this channel warn on every rendered frame.
KIT_ARGS += " --/log/channels/isaacsim.core.simulation_manager.plugin=error"
if _deprecated_ext_folder():
    KIT_ARGS = f"--ext-folder {_deprecated_ext_folder()} {KIT_ARGS}"


def enable(ext_root: str | None = None) -> bool:
    try:
        import omni.kit.app
    except ImportError as exc:
        raise ImportError("enable() needs a launched Isaac Sim app") from exc

    mgr = omni.kit.app.get_app().get_extension_manager()
    roots = [ext_root] if ext_root else []
    if not ext_root:
        try:
            import isaacsim
            for base in getattr(isaacsim, "__path__", []):
                cand = os.path.join(base, "extsUser")
                if os.path.isdir(cand):
                    roots.append(cand)
        except Exception:
            pass
    for r in roots:
        try:
            mgr.add_path(str(r))
        except Exception:
            pass
    for _ in range(3):
        omni.kit.app.get_app().update()
    for dep in KIT_EXTENSIONS:
        try:
            mgr.set_extension_enabled_immediate(dep, True)
        except Exception:
            pass
    for e in mgr.get_extensions():
        name = f"{e.get('id', '')} {e.get('name', '')}".lower()
        if "ocean" not in name:
            continue
        for key in (e.get("id"), e.get("name")):
            if not key:
                continue
            try:
                mgr.set_extension_enabled_immediate(key, True)
                if available():
                    return True
            except Exception:
                continue
    return available()


def _app_running() -> bool:
    import sys
    return "omni.kit.app" in sys.modules


def available() -> bool:
    if not _app_running():
        return False
    try:
        import isaacsim.oceansim  # noqa: F401
        return True
    except Exception:
        return False


def _require():
    return _oceansim_class("isaacsim.oceansim.sensors.UW_Camera", "UW_Camera")


def _ros_camera_info_compat() -> None:
    import sys

    bridge = sys.modules.get("isaacsim.ros2.bridge")
    if bridge is not None and not hasattr(bridge, "read_camera_info"):
        from isaacsim.ros2.core import read_camera_info
        bridge.read_camera_info = read_camera_info


def make_uw_camera(prim_path: str, water: OceanSimWater | None = None, *,
                   name: str = "isaacswim_uw_camera",
                   resolution=(640, 480),
                   depth_noise_sigma: float = 0.01,
                   max_range: float | None = None,
                   viewport: bool = True,
                   initialize: bool = True,
                   ros=None,
                   **camera_kwargs):
    if ros is None:
        UW_Camera = _require()
    else:
        _ros_camera_info_compat()
        UW_Camera = _oceansim_class("isaacsim.oceansim.sensors.UW_Camera_ROS",
                                    "UW_Camera_ROS")
        camera_kwargs.update(og_node=ros._rgb_node, depth_og_node=ros._depth_node,
                             pointcloud_og_node=ros._pointcloud_node)
    water = water if water is not None else OceanSimWater()

    with _warp_device():
        cam = UW_Camera(prim_path=prim_path, name=name, resolution=resolution,
                        **camera_kwargs)
        if initialize:
            cam.initialize(
                UW_param=water.to_uw_param(),
                depth_noise_sigma=depth_noise_sigma,
                max_range=float(max_range if max_range is not None
                                else water.visibility_range()),
                viewport=viewport,
            )
    return cam


def _warp_device(sensor=None):
    import contextlib
    try:
        import warp as wp
    except ImportError:
        return contextlib.nullcontext()
    dev = getattr(sensor, "_device", None) or wp.get_preferred_device()
    return wp.ScopedDevice(str(dev))


def _arrays_to_device(sensor, *, preferred: bool = False) -> None:
    try:
        import warp as wp
    except ImportError:
        return
    dev = wp.get_preferred_device() if preferred else getattr(sensor, "_device", None)
    if dev is None:
        return
    dev = wp.get_device(str(dev))
    for k, v in list(vars(sensor).items()):
        if isinstance(v, wp.array) and v.device != dev:
            setattr(sensor, k, v.to(dev))


def _build_annotators() -> None:
    from isaacswim.isaaclab._lab import lab_major
    if lab_major() < 3:
        return
    from importlib import import_module
    try:
        import_module("omni.kit.app")
    except ImportError:
        return
    from isaacswim.launch import update_app
    update_app()


def _water_material(surface_type):
    import sys
    from pathlib import Path

    module = sys.modules.get(surface_type.__module__)
    location = getattr(module, "__file__", None)
    if location is None:
        return None
    mdl = Path(location).resolve().parents[3] / "demo/water_material/Water.mdl"
    if not mdl.is_file():
        return None

    import omni.kit.commands
    import omni.usd
    from pxr import UsdShade

    stage = omni.usd.get_context().get_stage()
    base = "/World/Looks/isaacswim_water"
    path, suffix = base, 0
    while stage.GetPrimAtPath(path):
        suffix += 1
        path = f"{base}_{suffix}"
    ok, _ = omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url=str(mdl),
                                      mtl_name="Water", mtl_path=path)
    if not ok:
        raise RuntimeError(f"could not create OceanSim's water material from {mdl}")
    return UsdShade.Material.Get(stage, path)


def make_water_surface(prim_path: str = "/World/WaterSurface", *,
                       size=(60.0, 60.0), dim=(200, 200),
                       height: float = 0.0,
                       caustics: bool = True,
                       name: str = "isaacswim_water_surface",
                       **kwargs):
    WaterSurface = _oceansim_class("isaacsim.oceansim.watersurface.WaterSurface",
                                   "WaterSurface")
    if kwargs.get("visual_material") is None:
        material = _water_material(WaterSurface)
        if material is not None:
            kwargs["visual_material"] = material
    with _warp_device():
        surface = WaterSurface(prim_path=prim_path, name=name,
                               position=(0.0, 0.0, float(height)),
                               size=list(size), dim=list(dim),
                               enable_caustics=bool(caustics), **kwargs)
    _arrays_to_device(surface, preferred=True)
    return surface


WAVE_DEFAULTS = dict(amplitude=1.0, windSpeed=10.0, direction=0.0, directionality=0.0,
                     scale=1.0, waterDepth=50.0, clipmapCellSize=1.0)


class Waves:
    def __init__(self, prim_path: str = "/World/WaterSurface", *, height: float = 0.0,
                 size=(60.0, 60.0), dim=(200, 200), caustics: bool = True,
                 render: bool = True, iterations: int = 2, time: float = 0.0, **wave):
        unknown = set(wave) - set(WAVE_DEFAULTS)
        if unknown:
            raise TypeError(f"unknown wave parameters {sorted(unknown)}; "
                            f"OceanSim takes {sorted(WAVE_DEFAULTS)}")
        self.params = {**WAVE_DEFAULTS, **wave}
        self.height = float(height)
        self.iterations = int(iterations)
        self.time = float(time)
        self._kernel = _oceansim_class(
            "isaacsim.oceansim.watersurface.ocean_deform_kernels",
            "ocean_deform_launch_kernel")
        self.surface = (make_water_surface(prim_path, size=size, dim=dim, height=height,
                                           caustics=caustics) if render else None)
        self.redraw()

    def _deform(self, local: np.ndarray) -> np.ndarray:
        with _warp_device():
            out = self._kernel(np.ascontiguousarray(local, dtype=np.float32), self.time,
                               **self.params)
        return np.asarray(out.numpy() if hasattr(out, "numpy") else out, dtype=np.float64)

    def height_at(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        flat = xy.reshape(-1, 2)
        target = np.stack([flat[:, 0], np.zeros(len(flat)), -flat[:, 1]], -1)
        src = target.copy()
        for _ in range(self.iterations):
            moved = self._deform(src) - src
            src[:, [0, 2]] = target[:, [0, 2]] - moved[:, [0, 2]]
        return (self.height + self._deform(src)[:, 1]).reshape(xy.shape[:-1])

    def __call__(self, centre_w):
        import torch
        h = self.height_at(centre_w[..., :2].detach().cpu().numpy())
        return torch.as_tensor(h, dtype=centre_w.dtype, device=centre_w.device)

    def step(self, dt: float) -> None:
        self.time += float(dt)
        self.redraw()

    def redraw(self) -> None:
        if self.surface is not None:
            with _warp_device():
                self.surface.deform(time=self.time, **self.params)


def _oceansim_class(module: str, name: str):
    import importlib
    if not _app_running():
        raise ImportError(
            f"{name}: OceanSim needs a running Isaac Sim app -- launch it (e.g. "
            "isaaclab.app.AppLauncher) before building OceanSim sensors.\n"
            + OCEANSIM_INSTALL_HINT)
    try:
        return getattr(importlib.import_module(module), name)
    except Exception as exc:
        raise ImportError(f"{name}: {OCEANSIM_INSTALL_HINT}") from exc


def _mount_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    from isaacswim.np_math import rpy_to_quat
    return rpy_to_quat(np.radians([roll_deg, pitch_deg, yaw_deg]))


def _torch(x, device):
    import torch
    return torch.as_tensor(np.asarray(x), device=device)


def set_sonar_reflectivity(prim_path: str, reflectivity: float = 1.0) -> None:
    from isaaclab.sim.utils.legacy import get_prim_at_path

    prim = get_prim_at_path(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"no prim at {prim_path!r}")
    try:
        from isaacsim.core.utils.semantics import add_update_semantics
        add_update_semantics(prim=prim, type_label="reflectivity",
                             semantic_label=str(float(reflectivity)))
    except ImportError:
        from isaaclab.sim.utils.semantics import add_labels
        add_labels(prim, [str(float(reflectivity))], instance_name="reflectivity")


def ros_publishers(name: str = "isaacswim", *, camera: bool = False,
                   sonar: bool = False, baro: bool = False, imu: bool = False):
    cls = _oceansim_class("isaacsim.oceansim.sensors.ros2_helpers", "OmniHandler")
    return cls(name=name, use_camera=camera, use_sonar=sonar, use_imu=imu,
               use_dvl=False, use_baro=baro)


class Sonar:
    def __init__(self, link_prim: str, *, name: str = "sonar",
                 translation=(0.2, 0.0, 0.0), roll_deg: float = 0.0,
                 pitch_deg: float = 30.0, yaw_deg: float = 0.0,
                 include_unlabelled: bool = False, image: dict | None = None,
                 normalize: str = "all", device: str = "cpu", ros=None, **sonar_kwargs):
        if ros is None:
            cls = _oceansim_class("isaacsim.oceansim.sensors.ImagingSonarSensor",
                                  "ImagingSonarSensor")
        else:
            cls = _oceansim_class("isaacsim.oceansim.sensors.ImagingSonarSensor_ROS",
                                  "ImagingSonarSensor_ROS")
            sonar_kwargs["og_node"] = ros._sonar_node
        with _warp_device():
            self.sensor = cls(prim_path=f"{link_prim}/{name}", name=name,
                              translation=np.asarray(translation, dtype=np.float64),
                              orientation=_mount_quat(roll_deg, pitch_deg, yaw_deg),
                              **sonar_kwargs)
        self.include_unlabelled = include_unlabelled
        self.normalize = normalize
        self.image = {"central_peak": 0.0, **(image or {})}
        self.device = device
        self._ready = False

    @property
    def grid(self) -> dict:
        s = self.sensor
        return {"min_range": float(s.min_range), "range_res": float(s.range_res),
                "min_azimuth": float(s.min_azi), "azimuth_res": float(s.angular_res)}

    def fan(self, polar, size: int = 256):
        return sonar_fan(polar, size=size, **self.grid)

    def measure(self, state=None, *, generator=None):
        import torch

        if not self._ready:
            with _warp_device(self.sensor):
                wanted = {"normalizing_method": self.normalize, "viewport": False,
                          "include_unlabelled": self.include_unlabelled}
                params = inspect.signature(self.sensor.sonar_initialize).parameters
                if not any(q.kind is q.VAR_KEYWORD for q in params.values()):
                    wanted = {k: v for k, v in wanted.items() if k in params}
                self.sensor.sonar_initialize(**wanted)
            _build_annotators()
            _arrays_to_device(self.sensor)
            self._ready = True
        with _warp_device(self.sensor):
            before = self.sensor.id
            self.sensor.make_sonar_data(**self.image)
        valid = self.sensor.id > before
        data = getattr(self.sensor, "sonar_data", None)
        if data is None:
            img = torch.zeros(1, 1, 1, device=self.device)
        else:
            img = _torch(data.numpy()[..., 2], self.device).unsqueeze(0)
        return img, torch.tensor([valid], device=self.device)


def sonar_fan(polar, *, min_range: float, range_res: float, min_azimuth: float,
              azimuth_res: float, size: int = 256):
    import math

    import torch

    p = torch.as_tensor(np.asarray(polar) if not torch.is_tensor(polar) else polar,
                        dtype=torch.float32)
    single = p.ndim == 2
    if single:
        p = p[None]
    n_r, n_a = p.shape[-2:]
    max_range = min_range + n_r * range_res
    half = min(math.pi / 2 - min_azimuth, min_azimuth + n_a * azimuth_res - math.pi / 2)
    half = max(half, 1e-6)
    width = 2 * max_range * math.sin(min(half, math.pi / 2))
    w = max(1, round(size * width / max_range))
    u = (torch.arange(w, device=p.device) + 0.5) / w * width - width / 2
    v = (size - torch.arange(size, device=p.device) - 0.5) / size * max_range
    y, x = torch.meshgrid(v, u, indexing="ij")
    r = torch.hypot(x, y)
    azi = torch.atan2(y, x)
    ri = torch.floor((r - min_range) / range_res).long()
    ai = torch.floor((azi - min_azimuth) / azimuth_res).long()
    ok = (ri >= 0) & (ri < n_r) & (ai >= 0) & (ai < n_a)
    out = p[:, ri.clamp(0, n_r - 1), ai.clamp(0, n_a - 1)] * ok
    return out[0] if single else out


MOUNT_ROOT = "/World/oceansim"


class _Mount:
    def __init__(self, path: str):
        import omni.usd
        from pxr import UsdGeom

        self.stage = omni.usd.get_context().get_stage()
        xf = UsdGeom.Xform.Define(self.stage, path)
        xf.ClearXformOpOrder()
        self._t = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        self._q = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        self.path = path

    def set_pose(self, pos, quat) -> None:
        from pxr import Gf

        p = [float(v) for v in np.asarray(pos).reshape(-1)[:3]]
        q = [float(v) for v in np.asarray(quat).reshape(-1)[:4]]
        self._t.Set(Gf.Vec3d(*p))
        self._q.Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))

    def strip_physics(self) -> None:
        from pxr import UsdPhysics

        prim = self.stage.GetPrimAtPath(self.path)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)


def _make_mount(path: str) -> _Mount:
    return _Mount(path)


class _Follows:
    mount = None

    def follow(self, pos_w, quat_w) -> None:
        if self.mount is not None:
            self.mount.set_pose(pos_w, quat_w)

    def _follow_state(self, state) -> None:
        if self.mount is not None and state is not None:
            self.follow(state.pos_w[0].detach().cpu().numpy(),
                        state.quat_w[0].detach().cpu().numpy())


class DVL(_Follows):
    def __init__(self, link_prim: str, *, name: str = "DVL",
                 translation=(0.0, 0.0, -0.1), device: str = "cpu", follow: bool = True,
                 **dvl_kwargs):
        cls = _oceansim_class("isaacsim.oceansim.sensors.DVLsensor", "DVLsensor")
        if _scene_queries_off():
            raise RuntimeError(
                "OceanSim's DVL raycasts, and this simulation has PhysX scene queries "
                "off (IsaacLab's default), so every beam would miss.  Build it with "
                "SimulationCfg(enable_scene_query_support=True).")
        self.link_prim = link_prim
        parent = link_prim
        if follow:
            self.mount = _make_mount(f"{MOUNT_ROOT}/{name}")
            parent = self.mount.path
        self.sensor = cls(name=name, **dvl_kwargs)
        self.sensor.attachDVL(rigid_body_path=parent,
                              translation=np.asarray(translation, dtype=np.float64))
        if self.mount is not None:
            self.mount.strip_physics()
        self.threshold = int(dvl_kwargs.get("num_beams_out_range_threshold", 2))
        self.device = device
        self.beam_ranges = np.full(4, np.nan)

    def measure(self, state=None, *, generator=None):
        import torch

        self._follow_state(state)
        hits = [bool(h) for h in self.sensor.get_beam_hit()]
        self.beam_ranges = np.asarray(self.sensor.get_depth(), dtype=np.float64)
        valid = hits.count(False) < self.threshold
        if not valid:
            vel = np.zeros((1, 3))
        elif state is None:
            vel = np.asarray(self.sensor.get_linear_vel(), dtype=np.float64).reshape(1, 3)
        else:
            vel = state.lin_vel_b[:1].detach().cpu().numpy().astype(np.float64)
            mvn = getattr(self.sensor, "_mvn_vel", None)
            if mvn is not None and mvn.is_uncertain():
                vel = vel + (np.asarray(self.sensor._transform)
                             @ np.asarray(mvn.sample_array(), dtype=np.float64))[None]
        return (_torch(vel, self.device).float(),
                torch.tensor([valid], device=self.device))


def _scene_queries_off() -> bool:
    try:
        import omni.usd
        from pxr import Usd
    except ImportError:
        return False
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return False
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        attr = prim.GetAttribute("physxScene:enableSceneQuerySupport")
        if attr and attr.HasAuthoredValue():
            return not bool(attr.Get())
    return False


@contextlib.contextmanager
def _physics_rate_kept():
    try:
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics
    except ImportError:
        yield
        return
    stage = omni.usd.get_context().get_stage()
    kept = {}
    for prim in (stage.Traverse() if stage is not None else ()):
        if prim.IsA(UsdPhysics.Scene):
            attr = PhysxSchema.PhysxSceneAPI(prim).GetTimeStepsPerSecondAttr()
            kept[prim.GetPath()] = (attr, attr.Get())
    try:
        yield
    finally:
        for attr, hz in kept.values():
            if hz is not None and attr.Get() != hz:
                attr.Set(hz)


class Barometer(_Follows):
    def __init__(self, link_prim: str, medium, *, name: str = "baro",
                 translation=(0.0, 0.0, 0.0), noise_cov: float = 0.0,
                 as_depth: bool = False, atmosphere_pressure: float = 101325.0,
                 g: float = 9.81, device: str = "cpu", follow: bool = True,
                 ros=None):
        if medium.surface_height is None:
            raise ValueError("Barometer needs Medium.surface_height; unbounded water "
                             "has no surface to measure depth from")
        extra = {}
        if ros is None:
            cls = _oceansim_class("isaacsim.oceansim.sensors.BarometerSensor",
                                  "BarometerSensor")
        else:
            cls = _oceansim_class("isaacsim.oceansim.sensors.BarometerSensor_ROS",
                                  "BarometerSensor_ROS")
            extra = dict(og_node=ros._baro_node, frame_id=name)
        self.link_prim = link_prim
        parent = link_prim
        if follow:
            self.mount = _make_mount(f"{MOUNT_ROOT}/{name}")
            parent = self.mount.path
        with _physics_rate_kept():
            self.sensor = cls(prim_path=f"{parent}/{name}", name=name,
                              translation=np.asarray(translation, dtype=np.float64),
                              water_density=float(medium.density), g=g,
                              noise_cov=noise_cov,
                              water_surface_z=float(medium.surface_height),
                              atmosphere_pressure=atmosphere_pressure, **extra)
        self.density, self.g, self.p_atm = float(medium.density), g, atmosphere_pressure
        self.as_depth = as_depth
        self.device = device

    def measure(self, state=None, *, generator=None):
        self._follow_state(state)
        read = getattr(self.sensor, "read", self.sensor.get_pressure)
        p = float(read())
        if self.as_depth:
            p = (p - self.p_atm) / (self.density * self.g)
        return _torch([p], self.device).float()


class Camera:
    def __init__(self, link_prim: str, water: OceanSimWater | None = None, *,
                 name: str = "uw_camera", translation=(0.25, 0.0, 0.0),
                 roll_deg: float = 0.0, pitch_deg: float = 15.0,
                 yaw_deg: float = 0.0, resolution=(640, 480),
                 device: str = "cpu", ros=None, **init_kwargs):
        self.water = water if water is not None else OceanSimWater()
        self._args = dict(prim_path=f"{link_prim}/{name}", water=self.water,
                          name=name, resolution=resolution, viewport=False,
                          translation=np.asarray(translation, dtype=np.float64),
                          orientation=_mount_quat(roll_deg, pitch_deg, yaw_deg),
                          **init_kwargs)
        self.camera = make_uw_camera(initialize=False, **{
            k: v for k, v in self._args.items()
            if k not in ("water", "viewport")}, ros=ros)
        self.device = device
        self.depth = None
        self._ready = False

    def measure(self, state=None, *, generator=None):
        import torch

        if not self._ready:
            with _warp_device(self.camera):
                self.camera.initialize(
                    UW_param=self.water.to_uw_param(),
                    max_range=float(self.water.visibility_range()), viewport=False)
            _build_annotators()
            _arrays_to_device(self.camera)
            self._ready = True
        with _warp_device(self.camera):
            self.camera.render()
        frame = getattr(self.camera, "_uw_frame", None)
        if frame is None:
            w, h = self._args["resolution"]
            return torch.zeros(1, h, w, 4, dtype=torch.uint8, device=self.device)
        self.depth = getattr(self.camera, "_degraded_depth_frame", None)
        return _torch(frame, self.device).unsqueeze(0)


def __getattr__(name: str):
    passthrough = {
        "UW_Camera": "isaacsim.oceansim.sensors.UW_Camera",
        "ImagingSonarSensor": "isaacsim.oceansim.sensors.ImagingSonarSensor",
        "DVLsensor": "isaacsim.oceansim.sensors.DVLsensor",
        "BarometerSensor": "isaacsim.oceansim.sensors.BarometerSensor",
        "WaterSurface": "isaacsim.oceansim.watersurface.WaterSurface",
    }
    if name not in passthrough:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _oceansim_class(passthrough[name], name)
