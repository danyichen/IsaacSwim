from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation

from isaacswim import OceanSimWater
from isaacswim.actuators import JointServos, ThrusterArray, Thrusters
from isaacswim.assets.bluerov import (
    BLUEROV2_DAMPING,
    DEFAULT_USD,
    AlphaIK,
    StationKeeper,
    build_hydro,
    gripper_shapes,
    robot_cfg,
)
from isaacswim.flow import GustyCurrent, Uniform
from isaacswim.isaaclab import CameraRig, ChaseView, LinkView, ShowcaseVideo, Vehicle, _lab
from isaacswim.np_math import wrap_pi

from ..core import CurrentSpec, SceneSpec, Task, World, yaw_quat
from ..envbase import PROPRIO_KEYS, SURFACE_Z, BenchEnv
from ..nav import DeadReckoning, Instruments, NavConfig, yaw_rotate
from ..platform import (
    ARM_JOINTS,
    GRIP_CLOSED,
    GRIP_JOINT,
    GRIP_OPEN,
    TOOL_MOUNT_JOINT,
    TOOL_OFFSET,
    TOOL_TIP_LINK,
    tool_world,
)
from . import scene as sc

__all__ = ["SimEnv", "VIEWS"]

WATERS = {
    "clear_ocean": OceanSimWater(veil="seaclear"),
    "hazy_ocean": OceanSimWater(veil="seaclear", scale=2.6),
    "harbour": OceanSimWater.jerlov("9", veil="murky"),
}

VIEWS = {
    "front": LinkView("base_link", (0.22, 0.0, 0.03), (1.0, 0.0, 0.0), 12.0),
    "arm": LinkView(
        "alpha_m2_1_2_link",
        (0.130, 0.0, -0.020),
        (math.sin(math.radians(-26.1)), 0.0, math.cos(math.radians(-26.1))),
        14.0,
    ),
    "down": LinkView("base_link", (0.65, 0.0, 0.15), (-0.10, 0.0, -1.0), 10.0),
}


CARROT_TAU = 0.5


class SimEnv(BenchEnv):
    def __init__(
        self,
        task,
        *,
        usd=DEFAULT_USD,
        device: str = "cpu",
        views=("front", "arm"),
        image_size=(128, 128),
        water: str | OceanSimWater | None = "clear_ocean",
        video: bool = False,
        video_height: int = 540,
        video_close: bool = True,
        video_specks: dict | None = None,
        specks_in_views: bool = False,
        arrows_in_views: bool = False,
        antialiasing: str | None = "auto",
        sonar: bool = False,
        sonar_pitch: float = 30.0,
        oceansim: bool = False,
        nav: str = "dvl",
        nav_cfg: NavConfig | None = None,
        physics_dt: float = 1.0 / 120.0,
        layout_seed: int = 0,
        sim=None,
        root="/World",
        origin=(0.0, 0.0, 0.0),
        defer_initialize=False,
        **kw,
    ):
        self.root = root
        self.on_physics_step = None
        self.origin = np.asarray(origin, dtype=np.float64)
        self._shared = sim is not None
        self._defer_settle = defer_initialize
        usd = Path(usd)
        if not usd.is_file():
            raise FileNotFoundError(
                f"{usd} not found -- build the vehicle first: "
                "isaaclab.sh -p scripts/tools/build_bluerov.py"
            )
        urdf = usd.with_suffix(".urdf")
        super().__init__(
            task,
            arm=AlphaIK(
                urdf, tool_offset=TOOL_OFFSET, tip_link=TOOL_TIP_LINK, mount_joint=TOOL_MOUNT_JOINT
            ),
            **kw,
        )
        self.physics_dt = float(physics_dt)
        self.substeps = max(1, round(self.control_dt / self.physics_dt))
        self.control_dt = self.substeps * self.physics_dt
        self.actions.dt = self.control_dt
        self.views = tuple(views)
        unknown = set(self.views) - set(VIEWS) - {"chase"}
        if unknown:
            raise ValueError(f"unknown views {sorted(unknown)}; have {sorted(VIEWS)} and 'chase'")

        layout = task.scene(task.sample(np.random.default_rng(layout_seed)))
        layout.validate()
        self.layout = layout

        self.oceansim = bool(oceansim)
        if nav not in ("dvl", "truth"):
            raise ValueError(f"nav must be 'dvl' or 'truth', not {nav!r}")
        self.nav_mode = nav
        self._sensor_noise = float(kw.get("sensor_noise", 1.0))
        self.nav_cfg = (nav_cfg or NavConfig()).scaled(float(kw.get("sensor_noise", 1.0)))
        self.nav = DeadReckoning(self.nav_cfg)
        self.instruments: Instruments | None = None
        if antialiasing == "auto":
            antialiasing = "FXAA" if hasattr(sim_utils, "RenderCfg") else "DLAA"
        render = {}
        if hasattr(sim_utils, "RenderCfg"):
            render["render"] = sim_utils.RenderCfg(antialiasing_mode=antialiasing)
        self.sim = sim or sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=self.physics_dt,
                device=device,
                gravity=(0.0, 0.0, -9.81),
                **render,
                enable_scene_query_support=self.oceansim,
            )
        )
        if not render and antialiasing is not None:
            try:
                import omni.replicator.core as rep

                rep.settings.set_render_rtx_realtime(antialiasing=antialiasing)
            except Exception as exc:  # pragma: no cover
                print(f"[isaacswim_bench] antialiasing not set: {exc}", flush=True)
        dome = sim_utils.DomeLightCfg(intensity=900.0, color=(0.62, 0.78, 0.92))
        if not self._shared:
            dome.func("/World/DomeLight", dome)
        ground = sim_utils.GroundPlaneCfg(color=layout.seabed_color)
        if not self._shared:
            ground.func("/World/seabed", ground, translation=(0.0, 0.0, layout.seabed_z))
        self.objects = sc.spawn_props(layout, root=self.root, origin=self.origin)
        sc.add_joints(layout, root=self.root, origin=self.origin)
        import omni.usd

        self.scene_task = task
        self.scenery_roots = task.dress(
            omni.usd.get_context().get_stage(),
            seed=layout_seed,
            root=self.root,
            origin=self.origin,
        )
        self.sonar_pitch = float(sonar_pitch)
        cfg = robot_cfg(
            usd,
            pos=np.asarray(layout.spawn_pos) + self.origin,
            quat=yaw_quat(layout.spawn_yaw),
            joint_pos=self._joint_pose(layout),
        )
        cfg.prim_path = f"{self.root}/rov"
        self.robot = Articulation(cfg)

        water = WATERS[water] if isinstance(water, str) else water
        self.rig = None
        if self.views:
            self.rig = CameraRig(
                self.robot,
                {
                    v: ChaseView(**task.video_view) if v == "chase" else replace(VIEWS[v])
                    for v in self.views
                },
                size=tuple(int(x) for x in image_size),
                water=water,
                prim_root=f"{self.root}/cam",
            )
        self.showcase = None
        if video:
            self.showcase = ShowcaseVideo(
                self.robot,
                close=replace(VIEWS["arm"]) if video_close else None,
                chase=ChaseView(**task.video_view),
                height=video_height,
                water=water,
                seabed_z=task.seabed_z,
                prim_root=f"{self.root}/video_cam",
                arrows={
                    "prim_root": f"{self.root}/rov",
                    "marker_path": f"{self.root}/thrust_arrows",
                },
                **({"specks": {**(video_specks or {}), "marker_path": f"{self.root}/current"}}),
            )
        self.specks_in_views = bool(specks_in_views)
        self.arrows_in_views = bool(arrows_in_views)
        self.video_frame: np.ndarray | None = None

        mounted = {}
        if sonar:
            mounted["sonar"] = self._mount_sonar(device)
        if self.oceansim:
            import isaacswim.oceansim as osim

            self._enable_oceansim()
            k = float(kw.get("sensor_noise", 1.0))
            mounted["dvl_os"] = osim.DVL(
                self._base_prim(),
                name=self.root.replace("/", "_") + "_dvl",
                max_range=10.0,
                vel_cov=(0.003 * k) ** 2,
                device=device,
            )

        self._usd = usd
        self._mounted = mounted
        if not defer_initialize:
            self.sim.reset()
            self.initialize()

    def initialize(self):
        usd, task, layout = self._usd, self.task, self.layout
        urdf = usd.with_suffix(".urdf")
        mounted = self._mounted
        self.device = self.sim.device
        if self.oceansim:
            import isaacswim.oceansim as osim
            from isaacswim import Medium

            water = Medium(surface_height=SURFACE_Z)
            sigma_pa = 0.01 * self._sensor_noise * water.density * 9.81
            mounted["baro_os"] = osim.Barometer(
                self._base_prim(),
                water,
                name=self.root.replace("/", "_") + "_baro",
                as_depth=True,
                noise_cov=sigma_pa**2,
                device=self.device,
            )

        self.hydro, _ = build_hydro(
            self.robot,
            urdf,
            current=(0.01, 0.0, 0.0),
            surface_z=None,
            seabed_z=task.seabed_z,
            device=self.device,
            extra_geom=gripper_shapes(usd.parent),
        )
        self.prop_hydro = sc.prop_hydrodynamics(layout, self.objects)
        self._prop_moving: dict[str, bool] = {}
        self.dynamic = [p.name for p in layout.props if p.kind == "dynamic"]
        self.thrusters = Thrusters(ThrusterArray(usd.parent / "thrusters.json"), body="base_link")
        self.vehicle = Vehicle(
            self.robot,
            hydro=self.hydro,
            seabed_z=task.seabed_z,
            actuators=[
                self.thrusters,
                JointServos(list(ARM_JOINTS), rate=0.5, name="arm"),
                JointServos(["camera_mount_joint"], rate=0.5, name="camera"),
                JointServos([GRIP_JOINT], name="gripper"),
            ],
            sensors=mounted,
        )
        self.keeper = StationKeeper(ki=(task.keeper_ki,) * 3, damping=BLUEROV2_DAMPING)
        self._jidx = {n: i for i, n in enumerate(self.robot.joint_names)}
        self._mounted = mounted
        self._followers = [s for s in mounted.values() if getattr(s, "mount", None)]

        if self.showcase is not None:
            self.showcase.thrusters = self.thrusters
        self._static = {n: layout.pose(n) for n in layout.poses}
        self._check_physics_dt()

    @staticmethod
    def _joint_pose(spec: SceneSpec) -> dict:
        q = dict(spec.arm or {})
        q[GRIP_JOINT] = GRIP_OPEN
        q["camera_mount_joint"] = float(spec.camera_tilt)
        return q

    def _check_physics_dt(self) -> None:
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import PhysxSchema

        prim = get_current_stage().GetPrimAtPath(self.sim.cfg.physics_prim_path)
        hz = PhysxSchema.PhysxSceneAPI(prim).GetTimeStepsPerSecondAttr().Get()
        if hz is None or abs(float(hz) * self.physics_dt - 1.0) > 1e-3:
            raise RuntimeError(
                f"the PhysX scene steps at {hz} Hz, but this env integrates "
                f"at {1.0 / self.physics_dt:.0f} Hz"
            )

    @staticmethod
    def _enable_oceansim():
        import isaacswim.oceansim as osim

        if not osim.enable():
            raise RuntimeError("OceanSim could not be enabled.\n" + osim.OCEANSIM_INSTALL_HINT)

    def _base_prim(self) -> str:
        import omni.usd
        from pxr import Usd

        stage = omni.usd.get_context().get_stage()
        return next(
            str(p.GetPath())
            for p in Usd.PrimRange(stage.GetPrimAtPath(f"{self.root}/rov"))
            if p.GetName() == "base_link"
        )

    @property
    def sonar_grid(self) -> dict | None:
        sonar = self.vehicle.sensors.get("sonar") if hasattr(self, "vehicle") else None
        return sonar.grid if sonar is not None and hasattr(sonar, "grid") else None

    def _mount_sonar(self, device):
        import isaacswim.oceansim as osim

        self._enable_oceansim()
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        for root, refl in (("/World/seabed", 1.0), (f"{self.root}/props", None)):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
                if prim.IsA(UsdGeom.Gprim):
                    r = refl
                    if r is None:
                        name = str(prim.GetPath())[len(f"{self.root}/props") + 1 :].split("/")[0]
                        r = self.layout.prop(name).reflectivity
                    osim.set_sonar_reflectivity(str(prim.GetPath()), r)
        self.scene_task.label_sonar(osim, stage, self.scenery_roots)
        return osim.Sonar(
            self._base_prim(),
            pitch_deg=self.sonar_pitch,
            max_range=6.0,
            range_res=0.02,
            hori_res=1536,
            device=device,
        )

    def _reset_plant(self, spec: SceneSpec) -> World:
        if not self.layout.same_layout(spec):
            raise ValueError(
                f"{self.task.name}: this episode's props differ from the "
                "stage's; a task may only re-pose its props between episodes"
            )
        self._static = {n: spec.pose(n) for n in spec.poses}
        for name, obj in self.objects.items():
            pos, quat = spec.pose(name)
            sc.write_pose(obj, pos + self.origin, quat, self.device)
        if spec.joints:
            sc.set_joint_friction(spec, root=self.root)

        r = self.robot
        pose = torch.tensor(
            [[*(np.asarray(spec.spawn_pos) + self.origin), *yaw_quat(spec.spawn_yaw)]],
            dtype=torch.float32,
            device=self.device,
        )
        _lab.write_root_pose(r, pose)
        _lab.write_root_velocity(r, torch.zeros(1, 6, device=self.device))
        q = _lab.data(r).default_joint_pos.clone()
        for n, v in self._joint_pose(spec).items():
            if n in self._jidx:
                q[0, self._jidx[n]] = float(v)
        _lab.write_joint_state(r, q, torch.zeros_like(q))
        _lab.set_joint_position_target(r, q)
        r.reset()

        cur = spec.current
        if any(cur.sigma) or type(self.task).current is not Task.current:
            field = GustyCurrent(
                mean=cur.mean,
                sigma=cur.sigma,
                tau=cur.tau,
                seabed_z=spec.seabed_z,
                device=self.device,
                seed=int(self.rng.integers(2**31)),
            )
        else:
            field = Uniform(cur.mean, seabed_z=spec.seabed_z, layer=1.5, device=self.device)
        self._current = field
        for h in (self.hydro, *self.prop_hydro.values()):
            h.cfg.current_fn = field
            h.set_medium(h.cfg.medium)
        if self.showcase is not None:
            self.showcase.reset(current=field)

        self.vehicle.reset()
        self.keeper.reset()
        self.keeper.ki[:] = self.task.keeper_ki
        self.instruments = Instruments(
            self.nav_cfg,
            np.random.default_rng([int(self.seed or 0), 0x4E4156]),
            dvl=self._mounted.get("dvl_os"),
            barometer=self._mounted.get("baro_os"),
            surface_z=SURFACE_Z,
        )
        spawn_q = np.asarray(yaw_quat(spec.spawn_yaw))
        self.nav.reset(spec.spawn_pos, yaw_rotate(spawn_q, self.instruments.heading_bias))
        self._dvl_clock = math.inf
        self.vehicle.command(
            {
                "arm": {n: spec.arm[n] for n in ARM_JOINTS},
                "camera": [float(spec.camera_tilt)],
                "gripper": [GRIP_OPEN],
            }
        )
        self._grip = False
        self._tilt = float(spec.camera_tilt)
        self._scale = 1.0
        self._t = 0.0
        hold, hold_yaw = np.asarray(spec.spawn_pos, dtype=np.float64), spec.spawn_yaw
        if self.nav_mode == "dvl":
            hold, hold_yaw = self.nav.pos.copy(), self.nav.yaw
        self._settle_target = (hold, hold_yaw)
        if not self._defer_settle:
            for _ in range(int(0.5 / self.physics_dt)):
                self._physics_step(hold, hold_yaw, None)
        self._t = 0.0
        self.vehicle.time = 0.0
        self._carrot = None
        return self._snapshot()

    def _set_current(self, current: CurrentSpec) -> None:
        f = self._current
        f.mean = np.asarray(current.mean, dtype=np.float64)
        f.sigma = np.asarray(current.sigma, dtype=np.float64)

    def _set_keeper_ki(self, ki: float) -> None:
        self.keeper.ki[:] = float(ki)

    def _begin_control(self, sp, dt: float):
        self._grip = bool(sp.grip)
        self.vehicle.command({"arm": sp.arm, "gripper": [GRIP_CLOSED if self._grip else GRIP_OPEN]})
        self._scale = 1.0
        if self._carrot is None:
            self._carrot_vel = np.zeros(4)
        else:
            v_max = self.keeper.v_max
            raw = np.append(
                np.clip((sp.pos - self._carrot[0]) / dt, -v_max, v_max),
                np.clip(wrap_pi(sp.yaw - self._carrot[1]) / dt, -1.0, 1.0),
            )
            self._carrot_vel += (raw - self._carrot_vel) * dt / (CARROT_TAU + dt)
        self._carrot = (np.array(sp.pos, dtype=np.float64), float(sp.yaw))
        vel, rate = self._carrot_vel[:3], float(self._carrot_vel[3])
        return vel, rate

    def _advance(self, sp, dt: float) -> World:
        vel, rate = self._begin_control(sp, dt)
        for i in range(self.substeps):
            self._physics_step(sp.pos, sp.yaw, sp.wrench, vel, rate, props_all=i == 0)
        self._t += self.substeps * self.physics_dt
        return self._snapshot()

    @property
    def camera_tilt(self) -> float:
        return self._tilt

    def _physics_step(self, *args, **kwargs):
        self._before_physics(*args, **kwargs)
        self.sim.step(render=False)
        self._after_physics()
        if self.on_physics_step is not None:
            self.on_physics_step()

    def _before_physics(self, pos, yaw, wrench, vel=None, rate=0.0, props_all=True) -> None:
        d = _lab.data(self.robot)
        p, q = d.root_pos_w[0].cpu().numpy() - self.origin, d.root_quat_w[0].cpu().numpy()
        v, w = d.root_lin_vel_b[0].cpu().numpy(), d.root_ang_vel_b[0].cpu().numpy()
        if self.nav_mode == "dvl":
            self._navigate(p, q, v, w)
            p, q, v, w = self.nav.pos, self.nav.quat, self.nav.vel_b, self.nav.ang_vel_b
        if wrench is None:
            wrench, _ = self.keeper.command(
                pos_w=p,
                quat_w=q,
                lin_vel_b=v,
                ang_vel_b=w,
                target_pos=pos,
                target_yaw=yaw,
                dt=self.physics_dt,
                target_vel=vel,
                target_yaw_rate=rate,
            )
        self.vehicle.command({"thrusters": wrench})
        self.vehicle.apply(self.physics_dt)
        self._scale = min(self._scale, float(self.thrusters.info["scale"][0]))
        for n, h in self.prop_hydro.items():
            if props_all:
                h.apply()
                v = _lab.data(h.asset).body_com_vel_w[0, 0]
                self._prop_moving[n] = bool(v[:3].norm() > 5e-3 or v[3:].norm() > 5e-2)
            elif self._prop_moving.get(n, True):
                h.apply()
        for n in self.dynamic:
            self.objects[n].write_data_to_sim()

    def _after_physics(self):
        d = _lab.data(self.robot)
        self.vehicle.update(self.physics_dt)
        if self._followers:
            p, q = d.root_pos_w[0].cpu().numpy(), d.root_quat_w[0].cpu().numpy()
            for s in self._followers:
                s.follow(p, q)
        for n in self.dynamic:
            self.objects[n].update(self.physics_dt)

    def _navigate(self, p, q, v, w) -> None:
        ins, dt = self.instruments, self.physics_dt
        state = None
        if self.oceansim:
            t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32))[None]  # noqa: E731
            state = SimpleNamespace(pos_w=t(p + self.origin), quat_w=t(q), lin_vel_b=t(v))
        quat, rates = ins.ahrs(q, w)
        dvl = None
        ok = True
        self._dvl_clock += dt
        if self._dvl_clock >= 1.0 / self.nav_cfg.dvl_hz - 1e-9:
            self._dvl_clock = 0.0
            dvl, ok = ins.dvl(v, float(p[2]) - self.task.seabed_z, state)
        self.nav.update(
            dt, quat=quat, ang_vel_b=rates, depth_z=ins.depth_z(p, state), dvl=dvl, dvl_valid=ok
        )

    def _nav_pose(self):
        if self.nav_mode == "truth":
            return super()._nav_pose()
        return self.nav.pos.copy(), self.nav.yaw

    def _nav_heading_error(self) -> float:
        if self.nav_mode == "truth" or self.instruments is None:
            return 0.0
        return self.instruments.heading_bias

    def report(self):
        out = super().report()
        if self.nav_mode == "dvl":
            out["nav_drift_m"] = float(np.linalg.norm((self.nav.pos - self.world.pos)[:2]))
        return out

    def _snapshot(self) -> World:
        d = _lab.data(self.robot)
        np1 = lambda t: t[0].detach().cpu().numpy().astype(np.float64)  # noqa: E731
        pos, quat = np1(d.root_pos_w) - self.origin, np1(d.root_quat_w)
        jp = np1(d.joint_pos)
        arm_q = {n: float(jp[self._jidx[n]]) for n in ARM_JOINTS}
        tool, R = tool_world(pos, quat, self.arm, arm_q)
        objects = dict(self._static)
        for n in self.dynamic:
            o = _lab.data(self.objects[n])
            objects[n] = (np1(o.root_pos_w) - self.origin, np1(o.root_quat_w))
        joints = {}
        for j in self.layout.joints:
            _, q = objects[j.body]
            joints[j.name] = math.atan2(
                2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)
            )
        return World(
            t=self._t,
            pos=pos,
            quat=quat,
            lin_vel_b=np1(d.root_lin_vel_b),
            ang_vel_b=np1(d.root_ang_vel_b),
            arm_q=arm_q,
            grip_q=float(jp[self._jidx[GRIP_JOINT]]),
            grip_closed=self._grip,
            tool_pos=tool,
            tool_R=R,
            objects=objects,
            joints=joints,
            seabed_z=self.task.seabed_z,
            thrust_scale=self._scale,
        )

    def _extra_obs(self) -> dict:
        out = {}
        if self.showcase is not None:
            self.showcase.advance(self.thrusters.thrust[0], self.control_dt)
        if self.rig is not None:
            if self.showcase is not None:
                self.showcase.hide_drawings(
                    specks=not self.specks_in_views, arrows=not self.arrows_in_views
                )
            if "front" in self.rig.views:
                self.rig.views["front"].look = (math.cos(self._tilt), 0.0, math.sin(self._tilt))
            out.update(
                {
                    f"rgb_{k}": v
                    for k, v in self.rig.grab(self.sim.render, dt=self.physics_dt).items()
                }
            )
        if self.showcase is not None:
            self.video_frame = self.showcase.grab(self.sim.render, render_dt=self.physics_dt)
        if self.vehicle.sensors:
            if "sonar" in self.vehicle.sensors and self.rig is None and self.showcase is None:
                self.sim.render()
            o = self.vehicle.observe()
            np32 = lambda t: t.detach().cpu().numpy().astype(np.float32).reshape(-1)  # noqa: E731
            if "sonar" in o:
                out["sonar"] = o["sonar"][0].detach().cpu().numpy().astype(np.float32)
            if "dvl_os" in o:
                out["dvl"] = np32(o["dvl_os"][0])
                out["dvl_valid"] = np32(o["dvl_os/valid"])
                out["depth"] = np32(o["baro_os"])
        return out

    def observe(self):
        obs = super().observe()
        if self.nav_mode == "dvl" and self.instruments is not None:
            ins = self.instruments
            imu = obs["imu"].astype(np.float64)
            imu[:4] = yaw_rotate(imu[:4], ins.heading_bias)
            obs["imu"] = imu.astype(np.float32)
            obs["dvl"] = (obs["dvl"] * ins.dvl_scale).astype(np.float32)
        obs["proprio"] = np.concatenate([obs[k] for k in PROPRIO_KEYS])
        return obs

    def close(self) -> None:
        self.sim.stop()
