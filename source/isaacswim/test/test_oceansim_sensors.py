from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import isaacswim.oceansim as osim  # noqa: E402
from isaacswim import Medium  # noqa: E402
from isaacswim.np_math import quat_apply  # noqa: E402
from isaacswim.sensors import PressureDepth  # noqa: E402

LINK = "/World/rov/base_link"


class FakeSonar:
    def __init__(self, prim_path, name, translation, orientation, **kw):
        self.prim_path, self.translation, self.orientation, self.kw = \
            prim_path, translation, orientation, kw
        self.id = 0
        self.init_args = None
        self.sees_something = True

    def sonar_initialize(self, **kw):
        self.init_args = kw

    def make_sonar_data(self, **kw):
        self.image_kw = kw
        if self.sees_something:
            vec = np.zeros((5, 7, 3), dtype=np.float32)
            vec[..., 2] = 0.5
            self.sonar_data = type("wp", (), {"numpy": lambda s: vec})()
            self.id += 1


class FakeDVL:
    def __init__(self, name="DVL", **kw):
        self.kw = kw
        self.hits = [True] * 4

    def attachDVL(self, rigid_body_path, translation=None):
        self.path, self.translation = rigid_body_path, translation

    def get_beam_hit(self):
        return list(self.hits)

    def get_depth(self):
        return [1.0 if h else float("nan") for h in self.hits]

    def get_linear_vel(self):
        return np.zeros(3) if self.hits.count(False) >= 2 else np.array([0.3, 0.0, -0.1])


class FakeMount:
    def __init__(self, path):
        self.path, self.pose, self.stripped = path, None, False

    def set_pose(self, pos, quat):
        self.pose = (np.asarray(pos).tolist(), np.asarray(quat).tolist())

    def strip_physics(self):
        self.stripped = True


class FakeBaro:
    def __init__(self, prim_path, name, translation, water_density, g, noise_cov,
                 water_surface_z, atmosphere_pressure):
        self.prim_path = prim_path
        self.density, self.g, self.surface, self.p0 = \
            water_density, g, water_surface_z, atmosphere_pressure
        self.z = -3.0

    def get_pressure(self):
        return self.p0 + self.density * self.g * max(self.surface - self.z, 0.0)


@pytest.fixture
def fakes(monkeypatch):
    table = {"ImagingSonarSensor": FakeSonar, "DVLsensor": FakeDVL,
             "BarometerSensor": FakeBaro}
    monkeypatch.setattr(osim, "_oceansim_class", lambda module, name: table[name])
    monkeypatch.setattr(osim, "_make_mount", FakeMount)
    return table


def _state(pos=(1.0, 2.0, -3.0), quat=(1.0, 0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0)):
    t = lambda v: torch.tensor([v], dtype=torch.float32)  # noqa: E731
    return type("S", (), {"pos_w": t(pos), "quat_w": t(quat), "lin_vel_b": t(vel)})()


def test_sonar_mounts_under_the_link_pitched_down(fakes):
    s = osim.Sonar(LINK, pitch_deg=30.0, max_range=5.0)
    assert s.sensor.prim_path == f"{LINK}/sonar"
    assert s.sensor.kw == {"max_range": 5.0}
    fwd = quat_apply(s.sensor.orientation, [1.0, 0.0, 0.0])
    assert fwd[2] == pytest.approx(-math.sin(math.radians(30)))


def test_sonar_initialises_once_then_reads_intensity(fakes):
    s = osim.Sonar(LINK, image={"attenuation": 0.2})
    img, valid = s.measure()
    assert s.sensor.init_args == {"normalizing_method": "all", "viewport": False,
                                  "include_unlabelled": False}
    assert s.sensor.image_kw == {"central_peak": 0.0, "attenuation": 0.2}
    assert img.shape == (1, 5, 7) and torch.allclose(img, torch.full_like(img, 0.5))
    assert valid.tolist() == [True]
    s.measure()
    assert s.sensor.id == 2


def test_sonar_that_sees_nothing_says_so(fakes):
    s = osim.Sonar(LINK)
    s.sensor.sees_something = False
    img, valid = s.measure()
    assert valid.tolist() == [False]
    assert img.shape == (1, 1, 1)


def test_dvl_attaches_and_reports_lock(fakes):
    d = osim.DVL(LINK, max_range=10.0)
    assert d.sensor.path == f"{osim.MOUNT_ROOT}/DVL" and d.sensor.kw == {"max_range": 10.0}
    vel, valid = d.measure()
    assert valid.tolist() == [True] and vel.shape == (1, 3)
    assert torch.allclose(vel, torch.tensor([[0.3, 0.0, -0.1]]))


def test_dvl_velocity_comes_from_the_state_when_given(fakes):
    d = osim.DVL(LINK)
    state = _state(vel=(0.1, -0.2, 0.05))
    vel, valid = d.measure(state)
    assert valid.tolist() == [True]
    assert torch.allclose(vel, torch.tensor([[0.1, -0.2, 0.05]]))
    d.sensor.hits = [False, False, True, True]
    vel, valid = d.measure(state)
    assert valid.tolist() == [False] and torch.allclose(vel, torch.zeros(1, 3))


def test_dvl_dropout_is_flagged_not_just_zeroed(fakes):
    d = osim.DVL(LINK)
    d.sensor.hits = [True, False, False, True]
    vel, valid = d.measure()
    assert valid.tolist() == [False]
    assert torch.allclose(vel, torch.zeros(1, 3))
    assert np.isnan(d.beam_ranges[1]) and d.beam_ranges[0] == 1.0


def test_dvl_and_barometer_hang_from_a_mount_that_follows_the_state(fakes):
    water = Medium.sea_water()
    water.surface_height = 0.0
    d, b = osim.DVL(LINK), osim.Barometer(LINK, water)
    assert d.mount.stripped
    assert b.sensor.prim_path == f"{osim.MOUNT_ROOT}/baro/baro"
    s = _state(pos=(1.0, 2.0, -3.0), quat=(0.0, 0.0, 0.0, 1.0))
    d.measure(s)
    b.measure(s)
    for m in (d.mount, b.mount):
        assert m.pose == ([1.0, 2.0, -3.0], [0.0, 0.0, 0.0, 1.0])


def test_follow_off_parents_to_the_link(fakes):
    assert osim.DVL(LINK, follow=False).sensor.path == LINK


def test_barometer_takes_the_water_from_the_medium(fakes):
    water = Medium.sea_water()
    water.surface_height = 0.0
    b = osim.Barometer(LINK, water)
    assert b.sensor.density == water.density
    assert b.sensor.surface == 0.0


def test_barometer_depth_agrees_with_pressure_depth(fakes):
    water = Medium.sea_water()
    water.surface_height = 0.0
    b = osim.Barometer(LINK, water, as_depth=True)
    b.sensor.z = -3.0
    ours = PressureDepth(surface_height=0.0, density=water.density, sigma=0.0)
    got = b.measure()
    want = ours(torch.tensor([[0.0, 0.0, -3.0]]))
    assert got.item() == pytest.approx(want.item(), abs=1e-6)


def test_barometer_refuses_unbounded_water(fakes):
    with pytest.raises(ValueError, match="surface_height"):
        osim.Barometer(LINK, Medium.sea_water())


def test_without_oceansim_the_error_says_how_to_install():
    with pytest.raises(ImportError, match="extsUser"):
        osim.Sonar(LINK)


class FakeHandler:
    _sonar_node, _baro_node = "sonar-node", "baro-node"
    _rgb_node, _depth_node, _pointcloud_node = "rgb", "depth", "pc"


class FakeSonarROS(FakeSonar):
    pass


class FakeBaroROS(FakeBaro):
    def __init__(self, *a, og_node=None, frame_id=None, **kw):
        super().__init__(*a, **kw)
        self.og_node, self.frame_id, self.published = og_node, frame_id, 0

    def read(self):
        self.published += 1
        return self.get_pressure()


def test_ros_picks_oceansims_ros_classes_and_their_publishers(fakes):
    fakes.update(ImagingSonarSensor_ROS=FakeSonarROS, BarometerSensor_ROS=FakeBaroROS)
    sonar = osim.Sonar(LINK, ros=FakeHandler())
    assert type(sonar.sensor) is FakeSonarROS
    assert sonar.sensor.kw["og_node"] == "sonar-node"
    baro = osim.Barometer(LINK, Medium(surface_height=0.0), ros=FakeHandler())
    assert type(baro.sensor) is FakeBaroROS and baro.sensor.og_node == "baro-node"
    baro.measure()
    assert baro.sensor.published == 1, "the ROS barometer publishes from read()"


def test_ros_publishers_is_oceansims_handler_without_the_dvl(fakes):
    class Handler:
        def __init__(self, **kw):
            self.kw = kw
    fakes["OmniHandler"] = Handler
    h = osim.ros_publishers("rov", sonar=True, baro=True)
    assert h.kw == dict(name="rov", use_camera=False, use_sonar=True, use_imu=False,
                        use_dvl=False, use_baro=True)


def _gerstner(points, time, amplitude=1.0, **kw):
    p = np.asarray(points, dtype=np.float64).copy()
    phase = p[:, 0] - time
    p[:, 1] += amplitude * 0.2 * np.cos(phase)
    p[:, 0] -= amplitude * 0.2 * np.sin(phase)
    return p


def test_waves_height_is_the_kernels_surface_under_the_point(fakes):
    fakes["ocean_deform_launch_kernel"] = _gerstner
    waves = osim.Waves(height=3.0, render=False, iterations=6, time=0.4)
    xy = np.array([[0.0, 0.0], [1.3, -2.0], [-4.0, 5.0]])
    h = waves.height_at(xy)
    xs = np.linspace(-10, 10, 200001)
    src = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], -1)
    moved = _gerstner(src, 0.4)
    for (x, _), got in zip(xy, h):
        i = np.argmin(np.abs(moved[:, 0] - x))
        assert got == pytest.approx(3.0 + moved[i, 1], abs=1e-4)


def test_waves_are_a_surface_fn_and_step_with_the_vehicle(fakes):
    fakes["ocean_deform_launch_kernel"] = _gerstner
    waves = osim.Waves(height=0.0, render=False)
    centres = torch.zeros(2, 3, 3, dtype=torch.float32)
    h0 = waves(centres)
    assert h0.shape == (2, 3) and h0.dtype == torch.float32
    waves.step(1.0)
    assert waves.time == pytest.approx(1.0)
    assert not torch.allclose(waves(centres), h0)
    with pytest.raises(TypeError, match="unknown wave parameters"):
        osim.Waves(render=False, wind=3.0)


def test_wave_construction_sampling_and_redraw_scope_the_warp_device(fakes, monkeypatch):
    import contextlib

    active = []
    calls = []

    @contextlib.contextmanager
    def scope(sensor=None):
        active.append(sensor)
        try:
            yield
        finally:
            active.pop()

    class Surface:
        def __init__(self, **kwargs):
            assert active
            calls.append("construct")

        def deform(self, **kwargs):
            assert active == [None]
            calls.append("redraw")

    def sample(points, time, **kwargs):
        assert active == [None]
        calls.append("sample")
        return _gerstner(points, time, **kwargs)

    monkeypatch.setattr(osim, "_warp_device", scope)
    fakes["WaterSurface"] = Surface
    fakes["ocean_deform_launch_kernel"] = sample
    waves = osim.Waves()
    waves.height_at([[0.0, 0.0]])
    waves.step(0.1)
    assert calls.count("construct") == 1
    assert calls.count("sample") == 3
    assert calls.count("redraw") == 2
    assert not active


def test_ros_camera_info_uses_the_sim6_core_export(monkeypatch):
    import sys
    import types

    reader = object()
    bridge = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "isaacsim.ros2.bridge", bridge)
    monkeypatch.setitem(sys.modules, "isaacsim.ros2.core", types.SimpleNamespace(
        read_camera_info=reader))
    osim._ros_camera_info_compat()
    assert bridge.read_camera_info is reader
    existing = object()
    bridge.read_camera_info = existing
    osim._ros_camera_info_compat()
    assert bridge.read_camera_info is existing


GRID = dict(min_range=0.2, range_res=0.1, min_azimuth=np.radians(25.0),
            azimuth_res=np.radians(1.0))


def polar_with(range_bin, azimuth_deg, n_r=58, n_a=130):
    polar = np.zeros((n_r, n_a), dtype=np.float32)
    polar[range_bin, int(round(azimuth_deg - 25.0))] = 1.0
    return polar


def bright_pixel(fan):
    row, col = np.unravel_index(int(np.argmax(fan)), fan.shape)
    return row, col


def test_sonar_fan_puts_the_apex_at_the_bottom_centre_and_range_upward():
    near = osim.sonar_fan(polar_with(5, 90.0), size=200, **GRID).numpy()
    far = osim.sonar_fan(polar_with(50, 90.0), size=200, **GRID).numpy()
    (rn, cn), (rf, cf) = bright_pixel(near), bright_pixel(far)
    tol = 0.02 * near.shape[1]
    assert abs(cn - near.shape[1] / 2) <= tol and abs(cf - far.shape[1] / 2) <= tol
    assert rf < rn


def test_sonar_fan_shows_small_azimuth_on_the_right():
    fan = osim.sonar_fan(polar_with(40, 40.0), size=200, **GRID).numpy()
    assert bright_pixel(fan)[1] > fan.shape[1] / 2


def test_sonar_fan_is_black_outside_the_fan():
    fan = osim.sonar_fan(np.ones((58, 130), dtype=np.float32), size=200, **GRID).numpy()
    assert fan[-1, 0] == 0.0 and fan[-1, -1] == 0.0
    assert fan[-2, fan.shape[1] // 2] == 0.0
    assert fan[-20, fan.shape[1] // 2] == 1.0
    assert fan.shape[0] == 200 and fan.shape[1] > 200


def test_sonar_fan_batches():
    polar = np.stack([polar_with(10, 90.0), polar_with(30, 60.0)])
    assert osim.sonar_fan(polar, size=64, **GRID).shape[0] == 2
