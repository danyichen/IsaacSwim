from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isaacswim import (  # noqa: E402
    IMU,
    Altimeter,
    DopplerVelocityLog,
    FluidShape,
    Medium,
    Odometry,
    PressureDepth,
)
from isaacswim.actuators import JointServos, ThrusterArray, Thrusters  # noqa: E402
from isaacswim.control import Controller, VehicleState  # noqa: E402
from isaacswim.flow import Uniform  # noqa: E402
from isaacswim.isaaclab import Hydrodynamics, HydrodynamicsCfg, Vehicle  # noqa: E402

BODIES = ["base_link", "arm_link"]
JOINTS = ["shoulder", "elbow", "camera_tilt"]
E = 2


class FakeData:
    def __init__(self):
        B, J = len(BODIES), len(JOINTS)
        self.root_pos_w = torch.tensor([[0.0, 0.0, -2.0], [1.0, 0.0, -3.0]])
        self.root_quat_w = torch.tensor([[1.0, 0, 0, 0]] * E)
        s = np.sqrt(0.5)
        self.root_quat_w[1] = torch.tensor([s, 0.0, 0.0, s])
        self.root_lin_vel_b = torch.tensor([[0.2, 0.0, 0.0]] * E)
        self.root_ang_vel_b = torch.zeros(E, 3)
        self.joint_pos = torch.zeros(E, J)
        self.joint_vel = torch.zeros(E, J)
        self.default_joint_pos = torch.tensor([[0.5, 1.0, 0.0]] * E)
        self.body_link_quat_w = self.root_quat_w.unsqueeze(1).repeat(1, B, 1)
        self.body_link_pos_w = self.root_pos_w.unsqueeze(1).repeat(1, B, 1)
        self.body_link_lin_vel_w = torch.zeros(E, B, 3)
        self.body_link_ang_vel_w = torch.zeros(E, B, 3)


class FakeComposer:
    def __init__(self):
        self.calls = []

    def set_forces_and_torques(self, forces, torques):
        self.calls.append((forces.clone(), torques.clone()))


class FakeArticulation:
    device = "cpu"

    def __init__(self):
        self.body_names = list(BODIES)
        self.joint_names = list(JOINTS)
        self.data = FakeData()
        self.permanent_wrench_composer = FakeComposer()
        self.targets = []
        self.writes = 0
        self.updates = 0

    def set_joint_position_target(self, target, joint_ids=None):
        self.targets.append((target.clone(), joint_ids.clone()))

    def write_data_to_sim(self):
        self.writes += 1

    def update(self, dt):
        self.updates += 1


def thrusters():
    arr = ThrusterArray.from_geometry(
        [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0], [0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
        [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]], fwd=50.0, rev=40.0)
    return Thrusters(arr, body="base_link", input="thrust")


@pytest.fixture
def robot():
    return FakeArticulation()


def test_actuator_wrench_lands_on_its_body_in_one_write(robot):
    v = Vehicle(robot, actuators=[thrusters()])
    v.command({"thrusters": [10.0, 10.0, 0.0, 0.0]})
    F, T = v.apply(0.01)
    assert len(robot.permanent_wrench_composer.calls) == 1
    assert torch.allclose(F[:, 0], torch.tensor([[20.0, 0.0, 0.0]] * E))
    assert torch.allclose(F[:, 1], torch.zeros(E, 3)), "arm got the thrust"
    assert robot.writes == 1


def test_add_and_hydrodynamics_share_the_single_write(robot):
    shapes = [FluidShape.sphere("base_link", radius=0.1)]
    hydro = Hydrodynamics(HydrodynamicsCfg(shapes=shapes, medium=Medium.sea_water()),
                          robot)
    v = Vehicle(robot, hydro=hydro, actuators=[thrusters()])
    extra = (torch.zeros(E, 2, 3), torch.zeros(E, 2, 3))
    extra[0][:, 1, 2] = 3.0
    v.command({"thrusters": [5.0, 5.0, 0.0, 0.0]})
    F, _ = v.apply(0.01, add=extra)
    assert len(robot.permanent_wrench_composer.calls) == 1
    F_hydro, _ = hydro.compute()
    assert torch.allclose(F[:, 0, 0], F_hydro[:, 0, 0] + 10.0, atol=1e-5)
    assert torch.allclose(F[:, 1, 2], F_hydro[:, 1, 2] + 3.0, atol=1e-5)


def test_joint_targets_reach_the_named_joints(robot):
    arm = JointServos(["shoulder", "elbow"], rate=1.0, name="arm")
    v = Vehicle(robot, actuators=[arm])
    v.command({"arm": {"elbow": 2.0}})
    v.apply(0.1)
    target, ids = robot.targets[-1]
    assert ids.tolist() == [0, 1]
    assert torch.allclose(target[0], torch.tensor([0.5, 1.1])), "starts at the spawn pose"


def test_observe_names_every_sensor_and_keeps_validity_masks(robot):
    v = Vehicle(robot, seabed_z=-4.0, sensors={
        "depth": PressureDepth(sigma=0.0), "alt": Altimeter(sigma=0.0),
        "dvl": DopplerVelocityLog(sigma=0.0, scale_error=0.0),
        "imu": IMU(), "odom": Odometry()})
    obs = v.observe()
    assert torch.allclose(obs["depth"], torch.tensor([2.0, 3.0]))
    assert torch.allclose(obs["alt"], torch.tensor([2.0, 1.0]))
    assert obs["alt/valid"].all()
    assert torch.allclose(obs["dvl"], torch.tensor([[0.2, 0.0, 0.0]] * E))
    assert obs["imu"].shape == (E, 7) and obs["odom"].shape == (E, 13)
    assert obs["time"] == 0.0


def test_current_is_reported_in_the_body_frame(robot):
    shapes = [FluidShape.sphere("base_link", radius=0.1)]
    hydro = Hydrodynamics(HydrodynamicsCfg(shapes=shapes, medium=Medium.sea_water(),
                                           current_fn=Uniform((1.0, 0.0, 0.0))), robot)
    st = Vehicle(robot, hydro=hydro).state()
    assert torch.allclose(st.current_b[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(st.current_b[1], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6)


def test_uniform_medium_current_reaches_water_track_dvl(robot):
    hydro = Hydrodynamics(HydrodynamicsCfg(
        shapes=[FluidShape.sphere("base_link", radius=0.1)],
        medium=Medium(current=(1.0, 0.0, 0.0))), robot)
    v = Vehicle(robot, hydro=hydro, seabed_z=-4.0,
                sensors={"dvl": DopplerVelocityLog(water_track=True, sigma=0, scale_error=0)})
    obs = v.observe()
    expected = torch.tensor([[-0.8, 0.0, 0.0], [0.2, 1.0, 0.0]])
    assert torch.allclose(obs["dvl"], expected, atol=1e-6)


def test_a_sensor_that_needs_a_seabed_says_so(robot):
    v = Vehicle(robot, sensors={"dvl": DopplerVelocityLog()})
    with pytest.raises(ValueError, match="seabed_z"):
        v.observe()


def test_a_plain_function_is_a_controller(robot):
    def forward(obs, dt):
        return {"thrusters": torch.tensor([[1.0, 1.0, 0.0, 0.0]] * E)}

    assert isinstance(forward, Controller)
    v = Vehicle(robot, actuators=[thrusters()], sensors={"odom": Odometry()})
    obs = v.control(forward, 0.05)
    assert "odom" in obs
    F, _ = v.apply(0.01)
    assert torch.allclose(F[:, 0, 0], torch.full((E,), 2.0))


def test_commanding_an_unknown_actuator_is_an_error(robot):
    v = Vehicle(robot, actuators=[thrusters()])
    with pytest.raises(KeyError, match="thrusters"):
        v.command({"thruster": [0.0] * 4})


def test_duplicate_actuator_names_are_rejected(robot):
    with pytest.raises(ValueError, match="both named"):
        Vehicle(robot, actuators=[thrusters(), thrusters()])


def test_time_advances_and_resets(robot):
    v = Vehicle(robot)
    for _ in range(3):
        v.apply(0.25)
    assert v.state().time == pytest.approx(0.75)
    v.reset()
    assert v.state().time == 0.0


def test_current_field_is_stepped_with_the_physics(robot):
    class Counting(Uniform):
        n = 0

        def step(self, dt):
            Counting.n += 1

    shapes = [FluidShape.sphere("base_link", radius=0.1)]
    hydro = Hydrodynamics(HydrodynamicsCfg(shapes=shapes, medium=Medium.sea_water(),
                                           current_fn=Counting((0.1, 0, 0))), robot)
    v = Vehicle(robot, hydro=hydro)
    v.apply(0.01)
    v.apply(0.01)
    assert Counting.n == 2
    Vehicle(robot, hydro=hydro, step_current=False).apply(0.01)
    assert Counting.n == 2


def test_a_moving_surface_is_stepped_with_the_physics(robot):
    class Sea:
        time = 0.0

        def __call__(self, centres):
            return torch.zeros(centres.shape[:-1], dtype=centres.dtype)

        def step(self, dt):
            self.time += dt

    sea = Sea()
    shapes = [FluidShape.sphere("base_link", radius=0.1)]
    hydro = Hydrodynamics(HydrodynamicsCfg(shapes=shapes, medium=Medium.sea_water(),
                                           surface_fn=sea), robot)
    Vehicle(robot, hydro=hydro).apply(0.01)
    assert sea.time == 0.01


def test_env_snapshot_is_numpy_with_named_joints(robot):
    snap = Vehicle(robot).state().env(1)
    assert isinstance(snap["pos_w"], np.ndarray)
    assert snap["joint_pos"] == {"shoulder": 0.0, "elbow": 0.0, "camera_tilt": 0.0}
    assert isinstance(VehicleState, type)
