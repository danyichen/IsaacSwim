from __future__ import annotations

import numpy as np
import pytest
import torch

from isaacswim import FluidShape, Medium
from isaacswim.isaaclab import Hydrodynamics, HydrodynamicsCfg, shapes_from_articulation

BODIES = ["hull", "arm"]
E = 4


class FakeData:
    def __init__(self, with_com: bool):
        B = len(BODIES)
        self.body_link_quat_w = torch.zeros(E, B, 4, dtype=torch.float32)
        self.body_link_quat_w[..., 0] = 1.0
        self.body_link_pos_w = torch.zeros(E, B, 3, dtype=torch.float32)
        self.body_link_lin_vel_w = torch.zeros(E, B, 3, dtype=torch.float32)
        self.body_link_lin_vel_w[..., 0] = 0.5
        self.body_link_ang_vel_w = torch.zeros(E, B, 3, dtype=torch.float32)
        if with_com:
            self.com_pos_b = torch.zeros(E, B, 3, dtype=torch.float32)
            self.com_pos_b[..., 2] = 0.01


class FakeView:
    def __init__(self):
        B = len(BODIES)
        self._masses = torch.tensor([[8.0, 1.0]], dtype=torch.float32)
        self._inertias = torch.zeros(1, B, 9, dtype=torch.float32)
        for i, (m, semi) in enumerate([(8.0, (0.3, 0.2, 0.15)), (1.0, (0.2, 0.05, 0.05))]):
            a, b, c = np.array(semi) ** 2
            d = m / 3.0 * np.array([b + c, c + a, a + b])
            self._inertias[0, i] = torch.tensor(np.diag(d).reshape(-1), dtype=torch.float32)
        self._coms = torch.zeros(1, B, 7, dtype=torch.float32)
        self._coms[..., 3] = 1.0

    def get_masses(self):
        return self._masses

    def get_inertias(self):
        return self._inertias

    def get_coms(self):
        return self._coms


class FakeComposer:
    def __init__(self):
        self.calls = []

    def set_forces_and_torques(self, forces, torques):
        self.calls.append((forces.clone(), torques.clone()))


class FakeAsset:
    def __init__(self, api: str = "composer", with_com: bool = True):
        self.body_names = list(BODIES)
        self.device = "cpu"
        self.data = FakeData(with_com)
        self.root_physx_view = FakeView()
        self.writes = []
        if api == "composer":
            self.permanent_wrench_composer = FakeComposer()
        if api == "legacy":
            d = self.data
            d.body_quat_w = d.body_link_quat_w
            d.body_pos_w = d.body_link_pos_w
            d.body_lin_vel_w = d.body_link_lin_vel_w
            d.body_ang_vel_w = d.body_link_ang_vel_w
            for attr in ("body_link_quat_w", "body_link_pos_w",
                         "body_link_lin_vel_w", "body_link_ang_vel_w"):
                delattr(d, attr)
            self.set_external_force_and_torque = self._legacy_write
        elif api == "is_global":
            self.set_external_force_and_torque = self._modern_write

    def _legacy_write(self, forces, torques):
        self.writes.append((forces.clone(), torques.clone(), None))

    def _modern_write(self, forces, torques, is_global=False):
        self.writes.append((forces.clone(), torques.clone(), is_global))


def cfg(**kw):
    base = dict(
        shapes=[FluidShape(body="hull", semiaxes=(0.3, 0.2, 0.15)),
                FluidShape(body="arm", semiaxes=(0.2, 0.05, 0.05), pos=(0.1, 0, 0))],
        medium=Medium(density=1000.0, viscosity=0.001, gravity=(0.0, 0.0, 0.0)),
    )
    base.update(kw)
    return HydrodynamicsCfg(**base)


def test_compute_returns_a_wrench_per_body():
    hydro = Hydrodynamics(cfg(), FakeAsset())
    F, T = hydro.compute()
    assert F.shape == (E, len(BODIES), 3)
    assert T.shape == F.shape
    assert F[0, 0, 0].item() < 0.0
    assert torch.isfinite(F).all() and torch.isfinite(T).all()


@pytest.mark.parametrize("api", ["composer", "is_global", "legacy"])
def test_every_isaaclab_wrench_api_is_written_to(api):
    asset = FakeAsset(api)
    hydro = Hydrodynamics(cfg(), asset)
    F, T = hydro.apply()
    if api == "composer":
        assert len(asset.permanent_wrench_composer.calls) == 1
        wrote_F, wrote_T = asset.permanent_wrench_composer.calls[0]
    else:
        assert len(asset.writes) == 1
        wrote_F, wrote_T, is_global = asset.writes[0]
        assert is_global == (False if api == "is_global" else None)
    assert torch.allclose(wrote_F, F) and torch.allclose(wrote_T, T)


def test_legacy_isaaclab_body_attribute_names_are_found():
    modern = Hydrodynamics(cfg(), FakeAsset("is_global")).compute()
    legacy = Hydrodynamics(cfg(), FakeAsset("legacy")).compute()
    assert torch.allclose(modern[0], legacy[0])
    assert torch.allclose(modern[1], legacy[1])


def test_is_global_is_rejected_when_unsupported():
    with pytest.raises(ValueError, match="is_global"):
        Hydrodynamics(cfg(is_global=True), FakeAsset("legacy"))


def test_world_wrench_rotates_buoyancy_and_added_thrust():
    asset = FakeAsset("is_global")
    s = np.sqrt(0.5)
    asset.data.body_link_quat_w[:] = torch.tensor([s, 0.0, s, 0.0])
    hydro = Hydrodynamics(cfg(is_global=True, medium=Medium.sea_water()), asset)
    thrust = torch.zeros(E, 2, 3)
    thrust[..., 0] = 10.0
    local_f, local_t = hydro.compute()
    from isaacswim.torch_math import quat_apply
    q = asset.data.body_link_quat_w
    f, t = hydro.apply(add=(thrust, torch.zeros_like(thrust)))
    assert torch.allclose(f, quat_apply(q, local_f + thrust), atol=1e-5)
    assert torch.allclose(t, quat_apply(q, local_t), atol=1e-5)
    assert asset.writes[-1][2] is True


def test_density_change_updates_acceleration_added_mass():
    asset = FakeAsset()
    view = asset.root_physx_view
    view._masses = view._masses.repeat(E, 1)
    view._inertias = view._inertias.repeat(E, 1, 1)
    hydro = Hydrodynamics(cfg(added_mass=True, physics_dt=0.01), asset)
    before = hydro.added_mass.P.clone()
    hydro.set_medium(Medium(density=2000.0))
    fresh = Hydrodynamics(cfg(added_mass=True, physics_dt=0.01,
                              medium=Medium(density=2000.0)), asset)
    assert not torch.allclose(before, hydro.added_mass.P)
    assert torch.allclose(hydro.added_mass.P, fresh.added_mass.P)


def test_add_sums_before_the_single_write():
    asset = FakeAsset("composer")
    hydro = Hydrodynamics(cfg(), asset)
    thrust = (torch.zeros(E, len(BODIES), 3, dtype=torch.float32),
              torch.zeros(E, len(BODIES), 3, dtype=torch.float32))
    thrust[0][:, 0, 0] = 25.0

    F_only, _ = hydro.compute()
    F, T = hydro.apply(add=thrust)
    assert torch.allclose(F[:, 0, 0], F_only[:, 0, 0] + 25.0)
    assert len(asset.permanent_wrench_composer.calls) == 1
    assert torch.allclose(asset.permanent_wrench_composer.calls[0][0], F)


def test_body_com_is_read_from_the_asset():
    with_com = Hydrodynamics(cfg(), FakeAsset(with_com=True)).compute()
    without = Hydrodynamics(cfg(), FakeAsset(with_com=False)).compute()
    assert torch.allclose(with_com[0], without[0])
    assert not torch.allclose(with_com[1], without[1])


def test_auto_shapes_recover_the_box_they_were_built_from():
    shapes = shapes_from_articulation(FakeAsset())
    assert [s.body for s in shapes] == BODIES
    assert np.allclose(shapes[0].semiaxes, (0.3, 0.2, 0.15), rtol=1e-6)
    assert np.allclose(shapes[1].semiaxes, (0.2, 0.05, 0.05), rtol=1e-6)


def test_from_inertia_has_no_ellipsoid_escape_hatch():
    with pytest.raises(TypeError):
        FluidShape.from_inertia("b", 1.0, (0.01, 0.01, 0.01), convention="ellipsoid")


def test_auto_shapes_honour_body_density_for_displacement():
    shapes = shapes_from_articulation(FakeAsset(), body_density=1050.0)
    assert shapes[0].volume == pytest.approx(8.0 / 1050.0)


def test_bookkeeping_links_are_filtered_out():
    asset = FakeAsset()
    asset.body_names = BODIES + ["imu_link", "logo_link"]
    v = asset.root_physx_view
    v._masses = torch.tensor([[8.0, 1.0, 1e-6, 1e-3]], dtype=torch.float32)
    inert = torch.zeros(1, 4, 9, dtype=torch.float32)
    inert[0, :2] = v._inertias[0, :2]
    inert[0, 2] = torch.tensor(np.diag([1e-5] * 3).reshape(-1), dtype=torch.float32)
    inert[0, 3] = torch.tensor(np.diag([1e-4] * 3).reshape(-1), dtype=torch.float32)
    v._inertias = inert
    v._coms = torch.zeros(1, 4, 7, dtype=torch.float32)
    v._coms[..., 3] = 1.0

    shapes = shapes_from_articulation(asset, verbose=False)
    assert [s.body for s in shapes] == BODIES, "an artefact link survived the filter"
    assert max(float(np.max(s.semiaxes)) for s in shapes) < 1.0


def test_filtering_everything_is_an_error_not_a_silent_empty():
    asset = FakeAsset()
    with pytest.raises(ValueError, match="filtered out"):
        shapes_from_articulation(asset, min_mass=1e6, verbose=False)


def test_auto_shapes_pattern_selects_bodies():
    assert [s.body for s in shapes_from_articulation(FakeAsset(), pattern="arm")] == ["arm"]
    with pytest.raises(ValueError, match="matched none"):
        shapes_from_articulation(FakeAsset(), pattern="thruster")


def test_auto_shapes_path_through_the_config():
    hydro = Hydrodynamics(cfg(shapes=[], auto_shapes="inertia", body_density=1050.0), FakeAsset())
    assert hydro.wrench.n_slots == len(BODIES)


def test_auto_shapes_default_reads_geometry(monkeypatch):
    import isaacswim.usd as usd

    seen = {}

    def fake(asset, body_names, mesh_buoyancy):
        seen["bodies"] = body_names
        return [FluidShape(body=n, semiaxes=(0.1, 0.1, 0.1)) for n in body_names]

    monkeypatch.setattr(usd, "shapes_from_articulation_usd", fake)
    hydro = Hydrodynamics(cfg(shapes=[], auto_shapes=True, auto_body_pattern="arm"), FakeAsset())
    assert seen["bodies"] == ["arm"]
    assert hydro.wrench.n_slots == 1
    with pytest.raises(ValueError, match="auto_shapes"):
        Hydrodynamics(cfg(shapes=[], auto_shapes="box"), FakeAsset())


def test_missing_shapes_without_auto_is_an_error():
    with pytest.raises(ValueError, match="auto_shapes"):
        Hydrodynamics(cfg(shapes=[]), FakeAsset())


def test_uninitialised_asset_gives_a_useful_message():
    asset = FakeAsset()
    asset.body_names = []
    with pytest.raises(RuntimeError, match="not initialised"):
        Hydrodynamics(cfg(), asset)


def test_debug_check_nan_fires_on_a_poisoned_state():
    asset = FakeAsset()
    asset.data.body_link_lin_vel_w[2, 1, 0] = float("nan")
    hydro = Hydrodynamics(cfg(debug_check_nan=True), asset)
    with pytest.raises(FloatingPointError, match="non-finite"):
        hydro.compute()


def test_medium_can_be_swapped_at_runtime():
    hydro = Hydrodynamics(cfg(), FakeAsset())
    light = hydro.compute()[0].clone()
    hydro.set_medium(Medium(density=2000.0, viscosity=0.002, gravity=(0.0, 0.0, 0.0)))
    assert torch.allclose(hydro.compute()[0], 2.0 * light, rtol=1e-5)


def test_free_surface_reads_positions_without_being_asked_twice():
    surface = Medium(density=1000.0, viscosity=0.001, gravity=(0.0, 0.0, -9.81),
                     surface_height=0.5)
    asset = FakeAsset()
    hydro = Hydrodynamics(cfg(medium=surface), asset)
    assert hydro._needs_pos
    F, _ = hydro.compute()
    assert F[0, 0, 2].item() > 0.0


def test_surface_fn_is_a_moving_surface_through_the_binding():
    still = Medium(density=1000.0, viscosity=0.001, gravity=(0.0, 0.0, -9.81))
    flat = Medium(density=1000.0, viscosity=0.001, gravity=(0.0, 0.0, -9.81),
                  surface_height=0.05)
    seen = []

    def level(centres):
        seen.append(tuple(centres.shape))
        return torch.full(centres.shape[:-1], 0.05, dtype=centres.dtype)

    hydro = Hydrodynamics(cfg(medium=still, surface_fn=level), FakeAsset())
    assert hydro._needs_pos
    F, T = hydro.compute()
    assert seen and seen[0][-1] == 3
    F0, T0 = Hydrodynamics(cfg(medium=flat), FakeAsset()).compute()
    assert torch.allclose(F, F0) and torch.allclose(T, T0)
