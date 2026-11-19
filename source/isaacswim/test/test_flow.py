from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isaacswim.flow import (  # noqa: E402
    BoundaryLayer,
    Composite,
    GustyCurrent,
    Jet,
    Pipe,
    Uniform,
    shear_profile,
)


def at(field, *pts):
    p = torch.tensor([list(pts)], dtype=torch.float32)
    return field(p)[0].numpy()


@pytest.fixture
def jet():
    return Jet((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), radius=0.10, outlet_velocity=2.0)


@pytest.fixture
def pipe():
    return Pipe((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), r1=0.20, r2=0.10, v_in=1.0)


def section_flow(field, x, radius, n=4000):
    r = (np.arange(n) + 0.5) / n * radius
    p = torch.tensor(np.stack([np.full(n, x), r, np.zeros(n)], 1), dtype=torch.float64)
    u = field(p)[:, 0].numpy()
    return float(np.sum(u * 2 * np.pi * r) * radius / n), r, u


def test_jet_far_field_decays_with_the_measured_constant(jet):
    for x in (2.0, 4.0, 8.0):
        assert at(jet, x, 0.0, 0.0)[0] == pytest.approx(5.8 * 0.2 / x * 2.0, rel=1e-5)


def test_jet_potential_core_runs_at_the_outlet_speed(jet):
    for x in (0.01, 0.5, 1.1):
        assert at(jet, x, 0.0, 0.0)[0] == pytest.approx(2.0, rel=1e-5)


def test_jet_is_half_speed_at_its_half_width(jet):
    for x in (3.0, 6.0):
        u0 = at(jet, x, 0.0, 0.0)[0]
        assert at(jet, x, 0.094 * x, 0.0)[0] == pytest.approx(0.5 * u0, rel=1e-4)
        assert at(jet, x, 0.0, 0.094 * x)[0] == pytest.approx(0.5 * u0, rel=1e-4)


def test_jet_conserves_momentum_flux(jet):
    outlet = np.pi * 0.2 ** 2 / 4 * 2.0 ** 2
    fluxes = []
    for x in (2.0, 5.0, 10.0):
        n, R = 6000, 0.094 * x * 5
        r = (np.arange(n) + 0.5) / n * R
        p = torch.tensor(np.stack([np.full(n, x), r, np.zeros(n)], 1), dtype=torch.float64)
        u = jet(p)[:, 0].numpy()
        fluxes.append(np.sum(u * u * 2 * np.pi * r) * R / n)
    assert np.ptp(fluxes) < 1e-3 * fluxes[0]
    assert fluxes[0] / outlet == pytest.approx(0.858, abs=0.01)


def test_jet_does_not_blow_backwards(jet):
    assert np.linalg.norm(at(jet, -1.0, 0.0, 0.0)) == 0.0


def test_jet_follows_its_axis():
    j = Jet((1.0, 2.0, 3.0), (0.0, 0.0, -2.0), radius=0.05, outlet_velocity=1.0)
    v = at(j, 1.0, 2.0, 1.0)
    assert v[2] < 0.0 and abs(v[0]) + abs(v[1]) < 1e-7


def test_pipe_conserves_volume_flow(pipe):
    want = 1.0 * np.pi * 0.20 ** 2
    for x in (0.0, 0.7, 1.4, 2.0):
        r = 0.20 + (0.10 - 0.20) * x / 2.0
        q, _, _ = section_flow(pipe, x, r)
        assert q == pytest.approx(want, rel=2e-3), x


def test_pipe_picks_its_regime_from_the_reynolds_number():
    fast = Pipe((0, 0, 0), (1, 0, 0), 0.2, 0.2, v_in=1.0)
    slow = Pipe((0, 0, 0), (1, 0, 0), 0.01, 0.01, v_in=0.01)
    assert fast.profile == "turbulent" and fast.reynolds > 1e5
    assert slow.profile == "laminar" and slow.reynolds < 2300


def test_laminar_pipe_is_poiseuille():
    p = Pipe((0, 0, 0), (1, 0, 0), 0.1, 0.1, v_in=0.5, profile="laminar")
    assert at(p, 0.5, 0.0, 0.0)[0] == pytest.approx(1.0, rel=1e-5)
    assert at(p, 0.5, 0.05, 0.0)[0] == pytest.approx(0.75, rel=1e-5)


def test_turbulent_pipe_centreline_is_the_power_law_ratio():
    p = Pipe((0, 0, 0), (1, 0, 0), 0.1, 0.1, v_in=1.0, profile="turbulent", n=7)
    assert at(p, 0.5, 0.0, 0.0)[0] == pytest.approx(8 * 15 / 98, rel=1e-5)
    q, _, _ = section_flow(p, 0.5, 0.1)
    assert q == pytest.approx(np.pi * 0.01, rel=2e-3)


def test_pipe_has_compact_support_and_no_slip(pipe):
    assert np.linalg.norm(at(pipe, 3.0, 0.0, 0.0)) == 0.0, "runs past its end"
    assert np.linalg.norm(at(pipe, 1.0, 0.30, 0.0)) == 0.0, "leaks through the wall"
    assert at(pipe, 1.0, 0.14999, 0.0)[0] < 0.3 * at(pipe, 1.0, 0.0, 0.0)[0]


def test_pipe_rejects_bad_arguments():
    with pytest.raises(ValueError, match="zero length"):
        Pipe((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), r1=0.1, r2=0.1, v_in=1.0)
    with pytest.raises(ValueError, match="profile"):
        Pipe((0, 0, 0), (1, 0, 0), 0.1, 0.1, v_in=1.0, profile="plug")


@pytest.mark.parametrize("name", ["jet", "pipe"])
def test_tracer_seeds_land_in_moving_water(name, request):
    field = request.getfixturevalue(name)
    rng = np.random.default_rng(1)
    pts = field.tracer_seed(300, rng)
    v = field(torch.tensor(pts, dtype=torch.float32).unsqueeze(0))[0].numpy()
    speed = np.linalg.norm(v, axis=1)

    live = (speed > 1e-6).mean()
    assert live > 0.95, f"{1 - live:.0%} of seeds are in still water"
    assert np.median(speed) > 0.25 * speed.max(), "seeds crowd the slow edge"


def test_composite_adds_at_the_point(jet, pipe):
    both = Composite(jet, pipe)
    p = (1.0, 0.0, 0.0)
    assert at(both, *p)[0] == pytest.approx(at(jet, *p)[0] + at(pipe, *p)[0], abs=1e-5)


def test_composite_steps_every_field_that_can():
    gusts = GustyCurrent(seed=0)
    both = Composite(gusts, Jet((0, 0, 0), (1, 0, 0), 0.1, 1.0))
    before = gusts.gust.copy()
    both.step(0.1)
    assert not np.allclose(before, gusts.gust), "the gust did not advance"


def test_composite_needs_a_field():
    with pytest.raises(ValueError):
        Composite()


def test_shear_profile_is_clamped_at_both_ends():
    pos = torch.tensor([[0.0, 0.0, -10.0], [0.0, 0.0, 0.0], [0.0, 0.0, 100.0]])
    p = shear_profile(pos, seabed_z=0.0, thickness=1.5)
    assert p[0].item() == 0.0, "below the bed must be zero, not complex"
    assert p[1].item() == 0.0, "at the bed, no-slip"
    assert p[2].item() == pytest.approx(1.0), "above the layer, full free stream"


def test_boundary_layer_wraps_a_jet():
    jet = Jet((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), radius=0.10, outlet_velocity=2.0)
    bare = at(jet, 1.0, 0.0, 0.0)[0]
    assert bare > 0.0

    on_the_bed = BoundaryLayer(jet, seabed_z=0.0, thickness=1.5)
    assert at(on_the_bed, 1.0, 0.0, 0.0)[0] == pytest.approx(0.0), "no-slip"

    well_above = BoundaryLayer(jet, seabed_z=-1.5, thickness=1.5)
    assert at(well_above, 1.0, 0.0, 0.0)[0] == pytest.approx(bare, rel=1e-5)


def test_fields_do_not_hand_out_views_of_their_own_buffers():
    f = GustyCurrent(mean=(0.5, 0.0, 0.0), sigma=(0.2, 0.2, 0.2), seed=5,
                     seabed_z=None)
    pos = torch.zeros(3, 3)
    first = f(pos).clone()
    held = f(pos)
    for _ in range(20):
        f.step(1.0 / 30.0)
    assert torch.equal(held, first), "the returned tensor aliased the live buffer"

    u = Uniform((1.0, 0.0, 0.0))
    out = u(torch.zeros(4, 3))
    out += 1.0
    assert at(u, 0.0, 0.0, 0.0)[0] == pytest.approx(1.0)


def test_uniform_without_a_seabed_is_uniform():
    u = Uniform((0.3, -0.1, 0.0))
    for z in (-10.0, 0.0, 10.0):
        assert np.allclose(at(u, 0.0, 0.0, z), [0.3, -0.1, 0.0])


def test_uniform_with_a_seabed_shears():
    u = Uniform((1.0, 0.0, 0.0), seabed_z=0.0, layer=1.5)
    assert at(u, 0.0, 0.0, 0.0)[0] == pytest.approx(0.0)
    assert at(u, 0.0, 0.0, 5.0)[0] == pytest.approx(1.0)
    assert 0.0 < at(u, 0.0, 0.0, 0.5)[0] < 1.0


def test_gusty_current_matches_the_original():
    mean = np.array([0.06, 0.02, 0.0])
    sigma = np.array([0.07, 0.07, 0.03])
    tau, seed = 2.5, 7

    rng = np.random.default_rng(seed)
    gust = np.zeros(3)
    expected = []
    for dt in [1.0 / 30.0] * 50:
        a = np.exp(-dt / tau)
        gust = a * gust + sigma * np.sqrt(1.0 - a * a) * rng.standard_normal(3)
        expected.append(mean + gust)

    field = GustyCurrent(mean=tuple(mean), sigma=tuple(sigma), tau=tau, seed=seed,
                         seabed_z=None)
    got = []
    for dt in [1.0 / 30.0] * 50:
        field.step(dt)
        got.append(at(field, 0.0, 0.0, 10.0))

    assert np.allclose(np.array(got), np.array(expected), atol=1e-6)


def test_gusty_current_variance_is_independent_of_step_size():
    sigma = 0.07
    out = {}
    for dt in (1.0 / 30.0, 1.0 / 400.0):
        f = GustyCurrent(mean=(0, 0, 0), sigma=(sigma,) * 3, tau=2.5, seed=3,
                         seabed_z=None)
        n = int(600.0 / dt)
        acc = np.empty((n, 3))
        for i in range(n):
            f.step(dt)
            acc[i] = f.gust
        out[dt] = acc[int(0.1 * n):].std(axis=0).mean()
    assert out[1 / 30.0] == pytest.approx(sigma, rel=0.1)
    assert out[1 / 400.0] == pytest.approx(sigma, rel=0.1)


def test_gusty_current_boundary_layer_is_on_by_default():
    f = GustyCurrent(mean=(1.0, 0.0, 0.0), sigma=(0.0,) * 3, seabed_z=-0.55)
    f.step(0.01)
    assert at(f, 0.0, 0.0, -0.55)[0] == pytest.approx(0.0)
    assert at(f, 0.0, 0.0, 5.0)[0] == pytest.approx(1.0, rel=1e-5)


def test_gusty_current_stats_track_the_flow():
    f = GustyCurrent(mean=(0.5, 0.0, 0.0), sigma=(0.1, 0.1, 0.0), seed=11)
    for _ in range(200):
        f.step(1.0 / 60.0)
    s = f.stats
    assert s["peak_speed"] >= s["mean_speed"] > 0.0
    assert s["tau"] == f.tau


def test_fields_are_shape_agnostic(jet):
    a = jet(torch.rand(4, 7, 3))
    b = jet(torch.rand(11, 3))
    assert a.shape == (4, 7, 3)
    assert b.shape == (11, 3)
