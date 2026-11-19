from __future__ import annotations

import pytest
import torch

from isaacswim.sensors import Altimeter, DopplerVelocityLog, PressureDepth


def gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_depth_is_measured_down_from_the_surface():
    s = PressureDepth(surface_height=0.0, sigma=0.0)
    pos = torch.tensor([[0., 0., -10.], [0., 0., -2.5], [0., 0., 0.]])
    assert torch.allclose(s(pos), torch.tensor([10.0, 2.5, 0.0]), atol=1e-6)


def test_a_vehicle_in_the_air_reads_zero_not_negative():
    s = PressureDepth(sigma=0.0)
    assert s(torch.tensor([[0., 0., 3.0]])).item() == pytest.approx(0.0)


def test_a_nonzero_surface_height_shifts_the_datum():
    s = PressureDepth(surface_height=5.0, sigma=0.0)
    assert s(torch.tensor([[0., 0., 1.0]])).item() == pytest.approx(4.0)


def test_fresh_water_calibration_reads_shallow_at_sea():
    exact = PressureDepth(density=1025.0, assumed_density=None, sigma=0.0)
    miscal = PressureDepth(density=1025.0, assumed_density=997.0, sigma=0.0)
    pos = torch.tensor([[0., 0., -10.]])
    assert exact(pos).item() == pytest.approx(10.0)
    assert miscal(pos).item() == pytest.approx(10.0 * 1025.0 / 997.0, rel=1e-6)
    assert miscal(pos).item() - exact(pos).item() > 0.25


def test_noise_and_quantisation_are_applied():
    noisy = PressureDepth(sigma=0.05, resolution=0.0)
    pos = torch.zeros(512, 3)
    pos[:, 2] = -8.0
    d = noisy(pos, generator=gen())
    assert d.std().item() > 0.01
    assert abs(d.mean().item() - 8.0) < 0.02

    quant = PressureDepth(sigma=0.0, resolution=0.25)
    pos = torch.tensor([[0., 0., -3.31]])
    assert quant(pos).item() == pytest.approx(3.25, abs=1e-6)


def test_dvl_reports_velocity_over_ground_and_loses_lock_out_of_range():
    dvl = DopplerVelocityLog(max_altitude=30.0, sigma=0.0, scale_error=0.0)
    v = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    alt = torch.tensor([10.0, 60.0])
    out, valid = dvl(v, alt)
    assert valid.tolist() == [True, False]
    assert out[0].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert out[1].tolist() == [0.0, 0.0, 0.0], "no bottom lock must not fake a reading"


def test_dvl_scale_error_grows_with_speed():
    dvl = DopplerVelocityLog(sigma=0.0, scale_error=0.01)
    alt = torch.tensor([5.0])
    slow, _ = dvl(torch.tensor([[0.1, 0., 0.]]), alt)
    fast, _ = dvl(torch.tensor([[2.0, 0., 0.]]), alt)
    assert slow[0, 0].item() - 0.1 == pytest.approx(0.001, rel=1e-5)
    assert fast[0, 0].item() - 2.0 == pytest.approx(0.020, rel=1e-5)


def test_water_track_measures_through_the_water_not_over_the_ground():
    over_ground = DopplerVelocityLog(sigma=0.0, scale_error=0.0)
    through_water = DopplerVelocityLog(sigma=0.0, scale_error=0.0, water_track=True)
    v = torch.tensor([[1.0, 0.0, 0.0]])
    current = torch.tensor([[0.3, 0.0, 0.0]])
    alt = torch.tensor([5.0])
    g, _ = over_ground(v, alt)
    w, _ = through_water(v, alt, current_b=current)
    assert (g - w)[0].tolist() == pytest.approx([0.3, 0.0, 0.0])


def test_water_track_without_a_current_is_an_error():
    dvl = DopplerVelocityLog(water_track=True)
    with pytest.raises(ValueError, match="current_b"):
        dvl(torch.zeros(1, 3), torch.tensor([5.0]))


def test_altimeter_blanks_close_in_and_drops_out_far_off():
    a = Altimeter(max_range=50.0, min_range=0.3, sigma=0.0)
    alt = torch.tensor([0.1, 1.0, 49.0, 80.0])
    out, valid = a(alt)
    assert valid.tolist() == [False, True, True, False]
    assert out[1].item() == pytest.approx(1.0)
    assert out[0].item() == 0.0, "inside the blanking distance there is no reading"


def test_altimeter_never_returns_negative_range():
    a = Altimeter(sigma=0.5)
    out, _ = a(torch.full((256,), 0.35), generator=gen())
    assert (out >= 0).all()


def test_every_sensor_is_batched_over_environments():
    E = 64
    pos = torch.randn(E, 3)
    alt = torch.rand(E) * 40.0
    v = torch.randn(E, 3)
    assert PressureDepth()(pos, generator=gen()).shape == (E,)
    d, valid = DopplerVelocityLog()(v, alt, generator=gen())
    assert d.shape == (E, 3) and valid.shape == (E,)
    r, valid = Altimeter()(alt, generator=gen())
    assert r.shape == (E,) and valid.shape == (E,)
