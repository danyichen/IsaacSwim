from __future__ import annotations

import numpy as np
import pytest
import torch

from isaacswim.oceansim_render import (
    CAUSTICS_DEFAULT,
    JERLOV_ATTEN,
    JERLOV_BACKSCATTER,
    OCEANSIM_DEFAULT,
    VEILS,
    OceanSimWater,
    apply_underwater,
)


def test_defaults_match_oceansims_colour_picker():
    assert OCEANSIM_DEFAULT["veil"] == (0.0, 0.31, 0.24)
    assert OCEANSIM_DEFAULT["backscatter"] == (0.05, 0.05, 0.2)
    assert OCEANSIM_DEFAULT["attenuation"] == (0.05, 0.05, 0.05)


def test_caustics_defaults_match_their_ui():
    assert CAUSTICS_DEFAULT["blend_weight"] == 0.5
    assert CAUSTICS_DEFAULT["depth_max"] == 100.0
    assert CAUSTICS_DEFAULT["time_speed"] == 2.0


def test_their_attenuation_is_one_and_a_half_times_their_backscatter():
    for k in JERLOV_BACKSCATTER:
        assert JERLOV_ATTEN[k] == pytest.approx(
            tuple(1.5 * c for c in JERLOV_BACKSCATTER[k]))


def test_their_attenuation_is_near_neutral():
    for k, v in JERLOV_BACKSCATTER.items():
        a = np.asarray(v)
        assert a.max() / a.min() < 2.0, f"Jerlov {k} is not near-neutral"
    assert np.ptp(np.asarray(OCEANSIM_DEFAULT["attenuation"])) == 0.0


def test_veiling_colours_are_displayable_not_radiometric():
    for name, v in VEILS.items():
        a = np.asarray(v)
        assert a.min() >= 0.0 and a.max() <= 1.0, name
        assert 0.2 <= a.max() <= 0.75, f"{name} veil peak {a.max()} is not photographic"


def test_jerlov_constructor_uses_their_table():
    w = OceanSimWater.jerlov("II", veil="seaclear")
    assert w.attenuation == JERLOV_BACKSCATTER["II"]
    assert w.backscatter == JERLOV_ATTEN["II"]
    assert w.veil == VEILS["seaclear"]
    assert OceanSimWater.jerlov("Type II").attenuation == w.attenuation


def test_jerlov_defaults_to_half_scale_because_their_raw_values_are_murky():
    assert OceanSimWater.jerlov("II").scale == 0.5
    assert OceanSimWater(attenuation=JERLOV_BACKSCATTER["II"],
                         scale=1.0).visibility_range() < 5.0
    assert OceanSimWater.jerlov("II").visibility_range() > 7.0


def test_the_default_water_is_clear_enough_to_see_a_swimmer():
    w = OceanSimWater()
    assert w.visibility_range() > 50.0


def test_uw_param_is_the_nine_vector_oceansim_takes():
    w = OceanSimWater(veil="akdeniz")
    p = w.to_uw_param()
    assert p.shape == (9,)
    assert tuple(p[0:3]) == VEILS["akdeniz"]
    assert tuple(p[3:6]) == w.attenuation
    assert tuple(p[6:9]) == w.backscatter


def test_uw_param_folds_in_the_scale_and_round_trips():
    w = OceanSimWater.jerlov("3", veil="murky", scale=0.5)
    p = w.to_uw_param()
    assert p[3:6] == pytest.approx(0.5 * np.asarray(w.attenuation))
    back = OceanSimWater.from_uw_param(p)
    assert back.visibility_range() == pytest.approx(w.visibility_range())
    with pytest.raises(ValueError, match="9 entries"):
        OceanSimWater.from_uw_param([0.1, 0.2, 0.3])


def test_unknown_names_raise_rather_than_defaulting():
    with pytest.raises(KeyError, match="veil"):
        OceanSimWater(veil="not_a_water")
    with pytest.raises(KeyError, match="Jerlov"):
        OceanSimWater.jerlov("XVII")


def test_scale_multiplies_extinction_so_visibility_moves_inversely():
    a = OceanSimWater(scale=0.5).visibility_range()
    b = OceanSimWater(scale=1.0).visibility_range()
    assert a == pytest.approx(2.0 * b)


def test_module_has_no_undefined_names():
    pyflakes = pytest.importorskip("pyflakes.api")
    import io

    from pyflakes.reporter import Reporter

    import isaacswim.oceansim_render as m

    err, warn = io.StringIO(), io.StringIO()
    pyflakes.checkPath(m.__file__, Reporter(warn, err))
    bad = [ln for ln in (warn.getvalue() + err.getvalue()).splitlines()
           if "undefined name" in ln]
    assert not bad, chr(10).join(bad)


def test_zero_range_is_the_raw_image_and_far_range_is_the_veil():
    w = OceanSimWater.jerlov("II")
    rgb = torch.rand(2, 4, 5, 3)
    near = apply_underwater(rgb, torch.zeros(2, 4, 5), w)
    far = apply_underwater(rgb, torch.full((2, 4, 5), 1e4), w)
    assert torch.allclose(near, rgb)
    assert torch.allclose(far, torch.tensor(w.veil).expand_as(far), atol=1e-6)


def test_it_is_oceansims_formula():
    w = OceanSimWater(veil="seaclear", scale=0.7)
    rgb, z = torch.rand(3, 3, 3), torch.rand(3, 3) * 20
    a, b, v = (torch.tensor(x) for x in (w.attenuation, w.backscatter, w.veil))
    ref = rgb * torch.exp(-z[..., None] * a * 0.7) + v * (1 - torch.exp(-z[..., None] * b * 0.7))
    assert torch.allclose(apply_underwater(rgb, z, w), ref)


def test_uint8_in_uint8_out_and_no_light_from_a_miss():
    w = OceanSimWater()
    rgb = torch.full((4, 4, 3), 200, dtype=torch.uint8)
    z = torch.full((4, 4), float("inf"))
    out = apply_underwater(rgb, z, w)
    assert out.dtype == torch.uint8
    assert torch.equal(out[0, 0], (torch.tensor(w.veil) * 255).to(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_underwater_camera_runs_the_model_on_the_gpu_and_hands_back_on_the_cpu():
    from types import SimpleNamespace

    from isaacswim.isaaclab.camera import UnderwaterCamera

    g = torch.Generator().manual_seed(0)
    rgb = torch.randint(0, 256, (2, 8, 8, 4), dtype=torch.uint8, generator=g)
    rng = torch.rand(2, 8, 8, 1, generator=g) * 20.0
    cam = SimpleNamespace(data=SimpleNamespace(output={"rgb": rgb, "depth": rng}))
    gpu = UnderwaterCamera(cam, range_key="depth").rgb()
    cpu = UnderwaterCamera(cam, range_key="depth", model_device="cpu").rgb()
    assert gpu.device.type == "cpu" and gpu.dtype == cpu.dtype
    assert (gpu.int() - cpu.int()).abs().max() <= 1
    assert torch.equal(gpu[..., 3], rgb[..., 3])
