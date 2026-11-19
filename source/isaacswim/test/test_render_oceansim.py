from __future__ import annotations

import numpy as np
import pytest

from isaacswim import OceanSimWater
from isaacswim.render import (
    ISAAC_UNDERWATER_ANISOTROPY,
    ISAAC_UNDERWATER_SETTINGS,
    OCEAN_ANISOTROPY,
    PETZOLD_SINGLE_SCATTER_ANISOTROPY,
    water_render_settings,
)

WATER = OceanSimWater.jerlov("II")


def test_adapter_imports_without_oceansim_installed():
    import isaacswim.oceansim as a
    assert isinstance(a.available(), bool)
    with pytest.raises(AttributeError):
        _ = a.NotASensor


def test_transmittance_colour_is_beer_lambert_at_the_stated_distance():
    d = 10.0
    s = water_render_settings(WATER, measurement_distance=d)
    t = np.array(s["/rtx/raytracing/inscattering/transmittanceColor"])
    beta = np.asarray(WATER.attenuation) * WATER.scale
    assert np.allclose(t, np.exp(-beta * d), rtol=1e-12)
    assert s["/rtx/raytracing/inscattering/transmittanceMeasurementDistance"] == d


def test_transmittance_keeps_the_per_channel_ordering():
    red_dies = OceanSimWater(attenuation=(0.4, 0.1, 0.05))
    t = np.array(water_render_settings(red_dies)["/rtx/raytracing/inscattering/transmittanceColor"])
    assert t[0] < t[1] < t[2]


def test_scattering_albedo_is_a_fraction():
    for name in ("I", "II", "3", "9"):
        s = water_render_settings(OceanSimWater.jerlov(name))
        a = np.array(s["/rtx/raytracing/inscattering/singleScatteringAlbedo"])
        assert ((a > 0.0) & (a < 1.0)).all(), (name, a)


def test_fog_colour_is_the_veil():
    s = water_render_settings(OceanSimWater(veil="seaclear"))
    assert np.allclose(s["/rtx/fog/fogColor"], OceanSimWater(veil="seaclear").veil)


def test_settings_are_all_real_rtx_keys():
    known = {
        "/rtx/raytracing/inscattering/transmittanceColor",
        "/rtx/raytracing/inscattering/transmittanceMeasurementDistance",
        "/rtx/raytracing/inscattering/singleScatteringAlbedo",
        "/rtx/raytracing/inscattering/anisotropyFactor",
        "/rtx/raytracing/inscattering/densityMult",
        "/rtx/raytracing/inscattering/maxDistance",
        "/rtx/raytracing/inscattering/atmosphereHeight",
        "/rtx/fog/enabled", "/rtx/fog/fogColor", "/rtx/fog/fogColorIntensity",
        "/rtx/fog/fogDistanceDensity", "/rtx/fog/fogStartDist",
        "/rtx/fog/fogEndDist", "/rtx/fog/fogHeightDensity",
        "/rtx/fog/fogHeightFalloff",
        "/rtx/raytracing/globalVolumetricEffects/enabled",
        "/rtx/sceneDb/ambientLightColor",
    }
    assert set(water_render_settings(WATER)) == known


def test_zero_measurement_distance_is_rejected():
    with pytest.raises(ValueError, match="measurement_distance"):
        water_render_settings(WATER, measurement_distance=0.0)


def test_the_global_volumetric_master_switch_is_always_written():
    assert water_render_settings(WATER)["/rtx/raytracing/globalVolumetricEffects/enabled"]


def test_anisotropy_defaults_to_isaac_underwaters_not_petzolds():
    assert OCEAN_ANISOTROPY == ISAAC_UNDERWATER_ANISOTROPY == 0.07
    assert PETZOLD_SINGLE_SCATTER_ANISOTROPY == 0.9
    assert water_render_settings(WATER)["/rtx/raytracing/inscattering/anisotropyFactor"] > 0.0


def test_isaac_underwater_reference_settings_are_verbatim():
    S = ISAAC_UNDERWATER_SETTINGS
    assert S["/rtx/raytracing/globalVolumetricEffects/enabled"] is True
    assert S["/rtx/raytracing/inscattering/transmittanceMeasurementDistance"] == 2000.0
    assert S["/rtx/raytracing/inscattering/densityMult"] == pytest.approx(1.59, abs=1e-6)
    assert S["/rtx/sceneDb/ambientLightColor"] == (0.1, 0.1, 0.1)


def test_ambient_floor_is_lifted_off_black():
    assert min(water_render_settings(WATER)["/rtx/sceneDb/ambientLightColor"]) > 0.0
