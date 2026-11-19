"""Underwater rendering settings for Isaac Sim's renderer."""

from __future__ import annotations

import numpy as np

from isaacswim.oceansim_render import OceanSimWater

__all__ = ["apply_water_rendering", "clear_water_rendering",
           "water_render_settings", "apply_isaac_underwater_look",
           "ISAAC_UNDERWATER_SETTINGS"]

ISAAC_UNDERWATER_ANISOTROPY = 0.07
OCEAN_ANISOTROPY = ISAAC_UNDERWATER_ANISOTROPY
PETZOLD_SINGLE_SCATTER_ANISOTROPY = 0.9

ISAAC_UNDERWATER_SETTINGS = {
    "/rtx/raytracing/globalVolumetricEffects/enabled": True,
    "/rtx/raytracing/inscattering/anisotropyFactor": 0.06999999843537807,
    "/rtx/raytracing/inscattering/densityMult": 1.5899999644607306,
    "/rtx/raytracing/inscattering/singleScatteringAlbedo":
        (0.010575219988822937, 0.02484351582825184, 0.27848100662231445),
    "/rtx/raytracing/inscattering/transmittanceColor":
        (0.006836516782641411, 0.29922208189964294, 0.405063271522522),
    "/rtx/raytracing/inscattering/transmittanceMeasurementDistance": 2000.0,
    "/rtx/sceneDb/ambientLightColor": (0.1, 0.1, 0.1),
}

_TARGET_OPTICAL_DEPTH = 1.0


def water_render_settings(water: OceanSimWater, *,
                          measurement_distance: float | None = None,
                          anisotropy: float = OCEAN_ANISOTROPY,
                          density_mult: float = 1.0,
                          max_distance: float | None = None) -> dict:
    beta = np.asarray(water.attenuation, dtype=np.float64) * water.scale
    if measurement_distance is None:
        d = float(_TARGET_OPTICAL_DEPTH / max(float(np.max(beta)), 1e-6))
    else:
        d = float(measurement_distance)
    if d <= 0.0:
        raise ValueError(f"measurement_distance must be > 0, got {d}")

    transmittance = np.exp(-beta * d)
    if float(np.max(transmittance)) < 1e-6:
        raise ValueError(
            f"transmittanceColor underflowed to {transmittance} at "
            f"measurement_distance={d:g} m.  RTX renders a degenerate volume from "
            "this and the water silently disappears; shorten the distance or leave "
            "it None so it is chosen for you.")
    albedo = np.clip(np.asarray(water.backscatter, dtype=np.float64) * water.scale
                     / np.maximum(beta, 1e-9), 0.05, 0.95)

    veil = np.asarray(water.veil, dtype=np.float64)
    vis = water.visibility_range()
    far = float(max_distance) if max_distance is not None else float(min(vis * 10.0, 5.0e4))

    return {
        "/rtx/raytracing/globalVolumetricEffects/enabled": True,
        "/rtx/raytracing/inscattering/transmittanceColor": tuple(transmittance),
        "/rtx/raytracing/inscattering/transmittanceMeasurementDistance": d,
        "/rtx/raytracing/inscattering/singleScatteringAlbedo": tuple(albedo),
        "/rtx/raytracing/inscattering/anisotropyFactor": float(anisotropy),
        "/rtx/raytracing/inscattering/densityMult": float(density_mult),
        "/rtx/raytracing/inscattering/maxDistance": far,
        "/rtx/raytracing/inscattering/atmosphereHeight": far,
        "/rtx/fog/enabled": True,
        "/rtx/fog/fogColor": tuple(np.clip(veil, 0.0, 1.0)),
        "/rtx/fog/fogColorIntensity": 1.0,
        "/rtx/fog/fogDistanceDensity": float(np.mean(beta)),
        "/rtx/fog/fogStartDist": 0.0,
        "/rtx/fog/fogEndDist": far,
        "/rtx/fog/fogHeightDensity": 0.0,
        "/rtx/fog/fogHeightFalloff": 0.0,
        "/rtx/sceneDb/ambientLightColor": (0.1, 0.1, 0.1),
    }


def apply_water_rendering(water: OceanSimWater, **kwargs) -> dict:
    settings = water_render_settings(water, **kwargs)
    _write(settings)
    return settings


def clear_water_rendering() -> None:
    _write({
        "/rtx/fog/enabled": False,
        "/rtx/raytracing/globalVolumetricEffects/enabled": False,
        "/rtx/raytracing/inscattering/densityMult": 0.0,
        "/rtx/raytracing/inscattering/anisotropyFactor": 0.0,
    })


def apply_isaac_underwater_look() -> dict:
    _write(ISAAC_UNDERWATER_SETTINGS)
    return dict(ISAAC_UNDERWATER_SETTINGS)


def _write(settings: dict) -> None:
    try:
        import carb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "isaacswim.render drives Isaac Sim's renderer and needs `carb`, which "
            "only exists inside a launched Isaac Sim app.  For a post-process that "
            "runs anywhere, use isaacswim.apply_underwater instead."
        ) from exc
    s = carb.settings.get_settings()
    for key, value in settings.items():
        if isinstance(value, tuple):
            s.set(key, list(value))
        elif isinstance(value, bool):
            s.set_bool(key, value)
        elif isinstance(value, float):
            s.set_float(key, value)
        else:
            s.set(key, value)
