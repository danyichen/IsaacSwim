from __future__ import annotations

import numpy as np
import pytest

from isaacswim.trim import TrimVolume, displaced_volume, solve_trim

FOAM = TrimVolume((0.170, 0.150, 0.030), z=0.118, max_volume=2.0e-3)
MAKEUP = TrimVolume((0.215, 0.275, 0.045), z=-0.030)


def original(geom_volume, geom_cob, total_mass, com, density, ballast=None,
             level_trim=True):
    want_volume = total_mass / density
    deficit = want_volume - geom_volume
    foam_volume = min(2.0e-3, deficit)
    makeup_volume = deficit - foam_volume
    makeup_moment = np.zeros(3)
    if makeup_volume > 0.0:
        makeup_moment = np.array((0.0, 0.0, -0.030), dtype=float) * makeup_volume
    fixed_moment = geom_cob[:2] * geom_volume + makeup_moment[:2]
    if ballast is not None:
        foam_xy = np.asarray(ballast, dtype=float)[:2]
    elif level_trim:
        foam_xy = (com[:2] * want_volume - fixed_moment) / foam_volume
    else:
        foam_xy = np.zeros(2)
    foam_pos = np.array((foam_xy[0], foam_xy[1], 0.118), dtype=float)
    cob = (geom_cob * geom_volume + makeup_moment + foam_pos * foam_volume) / want_volume
    return cob, foam_volume, makeup_volume, foam_pos


def test_matches_the_original_build_hydro_arithmetic():
    rng = np.random.default_rng(0)
    for _ in range(200):
        geom_volume = rng.uniform(8e-3, 12e-3)
        geom_cob = rng.normal(scale=0.02, size=3)
        total_mass = rng.uniform(14.0, 15.0)
        com = rng.normal(scale=0.01, size=3)
        _, info = solve_trim(geom_volume=geom_volume, geom_cob=geom_cob,
                             total_mass=total_mass, com=com, density=1025.0,
                             foam=FOAM, makeup=MAKEUP)
        cob, fv, mv, fpos = original(geom_volume, geom_cob, total_mass, com, 1025.0)
        assert np.array_equal(info["cob"], cob)
        assert info["foam_volume"] == fv and info["makeup_volume"] == mv
        assert np.array_equal(info["foam_pos"], fpos)


def test_trimmed_vehicle_is_neutral_and_level():
    shapes, info = solve_trim(geom_volume=10.008e-3, geom_cob=[0.01, -0.004, 0.048],
                              total_mass=14.478, com=[0.012, 0.0, 0.0076],
                              density=1025.0, foam=FOAM, makeup=MAKEUP)
    total = 10.008e-3 + sum(s.volume for s in shapes)
    assert total * 1025.0 == pytest.approx(14.478, rel=1e-12), "not neutral"
    assert np.allclose(info["bg"][:2], 0.0, atol=1e-12), "not level"
    assert info["bg"][2] > 0.0, "CoB below CoM: the vehicle would capsize"


def test_held_ballast_is_used_verbatim():
    _, first = solve_trim(geom_volume=10e-3, geom_cob=[0, 0, 0.05], total_mass=14.5,
                          com=[0.01, 0.0, 0.0], density=1025.0, foam=FOAM, makeup=MAKEUP)
    _, second = solve_trim(geom_volume=10e-3, geom_cob=[0, 0, 0.05], total_mass=14.5,
                           com=[0.03, 0.0, 0.0], density=1025.0, foam=FOAM,
                           makeup=MAKEUP, ballast=first["ballast"])
    assert np.array_equal(second["foam_pos"][:2], first["foam_pos"][:2])
    assert second["bg"][0] < 0.0, "the CoM moved forward of an unmoved CoB"


def test_uncapped_foam_takes_the_whole_deficit():
    shapes, info = solve_trim(geom_volume=10e-3, geom_cob=[0, 0, 0], total_mass=14.5,
                              com=[0, 0, 0], density=1025.0,
                              foam=TrimVolume((0.1, 0.1, 0.02), z=0.1))
    assert info["makeup_volume"] == 0.0 and len(shapes) == 1


def test_a_vehicle_that_floats_needs_weight_not_foam():
    with pytest.raises(ValueError, match="no room for foam"):
        solve_trim(geom_volume=20e-3, geom_cob=[0, 0, 0], total_mass=14.5,
                   com=[0, 0, 0], density=1025.0, foam=FOAM, makeup=MAKEUP)


def test_capped_foam_without_makeup_is_an_error():
    with pytest.raises(ValueError, match="no makeup"):
        solve_trim(geom_volume=10e-3, geom_cob=[0, 0, 0], total_mass=14.5,
                   com=[0, 0, 0], density=1025.0, foam=FOAM)


def test_displaced_volume_is_the_volume_weighted_centroid():
    v, c = displaced_volume([1.0, 3.0], [[0, 0, 0], [4, 0, 0]])
    assert v == 4.0 and np.allclose(c, [3.0, 0.0, 0.0])


def test_trim_volumes_are_buoyancy_only_unless_asked():
    kw = dict(geom_volume=9e-3, geom_cob=np.zeros(3), total_mass=14.5,
              com=np.zeros(3), density=1025.0, foam=FOAM, makeup=MAKEUP)
    shapes, _ = solve_trim(**kw)
    assert shapes and all(s.volume > 0 for s in shapes)
    for s in shapes:
        assert not np.any(s.coef) and not np.any(s.virtual_mass) \
            and not np.any(s.virtual_inertia)
    with_drag, _ = solve_trim(**{**kw, "foam": TrimVolume(FOAM.semiaxes, FOAM.z,
                                                          FOAM.max_volume, drag=True)})
    assert np.any(with_drag[-1].coef) and np.any(with_drag[-1].virtual_mass)
