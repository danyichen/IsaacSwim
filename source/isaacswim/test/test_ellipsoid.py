from __future__ import annotations

import numpy as np
import pytest

from isaacswim.ellipsoid import added_mass, carlson_rd, ellipsoid_volume, lamb_coefficients

MUJOCO_GOLDEN = [
    ([0.0959639200583177, 0.09596308372644559, 0.18976092399377215],
     [0.005132079633595246, 0.005132164437489867, 0.0015602419548689833],
     [1.5435156434868044e-05, 1.5434726914675597e-05, 1.7401939249456872e-15]),
    ([0.015290998495576164, 0.022823444248937348, 0.022823360454260807],
     [2.669788074470522e-05, 1.2830625160789058e-05, 1.2830707048917416e-05],
     [5.017485947469669e-20, 5.687958775979204e-10, 5.688075666709942e-10]),
    ([0.3, 0.3, 0.3],
     [0.05654863, 0.05654863, 0.05654863],
     [0.0, 0.0, 0.0]),
    ([0.1, 0.1, 0.5],
     [0.01872935, 0.01872935, 0.00123823],
     [0.0007622, 0.0007622, 0.0]),
    ([0.1, 0.2, 0.3],
     [0.03421892, 0.00916197, 0.004656],
     [3.91419015e-05, 3.55998376e-04, 9.63256192e-05]),
]

MUJOCO_TOL = 2e-4


def test_lamb_coefficients_sum_to_two():
    rng = np.random.default_rng(0)
    s = 10.0 ** rng.uniform(-3, 1, size=(500, 3))
    total = lamb_coefficients(s).sum(-1)
    assert np.allclose(total, 2.0, atol=1e-12)


def test_sphere_added_mass_is_half_the_displaced_volume():
    vm, vi = added_mass([0.25, 0.25, 0.25])
    assert np.allclose(vm, 0.5 * ellipsoid_volume([0.25, 0.25, 0.25]), rtol=1e-12)
    assert np.allclose(vi, 0.0, atol=1e-18)


def test_spheroid_has_no_added_inertia_about_its_symmetry_axis():
    for semi, axis in (([0.1, 0.1, 0.4], 2), ([0.4, 0.1, 0.1], 0), ([0.1, 0.4, 0.1], 1)):
        _, vi = added_mass(semi)
        assert vi[axis] == pytest.approx(0.0, abs=1e-18)
        assert np.all(np.delete(vi, axis) > 0.0)


def test_carlson_matches_independent_quadrature():
    x, w = np.polynomial.legendre.leggauss(4000)

    def alpha_quad(s):
        a, b, c = s
        t = 0.5 * (x + 1.0)
        wt = 0.5 * w
        lam = t / (1.0 - t)
        jac = 1.0 / (1.0 - t) ** 2
        d = np.sqrt((a * a + lam) * (b * b + lam) * (c * c + lam))
        return np.array([a * b * c * np.sum(wt * jac / ((q + lam) * d))
                         for q in (a * a, b * b, c * c)])

    rng = np.random.default_rng(1)
    for _ in range(20):
        s = 10.0 ** rng.uniform(-2, 0, 3)
        assert np.allclose(lamb_coefficients(s), alpha_quad(s), atol=1e-8)


def test_carlson_is_scale_covariant():
    s = np.array([0.03, 0.11, 0.4])
    base = lamb_coefficients(s)
    for k in (1e-3, 1e-1, 10.0, 1e3):
        assert np.allclose(lamb_coefficients(s * k), base, atol=1e-12)


@pytest.mark.parametrize("semi,vm_ref,vi_ref", MUJOCO_GOLDEN)
def test_matches_mujoco_golden_values(semi, vm_ref, vi_ref):
    vm, vi = added_mass(semi)
    assert np.allclose(vm, vm_ref, rtol=MUJOCO_TOL)
    scale = max(np.max(np.abs(vi_ref)), 1e-12)
    assert np.max(np.abs(vi - np.asarray(vi_ref))) < MUJOCO_TOL * scale


def test_textbook_scaling_differs_by_the_documented_factor():
    semi = np.array([0.1, 0.2, 0.3])
    _, vi_mj = added_mass(semi, mujoco_scaling=True)
    _, vi_tb = added_mass(semi, mujoco_scaling=False)
    a2, b2, c2 = semi ** 2
    factor = np.array([b2 + c2, c2 + a2, a2 + b2])
    assert np.allclose(vi_tb, vi_mj * factor, rtol=1e-12)


def test_rejects_degenerate_shapes():
    with pytest.raises(ValueError):
        added_mass([0.1, 0.1, 0.0])
    with pytest.raises(ValueError):
        added_mass([0.1, -0.1, 0.2])
    with pytest.raises(ValueError):
        carlson_rd(1.0, 1.0, 0.0)


def test_vectorises_over_leading_dimensions():
    s = np.array([[0.1, 0.2, 0.3], [0.4, 0.4, 0.4], [0.05, 0.5, 0.2]])
    vm, vi = added_mass(s)
    assert vm.shape == vi.shape == (3, 3)
    for i in range(3):
        vm_i, vi_i = added_mass(s[i])
        assert np.allclose(vm[i], vm_i) and np.allclose(vi[i], vi_i)
