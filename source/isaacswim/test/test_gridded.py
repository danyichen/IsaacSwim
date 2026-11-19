from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isaacswim.flow import Composite, Gridded, Uniform  # noqa: E402

X = np.array([-2.0, -1.0, 0.5, 3.0])
Y = np.array([0.0, 1.0, 2.0])
Z = np.array([-30.0, -12.0, -5.0, -1.0, 0.0])


@pytest.mark.parametrize("x,times", [([float("nan")], None), ([0], [float("inf")])])
def test_nonfinite_grid_coordinates_are_rejected(x, times):
    with pytest.raises(ValueError, match="finite"):
        Gridded(np.zeros((1, 1, 1, 3)), x, [0], [0], times=times)


def test_nonfinite_current_samples_are_treated_as_missing():
    field = Gridded(np.full((1, 1, 1, 3), np.inf), [0], [0], [0])
    assert torch.equal(field(torch.zeros(2, 3)), torch.zeros(2, 3))


def linear(x, y, z, t=0.0):
    return np.stack([0.1 + 0.2 * x - 0.05 * z + 0.01 * t,
                     -0.3 * y + 0.02 * x * 0 + 0.0 * t,
                     0.01 * z + 0.002 * t], -1)


def grid_of(fn, t=0.0):
    G = np.stack(np.meshgrid(X, Y, Z, indexing="ij"), -1)
    return fn(G[..., 0], G[..., 1], G[..., 2], t)


def random_points(rng, n):
    return np.stack([rng.uniform(X[0], X[-1], n), rng.uniform(Y[0], Y[-1], n),
                     rng.uniform(Z[0], Z[-1], n)], 1)


def test_trilinear_is_exact_on_an_uneven_grid():
    f = Gridded(grid_of(linear), X, Y, Z)
    p = random_points(np.random.default_rng(0), 500)
    got = f(torch.tensor(p, dtype=torch.float32)).numpy()
    assert np.allclose(got, linear(p[:, 0], p[:, 1], p[:, 2]), atol=2e-6)


def test_samples_are_returned_exactly_at_the_nodes():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(len(X), len(Y), len(Z), 3)).astype(np.float32)
    f = Gridded(v, X, Y, Z)
    G = np.stack(np.meshgrid(X, Y, Z, indexing="ij"), -1).reshape(-1, 3)
    got = f(torch.tensor(G, dtype=torch.float64)).numpy()
    assert np.allclose(got, v.reshape(-1, 3), atol=1e-6)


def test_time_series_interpolates_holds_and_loops():
    frames = np.stack([grid_of(linear, t) for t in (0.0, 10.0, 20.0)])
    p = torch.tensor([[0.0, 1.0, -3.0]])
    f = Gridded(frames, X, Y, Z, times=[0.0, 10.0, 20.0])
    f.step(14.0)
    assert np.allclose(f(p).numpy(), linear(0.0, 1.0, -3.0, 14.0), atol=1e-6)
    f.step(100.0)
    assert np.allclose(f(p).numpy(), linear(0.0, 1.0, -3.0, 20.0), atol=1e-6)

    g = Gridded(frames, X, Y, Z, times=[0.0, 10.0, 20.0], loop=True, t0=30.0)
    assert np.allclose(g(p).numpy(), linear(0.0, 1.0, -3.0, 0.0), atol=1e-6)
    g.step(5.0)
    assert np.allclose(g(p).numpy(), linear(0.0, 1.0, -3.0, 5.0), atol=1e-6)
    h = Gridded(frames, X, Y, Z, times=[0.0, 10.0, 20.0], loop=True, t0=25.0)
    mid = 0.5 * (linear(0.0, 1.0, -3.0, 20.0) + linear(0.0, 1.0, -3.0, 0.0))
    assert np.allclose(h(p).numpy(), mid, atol=1e-6)


def test_outside_the_grid():
    v = np.ones((len(X), len(Y), len(Z), 3), dtype=np.float32)
    out = torch.tensor([[10.0, 1.0, -3.0], [0.0, 1.0, 5.0]])
    assert torch.equal(Gridded(v, X, Y, Z)(out), torch.zeros(2, 3))
    assert torch.allclose(Gridded(v, X, Y, Z, outside="clamp")(out), torch.ones(2, 3))


def test_land_reads_as_still_water():
    v = np.ones((len(X), len(Y), len(Z), 3), dtype=np.float32)
    v[0] = np.nan
    f = Gridded(v, X, Y, Z)
    assert torch.equal(f(torch.tensor([[X[0], 1.0, -3.0]])), torch.zeros(1, 3))
    assert not torch.isnan(f(torch.tensor([[-1.5, 1.0, -3.0]]))).any()


def test_a_single_layer_is_constant_in_that_axis():
    v = grid_of(linear)[:, :, -1:]
    f = Gridded(v, X, Y, [0.0])
    a = f(torch.tensor([[1.0, 1.0, -100.0], [1.0, 1.0, 50.0]]))
    assert torch.allclose(a[0], a[1])


def test_shape_agnostic_and_composes():
    f = Gridded(grid_of(linear), X, Y, Z)
    assert f(torch.zeros(4, 7, 3)).shape == (4, 7, 3)
    both = Composite(f, Uniform((1.0, 0.0, 0.0)))
    p = torch.tensor([[0.0, 1.0, -2.0]])
    assert torch.allclose(both(p), f(p) + torch.tensor([1.0, 0.0, 0.0]))


def test_rejects_malformed_grids():
    v = grid_of(linear)
    with pytest.raises(ValueError, match="increasing"):
        Gridded(v, X[::-1], Y, Z)
    with pytest.raises(ValueError, match="samples"):
        Gridded(v, X[:-1], Y, Z)
    with pytest.raises(ValueError, match="times"):
        Gridded(np.stack([v, v]), X, Y, Z)


def test_from_scattered_recovers_a_linear_field():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(2)
    pts = random_points(rng, 3000)
    f = Gridded.from_scattered(pts, linear(*pts.T), spacing=0.5,
                               bounds=((-1.5, 0.2, -25.0), (2.5, 1.8, -0.5)))
    q = np.stack([rng.uniform(-1.4, 2.4, 200), rng.uniform(0.3, 1.7, 200),
                  rng.uniform(-24.0, -1.0, 200)], 1)
    got = f(torch.tensor(q, dtype=torch.float64)).numpy()
    assert np.allclose(got, linear(*q.T), atol=1e-4)


def test_from_dataset_reads_an_ocean_model_layout():
    xr = pytest.importorskip("xarray")
    lon = np.linspace(-70.70, -70.68, 5)
    lat = np.linspace(41.52, 41.53, 4)
    depth = np.array([0.5, 2.0, 6.0, 15.0])
    time = np.array(["2026-10-01T00", "2026-10-01T01"], dtype="datetime64[ns]")
    shape = (2, 4, 4, 5)
    uo = np.zeros(shape, np.float32)
    uo += np.array([0.1, 0.3])[:, None, None, None]
    vo = np.zeros(shape, np.float32) - np.asarray(depth)[None, :, None, None] * 0.01
    ds = xr.Dataset({"uo": (("time", "depth", "latitude", "longitude"), uo),
                     "vo": (("time", "depth", "latitude", "longitude"), vo)},
                    coords={"time": time, "depth": depth, "latitude": lat,
                            "longitude": lon})
    f = Gridded.from_dataset(ds, origin_lonlat=(-70.69, 41.525))
    assert list(f.times) == [0.0, 3600.0]
    p = torch.tensor([[0.0, 0.0, -6.0]], dtype=torch.float64)
    assert np.allclose(f(p).numpy(), [[0.1, -0.06, 0.0]], atol=1e-6)
    f.step(1800.0)
    assert np.allclose(f(p).numpy(), [[0.2, -0.06, 0.0]], atol=1e-6)
    east = 0.01 * np.pi / 180 * 6371008.8 * np.cos(np.radians(41.525))
    assert f.axes[0][-1] == pytest.approx(east, rel=1e-9)
    near = torch.tensor([[0.0, 0.0, -0.1]], dtype=torch.float64)
    assert np.allclose(f(near).numpy(), [[0.2, -0.005, 0.0]], atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_runs_on_the_gpu():
    f = Gridded(grid_of(linear), X, Y, Z)
    p = random_points(np.random.default_rng(3), 64)
    g = f(torch.tensor(p, dtype=torch.float32, device="cuda"))
    assert g.device.type == "cuda"
    assert np.allclose(g.cpu().numpy(), linear(*p.T), atol=2e-6)
