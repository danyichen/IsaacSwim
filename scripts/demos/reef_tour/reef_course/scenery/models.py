from __future__ import annotations

import math
from pathlib import Path

import numpy as np

FLARE_ANGLE = math.radians(35.0)


def preview_material(stage, path, *, color=(0.5, 0.5, 0.5), roughness=0.6, metallic=0.0,
                     emissive=None, diffuse_tex=None, normal_tex=None, uv_scale=1.0):
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if emissive is not None:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(tuple(emissive))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    if diffuse_tex is None and normal_tex is None:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tuple(color))
        return mat
    reader = UsdShade.Shader.Define(stage, f"{path}/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    scale = UsdShade.Shader.Define(stage, f"{path}/st_scale")
    scale.CreateIdAttr("UsdTransform2d")
    scale.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result")
    scale.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set((uv_scale, uv_scale))

    def tex(name, file, *, raw=False):
        t = UsdShade.Shader.Define(stage, f"{path}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(file))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            scale.ConnectableAPI(), "result")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        if raw:
            t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        return t

    if diffuse_tex is not None:
        t = tex("albedo", diffuse_tex)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tuple(color))
    if normal_tex is not None:
        t = tex("normal", normal_tex, raw=True)
        t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2.0, 2.0, 2.0, 1.0))
        t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1.0, -1.0, -1.0, 0.0))
        sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    return mat


def bind(prim, mat):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _smooth_noise(n, cells, rng):
    lattice = rng.uniform(-1, 1, (cells, cells))
    x = np.arange(n) / n * cells
    i0 = np.floor(x).astype(int)
    f = x - i0
    w = f * f * (3 - 2 * f)
    a, b = i0 % cells, (i0 + 1) % cells
    rows = lattice[a] * (1 - w)[:, None] + lattice[b] * w[:, None]
    return rows[:, a] * (1 - w)[None] + rows[:, b] * w[None]


def fbm(n, rng, octaves=5, base=4):
    out, amp, total = np.zeros((n, n)), 1.0, 0.0
    for k in range(octaves):
        out += amp * _smooth_noise(n, base * 2 ** k, rng)
        total += amp
        amp *= 0.5
    return out / total


def seabed_textures(texture_dir, *, size=1024, seed=0):
    from PIL import Image

    d = Path(texture_dir)
    d.mkdir(parents=True, exist_ok=True)
    albedo, normal = d / f"sand_albedo_v2_{seed}.png", d / f"sand_normal_{seed}.png"
    if albedo.exists() and normal.exists():
        return albedo, normal
    rng = np.random.default_rng(seed)
    patches = fbm(size, rng, octaves=4, base=3)
    grain = rng.normal(0, 1, (size, size))
    base = np.array([0.47, 0.42, 0.33])
    tint = base[None, None] * (1 + 0.22 * patches[..., None]) * (1 + 0.08 * grain[..., None])
    flecks = rng.random((size, size))
    tint[flecks > 0.997] *= 0.45
    tint[flecks < 0.0015] = tint[flecks < 0.0015] * 0.3 + 0.62
    Image.fromarray((np.clip(tint, 0, 1) * 255).astype(np.uint8)).save(albedo)
    y, x = np.mgrid[0:size, 0:size] / size
    phase = 2 * math.pi * (10 * x + 0.6 * _smooth_noise(size, 4, rng))
    h = np.sin(phase) ** 2 + 0.25 * fbm(size, rng, octaves=3, base=16)
    gy, gx = np.gradient(h)
    k = 6.0
    nrm = np.stack([-k * gx, -k * gy, np.ones_like(h)], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    Image.fromarray(((nrm * 0.5 + 0.5) * 255).astype(np.uint8)).save(normal)
    return albedo, normal


def seabed_height(x, y, *, seed=0, relief=0.35, flat_radius=3.0, blend=3.0):
    rng = np.random.default_rng(seed)
    h = np.zeros_like(x, dtype=np.float64)
    for k in range(5):
        f = 0.12 * 2 ** k
        a = relief * 0.5 ** k
        th, ph = rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)
        h += a * np.sin(f * 2 * math.pi * (x * math.cos(th) + y * math.sin(th)) + ph)
    r = np.hypot(x, y)
    t = np.clip((r - flat_radius) / blend, 0, 1)
    return h * t * t * (3 - 2 * t)


def build_seabed(stage, path, *, z0, texture_dir, half=15.0, step=0.1, seed=0,
                 flat_radius=3.0, tile=2.0):
    from pxr import Gf, Sdf, UsdGeom, Vt

    n = int(round(2 * half / step)) + 1
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    Z = z0 + seabed_height(X, Y, seed=seed, flat_radius=flat_radius)
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    idx = np.arange(n * n).reshape(n, n)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(quads)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(quads.reshape(-1).tolist()))
    mesh.CreateSubdivisionSchemeAttr("none")
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy((pts[:, :2] / tile).astype(np.float32)))
    mesh.CreateExtentAttr([Gf.Vec3f(*pts.min(0)), Gf.Vec3f(*pts.max(0))])
    albedo, normal = seabed_textures(texture_dir, seed=seed)
    bind(mesh.GetPrim(), preview_material(stage, f"{path}_mat", roughness=0.95,
                                          diffuse_tex=albedo, normal_tex=normal))
    return mesh.GetPrim()


def _tube(stage, path, p0, p1, r, mat):
    from pxr import Gf, UsdGeom

    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(r))
    cyl.CreateHeightAttr(length)
    cyl.CreateAxisAttr("Z")
    z = np.array([0.0, 0.0, 1.0])
    u = d / length
    axis = np.cross(z, u)
    s = float(np.linalg.norm(axis))
    angle = math.degrees(math.atan2(s, float(z @ u)))
    rot = Gf.Rotation(Gf.Vec3d(*(axis / s)) if s > 1e-9 else Gf.Vec3d(1, 0, 0),
                      angle if s > 1e-9 else (0.0 if u[2] > 0 else 180.0))
    xf = UsdGeom.Xformable(cyl)
    xf.AddTranslateOp().Set(Gf.Vec3d(*((p0 + p1) / 2)))
    xf.AddOrientOp().Set(Gf.Quatf(rot.GetQuat()))
    bind(cyl.GetPrim(), mat)
    return cyl


def _box(stage, path, center, size, mat, yaw=0.0):
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateZOp().Set(math.degrees(yaw))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    bind(cube.GetPrim(), mat)
    return cube


def _quad(stage, path, center, size_yz, mat, normal_x=-1.0):
    from pxr import Sdf, UsdGeom, Vt

    cx, cy, cz = center
    hy, hz = size_yz[0] / 2, size_yz[1] / 2
    pts = [(cx, cy + hy, cz - hz), (cx, cy - hy, cz - hz),
           (cx, cy - hy, cz + hz), (cx, cy + hy, cz + hz)]
    if normal_x > 0:
        pts = pts[::-1]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    m.CreateSubdivisionSchemeAttr("none")
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                               UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([(0, 0), (1, 0), (1, 1), (0, 1)] if normal_x < 0
                         else [(0, 1), (1, 1), (1, 0), (0, 0)]))
    bind(m.GetPrim(), mat)
    return m


def fiducial_png(texture_dir, marker_id=7, px=512):
    import cv2

    out = Path(texture_dir) / f"aruco4x4_{marker_id}.png"
    if out.exists():
        return out
    Path(texture_dir).mkdir(parents=True, exist_ok=True)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    inner = cv2.aruco.generateImageMarker(d, marker_id, int(px * 0.75))
    img = np.full((px, px), 255, np.uint8)
    o = (px - inner.shape[0]) // 2
    img[o:o + inner.shape[0], o:o + inner.shape[1]] = inner
    cv2.imwrite(str(out), img)
    return out


def build_dock_station(stage, path, *, bay_center, seabed_z, texture_dir, inner_w=0.84,
                       depth=1.0, roof=0.24, flare_len=0.6, flare_angle=FLARE_ANGLE,
                       tube_r=0.03, seed=0):
    from pxr import UsdGeom, UsdLux

    rng = np.random.default_rng(seed)
    UsdGeom.Xform.Define(stage, path)
    cx, cy, zb = (float(v) for v in bay_center)
    paint = [preview_material(stage, f"{path}/mat/paint{k}", roughness=0.55,
                              color=tuple(np.array([0.88, 0.66, 0.08]) * f + g))
             for k, (f, g) in enumerate([(0.85, 0), (0.7, np.array([0.02, 0.0, -0.02])),
                                         (0.55, np.array([0.02, 0.06, 0.03])),
                                         (0.6, np.array([0.08, -0.05, -0.03]))])]
    steel = preview_material(stage, f"{path}/mat/steel", color=(0.35, 0.36, 0.37),
                             roughness=0.5, metallic=0.6)
    grate = preview_material(stage, f"{path}/mat/grate", color=(0.22, 0.23, 0.22),
                             roughness=0.8)
    rubber = preview_material(stage, f"{path}/mat/rubber", color=(0.04, 0.04, 0.04),
                              roughness=0.9)
    marker = preview_material(stage, f"{path}/mat/marker", roughness=0.7,
                              diffuse_tex=fiducial_png(texture_dir))
    led = preview_material(stage, f"{path}/mat/led", color=(1.0, 1.0, 1.0),
                           emissive=(0.4, 1.0, 0.5))

    def pick():
        return paint[int(rng.integers(len(paint)))]

    x0, x1 = cx - depth / 2, cx + depth / 2
    yw = inner_w / 2 + tube_r
    floor, top = zb - 0.20, zb + roof
    legs_bottom = seabed_z + 0.05
    k = 0

    def tube(p0, p1, r=tube_r, mat=None):
        nonlocal k
        k += 1
        return _tube(stage, f"{path}/frame/t{k:03d}", p0, p1, r, mat or pick())

    _box(stage, f"{path}/mudmat", (cx, cy, seabed_z + 0.025),
         (depth + 0.5, inner_w + 0.6, 0.05), grate)
    for x in (x0, x1):
        for y in (-yw, yw):
            tube((x, cy + y, legs_bottom), (x, cy + y, top), r=0.045)
            tube((x, cy + y, legs_bottom + 0.1),
                 (x + (0.3 if x == x0 else -0.3), cy + y, floor - 0.05), r=0.02)
    for y in (-yw, yw):
        for z in (floor, (floor + top) / 2, top):
            tube((x0, cy + y, z), (x1, cy + y, z))
        for x in np.linspace(x0, x1, 6)[1:-1]:
            tube((x, cy + y, floor), (x, cy + y, top), r=0.02)
    for x in np.linspace(x0, x1, 5):
        tube((x, cy - yw, top), (x, cy + yw, top), r=0.025)
    for z in (floor, top):
        tube((x1, cy - yw, z), (x1, cy + yw, z))
    _box(stage, f"{path}/back_plate", (x1 + 0.02, cy, (floor + top) / 2),
         (0.01, inner_w, top - floor), grate)
    _quad(stage, f"{path}/fiducial", (x1 + 0.012, cy, zb + 0.02), (0.26, 0.26), marker)
    _tube(stage, f"{path}/stab", (x1 + 0.01, cy, floor + 0.08),
          (x1 - 0.12, cy, floor + 0.08), 0.035, steel)
    _tube(stage, f"{path}/stab_tip", (x1 - 0.12, cy, floor + 0.08),
          (x1 - 0.16, cy, floor + 0.08), 0.02, steel)
    for y in (-0.18, 0.18):
        tube((x0, cy + y, floor), (x1, cy + y, floor), r=0.025, mat=steel)
        for x in np.linspace(x0 + 0.15, x1 - 0.15, 3):
            _box(stage, f"{path}/pad_{k:03d}",
                 (x, cy + y, floor + 0.03), (0.1, 0.06, 0.02), rubber)
            k += 1
        tube((x0, cy + y, floor - 0.0), (x0, cy + y, legs_bottom + 0.15), r=0.02)
    c, s = math.cos(flare_angle), math.sin(flare_angle)
    for side in (-1, 1):
        y0 = cy + side * yw
        tip = (x0 - flare_len * c, y0 + side * flare_len * s)
        for z in (floor, (floor + top) / 2, top):
            tube((x0, y0, z), (tip[0], tip[1], z))
        tube((tip[0], tip[1], floor), (tip[0], tip[1], top), r=0.025)
        tube((tip[0], tip[1], floor), (tip[0], tip[1], legs_bottom), r=0.035)
        ledp = f"{path}/led_{'port' if side > 0 else 'stbd'}"
        _box(stage, ledp, (x0, y0, top + 0.04), (0.05, 0.05, 0.04), led)
        light = UsdLux.SphereLight.Define(stage, f"{ledp}_light")
        light.CreateRadiusAttr(0.02)
        light.CreateIntensityAttr(3000.0)
        light.CreateColorAttr((0.3, 1.0, 0.4) if side > 0 else (1.0, 0.25, 0.2))
        UsdGeom.Xformable(light).AddTranslateOp().Set((x0, y0, top + 0.09))
    for x in (x0, x1):
        for y in (-yw, yw):
            _box(stage, f"{path}/anode_{k}", (x, cy + y + np.sign(y) * 0.06, floor - 0.25),
                 (0.06, 0.05, 0.18), steel)
            k += 1
    return stage.GetPrimAtPath(path)


def _icosphere(subdiv=3):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
         (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b, cache=cache):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(verts), np.array(f)


def _noise3(p, rng, octaves=4, freq=1.5):
    out, amp, total = np.zeros(len(p)), 1.0, 0.0
    for _ in range(octaves):
        for _ in range(6):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            out += amp * np.sin(freq * (p @ d) * 2 * math.pi + rng.uniform(0, 2 * math.pi))
            total += amp
        amp *= 0.5
        freq *= 2.1
    return out / total * 2.5


def rock_mesh(seed, size=0.3):
    rng = np.random.default_rng(seed)
    v, f = _icosphere(3)
    v = v * (1 + 0.22 * _noise3(v, rng))[:, None]
    v *= np.array([1.0, rng.uniform(0.6, 1.0), rng.uniform(0.35, 0.7)]) * size / 2
    v[:, 2] -= v[:, 2].min() + 0.15 * np.ptp(v[:, 2])
    palette = [(0.34, 0.33, 0.30), (0.42, 0.38, 0.32), (0.26, 0.27, 0.27)]
    base = np.array(palette[int(rng.integers(len(palette)))])
    shade = 1 + 0.25 * _noise3(v * 6 / size, rng, octaves=2)
    col = base[None] * shade[:, None]
    top = (v[:, 2] - v[:, 2].min()) / (np.ptp(v[:, 2]) + 1e-9)
    col = col * (1 - 0.35 * top[:, None]) + 0.35 * top[:, None] * np.array([0.25, 0.32, 0.18])
    return v, f, np.clip(col, 0, 1)


def sponge_mesh(seed, height=0.3):
    rng = np.random.default_rng(seed)
    nz, na = 16, 40
    z = np.linspace(0, 1, nz)
    r = 0.35 + 0.25 * np.sin(math.pi * z * 0.9) + 0.1 * z
    a = np.linspace(0, 2 * math.pi, na, endpoint=False)
    ridge = 1 + 0.08 * np.sin(a * rng.integers(9, 15))[None]
    lump = (1 + 0.15 * np.sin(a[None] * 2 + z[:, None] * 5 + rng.uniform(0, 6))
            + 0.1 * np.sin(a[None] * 3 - z[:, None] * 3 + rng.uniform(0, 6)))
    R = r[:, None] * ridge * lump * height * rng.uniform(0.5, 0.8)
    v = np.stack([R * np.cos(a)[None], R * np.sin(a)[None],
                  np.repeat(z[:, None], na, 1) * height], -1).reshape(-1, 3)
    f = []
    for i in range(nz - 1):
        for j in range(na):
            a0, a1 = i * na + j, i * na + (j + 1) % na
            f += [(a0, a1, a1 + na), (a0, a1 + na, a0 + na)]
    palette = [(0.80, 0.45, 0.12), (0.62, 0.30, 0.38), (0.78, 0.66, 0.30)]
    base = np.array(palette[int(rng.integers(len(palette)))])
    col = base[None] * (0.85 + 0.15 * rng.random(len(v)))[:, None]
    return v, np.array(f), col


def seaweed_mesh(seed, height=0.6, blades=7):
    rng = np.random.default_rng(seed)
    vs, fs, cs, off = [], [], [], 0
    for _ in range(blades):
        n = 12
        h = height * rng.uniform(0.6, 1.1)
        t = np.linspace(0, 1, n)
        lean, yaw = rng.uniform(0.1, 0.4), rng.uniform(0, 2 * math.pi)
        x = lean * h * t ** 2
        c = np.stack([x * math.cos(yaw), x * math.sin(yaw), h * t], -1)
        c[:, :2] += rng.normal(0, 0.03, 2)
        w = 0.03 * (1 - t) ** 0.5 + 0.004
        side = np.array([-math.sin(yaw + 0.8), math.cos(yaw + 0.8), 0.0])
        wave = 0.02 * np.sin(t * 9 + rng.uniform(0, 6))[:, None] * side
        v = np.stack([c + wave + w[:, None] * side, c + wave - w[:, None] * side],
                     1).reshape(-1, 3)
        for i in range(n - 1):
            a, b = off + 2 * i, off + 2 * i + 2
            fs += [(a, a + 1, b + 1), (a, b + 1, b)]
        base = np.array([0.30, 0.34, 0.10]) * rng.uniform(0.8, 1.2)
        cs.append(np.repeat(base * (0.7 + 0.5 * t[:, None]), 2, 0))
        vs.append(v)
        off += len(v)
    return np.concatenate(vs), np.array(fs), np.clip(np.concatenate(cs), 0, 1)


def _tube_mesh(points, radii, sides=6):
    vs, fs = [], []
    n = len(points)
    for i, (p, r) in enumerate(zip(points, radii, strict=True)):
        d = points[min(i + 1, n - 1)] - points[max(i - 1, 0)]
        d = d / (np.linalg.norm(d) + 1e-12)
        a = np.cross(d, [0, 0, 1.0]) if abs(d[2]) < 0.9 else np.cross(d, [1.0, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        for k in range(sides):
            th = 2 * math.pi * k / sides
            vs.append(p + r * (math.cos(th) * a + math.sin(th) * b))
    for i in range(n - 1):
        for k in range(sides):
            a0, a1 = i * sides + k, i * sides + (k + 1) % sides
            fs += [(a0, a1, a1 + sides), (a0, a1 + sides, a0 + sides)]
    return np.array(vs), np.array(fs)


def coral_mesh(seed, size=0.35):
    rng = np.random.default_rng(seed)
    vs, fs = [], []
    off = 0

    def grow(p, d, r, depth):
        nonlocal off
        length = size * 0.35 * (0.75 ** (3 - depth)) * rng.uniform(0.8, 1.2)
        pts = [p]
        for _ in range(4):
            d = d + rng.normal(0, 0.15, 3)
            d[2] = abs(d[2]) + 0.2
            d = d / np.linalg.norm(d)
            pts.append(pts[-1] + d * length / 4)
        pts = np.array(pts)
        v, f = _tube_mesh(pts, np.linspace(r, r * 0.7, len(pts)))
        vs.append(v)
        fs.append(f + off)
        off += len(v)
        if depth > 0:
            for _ in range(int(rng.integers(2, 4))):
                grow(pts[-1], d + rng.normal(0, 0.5, 3), r * 0.7, depth - 1)

    grow(np.zeros(3), np.array([0, 0, 1.0]), size * 0.03, 3)
    v = np.concatenate(vs)
    palette = [(0.75, 0.25, 0.30), (0.85, 0.55, 0.20), (0.55, 0.30, 0.60)]
    base = np.array(palette[int(rng.integers(len(palette)))])
    col = base[None] * (0.8 + 0.4 * (v[:, 2] / (v[:, 2].max() + 1e-9)))[:, None]
    return v, np.concatenate(fs), np.clip(col, 0, 1)


def pebble_mesh(seed, size=0.05):
    v, f, c = rock_mesh(seed, size)
    return v, f, c * 0.9


GENERATORS = {"rock": rock_mesh, "sponge": sponge_mesh, "seaweed": seaweed_mesh,
              "coral": coral_mesh, "pebble": pebble_mesh}


def add_mesh(stage, path, verts, faces, colors, *, pos=(0, 0, 0), yaw=0.0, mat=None,
             double_sided=False):
    from pxr import Gf, Sdf, UsdGeom, Vt

    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(verts, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(np.asarray(faces).reshape(-1).tolist()))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(bool(double_sided))
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                               UsdGeom.Tokens.vertex)
    linear = np.clip(np.asarray(colors, np.float32), 0, 1) ** 2.2
    pv.Set(Vt.Vec3fArray.FromNumpy(linear.astype(np.float32)))
    xf = UsdGeom.Xformable(m)
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(x) for x in pos]))
    xf.AddRotateZOp().Set(math.degrees(yaw))
    if mat is not None:
        bind(m.GetPrim(), mat)
    return m


def scatter_scenery(stage, root, *, seabed_z, seed=0, half=12.0, keep_out=None, counts=None,
                    flat_radius=3.0):
    from pxr import UsdGeom

    rng = np.random.default_rng(seed)
    keep_out = keep_out or [(0.0, 0.0, 2.5)]
    counts = counts or {"rock": 60, "sponge": 18, "seaweed": 60, "coral": 30, "pebble": 400}
    UsdGeom.Xform.Define(stage, root)
    rocks = []

    def clear(x, y):
        return all(math.hypot(x - kx, y - ky) > kr for kx, ky, kr in keep_out)

    for kind, n in counts.items():
        for i in range(n):
            if kind in ("seaweed", "coral") and rocks and rng.random() < 0.6:
                px, py = rocks[int(rng.integers(len(rocks)))]
                x, y = px + rng.normal(0, 0.4), py + rng.normal(0, 0.4)
            else:
                x, y = rng.uniform(-half, half, 2)
            if not clear(x, y):
                continue
            size = {"rock": rng.lognormal(math.log(0.35), 0.5),
                    "sponge": rng.uniform(0.2, 0.45), "seaweed": rng.uniform(0.3, 0.8),
                    "coral": rng.uniform(0.2, 0.5),
                    "pebble": rng.lognormal(math.log(0.05), 0.4)}[kind]
            v, f, c = GENERATORS[kind](int(rng.integers(1 << 30)), size)
            z = seabed_z + float(seabed_height(np.array([x]), np.array([y]), seed=seed,
                                              flat_radius=flat_radius)[0])
            add_mesh(stage, f"{root}/{kind}_{i:03d}", v, f, c, pos=(x, y, z),
                     yaw=float(rng.uniform(0, 2 * math.pi)),
                     double_sided=kind == "seaweed")
            if kind == "rock":
                rocks.append((x, y))
    return stage.GetPrimAtPath(root)
