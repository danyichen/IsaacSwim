from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .models import _icosphere, _noise3, bind, fbm, preview_material


def _save(img, path):
    from PIL import Image

    Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)).save(path)
    return path


def tex_paint(tdir, name, rgb, *, wear=0.25, rust=0.25, fouling=0.15, seed=0, size=512):
    out = Path(tdir) / f"paint_{name}_{seed}.png"
    if out.exists():
        return out
    Path(tdir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img = np.ones((size, size, 3)) * np.asarray(rgb)[None, None]
    img *= (1 + 0.08 * fbm(size, rng, octaves=4, base=4))[..., None]
    streak = fbm(size, rng, octaves=3, base=24)
    streak = np.repeat(streak[::8], 8, axis=0)[:size]
    runs = np.clip((streak - 0.35) * 3, 0, 1) * rust
    img = img * (1 - runs[..., None]) + runs[..., None] * np.array([0.42, 0.20, 0.08])
    chips = fbm(size, rng, octaves=5, base=8) > (1 - wear)
    img[chips] = img[chips] * 0.3 + np.array([0.30, 0.18, 0.10]) * 0.7
    growth = np.clip((fbm(size, rng, octaves=4, base=3) - 0.2) * 2, 0, 1) * fouling
    img = img * (1 - growth[..., None]) + growth[..., None] * np.array([0.30, 0.33, 0.20])
    return _save(img, out)


def tex_concrete(tdir, seed=0, size=512):
    out = Path(tdir) / f"concrete_{seed}.png"
    if out.exists():
        return out
    Path(tdir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    g = 0.42 + 0.06 * fbm(size, rng, octaves=5, base=6) + 0.05 * rng.normal(size=(size, size))
    img = np.stack([g, g * 0.98, g * 0.94], -1)
    growth = np.clip(fbm(size, rng, octaves=4, base=4) * 1.5, 0, 1)
    img = img * (1 - 0.5 * growth[..., None]) + 0.5 * growth[..., None] * np.array(
        [0.28, 0.32, 0.20])
    return _save(img, out)


def tex_dial(tdir, size=512):
    from PIL import Image, ImageDraw

    out = Path(tdir) / "valve_dial.png"
    if out.exists():
        return out
    Path(tdir).mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (size, size), (235, 232, 220))
    d = ImageDraw.Draw(im)
    c, r = size / 2, size * 0.47
    for k in range(24):
        a = math.radians(k * 15)
        r0 = r * (0.78 if k % 6 == 0 else 0.88)
        d.line([(c + r0 * math.cos(a), c + r0 * math.sin(a)),
                (c + r * math.cos(a), c + r * math.sin(a))], fill=(20, 20, 20),
               width=10 if k % 6 == 0 else 4)
    d.text((c - 40, c - r * 0.6), "OPEN", fill=(10, 120, 30), font_size=36)
    d.text((c + r * 0.25, c - 18), "SHUT", fill=(170, 20, 20), font_size=36)
    d.arc([c - r * 0.45, c - r * 0.45, c + r * 0.45, c + r * 0.45], 200, 330,
          fill=(20, 20, 20), width=8)
    d.text((c - 30, c + r * 0.2), "CW", fill=(20, 20, 20), font_size=40)
    im.save(out)
    return out


def tex_label(tdir, text, *, bg=(240, 240, 235), fg=(15, 15, 15), size=(512, 160)):
    from PIL import Image, ImageDraw

    out = Path(tdir) / f"label_{text}.png"
    if out.exists():
        return out
    Path(tdir).mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    d.rectangle([6, 6, size[0] - 7, size[1] - 7], outline=fg, width=6)
    d.text((size[0] * 0.12, size[1] * 0.22), text, fill=fg, font_size=int(size[1] * 0.55))
    im.save(out)
    return out


def tex_gauge(tdir, size=512):
    from PIL import Image, ImageDraw

    out = Path(tdir) / "gauge.png"
    if out.exists():
        return out
    Path(tdir).mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (size, size), (245, 245, 240))
    d = ImageDraw.Draw(im)
    c, r = size / 2, size * 0.46
    for k in range(11):
        a = math.radians(135 + k * 27)
        d.line([(c + 0.8 * r * math.cos(a), c + 0.8 * r * math.sin(a)),
                (c + r * math.cos(a), c + r * math.sin(a))], fill=(10, 10, 10), width=6)
    d.arc([c - 0.9 * r, c - 0.9 * r, c + 0.9 * r, c + 0.9 * r], 315, 405,
          fill=(200, 30, 30), width=14)
    a = math.radians(250)
    d.line([(c, c), (c + 0.75 * r * math.cos(a), c + 0.75 * r * math.sin(a))],
           fill=(10, 10, 10), width=10)
    d.ellipse([c - 18, c - 18, c + 18, c + 18], fill=(10, 10, 10))
    im.save(out)
    return out


def _box_geom(size, tile=None):
    sx, sy, sz = (s / 2 for s in size)
    corners = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4),
             (1, 5, 7, 3)]
    uvs = []
    for q in quads:
        p = corners[list(q)]
        ext = np.ptp(p, 0)
        a, b = [i for i in range(3) if ext[i] > 1e-9][:2]
        for v in p:
            if tile:
                uvs.append((v[a] / tile, v[b] / tile))
            else:
                uvs.append(((v[a] + size[a] / 2) / size[a], (v[b] + size[b] / 2) / size[b]))
    return corners, [4] * 6, [i for q in quads for i in q], uvs


def _cyl_geom(r, h, *, seg=28, caps=True, tile=None, r_top=None):
    r_top = r if r_top is None else r_top
    a = np.linspace(0, 2 * math.pi, seg + 1)
    pts, counts, idx, uvs = [], [], [], []
    for z, rr in ((0.0, r), (h, r_top)):
        for t in a:
            pts.append((rr * math.cos(t), rr * math.sin(t), z))
    n = seg + 1
    circ = 2 * math.pi * r
    for i in range(seg):
        q = (i, i + 1, n + i + 1, n + i)
        counts.append(4)
        idx += q
        for k in q:
            u = (k % n) / seg
            v = (k // n)
            uvs.append((u * circ / tile, v * h / tile) if tile else (u, v))
    if caps:
        for z, rr, flip in ((0.0, r, True), (h, r_top, False)):
            c0 = len(pts)
            pts.append((0, 0, z))
            ring = list(range(len(pts), len(pts) + seg))
            for t in a[:-1]:
                pts.append((rr * math.cos(t), rr * math.sin(t), z))
            for i in range(seg):
                tri = (c0, ring[i], ring[(i + 1) % seg])
                tri = tri[::-1] if flip else tri
                counts.append(3)
                idx += tri
                for k in tri:
                    x, y = pts[k][0], pts[k][1]
                    uvs.append((0.5 + x / (2 * rr + 1e-9), 0.5 + y / (2 * rr + 1e-9)))
    return np.array(pts), counts, idx, uvs


_AXIS = {"z": (1, 0, 0, 0), "x": (math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0),
         "y": (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0, 0)}


def safe(path):
    head, _, name = path.rpartition("/")
    return f"{head}/{name.replace('-', 'm').replace('.', 'p').replace('+', '')}"


def mesh(stage, path, geom, mat, *, pos=(0, 0, 0), quat=(1, 0, 0, 0), double_sided=False):
    from pxr import Gf, Sdf, UsdGeom, Vt

    path = safe(path)
    pts, counts, idx, uvs = geom
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(pts, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(list(counts)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in idx]))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(bool(double_sided))
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                               UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uvs, np.float32)))
    xf = UsdGeom.Xformable(m)
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    xf.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in quat]))
    bind(m.GetPrim(), mat)
    return m


def box(stage, path, center, size, mat, *, tile=None, quat=(1, 0, 0, 0)):
    return mesh(stage, path, _box_geom(size, tile), mat, pos=center, quat=quat)


def cyl(stage, path, base, r, h, mat, *, axis="z", tile=None, r_top=None, caps=True, seg=28):
    return mesh(stage, path, _cyl_geom(r, h, seg=seg, caps=caps, tile=tile, r_top=r_top), mat,
                pos=base, quat=_AXIS[axis])


def group(stage, path, pos, yaw):
    from pxr import Gf, UsdGeom

    g = UsdGeom.Xform.Define(stage, path)
    g.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    g.AddRotateZOp().Set(math.degrees(yaw))
    return g


def _mats(stage, root, tdir, seed=0):
    from pxr import UsdShade

    if stage.GetPrimAtPath(f"{root}/steel"):
        return {p.GetName(): UsdShade.Material(p) for p in stage.GetPrimAtPath(root).GetChildren()}
    pm = preview_material
    return {
        "yellow": pm(stage, f"{root}/yellow", roughness=0.55,
                     diffuse_tex=tex_paint(tdir, "yellow", (0.80, 0.62, 0.08), seed=seed)),
        "orange": pm(stage, f"{root}/orange", roughness=0.5,
                     diffuse_tex=tex_paint(tdir, "orange", (0.90, 0.38, 0.06), rust=0.1,
                                           wear=0.15, seed=seed)),
        "grey_paint": pm(stage, f"{root}/grey_paint", roughness=0.6,
                         diffuse_tex=tex_paint(tdir, "grey", (0.38, 0.40, 0.42), seed=seed)),
        "steel": pm(stage, f"{root}/steel", color=(0.42, 0.43, 0.44), roughness=0.4,
                    metallic=0.8),
        "anode": pm(stage, f"{root}/anode", color=(0.55, 0.56, 0.55), roughness=0.85,
                    metallic=0.3),
        "black": pm(stage, f"{root}/black", color=(0.03, 0.03, 0.03), roughness=0.7),
        "concrete": pm(stage, f"{root}/concrete", roughness=0.95,
                       diffuse_tex=tex_concrete(tdir, seed)),
        "sleeve": pm(stage, f"{root}/sleeve", color=(0.08, 0.08, 0.08), roughness=0.6),
        "hdpe": pm(stage, f"{root}/hdpe", color=(0.30, 0.31, 0.30), roughness=0.5),
        "dial": pm(stage, f"{root}/dial", roughness=0.4, diffuse_tex=tex_dial(tdir)),
        "gauge": pm(stage, f"{root}/gauge", roughness=0.2, diffuse_tex=tex_gauge(tdir)),
    }


SAMPLE_COLORS = {"red": (0.58, 0.16, 0.10), "yellow": (0.80, 0.64, 0.18),
                 "green": (0.20, 0.44, 0.26)}


def sample_rock_geom(color, seed, size=0.045):
    rng = np.random.default_rng(seed)
    v, f = _icosphere(3)
    v = v * (1 + 0.12 * _noise3(v, rng, octaves=3))[:, None]
    v = v / np.abs(v).max(0) * size / 2 * np.array([1.0, 0.95, 0.85])
    v[:, 2] += size / 2 * 0.85 - size / 2
    base = np.asarray(SAMPLE_COLORS[color])
    vein = np.abs(np.sin(_noise3(v * 60, rng, octaves=2) * 3))
    col = base[None] * (0.8 + 0.3 * vein[:, None])
    return v, f, np.clip(col, 0, 1)


def build_sample(stage, path, color, pos, yaw, *, seed=0, tdir=None):
    from .models import add_mesh

    v, f, c = sample_rock_geom(color, seed)
    return add_mesh(stage, path, v, f, c, pos=pos, yaw=yaw)


def build_sample_crate(stage, path, pos, yaw, *, inner=0.22, wall=(0.012, 0.05), floor_t=0.01,
                       tdir=None, seed=0):
    mats = _mats(stage, "/World/Looks", tdir, seed)
    group(stage, path, pos, yaw)
    t, h = wall
    out = inner / 2 + t / 2
    box(stage, f"{path}/floor", (0, 0, floor_t / 2), (inner + 2 * t, inner + 2 * t, floor_t),
        mats["hdpe"])
    k = 0
    for side in range(4):
        ang = side * math.pi / 2
        c, s = math.cos(ang), math.sin(ang)
        for u in np.linspace(-inner / 2, inner / 2, 11):
            x, y = c * out - s * u, s * out + c * u
            cyl(stage, f"{path}/bar{k:03d}", (x, y, floor_t), 0.0025, h, mats["steel"], seg=6,
                caps=False)
            k += 1
        for z in (floor_t + h * 0.5,):
            x0, y0 = c * out - s * (-inner / 2 - t), s * out + c * (-inner / 2 - t)
            q = (math.cos((ang + math.pi / 2) / 2), 0, 0, math.sin((ang + math.pi / 2) / 2))
            mesh(stage, f"{path}/rail{k:03d}", _cyl_geom(0.003, inner + 2 * t, seg=6),
                 mats["steel"], pos=(x0, y0, z),
                 quat=_quat_mul(q, _AXIS["x"]))
            k += 1
        x0, y0 = c * out - s * (-inner / 2 - t), s * out + c * (-inner / 2 - t)
        q = (math.cos((ang + math.pi / 2) / 2), 0, 0, math.sin((ang + math.pi / 2) / 2))
        mesh(stage, f"{path}/rim{side}", _cyl_geom(0.007, inner + 2 * t, seg=10),
             mats["orange"], pos=(x0, y0, floor_t + h), quat=_quat_mul(q, _AXIS["x"]))
    return stage.GetPrimAtPath(path)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def _yaw_q(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def build_valve_manifold(stage, path, pos, yaw, *, size=(0.50, 0.50, 0.25), tdir=None, seed=0):
    mats = _mats(stage, "/World/Looks", tdir, seed)
    group(stage, path, pos, yaw)
    sx, sy, sz = size
    box(stage, f"{path}/mudmat", (0, 0, 0.012), (sx + 0.4, sy + 0.4, 0.024), mats["grey_paint"],
        tile=0.5)
    box(stage, f"{path}/body", (0, 0, sz / 2), (sx, sy, sz), mats["yellow"], tile=0.5)
    box(stage, f"{path}/top_plate", (0, 0, sz + 0.004), (sx * 0.8, sy * 0.8, 0.008),
        mats["grey_paint"], tile=0.4)
    cyl(stage, f"{path}/dial", (0, 0, sz + 0.006), 0.10, 0.004, mats["dial"])
    for side in (-1, 1):
        x = side * (sx / 2)
        mesh(stage, f"{path}/pipe_in{side}", _cyl_geom(0.055, 0.35, tile=0.4),
             mats["grey_paint"], pos=(x, 0, sz * 0.45),
             quat=_AXIS["x"] if side > 0 else _quat_mul(_yaw_q(math.pi), _AXIS["x"]))
        fx = x + side * 0.02
        mesh(stage, f"{path}/flange{side}", _cyl_geom(0.085, 0.03), mats["steel"],
             pos=(fx, 0, sz * 0.45),
             quat=_AXIS["x"] if side > 0 else _quat_mul(_yaw_q(math.pi), _AXIS["x"]))
        for b in range(8):
            a = 2 * math.pi * b / 8
            mesh(stage, f"{path}/bolt{side}_{b}", _cyl_geom(0.008, 0.05, seg=6), mats["steel"],
                 pos=(fx - side * 0.01, 0.07 * math.cos(a), sz * 0.45 + 0.07 * math.sin(a)),
                 quat=_AXIS["x"])
        cyl(stage, f"{path}/riser{side}", (x + side * 0.33, 0, -0.1), 0.055, sz * 0.45 + 0.155,
            mats["grey_paint"], tile=0.4)
    for cx in (-1, 1):
        for cy in (-1, 1):
            box(stage, f"{path}/padeye{cx}{cy}", (cx * (sx / 2 - 0.04), cy * (sy / 2 - 0.04),
                                                   sz + 0.03), (0.015, 0.05, 0.05), mats["steel"])
    tag = preview_material(stage, f"{path}/tag_mat", roughness=0.5,
                           diffuse_tex=tex_label(tdir, "XV-104"))
    box(stage, f"{path}/tag", (-sx / 2 - 0.002, 0, sz * 0.75), (0.004, 0.18, 0.055), tag)
    return stage.GetPrimAtPath(path)


def build_tbar(stage, path, pos, yaw, *, size=(0.16, 0.03, 0.03), tdir=None, seed=0):
    mats = _mats(stage, "/World/Looks", tdir, seed)
    group(stage, path, pos, yaw)
    lx, ly, lz = size
    box(stage, f"{path}/bar", (0, 0, 0), (lx, ly * 0.9, lz * 0.9), mats["orange"])
    for s in (-1, 1):
        mesh(stage, f"{path}/grip{s}", _cyl_geom(lz * 0.55, lx * 0.3, seg=16), mats["black"],
             pos=(s * lx * 0.2 if s > 0 else -lx * 0.5, 0, 0), quat=_AXIS["x"])
    cyl(stage, f"{path}/boss", (0, 0, -lz / 2 - 0.02), 0.022, 0.02 + lz, mats["steel"])
    return stage.GetPrimAtPath(path)


def build_pipe_segment(stage, path, start, end, *, r=0.10, z0, anode=True, tdir=None, seed=0,
                       burial=0.35):
    from .models import add_mesh

    mats = _mats(stage, "/World/Looks", tdir, seed)
    p0, p1 = np.array([*start, z0 + r]), np.array([*end, z0 + r])
    d = p1 - p0
    length = float(np.linalg.norm(d))
    yaw = math.atan2(d[1], d[0])
    q = _quat_mul(_yaw_q(yaw), _AXIS["x"])
    u = d / length
    joint = 0.12
    mesh(stage, f"{path}/coat", _cyl_geom(r, length - 2 * joint, tile=0.6, caps=False),
         mats["concrete"], pos=p0 + u * joint, quat=q)
    for k, a in enumerate((p0, p1 - u * joint)):
        mesh(stage, f"{path}/joint{k}", _cyl_geom(r * 0.93, joint, caps=False),
             mats["sleeve"], pos=a, quat=q)
    if anode:
        mesh(stage, f"{path}/anode", _cyl_geom(r * 1.06, 0.14, caps=False), mats["anode"],
             pos=p0 + u * (length / 2 - 0.07), quat=q)
    rng = np.random.default_rng(seed)
    n = 24
    t = np.linspace(0, 1, n)
    side = np.array([-u[1], u[0], 0.0])
    verts, faces, cols = [], [], []
    for sgn in (-1, 1):
        off = len(verts)
        for ti in t:
            c = p0 + d * ti
            hh = r * 2 * burial * (0.8 + 0.4 * rng.random())
            for w, z in ((r * 0.95, hh), (r * 2.2, 0.0)):
                verts.append(c + sgn * side * w + np.array([0, 0, z - r - 0.002]))
                cols.append((0.45, 0.40, 0.31) if z == 0 else (0.42, 0.38, 0.30))
        for i in range(n - 1):
            a, b = off + 2 * i, off + 2 * i + 2
            faces += [(a, a + 1, b + 1), (a, b + 1, b)] if sgn > 0 else \
                [(a, b + 1, a + 1), (a, b, b + 1)]
    add_mesh(stage, f"{path}/berm", np.array(verts), np.array(faces), np.array(cols),
             double_sided=True)
    return stage.GetPrimAtPath(path)


def build_inspection_panel(stage, path, pos, yaw, *, panel=(0.10, 0.80, 0.70), centre_z=1.0,
                           tdir=None, seed=0):
    from .models import fiducial_png

    mats = _mats(stage, "/World/Looks", tdir, seed)
    group(stage, path, pos, yaw)
    tx, sy, sz = panel
    zc = centre_z
    box(stage, f"{path}/mudmat", (0, 0, 0.015), (0.7, 0.8, 0.03), mats["grey_paint"], tile=0.5)
    for y in (-0.2, 0.2):
        cyl(stage, f"{path}/leg{y:+.1f}", (-0.03, y, 0.03), 0.04, zc - sz / 2 - 0.03,
            mats["grey_paint"], tile=0.4)
        mesh(stage, f"{path}/brace{y:+.1f}", _cyl_geom(0.02, 0.55), mats["grey_paint"],
             pos=(-0.35, y, 0.04), quat=(math.cos(0.45), 0, math.sin(0.45), 0))
    box(stage, f"{path}/face", (0, 0, zc), (tx, sy, sz), mats["yellow"], tile=0.6)
    fx = tx / 2 + 0.01
    for z in (zc - sz / 2 + 0.02, zc + sz / 2 - 0.02):
        box(stage, f"{path}/frame_h{int(z * 100)}", (fx, 0, z), (0.02, sy, 0.04),
            mats["grey_paint"])
    for y in (-sy / 2 + 0.02, sy / 2 - 0.02):
        box(stage, f"{path}/frame_v{int(y * 100)}", (fx, y, zc), (0.02, 0.04, sz),
            mats["grey_paint"])
    marker = preview_material(stage, f"{path}/marker_mat", roughness=0.6,
                              diffuse_tex=fiducial_png(tdir, marker_id=11))
    box(stage, f"{path}/marker", (tx / 2 + 0.003, 0, zc), (0.004, 0.20, 0.20), marker)
    mesh(stage, f"{path}/gauge", _cyl_geom(0.07, 0.03), mats["gauge"],
         pos=(tx / 2, 0.22, zc + 0.2), quat=_AXIS["x"])
    mesh(stage, f"{path}/gauge_rim", _cyl_geom(0.075, 0.02, caps=False), mats["steel"],
         pos=(tx / 2, 0.22, zc + 0.2), quat=_AXIS["x"])
    mesh(stage, f"{path}/hotstab", _cyl_geom(0.05, 0.08, r_top=0.03, caps=False),
         mats["steel"], pos=(tx / 2 + 0.08, -0.22, zc - 0.2),
         quat=_quat_mul(_yaw_q(math.pi), _AXIS["x"]))
    mesh(stage, f"{path}/cp_point", _cyl_geom(0.015, 0.05), mats["steel"],
         pos=(tx / 2, -0.22, zc + 0.2), quat=_AXIS["x"])
    tag = preview_material(stage, f"{path}/tag_mat", roughness=0.5,
                           diffuse_tex=tex_label(tdir, "IP-03"))
    box(stage, f"{path}/tag", (tx / 2 + 0.003, 0.0, zc - 0.25), (0.004, 0.22, 0.07), tag)
    for y in np.linspace(-sy / 2 + 0.06, sy / 2 - 0.06, 6):
        for z in (zc - sz / 2 + 0.05, zc + sz / 2 - 0.05):
            mesh(stage, f"{path}/bolt{int(y * 100)}_{int(z * 100)}", _cyl_geom(0.009, 0.015, seg=6),
                 mats["steel"], pos=(tx / 2 + 0.02, y, z), quat=_AXIS["x"])
    return stage.GetPrimAtPath(path)
