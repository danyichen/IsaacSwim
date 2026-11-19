from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("trimesh")

from isaacswim.assets.bluerov.cfg import FRAME_MESH, _frame_split, frame_part_shapes  # noqa: E402
from isaacswim.urdf import shapes_from_urdf  # noqa: E402

URDF = """<robot name="r"><link name="base_link"><visual><origin xyz="0 0 0"/>
<geometry><mesh filename="meshes/{mesh}" scale="1 1 1"/></geometry></visual></link></robot>"""


def _box(lo, hi):
    import trimesh
    return trimesh.creation.box(bounds=[lo, hi])


@pytest.fixture
def robot(tmp_path):
    import trimesh
    parts = [_box([-0.2, -0.05, -0.05], [0.2, 0.05, 0.05]),
             _box([0.0, 0.2, 0.0], [0.1, 0.3, 0.1]),
             _box([0.0, -0.3, 0.0], [0.05, -0.25, 0.05]),
             _box([0.0, 0.0, 0.2], [0.01, 0.01, 0.21])]
    (tmp_path / "meshes").mkdir()
    trimesh.util.concatenate(parts).export(tmp_path / "meshes" / FRAME_MESH)
    urdf = tmp_path / "r.urdf"
    urdf.write_text(URDF.format(mesh=FRAME_MESH))
    return urdf


def test_parts_are_ranked_largest_first_and_drag_only(robot):
    shapes = frame_part_shapes(robot, n_parts=2)
    assert len(shapes) == 2
    assert np.prod(shapes[0].semiaxes) > np.prod(shapes[1].semiaxes)
    assert shapes[0].semiaxes.max() == pytest.approx(0.2, rel=1e-6)
    assert all(s.volume == 0.0 and s.mesh is None for s in shapes)
    assert all(s.coef[3] == 0.0 and s.coef[0] > 0 for s in shapes)
    assert robot.with_name("frame_parts_2.json").exists()
    again = frame_part_shapes(robot, n_parts=2)
    assert np.allclose(again[1].pos, shapes[1].pos)


def test_screws_are_left_out(robot):
    assert len(frame_part_shapes(robot, n_parts=10)) == 3


def test_split_keeps_the_buoyancy_and_moves_the_drag(robot):
    geom = shapes_from_urdf(robot, verbose=False)
    split = _frame_split(geom, robot, 3)
    assert sum(s.volume for s in split) == pytest.approx(sum(s.volume for s in geom))
    buoy = [s for s in split if s.volume > 0]
    assert len(buoy) == 1 and not np.any(buoy[0].coef) and not np.any(buoy[0].virtual_mass)
    assert len([s for s in split if s.volume == 0.0]) == 3


def test_a_split_cached_from_another_hull_is_not_reused(robot):
    import json

    frame_part_shapes(robot, n_parts=2)
    cache = robot.with_name("frame_parts_2.json")
    assert json.loads(cache.read_text())["key"]["mesh"] == FRAME_MESH
    _box([-0.4, -0.05, -0.05], [0.4, 0.05, 0.05]).export(robot.parent / "meshes" / FRAME_MESH)
    shapes = frame_part_shapes(robot, n_parts=2)
    assert len(shapes) == 1 and shapes[0].semiaxes.max() == pytest.approx(0.4, rel=1e-6)
