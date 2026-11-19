from __future__ import annotations

import math

import numpy as np
import pytest

from isaacswim.urdf import load_obj, shapes_from_urdf

CUBE_OBJ = """\
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""


def write_urdf(tmp_path, joint_type="fixed"):
    (tmp_path / "cube.obj").write_text(CUBE_OBJ)
    (tmp_path / "robot.urdf").write_text(f"""<robot name="r">
  <link name="hull">
    <visual><geometry><mesh filename="cube.obj" scale="0.2 0.2 0.2"/></geometry></visual>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
  </link>
  <link name="sensor_pod">
    <visual><origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry><mesh filename="cube.obj" scale="0.1 0.1 0.1"/></geometry></visual>
  </link>
  <joint name="pod_mount" type="{joint_type}">
    <parent link="hull"/><child link="sensor_pod"/>
    <origin xyz="0.3 0 0" rpy="0 0 {math.pi / 2}"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>""")
    return tmp_path / "robot.urdf"


def test_obj_reader_triangulates_quads(tmp_path):
    (tmp_path / "c.obj").write_text(CUBE_OBJ)
    v, f = load_obj(tmp_path / "c.obj")
    assert v.shape == (8, 3) and f.shape == (12, 3)


def test_obj_reader_handles_slashes_and_negative_indices(tmp_path):
    (tmp_path / "t.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3/1/1 -2/2/2 -1/3/3\n")
    _, f = load_obj(tmp_path / "t.obj")
    assert f.tolist() == [[0, 1, 2]]


def test_fixed_joint_folds_the_shape_onto_the_surviving_body(tmp_path):
    shapes = shapes_from_urdf(write_urdf(tmp_path, "fixed"), verbose=False)
    assert [s.body for s in shapes] == ["hull", "hull"]
    pod = shapes[1]
    assert np.allclose(pod.pos, [0.3, 0.0, 0.1])


def test_movable_joint_keeps_its_own_body(tmp_path):
    shapes = shapes_from_urdf(write_urdf(tmp_path, "revolute"), verbose=False)
    assert [s.body for s in shapes] == ["hull", "sensor_pod"]
    assert np.allclose(shapes[1].pos, [0.0, 0.0, 0.1])


def test_visual_meshes_carry_their_scaled_volume(tmp_path):
    hull = shapes_from_urdf(write_urdf(tmp_path), verbose=False)[0]
    assert hull.volume == pytest.approx(0.2 ** 3, rel=1e-9)


def test_collision_primitives_are_not_meshes(tmp_path):
    assert shapes_from_urdf(write_urdf(tmp_path), source="collision",
                            verbose=False) == []


def test_non_obj_meshes_are_reported_not_silently_dropped(tmp_path, capsys):
    urdf = write_urdf(tmp_path)
    urdf.write_text(urdf.read_text().replace('filename="cube.obj" scale="0.1',
                                             'filename="pod.stl" scale="0.1'))
    shapes = shapes_from_urdf(urdf, verbose=True)
    assert len(shapes) == 1
    assert "pod.stl" in capsys.readouterr().out
