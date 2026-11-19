import pytest

Usd = pytest.importorskip("pxr.Usd")

from isaacswim.assets.bluerov.usd_cleanup import remove_dangling_references  # noqa: E402


def test_only_missing_internal_targets_are_removed(tmp_path):
    external = tmp_path / "external.usda"
    other = Usd.Stage.CreateNew(str(external))
    other.DefinePrim("/mesh", "Xform")
    other.GetRootLayer().Save()
    path = tmp_path / "vehicle.usda"
    stage = Usd.Stage.CreateNew(str(path))
    stage.DefinePrim("/mesh", "Xform")
    valid = stage.DefinePrim("/valid", "Xform")
    valid.GetReferences().AddInternalReference("/mesh")
    missing = stage.DefinePrim("/missing", "Xform")
    missing.GetReferences().AddInternalReference("/absent")
    linked = stage.DefinePrim("/linked", "Xform")
    linked.GetReferences().AddReference("external.usda", "/mesh")
    stage.GetRootLayer().Save()

    assert remove_dangling_references(path) == 1
    assert not missing.HasAuthoredReferences()
    assert valid.HasAuthoredReferences()
    assert linked.HasAuthoredReferences()
    assert remove_dangling_references(path) == 0
