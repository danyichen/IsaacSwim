from pathlib import Path


def remove_dangling_references(usd_path):
    from pxr import Usd

    path = Path(usd_path)
    count = 0
    for layer_file in [path, *sorted((path.parent / "configuration").glob("*.usd"))]:
        stage = Usd.Stage.Open(str(layer_file))
        for layer in stage.GetLayerStack(includeSessionLayers=False):
            missing = []

            def visit(prim_path, layer=layer, stage=stage, missing=missing):
                if prim_path.IsPrimPath():
                    refs = layer.GetPrimAtPath(prim_path).referenceList
                    missing.extend(
                        (refs, ref) for ref in refs.prependedItems
                        if not ref.assetPath and not stage.GetPrimAtPath(ref.primPath)
                    )

            layer.Traverse(layer.pseudoRoot.path, visit)
            for refs, ref in missing:
                refs.prependedItems.remove(ref)
            if missing:
                layer.Save()
                count += len(missing)
    return count
