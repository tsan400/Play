"""Build lightweight single-file viewers for phones.

Usage: python bundle_mobile.py [--ratio 0.25] [--out DIR]

Compared with bundle.py: Info clashes are dropped, furniture/finishes are
removed, meshes are decimated, and every element not involved in a clash is
merged into one mesh per discipline so the phone draws a handful of objects.
"""
import argparse, base64, io, json
from pathlib import Path

import numpy as np
import trimesh

from bundle import SHIM

HERE = Path(__file__).parent
DROP_KINDS = {"Furniture", "Ceiling/Covering", "Railing"}
MOBILE_CSS = """<style>
  @media (max-width: 900px) {
    html, body { overflow-x: hidden; }
    body { font-size: 13px; }
    header { padding: 8px 12px; gap: 10px; }
    .mark { width: 28px; height: 28px; } .mark::after { width: 10px; height: 18px; right: 4px; top: 5px; }
    .brand b { font-size: 14px; } .brand span { font-size: 12px; }
    .right { gap: 6px; min-width: 0; } select { max-width: 40vw; padding: 6px; } .btn { padding: 8px 9px; font-size: 11px; }
    main { grid-template-columns: minmax(0, 1fr); grid-template-rows: 52vh minmax(0, 1fr); }
    aside { min-width: 0; }
    #legend { display: none; }
    #card { left: 8px; right: 8px; bottom: 8px; width: auto; max-height: 38%; overflow: auto; font-size: 12px; }
    #card dl { display: none !important; }
    #card h3 { padding: 8px 12px; font-size: 14px; } #card .body { padding: 8px 12px; }
    #card dl { grid-template-columns: 74px 1fr; margin-top: 6px; }
    .action { margin-top: 6px; padding: 6px 8px; }
    .sect { padding: 10px 12px; } .chips { grid-template-columns: repeat(3, 1fr); }
    .chip[data-s="Info"] { display: none; } .chip .n { font-size: 17px; }
    .item { padding: 9px 12px; }
  }
</style>
"""


def simplify(m, ratio):
    if len(m.faces) < 60:
        return m
    try:
        return m.simplify_quadric_decimation(percent=1 - ratio)
    except Exception:
        return m


def build_glb(path, keep, ratio, kinds, kind_ids):
    scene = trimesh.load(path, force="scene")
    meta = {}
    out, rest = trimesh.Scene(), {}
    for node in scene.graph.nodes_geometry:
        tf, gname = scene.graph[node]
        m = scene.geometry[gname].copy()
        m.apply_transform(tf)
        guid = node
        if guid in keep:
            out.add_geometry(simplify(m, min(1, ratio * 1.5)), node_name=guid, geom_name=guid)
        else:
            rest.setdefault(kinds.get(guid, "Other"), []).append(simplify(m, ratio))
    # One merged mesh per category keeps draw calls low but layers usable.
    for kind, parts in rest.items():
        out.add_geometry(trimesh.util.concatenate(parts), node_name=f"ctx_{kind_ids[kind]}", geom_name=f"ctx_{kind_ids[kind]}")
    buf = io.BytesIO()
    out.export(buf, file_type="glb", include_normals=False)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.15, help="fraction of triangles kept for context geometry")
    ap.add_argument("--out", default=str(HERE / "dist"))
    args = ap.parse_args()
    data = HERE / "viewer" / "data"
    html = (HERE / "viewer" / "index.html").read_text()
    html = html.replace("new THREE.WebGLRenderer({ antialias: true })", "new THREE.WebGLRenderer({ antialias: false, powerPreference: 'low-power' })")
    html = html.replace("renderer.setPixelRatio(devicePixelRatio)", "renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5))")
    html = html.replace(">Download CSV<", ">CSV<").replace("</head>", MOBILE_CSS + "</head>", 1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for proj in json.loads((data / "index.json").read_text()):
        slug = proj["slug"]
        meta = json.loads((data / f"{slug}.json").read_text())
        meta["clashes"] = [c for c in meta["clashes"] if c["severity"] != "Info"]
        keep = {g for c in meta["clashes"] for g in (c["a"], c["b"], *c["parts"])}
        # Drop furniture/finishes not involved in a clash.
        drop = {g for g, e in meta["elements"].items() if e["kind"] in DROP_KINDS and g not in keep}
        kinds = {g: e["kind"] for g, e in meta["elements"].items()}
        meta["ctx_kinds"] = sorted(set(kinds.values()) | {"Other"})
        kind_ids = {k: i for i, k in enumerate(meta["ctx_kinds"])}
        meta["elements"] = {g: e for g, e in meta["elements"].items() if g in keep}
        meta["summary"].pop("Info", None)
        for c in meta["clashes"]:
            c["point"] = [round(v, 2) for v in c["point"]]
        bundle = {}
        for disc, fname in meta["glb"].items():
            scene = trimesh.load(data / fname, force="scene")
            for node in list(scene.graph.nodes_geometry):
                if node in drop:
                    scene.delete_geometry(scene.graph[node][1])
            tmp = out / f".{fname}"
            scene.export(tmp)
            bundle[fname] = base64.b64encode(build_glb(tmp, keep, args.ratio, kinds, kind_ids)).decode()
            tmp.unlink()
        lines = (data / meta["csv"]).read_text(encoding="utf-8-sig").splitlines(keepends=True)
        csv_bytes = ("\ufeff" + lines[0] + "".join(l for l in lines[1:] if ",Info," not in l)).encode()
        bundle[meta["csv"]] = base64.b64encode(csv_bytes).decode()
        bundle[f"{slug}.json"] = base64.b64encode(json.dumps(meta).encode()).decode()
        proj = {**proj, "total": len(meta["clashes"])}
        bundle["index.json"] = base64.b64encode(json.dumps([proj]).encode()).decode()
        page = html.replace('<script type="importmap">', SHIM % json.dumps(bundle) + '<script type="importmap">', 1)
        dest = out / f"{slug}-clash-review-mobile.html"
        dest.write_text(page)
        print(f"{dest}  {dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
