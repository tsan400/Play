import os
import shutil
import subprocess
import xml.dom.minidom

import numpy as np
import pytest

from app.pipeline import export
from app.pipeline.lod import LAYERS, build_lod, newell
from app.pipeline.pointcloud import detect_up_axis, load_scan, to_z_up
from app.pipeline.room import extract_room
from app.pipeline.synthetic import DEMO_HEIGHT, DEMO_OPENINGS, DEMO_OUTLINE, demo_scan, write_ply


def _edge_lengths(ring):
    return sorted(round(float(np.linalg.norm(np.subtract(ring[(i + 1) % len(ring)], ring[i]))), 2)
                  for i in range(len(ring)))


@pytest.fixture(scope="module")
def demo_room():
    return extract_room(demo_scan())


def test_demo_outline_and_height(demo_room):
    assert len(demo_room["polygon"]) == len(DEMO_OUTLINE)
    assert _edge_lengths(demo_room["polygon"]) == _edge_lengths(DEMO_OUTLINE)
    assert abs(demo_room["height"] - DEMO_HEIGHT) < 0.01
    assert abs(demo_room["metrics"]["floor_area"] - 25.68) < 0.1


def test_demo_openings(demo_room):
    found = sorted((o["kind"], round(o["u1"] - o["u0"], 2), round(o["v1"] - o["v0"], 2))
                   for o in demo_room["openings"])
    truth = sorted((k, round(u1 - u0, 2), round(v1 - v0, 2)) for _, u0, u1, v0, v1, k in DEMO_OPENINGS)
    assert len(found) == len(truth)
    for (fk, fw, fh), (tk, tw, th) in zip(found, truth):
        assert fk == tk
        assert abs(fw - tw) < 0.03 and abs(fh - th) < 0.03


def test_demo_furniture(demo_room):
    sizes = sorted((round(f["w"], 1), round(f["d"], 1)) for f in demo_room["furniture"])
    assert sizes == [(1.6, 0.9), (2.0, 0.9)]


def test_ceiling_missing_is_estimated():
    r = extract_room(demo_scan(scan_ceiling=False, density=1500))
    assert r["ceiling_estimated"]
    assert _edge_lengths(r["polygon"]) == _edge_lengths(DEMO_OUTLINE)


def test_y_up_ply_roundtrip(tmp_path):
    p = tmp_path / "scan.ply"
    write_ply(p, demo_scan(density=1500), y_up=True)
    pts, nrm = load_scan(p)
    assert nrm is None
    axis = detect_up_axis(pts)
    assert axis == "y"
    pts, _ = to_z_up(pts, axis)
    r = extract_room(pts)
    assert _edge_lengths(r["polygon"]) == _edge_lengths(DEMO_OUTLINE)


def test_mesh_scan_is_sampled(tmp_path):
    import trimesh
    box = trimesh.creation.box(extents=(4.0, 3.0, 2.5))
    box.invert()
    p = tmp_path / "room.obj"
    box.export(p)
    pts, nrm = load_scan(p)
    assert nrm is not None and len(pts) == len(nrm) >= 50_000
    r = extract_room(pts)
    assert _edge_lengths(r["polygon"]) == [3.0, 3.0, 4.0, 4.0]
    assert abs(r["height"] - 2.5) < 0.01


def test_lod_faces_are_planar_and_oriented(demo_room):
    model = build_lod(demo_room)
    assert {f["layer"] for f in model["faces"]} == set(LAYERS)
    for f in model["faces"]:
        P = np.array(f["outer"])
        n = newell(f["outer"])
        assert np.abs((P - P[0]) @ n).max() < 1e-6
    # Floor slab top faces up; wall tops are a ring with the room as a hole.
    tops = [f for f in model["faces"] if f["layer"] == "walls" and len(f["holes"]) == 1
            and all(abs(p[2] - demo_room["height"]) < 1e-9 for p in f["outer"])]
    assert len(tops) == 1 and newell(tops[0]["outer"])[2] > 0.99


def test_scale_multiplies_lengths(demo_room):
    a = build_lod(demo_room, scale=1.0)
    b = build_lod(demo_room, scale=2.0)
    assert np.allclose(np.array(b["inner_outline"]), 2 * np.array(a["inner_outline"]))
    assert abs(b["height"] - 2 * a["height"]) < 1e-9


def test_exports(demo_room, tmp_path):
    model = build_lod(demo_room)
    xml.dom.minidom.parseString(export.to_dae(model, LAYERS))
    xml.dom.minidom.parseString(export.to_svg_plan(demo_room, model))
    dxf = export.to_dxf(model, LAYERS)
    assert dxf.rstrip().endswith("EOF") and "3DFACE" in dxf and "WALLS_LINES" in dxf
    obj = export.to_obj(model, LAYERS)
    nv = sum(1 for l in obj.splitlines() if l.startswith("v "))
    idx = [int(x) for l in obj.splitlines() if l.startswith(("f ", "l ")) for x in l.split()[1:]]
    assert max(idx) <= nv and min(idx) >= 1
    rb = export.to_sketchup_rb(model, LAYERS)
    assert rb.count("rs_face(g.entities") == sum(1 for f in model["faces"])
    if shutil.which("ruby"):
        p = tmp_path / "room.rb"
        p.write_text(rb)
        assert subprocess.run(["ruby", "-c", str(p)], capture_output=True).returncode == 0


@pytest.mark.skipif(not os.environ.get("ROOMSKETCH_SLOW"), reason="set ROOMSKETCH_SLOW=1 (runs COLMAP, ~2 min)")
def test_photogrammetry_end_to_end(tmp_path):
    from app.pipeline import photogrammetry as pg

    from .synth_photos import MARKER, ROOM, render_room

    frames = render_room(tmp_path / "frames")
    names = pg.prepare_images(frames, tmp_path / "images")
    pts, info = pg.reconstruct(tmp_path / "images", names, tmp_path / "work", MARKER)
    assert info["scaled"] and info["registered"] >= 0.9 * len(names)
    r = extract_room(pts, None, sparse=True)
    got = _edge_lengths(r["polygon"])
    want = sorted([ROOM[0], ROOM[0], ROOM[1], ROOM[1]])
    assert len(got) == 4
    for g, w in zip(got, want):
        assert abs(g - w) / w < 0.03, (got, want)
    assert abs(r["height"] - ROOM[2]) / ROOM[2] < 0.03
