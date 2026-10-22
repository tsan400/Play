import json
import os
import struct

import numpy as np
import pytest

os.environ.setdefault("ROOMSKETCH_NO_PHOTOGRAMMETRY", "1")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.pipeline.pointcloud import voxel_downsample  # noqa: E402
from app.pipeline.synthetic import DEMO_OUTLINE, demo_scan  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _body(pts, normals=None):
    pts = np.asarray(pts, "<f4")
    out = struct.pack("<I", len(pts)) + pts.tobytes()
    if normals is not None:
        out += np.asarray(normals, np.int8).tobytes()
    return out


def _lengths(ring):
    return sorted(round(float(np.linalg.norm(np.subtract(ring[(i + 1) % len(ring)], ring[i]))), 2) for i in range(len(ring)))


def test_lidar_upload_y_up(client):
    pts, _ = voxel_downsample(demo_scan(density=1500), 0.025)
    yup = np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1)
    r = client.post("/api/room?up=auto&name=scan.ply", content=_body(yup))
    assert r.status_code == 200
    room = r.json()["room"]
    assert room["meta"]["up_axis"] == "y"
    assert _lengths(room["polygon"]) == _lengths(DEMO_OUTLINE)


def test_browser_photogrammetry_upload(client):
    # A sparse, metric, Z-up cloud like the in-browser SfM produces.
    rng = np.random.default_rng(1)
    pts = demo_scan(density=2500)
    pts = pts[rng.choice(len(pts), 15000, replace=False)] + rng.normal(0, 0.008, (15000, 3))
    meta = {"scaled": True, "scale_source": "aruco", "registered": 80, "images": 84}
    r = client.post("/api/room?source=photogrammetry&meta=" + json.dumps(meta), content=_body(pts))
    assert r.status_code == 200, r.text
    room = r.json()["room"]
    assert room["meta"]["source"] == "photogrammetry" and room["meta"]["processed"] == "browser"
    assert room["meta"]["scaled"] is True
    got, want = _lengths(room["polygon"]), _lengths(DEMO_OUTLINE)
    assert len(got) == len(want) and max(abs(g - w) for g, w in zip(got, want)) < 0.05


def test_model_and_exports_roundtrip(client):
    room = client.post("/api/demo").json()["room"]
    m = client.post("/api/model", json={"room": room, "scale": 2.0})
    assert m.status_code == 200 and abs(m.json()["metrics"]["height"] - 2 * room["height"]) < 1e-6
    for fmt in ["rb", "dae", "dxf", "obj", "svg", "json"]:
        assert client.post("/api/export", json={"room": room, "fmt": fmt}).status_code == 200


def test_rejects_bad_input(client):
    assert client.post("/api/room", content=b"abc").status_code == 400
    assert client.post("/api/room?source=nope", content=_body(np.zeros((100, 3)))).status_code == 400
    assert client.post("/api/model", json={"room": {"polygon": [1]}}).status_code == 400
    assert client.get("/api/config").json()["photogrammetry"] is False
    assert client.post("/api/photos", files={"files": ("a.jpg", b"x")}).status_code == 501
