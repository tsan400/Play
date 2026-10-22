"""Synthetic LiDAR-like room scans with known ground truth (tests + demo)."""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import Polygon, box

# L-shaped room, metres, CCW. Ground truth used by the tests.
DEMO_OUTLINE = [(0.0, 0.0), (6.0, 0.0), (6.0, 3.2), (3.6, 3.2), (3.6, 5.0), (0.0, 5.0)]
DEMO_HEIGHT = 2.6
# (edge index, u0, u1, v0, v1, kind) measured along the edge from its start vertex.
DEMO_OPENINGS = [
    (0, 1.0, 1.9, 0.0, 2.1, "door"),
    (1, 0.9, 2.3, 0.9, 2.1, "window"),
    (5, 1.5, 3.3, 0.8, 2.2, "window"),
]
# (cx, cy, w, d, height)
DEMO_FURNITURE = [
    (1.6, 3.6, 1.6, 0.9, 0.75),  # table
    (4.6, 1.2, 2.0, 0.9, 0.85),  # sofa
]


def _sample_poly(origin, U, V, poly: Polygon, density: float, rng) -> np.ndarray:
    n = rng.poisson(poly.area * density)
    if n == 0:
        return np.zeros((0, 3))
    minx, miny, maxx, maxy = poly.bounds
    out = []
    need = n
    while need > 0:
        cand = rng.uniform([minx, miny], [maxx, maxy], size=(int(need * 1.6) + 16, 2))
        cand = cand[shapely.contains_xy(poly, cand[:, 0], cand[:, 1])]
        out.append(cand[:need])
        need -= len(out[-1])
    uv = np.vstack(out)
    return origin + uv[:, :1] * U + uv[:, 1:] * V


def demo_scan(density: float = 2500.0, noise: float = 0.006, yaw_deg: float = 23.0,
              offset=(2.0, -1.5, 0.4), scan_ceiling: bool = True, seed: int = 0) -> np.ndarray:
    """Points of the demo room as a LiDAR app would export them (+Z up, rotated, offset)."""
    rng = np.random.default_rng(seed)
    H = DEMO_HEIGHT
    outline = Polygon(DEMO_OUTLINE)
    parts = []
    # Floor, minus furniture footprints (occluded).
    floor = outline
    for cx, cy, w, d, h in DEMO_FURNITURE:
        floor = floor.difference(box(cx - w / 2, cy - d / 2, cx + w / 2, cy + d / 2))
    Z = np.array([0, 0, 1.0])
    parts.append(_sample_poly(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), floor, density, rng))
    if scan_ceiling:
        parts.append(_sample_poly(np.array([0, 0, H]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                                  outline, density * 0.6, rng))
    ring = DEMO_OUTLINE
    for i in range(len(ring)):
        a, b = np.array(ring[i]), np.array(ring[(i + 1) % len(ring)])
        L = float(np.linalg.norm(b - a))
        d = (b - a) / L
        face = box(0, 0, L, H)
        for e, u0, u1, v0, v1, kind in DEMO_OPENINGS:
            if e == i:
                face = face.difference(box(u0, v0, u1, v1))
                # Something seen through the opening, 2 m outside.
                n_out = np.array([d[1], -d[0], 0])
                parts.append(_sample_poly(np.array([a[0], a[1], 0]) + n_out * 2.0, np.array([d[0], d[1], 0]),
                                          Z, box(u0 - 0.5, max(0, v0 - 0.3), u1 + 0.5, v1 + 0.3),
                                          density * 0.15, rng))
        parts.append(_sample_poly(np.array([a[0], a[1], 0]), np.array([d[0], d[1], 0]), Z, face, density, rng))
    for cx, cy, w, d, h in DEMO_FURNITURE:
        x0, y0 = cx - w / 2, cy - d / 2
        X, Y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        parts.append(_sample_poly(np.array([x0, y0, h]), X, Y, box(0, 0, w, d), density, rng))
        for org, U, L in [((x0, y0), X, w), ((x0, y0 + d), X, w), ((x0, y0), Y, d), ((x0 + w, y0), Y, d)]:
            parts.append(_sample_poly(np.array([org[0], org[1], 0]), U, Z, box(0, 0, L, h), density, rng))
    pts = np.vstack(parts)
    pts += rng.normal(0, noise, pts.shape)
    a = math.radians(yaw_deg)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return pts @ R.T + np.asarray(offset)


def write_ply(path, pts: np.ndarray, y_up: bool = False) -> None:
    """Binary little-endian PLY, optionally converted to Y-up like ARKit exports."""
    P = pts.astype(np.float32)
    if y_up:
        P = np.stack([P[:, 0], P[:, 2], -P[:, 1]], axis=1)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(P.tobytes())
