"""Point cloud loading, downsampling and normal estimation.

Internal convention everywhere in the pipeline: metres, right-handed, +Z up.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# Target sample spacing when converting a scanned mesh to points.
MESH_SAMPLE_SPACING = 0.015
MAX_MESH_SAMPLES = 1_500_000


def load_scan(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a LiDAR scan export (PLY / OBJ / GLB / GLTF / STL / XYZ / PTS / TXT / CSV).

    Meshes are converted to an evenly sampled point cloud (so dense and sparse
    regions of the mesh contribute in proportion to their real area) and the
    face normals of the samples are returned. Returns points in the file's own
    axes; call `to_z_up` afterwards.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".xyz", ".pts", ".txt", ".csv"}:
        return _load_ascii_points(path), None

    import trimesh

    geom = trimesh.load(str(path), process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.dump() if isinstance(g, trimesh.Trimesh)]
        clouds = [g for g in geom.dump() if isinstance(g, trimesh.PointCloud)]
        if meshes:
            geom = trimesh.util.concatenate(meshes)
        elif clouds:
            geom = trimesh.PointCloud(np.vstack([c.vertices for c in clouds]))
        else:
            raise ValueError("No geometry found in file")

    if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
        area = float(geom.area)
        count = int(np.clip(area / MESH_SAMPLE_SPACING**2, 50_000, MAX_MESH_SAMPLES))
        pts, face_idx = trimesh.sample.sample_surface(geom, count, seed=0)
        normals = geom.face_normals[face_idx]
        return np.asarray(pts, dtype=np.float64), np.asarray(normals, dtype=np.float64)

    verts = np.asarray(getattr(geom, "vertices", []), dtype=np.float64)
    if len(verts) == 0:
        raise ValueError("File contains no points")
    return verts, None


def _load_ascii_points(path: Path) -> np.ndarray:
    rows = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue  # header line
    if not rows:
        raise ValueError("No points found in text file")
    return np.asarray(rows, dtype=np.float64)


def to_z_up(points: np.ndarray, up: str, normals: np.ndarray | None = None):
    """Rotate from a file's up axis ('y', '-y', 'z', '-z', 'x', '-x') to +Z up."""
    R = up_rotation(up)
    pts = points @ R.T
    nrm = normals @ R.T if normals is not None else None
    return pts, nrm


def up_rotation(up: str) -> np.ndarray:
    # Each matrix maps the named axis onto +Z with a proper rotation.
    table = {
        "z": np.eye(3),
        "-z": np.diag([1.0, -1.0, -1.0]),
        "y": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]),
        "-y": np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]),
        "x": np.array([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]]),
        "-x": np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]),
    }
    if up not in table:
        raise ValueError(f"Unknown up axis {up!r}")
    return table[up]


def voxel_downsample(points: np.ndarray, voxel: float, normals: np.ndarray | None = None):
    """Average points (and normals) that fall in the same voxel."""
    keys = np.floor(points / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    lin = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, inv, counts = np.unique(lin, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((counts.size, 3))
    np.add.at(out, inv, points)
    out /= counts[:, None]
    out_n = None
    if normals is not None:
        # Normals from a mesh may point either way; align to a per-voxel reference first.
        ref = np.zeros((counts.size, 3))
        ref[inv] = normals
        sign = np.sign(np.einsum("ij,ij->i", normals, ref[inv]))
        sign[sign == 0] = 1
        out_n = np.zeros((counts.size, 3))
        np.add.at(out_n, inv, normals * sign[:, None])
        length = np.linalg.norm(out_n, axis=1, keepdims=True)
        length[length == 0] = 1
        out_n /= length
    return out, out_n


def estimate_normals(points: np.ndarray, k: int = 16):
    """PCA normals. Returns (normals, curvature, median nearest-neighbour spacing)."""
    k = min(k, len(points))
    tree = cKDTree(points)
    dist, idx = tree.query(points, k=k, workers=-1)
    normals = np.empty_like(points)
    curvature = np.empty(len(points))
    chunk = 200_000
    for s in range(0, len(points), chunk):
        nb = points[idx[s : s + chunk]]
        nb = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb, nb)
        w, v = np.linalg.eigh(cov)
        normals[s : s + chunk] = v[:, :, 0]
        curvature[s : s + chunk] = w[:, 0] / np.maximum(w.sum(axis=1), 1e-12)
    spacing = float(np.median(dist[:, 1])) if k > 1 else 0.01
    return normals, curvature, spacing


def detect_up_axis(points: np.ndarray) -> str:
    """Guess which file axis is 'up' for a scan export.

    ARKit-based scanners are gravity aligned, so up is one of the file axes;
    only the axis and its sign are unknown. Two cues are combined:
      * floor + ceiling + furniture tops make the horizontal surface area the largest,
      * clutter (furniture) sits in the lower half of a room, which fixes the sign.
    """
    sample = points
    if len(sample) > 120_000:
        sample = sample[np.random.default_rng(0).choice(len(sample), 120_000, replace=False)]
    voxel = max(_extent(sample) / 250.0, 0.02)
    sample, _ = voxel_downsample(sample, voxel)
    normals, _, _ = estimate_normals(sample, k=12)

    area = np.abs(normals).astype(float)
    score = (area > 0.9).sum(axis=0).astype(float)  # count of points facing each axis
    axis = int(np.argmax(score))
    # Prefer Y (ARKit/glTF convention) when Y and the winner are close.
    if axis != 1 and score[1] > 0.85 * score[axis]:
        axis = 1

    coord = sample[:, axis]
    lo, hi = np.percentile(coord, [2, 98])
    mid = 0.5 * (lo + hi)
    # Clutter = points not lying on planes perpendicular to any axis.
    clutter = np.abs(normals).max(axis=1) < 0.9
    near_band = (coord > lo + 0.05 * (hi - lo)) & (coord < hi - 0.05 * (hi - lo))
    sel = clutter & near_band
    below = np.count_nonzero(coord[sel] < mid)
    above = np.count_nonzero(coord[sel] >= mid)
    name = "xyz"[axis]
    return name if below >= above else "-" + name


def _extent(points: np.ndarray) -> float:
    lo, hi = np.percentile(points, [1, 99], axis=0)
    return float(np.max(hi - lo))
