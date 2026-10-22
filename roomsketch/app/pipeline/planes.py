"""Sequential plane detection (normal-guided RANSAC + connected components)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class Plane:
    normal: np.ndarray  # unit vector
    d: float  # plane: normal . x + d = 0
    inliers: np.ndarray  # indices into the point array
    area: float = 0.0  # occupied area estimate (m^2)
    extra: dict = field(default_factory=dict)

    def distance(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.normal + self.d


def fit_plane(pts: np.ndarray) -> tuple[np.ndarray, float]:
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    return n, float(-n @ c)


def plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def largest_component(pts: np.ndarray, n: np.ndarray, cell: float) -> tuple[np.ndarray, float]:
    """Mask of the points in the largest 8-connected patch of the plane, plus its area."""
    u, v = plane_basis(n)
    uv = np.stack([pts @ u, pts @ v], axis=1)
    ij = np.floor((uv - uv.min(axis=0)) / cell).astype(np.int64)
    shape = ij.max(axis=0) + 1
    if shape[0] * shape[1] > 25_000_000:  # absurdly spread "plane": refuse
        return np.zeros(len(pts), bool), 0.0
    grid = np.zeros(shape, dtype=bool)
    grid[ij[:, 0], ij[:, 1]] = True
    # Bridge single-cell gaps so sparse scans do not fragment.
    closed = ndimage.binary_closing(grid, structure=np.ones((3, 3)), border_value=0) | grid
    labels, count = ndimage.label(closed, structure=np.ones((3, 3)))
    if count == 0:
        return np.zeros(len(pts), bool), 0.0
    pl = labels[ij[:, 0], ij[:, 1]]
    best = np.bincount(pl, minlength=count + 1)[1:].argmax() + 1
    mask = pl == best
    area = float(np.count_nonzero(grid & (labels == best))) * cell * cell
    return mask, area


def detect_planes(
    points: np.ndarray,
    normals: np.ndarray,
    curvature: np.ndarray,
    dist: float,
    cell: float,
    min_inliers: int,
    angle_deg: float = 20.0,
    max_planes: int = 60,
    hypotheses: int = 256,
    seed: int = 0,
) -> tuple[list[Plane], np.ndarray]:
    """Extract planes largest-first. Returns (planes, indices of unassigned points)."""
    rng = np.random.default_rng(seed)
    cos_t = np.cos(np.radians(angle_deg))
    remaining = np.arange(len(points))
    flat = curvature < max(np.percentile(curvature, 60), 1e-4)
    planes: list[Plane] = []
    fails = 0
    while len(planes) < max_planes and remaining.size >= min_inliers and fails < 6:
        cand = remaining[flat[remaining]]
        if cand.size < 10:
            cand = remaining
        seeds = rng.choice(cand, min(hypotheses, cand.size), replace=False)
        sub = remaining if remaining.size <= 25_000 else rng.choice(remaining, 25_000, replace=False)
        hn = normals[seeds]
        hd = -np.einsum("ij,ij->i", hn, points[seeds])
        ok = (np.abs(points[sub] @ hn.T + hd) < dist) & (np.abs(normals[sub] @ hn.T) > cos_t)
        best = int(ok.sum(axis=0).argmax())
        n, d = hn[best], hd[best]

        P, N = points[remaining], normals[remaining]
        mask = None
        for _ in range(3):
            mask = (np.abs(P @ n + d) < dist) & (np.abs(N @ n) > cos_t)
            if mask.sum() < 3:
                break
            n, d = fit_plane(P[mask])
        if mask is None or mask.sum() < min_inliers:
            fails += 1
            continue
        inl = remaining[mask]
        comp, area = largest_component(points[inl], n, cell)
        inl = inl[comp]
        if inl.size < min_inliers:
            fails += 1
            continue
        n, d = fit_plane(points[inl])
        planes.append(Plane(normal=n, d=d, inliers=inl, area=area))
        remaining = np.setdiff1d(remaining, inl, assume_unique=True)
        fails = 0
    return planes, remaining
