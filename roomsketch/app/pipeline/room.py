"""Turn a gravity-aligned (+Z up, metres) point cloud into a parametric room.

Output is a plain dict (JSON friendly) describing:
  floor/ceiling heights, the room outline polygon (inner face of walls),
  openings (doors / windows) per outline edge, and furniture boxes.
All coordinates are in a "model" frame: walls squared to the X/Y axes,
floor at z = 0 and the outline's bounding box starting at the origin.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import shapely
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union

from .planes import detect_planes
from .pointcloud import estimate_normals, voxel_downsample

Progress = Callable[[float, str], None]


def _noop(_f: float, _m: str) -> None:
    pass


def extract_room(points: np.ndarray, normals: np.ndarray | None = None, *, sparse: bool = False,
                 progress: Progress = _noop) -> dict:
    if len(points) < 50:
        raise ValueError("Not enough points to reconstruct a room")

    # ---- 1. Downsample + normals -------------------------------------------------
    progress(0.05, "Downsampling point cloud")
    pts, nrm = points, normals
    if not sparse:
        pts, nrm = voxel_downsample(points, 0.025, normals)
    progress(0.12, f"Estimating normals for {len(pts):,} points")
    pca_n, curv, spacing = estimate_normals(pts, k=10 if sparse else 16)
    if nrm is None:
        nrm = pca_n
    spacing = max(spacing, 0.005)
    dist = float(np.clip(2.5 * spacing, 0.02, 0.06))
    if sparse:
        # SfM points cluster on texture, so nearest-neighbour spacing understates the gaps.
        # Use the mean spacing implied by local density instead: r_k ~ sqrt(k / (pi * rho)).
        k = 8
        rk, _ = cKDTree(pts).query(pts, k=k + 1, workers=-1)
        spacing = max(spacing, float(np.median(rk[:, k])) * np.sqrt(np.pi / k))
        dist = max(dist, 0.03)
    cell = float(max(0.05, 3.0 * spacing))
    min_inliers = int(np.clip(0.12 / spacing**2, 25, 400))

    # ---- 2. Planes ----------------------------------------------------------------
    progress(0.2, "Detecting planes")
    planes, _ = detect_planes(pts, nrm, curv, dist=dist, cell=cell, min_inliers=min_inliers,
                              angle_deg=25 if sparse else 20)
    if not planes:
        raise ValueError("No planar surfaces found - is this a room scan?")
    progress(0.45, f"Found {len(planes)} planes")

    horiz = [p for p in planes if abs(p.normal[2]) > math.cos(math.radians(12))]
    vert = [p for p in planes if abs(p.normal[2]) < math.sin(math.radians(12))]

    # ---- 3. Floor + ceiling -------------------------------------------------------
    for p in horiz:
        if p.normal[2] < 0:
            p.normal, p.d = -p.normal, -p.d
        p.extra["z"] = -p.d
    big_h = [p for p in horiz if p.area >= 0.8]
    floor = min(big_h, key=lambda p: p.extra["z"]) if big_h else None
    floor_z = floor.extra["z"] if floor else float(np.percentile(pts[:, 2], 1.5))
    ceil_cands = [p for p in big_h if p.extra["z"] - floor_z > 1.8]
    ceiling = max(ceil_cands, key=lambda p: p.extra["z"]) if ceil_cands else None
    ceiling_estimated = ceiling is None
    ceiling_z = ceiling.extra["z"] if ceiling else float(np.percentile(pts[:, 2], 99))
    height = ceiling_z - floor_z
    if height < 1.0:
        raise ValueError("Could not find a floor and ceiling at least 1 m apart")

    # ---- 4. Walls -------------------------------------------------------------------
    progress(0.5, "Finding walls")
    walls = []
    for p in vert:
        P = pts[p.inliers]
        z_lo, z_hi = np.percentile(P[:, 2], [3, 97])
        n2 = p.normal[:2] / np.linalg.norm(p.normal[:2])
        t = np.array([-n2[1], n2[0]])
        s = P[:, :2] @ t
        length = float(np.percentile(s, 98) - np.percentile(s, 2))
        if z_hi - z_lo < max(0.8, 0.4 * height) or length < 0.4:
            continue
        walls.append({"n": n2, "c": float(np.median(P[:, :2] @ n2)), "idx": p.inliers,
                      "w": len(p.inliers), "plane": p})

    # Dominant orientation (Manhattan) from wall normals, weighted by support.
    if walls:
        ang = np.array([math.atan2(w["n"][1], w["n"][0]) for w in walls])
        wts = np.array([w["w"] for w in walls], float)
        phi = float(np.angle(np.sum(wts * np.exp(4j * ang))) / 4.0)
    else:
        phi = 0.0
    Rz = _rot_z(-phi)
    pts = pts @ Rz.T
    nrm = nrm @ Rz.T
    for w in walls:
        w["n"] = Rz[:2, :2] @ w["n"]
        _snap_wall(w, pts)
    walls = _merge_walls(walls, pts)
    progress(0.6, f"{len(walls)} walls")

    plane_role = {}
    if floor is not None:
        plane_role[id(floor)] = "floor"
    if ceiling is not None:
        plane_role[id(ceiling)] = "ceiling"

    # ---- 5. Room outline from the wall-line arrangement ---------------------------
    progress(0.65, "Solving room outline")
    evidence = np.zeros(len(pts), bool)
    if floor is not None:
        evidence[floor.inliers] = True
    if ceiling is not None:
        evidence[ceiling.inliers] = True
    if evidence.sum() < 0.05 * len(pts):
        evidence |= (pts[:, 2] < floor_z + 0.1) | (pts[:, 2] > ceiling_z - 0.1)
    in_band = (pts[:, 2] > floor_z - 0.2) & (pts[:, 2] < ceiling_z + 0.2)
    polygon = None
    if len(walls) >= 3:
        polygon = _outline_from_arrangement(walls, pts[evidence & in_band, :2], pts[in_band, :2], spacing)
    if polygon is None:
        polygon = _outline_fallback(pts[evidence & in_band, :2])
    ring = _clean_ring(list(polygon.exterior.coords))
    if len(ring) < 3:
        raise ValueError("Could not close the room outline")

    # ---- 6. Openings -----------------------------------------------------------------
    progress(0.75, "Detecting doors and windows")
    openings = []
    for i in range(len(ring)):
        a, b = np.array(ring[i]), np.array(ring[(i + 1) % len(ring)])
        for op in _detect_openings(pts, a, b, floor_z, ceiling_z, spacing):
            op["edge"] = i
            openings.append(op)

    # ---- 7. Furniture ----------------------------------------------------------------
    progress(0.85, "Blocking out furniture")
    furniture = _detect_furniture(pts, ring, walls, floor_z, ceiling_z, spacing, plane_role, planes)

    # ---- 8. Move into model frame ------------------------------------------------------
    ring_np = np.array(ring)
    origin = np.array([ring_np[:, 0].min(), ring_np[:, 1].min(), floor_z])
    ring_np = ring_np - origin[:2]
    for f in furniture:
        f["cx"] -= origin[0]
        f["cy"] -= origin[1]
        f["z0"] -= floor_z
        f["z1"] -= floor_z
    transform = np.eye(4)
    transform[:3, :3] = Rz
    transform[:3, 3] = -origin

    poly = Polygon(ring_np)
    edge_lengths = [float(np.linalg.norm(ring_np[(i + 1) % len(ring_np)] - ring_np[i]))
                    for i in range(len(ring_np))]
    progress(0.95, "Room solved")
    return {
        "height": round(height, 4),
        "ceiling_estimated": ceiling_estimated,
        "polygon": ring_np.round(4).tolist(),
        "openings": openings,
        "furniture": furniture,
        "transform": transform.tolist(),  # input frame -> model frame
        "stats": {
            "points": int(len(points)),
            "working_points": int(len(pts)),
            "spacing": round(spacing, 4),
            "planes": len(planes),
            "walls": len(walls),
        },
        "metrics": {
            "floor_area": round(poly.area, 3),
            "perimeter": round(poly.length, 3),
            "volume": round(poly.area * height, 3),
            "edge_lengths": [round(x, 3) for x in edge_lengths],
        },
    }


# ---------------------------------------------------------------------------------
# Walls
# ---------------------------------------------------------------------------------

def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _snap_wall(w: dict, pts: np.ndarray, tol_deg: float = 6.0) -> None:
    ang = math.atan2(w["n"][1], w["n"][0])
    q = round(ang / (math.pi / 2)) * (math.pi / 2)
    if abs(ang - q) < math.radians(tol_deg):
        w["n"] = np.array([math.cos(q), math.sin(q)])
        w["n"][np.abs(w["n"]) < 1e-12] = 0.0
    w["c"] = float(np.median(pts[w["idx"], :2] @ w["n"]))


def _merge_walls(walls: list[dict], pts: np.ndarray) -> list[dict]:
    merged: list[dict] = []
    for w in sorted(walls, key=lambda w: -w["w"]):
        for m in merged:
            dot = float(m["n"] @ w["n"])
            if abs(dot) < math.cos(math.radians(3)):
                continue
            c = w["c"] * np.sign(dot)
            if abs(c - m["c"]) < 0.06:
                m["idx"] = np.concatenate([m["idx"], w["idx"]])
                m["w"] += w["w"]
                m["c"] = float(np.median(pts[m["idx"], :2] @ m["n"]))
                break
        else:
            merged.append(dict(w))
    return merged


def _outline_from_arrangement(walls, evidence_xy, band_xy, spacing) -> Polygon | None:
    if len(evidence_xy) < 20:
        return None
    lo = band_xy.min(axis=0) - 1.0
    hi = band_xy.max(axis=0) + 1.0
    frame = box(lo[0], lo[1], hi[0], hi[1])
    R = float(np.linalg.norm(hi - lo)) * 2
    lines = [frame.boundary]
    for w in walls:
        n = w["n"]
        t = np.array([-n[1], n[0]])
        p0 = n * w["c"]
        seg = LineString([p0 - t * R, p0 + t * R]).intersection(frame)
        if not seg.is_empty:
            lines.append(seg)
    cells = list(polygonize(unary_union(lines)))
    if not cells:
        return None

    g = float(max(0.08, 2.5 * spacing))
    ij = np.floor((evidence_xy - lo) / g).astype(int)
    shape = np.floor((hi - lo) / g).astype(int) + 1
    occ = np.zeros(shape, bool)
    occ[ij[:, 0], ij[:, 1]] = True
    occ = ndimage.binary_closing(occ, np.ones((3, 3))) | occ
    gx, gy = np.meshgrid(lo[0] + (np.arange(shape[0]) + 0.5) * g,
                         lo[1] + (np.arange(shape[1]) + 0.5) * g, indexing="ij")
    inside = []
    for c in cells:
        if c.area < 0.02:
            continue
        minx, miny, maxx, maxy = c.bounds
        i0, i1 = int((minx - lo[0]) / g), int((maxx - lo[0]) / g) + 1
        j0, j1 = int((miny - lo[1]) / g), int((maxy - lo[1]) / g) + 1
        sx, sy = gx[i0:i1, j0:j1], gy[i0:i1, j0:j1]
        m = shapely.contains_xy(c, sx, sy)
        total = int(m.sum())
        if total == 0:
            # Cell smaller than a raster cell: sample its centroid instead.
            pt = c.representative_point()
            ii, jj = int((pt.x - lo[0]) / g), int((pt.y - lo[1]) / g)
            cov = 1.0 if occ[min(ii, shape[0] - 1), min(jj, shape[1] - 1)] else 0.0
        else:
            cov = float(occ[i0:i1, j0:j1][m].sum()) / total
        if cov >= 0.3:
            inside.append(c)
    if not inside:
        return None
    room = unary_union(inside)
    if isinstance(room, MultiPolygon):
        room = max(room.geoms, key=lambda p: p.area)
    if room.area < 1.0:
        return None
    return orient(Polygon(room.exterior), 1.0)


def _outline_fallback(xy: np.ndarray) -> Polygon:
    """Min-area rectangle of the floor/ceiling evidence."""
    hull = shapely.MultiPoint(xy).convex_hull
    rect = hull.minimum_rotated_rectangle
    return orient(Polygon(rect.exterior), 1.0)


def _clean_ring(coords, min_len: float = 0.03, sin_tol: float = 0.03) -> list[tuple[float, float]]:
    pts = [np.array(c[:2], float) for c in coords]
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            ab, bc = b - a, c - b
            la, lb = np.linalg.norm(ab), np.linalg.norm(bc)
            if la < min_len or lb < min_len:
                del pts[i]
                changed = True
                break
            cross = (ab[0] * bc[1] - ab[1] * bc[0]) / (la * lb)
            if abs(cross) < sin_tol and ab @ bc > 0:
                del pts[i]
                changed = True
                break
    ring = [tuple(map(float, p)) for p in pts]
    if Polygon(ring).exterior.is_ccw is False:
        ring.reverse()
    return ring


# ---------------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------------

def _detect_openings(pts, a, b, floor_z, ceiling_z, spacing) -> list[dict]:
    L = float(np.linalg.norm(b - a))
    H = ceiling_z - floor_z
    if L < 0.6:
        return []
    d = (b - a) / L
    n_out = np.array([d[1], -d[0]])  # CCW outline -> interior on the left
    rel = pts[:, :2] - a
    u = rel @ d
    s = rel @ n_out
    v = pts[:, 2] - floor_z
    in_span = (u > 0) & (u < L) & (v > 0) & (v < H)
    tol = max(0.04, 2.0 * spacing)
    on_wall = in_span & (np.abs(s) < tol)
    occluder = in_span & (s < -tol) & (s > -1.2) & (v > 0.06) & (v < H - 0.06)
    if on_wall.sum() < 30:
        return []

    # Raster cell sized so a scanned wall cell holds a few points. Nearest-neighbour
    # spacing underestimates this for photogrammetry (points cluster on features).
    density = on_wall.sum() / (L * H)
    c = float(np.clip(np.sqrt(4.0 / density), max(0.05, 2.0 * spacing), 0.2))
    nu, nv = int(math.ceil(L / c)), int(math.ceil(H / c))

    def raster(mask):
        g = np.zeros((nu, nv), bool)
        iu = np.clip((u[mask] / c).astype(int), 0, nu - 1)
        iv = np.clip((v[mask] / c).astype(int), 0, nv - 1)
        g[iu, iv] = True
        return g

    W = raster(on_wall)
    W = ndimage.binary_closing(W, np.ones((3, 3))) | W
    O = ndimage.binary_dilation(raster(occluder), np.ones((3, 3)))
    empty = ~W & ~O
    labels, count = ndimage.label(empty)
    out = []
    for lab in range(1, count + 1):
        cells = np.argwhere(labels == lab)
        i0, j0 = cells.min(axis=0)
        i1, j1 = cells.max(axis=0) + 1
        u0, u1, v0, v1 = i0 * c, min(i1 * c, L), j0 * c, min(j1 * c, H)
        w, h = u1 - u0, v1 - v0
        fill = len(cells) / float((i1 - i0) * (j1 - j0))
        if w < 0.5 or h < 0.5 or fill < 0.6:
            continue
        touches_floor = v0 < 0.12
        touches_ceiling = v1 > H - 0.12
        if w > 0.9 * L and h > 0.85 * H:
            continue  # whole wall missing: unscanned, not an opening
        if touches_ceiling and not touches_floor:
            continue  # usually an unscanned top band
        if touches_floor:
            if h < 1.75:
                continue  # furniture shadow
            kind = "opening" if touches_ceiling else "door"
            if kind == "opening" and w > 2.6:
                continue
        else:
            if v0 < 0.2:
                continue
            kind = "window"
        # Require a wall frame around it (rejects unscanned blobs).
        frame = []
        if i0 > 0:
            frame.append(W[i0 - 1, j0:j1])
        if i1 < nu:
            frame.append(W[i1, j0:j1])
        if j1 < nv:
            frame.append(W[i0:i1, j1])
        if not touches_floor and j0 > 0:
            frame.append(W[i0:i1, j0 - 1])
        if not frame or np.concatenate(frame).mean() < 0.5:
            continue

        u0, u1, v0, v1 = _refine_opening(u[on_wall], v[on_wall], u0, u1, v0, v1, c, L, H)
        if touches_floor:
            v0 = 0.0
        if kind == "opening":
            v1 = H
        out.append({"kind": kind, "u0": round(u0, 3), "u1": round(u1, 3),
                    "v0": round(v0, 3), "v1": round(v1, 3)})
    return out


def _refine_opening(wu, wv, u0, u1, v0, v1, c, L, H):
    """Tighten the raster bbox to the actual wall-point edges around the opening."""
    def edge(vals, mask, pick, default):
        sel = vals[mask]
        if sel.size < 3:
            return default
        return float(np.percentile(sel, 95 if pick == "max" else 5))

    mid_v = (wv > v0 + c) & (wv < v1 - c)
    mid_u = (wu > u0 + c) & (wu < u1 - c)
    nu0 = edge(wu, mid_v & (wu > u0 - 2 * c) & (wu < u0 + c), "max", u0) if u0 > 0 else 0.0
    nu1 = edge(wu, mid_v & (wu < u1 + 2 * c) & (wu > u1 - c), "min", u1) if u1 < L else L
    nv0 = edge(wv, mid_u & (wv > v0 - 2 * c) & (wv < v0 + c), "max", v0) if v0 > 0.12 else v0
    nv1 = edge(wv, mid_u & (wv < v1 + 2 * c) & (wv > v1 - c), "min", v1) if v1 < H else H
    if nu1 - nu0 < 0.3 or nv1 - nv0 < 0.3:
        return u0, u1, v0, v1
    return nu0, nu1, nv0, nv1


# ---------------------------------------------------------------------------------
# Furniture
# ---------------------------------------------------------------------------------

def _detect_furniture(pts, ring, walls, floor_z, ceiling_z, spacing, plane_role, planes) -> list[dict]:
    room = Polygon(ring)
    inner = room.buffer(-0.06)
    if inner.is_empty:
        return []
    keep = np.ones(len(pts), bool)
    for p in planes:
        if plane_role.get(id(p)) in ("floor", "ceiling"):
            keep[p.inliers] = False
    keep &= (pts[:, 2] > floor_z + 0.04) & (pts[:, 2] < ceiling_z - 0.1)
    # Drop points on the outline walls.
    ring_np = np.array(ring)
    for i in range(len(ring_np)):
        a, b = ring_np[i], ring_np[(i + 1) % len(ring_np)]
        L = np.linalg.norm(b - a)
        d = (b - a) / L
        rel = pts[:, :2] - a
        near = (np.abs(rel @ np.array([d[1], -d[0]])) < 0.06) & (rel @ d > -0.1) & (rel @ d < L + 0.1)
        keep &= ~near
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        return []
    idx = idx[shapely.contains_xy(inner, pts[idx, 0], pts[idx, 1])]
    if idx.size < 30:
        return []
    P = pts[idx]

    c = float(max(0.05, 2.0 * spacing))
    lo = P.min(axis=0)
    ijk = np.floor((P - lo) / c).astype(int)
    grid = np.zeros(ijk.max(axis=0) + 1, bool)
    grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    grid = ndimage.binary_closing(grid, np.ones((3, 3, 3))) | grid
    labels, count = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    pl = labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    min_pts = max(25, int(0.02 / (spacing * spacing)))
    boxes = []
    for lab in range(1, count + 1):
        Q = P[pl == lab]
        if len(Q) < min_pts:
            continue
        z0, z1 = np.percentile(Q[:, 2], [1, 99])
        if z1 - z0 < 0.15:
            continue
        hull = shapely.MultiPoint(Q[:, :2]).convex_hull
        if hull.geom_type != "Polygon" or hull.area < 0.04:
            continue
        rect = hull.minimum_rotated_rectangle
        xy = np.array(rect.exterior.coords)[:4]
        e1, e2 = xy[1] - xy[0], xy[2] - xy[1]
        yaw = math.atan2(e1[1], e1[0])
        q = round(yaw / (math.pi / 2)) * (math.pi / 2)
        if abs(yaw - q) < math.radians(8):  # square to the room
            yaw = q
            ca, sa = math.cos(-yaw), math.sin(-yaw)
            loc = Q[:, :2] @ np.array([[ca, -sa], [sa, ca]]).T
            mn, mx = np.percentile(loc, 1, axis=0), np.percentile(loc, 99, axis=0)
            w, dpt = mx - mn
            ctr_loc = (mn + mx) / 2
            ca, sa = math.cos(yaw), math.sin(yaw)
            ctr = np.array([[ca, -sa], [sa, ca]]) @ ctr_loc
        else:
            w, dpt = np.linalg.norm(e1), np.linalg.norm(e2)
            ctr = xy.mean(axis=0)
        if max(w, dpt) < 0.25:
            continue
        if z0 < floor_z + 0.2:
            z0 = floor_z
        boxes.append({"cx": float(ctr[0]), "cy": float(ctr[1]), "w": round(float(w), 3),
                      "d": round(float(dpt), 3), "yaw": round(float(yaw), 5),
                      "z0": float(z0), "z1": float(z1)})
    boxes.sort(key=lambda b: -b["w"] * b["d"])
    return boxes[:40]
