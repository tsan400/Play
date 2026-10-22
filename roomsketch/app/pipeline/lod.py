"""Build a low-detail (LOD) planar model from a parametric room.

The model is a list of planar faces (outer ring + holes, 3D, CCW seen from the
front) on named layers, plus the deduplicated edge set. This is what the toon
viewer draws and what every exporter writes.
"""
from __future__ import annotations

import math

import mapbox_earcut
import numpy as np
from shapely.affinity import translate
from shapely.geometry import box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

LAYERS = ["walls", "floor", "ceiling", "openings", "furniture"]


def build_lod(room: dict, wall_thickness: float = 0.15, slab: float = 0.12, scale: float = 1.0) -> dict:
    """`scale` multiplies every length (used after the user calibrates a dimension)."""
    s = float(scale)
    P = np.array(room["polygon"], float) * s
    H = float(room["height"]) * s
    t = wall_thickness  # wall/slab thickness are design choices, not measured: not scaled
    n = len(P)
    D = np.array([(P[(i + 1) % n] - P[i]) / np.linalg.norm(P[(i + 1) % n] - P[i]) for i in range(n)])
    N_out = np.stack([D[:, 1], -D[:, 0]], axis=1)
    Q = _offset_ring(P, D, N_out, t)

    faces: list[dict] = []

    def add(layer, outer, holes=(), normal=None):
        outer = [list(map(float, p)) for p in outer]
        holes = [[list(map(float, p)) for p in h] for h in holes]
        if normal is not None:
            outer, holes = _orient(outer, holes, np.asarray(normal, float))
        faces.append({"layer": layer, "outer": outer, "holes": holes})

    by_edge: dict[int, list[dict]] = {}
    for op in room.get("openings", []):
        by_edge.setdefault(op["edge"], []).append(op)

    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        L = float(np.linalg.norm(b - a))
        d, no = D[i], N_out[i]
        off = float((Q[i] - P[i]) @ d)  # where the outer face starts along the edge
        Lo = float(np.linalg.norm(Q[(i + 1) % n] - Q[i]))
        rects = []
        for op in by_edge.get(i, []):
            r = box(op["u0"] * s, op["v0"] * s, op["u1"] * s, op["v1"] * s).intersection(box(0, 0, L, H))
            if not r.is_empty and r.area > 1e-4:
                rects.append(r)
        cut = unary_union(rects) if rects else None

        def wall_face(origin2, length, shift, normal):
            face = box(0, 0, length, H)
            if cut is not None:
                face = face.difference(translate(cut, -shift, 0))
            for poly in _polys(face):
                to3 = lambda uv: [origin2[0] + d[0] * uv[0], origin2[1] + d[1] * uv[0], uv[1]]
                add("walls", [to3(c) for c in poly.exterior.coords[:-1]],
                    [[to3(c) for c in h.coords[:-1]] for h in poly.interiors], normal)

        wall_face(a, L, 0.0, [-no[0], -no[1], 0])  # inner face looks into the room
        wall_face(Q[i], Lo, off, [no[0], no[1], 0])

        # Reveals (jambs, head, sill) through the wall thickness.
        for r in rects:
            u0, v0, u1, v1 = r.bounds
            pin = lambda u, v: [a[0] + d[0] * u, a[1] + d[1] * u, v]
            pout = lambda u, v: [a[0] + d[0] * u + no[0] * t, a[1] + d[1] * u + no[1] * t, v]
            sides = []
            if u0 > 1e-3:
                sides.append(((u0, v0), (u0, v1), [d[0], d[1], 0]))
            if u1 < L - 1e-3:
                sides.append(((u1, v0), (u1, v1), [-d[0], -d[1], 0]))
            if v1 < H - 1e-3:
                sides.append(((u0, v1), (u1, v1), [0, 0, -1]))
            if v0 > 1e-3:
                sides.append(((u0, v0), (u1, v0), [0, 0, 1]))
            for (pu, pv), (qu, qv), nrm in sides:
                add("walls", [pin(pu, pv), pin(qu, qv), pout(qu, qv), pout(pu, pv)], normal=nrm)

    # Wall tops: ring between outer and inner outline.
    add("walls", [[x, y, H] for x, y in Q], [[[x, y, H] for x, y in P]], [0, 0, 1])

    # Floor slab (top face covers the whole outline incl. under walls / thresholds).
    _slab(add, "floor", Q, 0.0, -slab, inner=None)
    # Ceiling slab sits on the walls; its underside covers the room only.
    _slab(add, "ceiling", Q, H + slab, H, inner=P)

    # Door / window markers: simple panes so openings read in the toon view.
    for i in range(n):
        a, d, no = P[i], D[i], N_out[i]
        for op in by_edge.get(i, []):
            if op["kind"] != "window":
                continue
            u0, u1, v0, v1 = op["u0"] * s, op["u1"] * s, op["v0"] * s, op["v1"] * s
            mid = t * 0.5
            pt = lambda u, v: [a[0] + d[0] * u + no[0] * mid, a[1] + d[1] * u + no[1] * mid, v]
            um = (u0 + u1) / 2
            add("openings", [pt(u0, v0), pt(u1, v0), pt(u1, v1), pt(u0, v1)],
                [[pt(u0 + .04, v0 + .04), pt(um - .02, v0 + .04), pt(um - .02, v1 - .04), pt(u0 + .04, v1 - .04)],
                 [pt(um + .02, v0 + .04), pt(u1 - .04, v0 + .04), pt(u1 - .04, v1 - .04), pt(um + .02, v1 - .04)]],
                [-no[0], -no[1], 0])

    for f in room.get("furniture", []):
        _box(add, f, s)

    edges = _edges(faces)
    return {"faces": faces, "edges": edges, "layers": LAYERS,
            "bounds": _bounds(faces), "inner_outline": P.round(4).tolist(),
            "outer_outline": Q.round(4).tolist(), "height": H, "wall_thickness": t}


# ---------------------------------------------------------------------------------

def _offset_ring(P, D, N_out, t):
    n = len(P)
    Q = np.zeros_like(P)
    for i in range(n):
        j = i - 1
        p1, d1 = P[j] + N_out[j] * t, D[j]
        p2, d2 = P[i] + N_out[i] * t, D[i]
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-6:
            Q[i] = p2
        else:
            diff = p2 - p1
            k = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
            Q[i] = p1 + d1 * k
    return Q


def _polys(g):
    if g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [orient(g, 1.0)]
    return [orient(p, 1.0) for p in getattr(g, "geoms", []) if p.geom_type == "Polygon" and p.area > 1e-5]


def _slab(add, layer, Q, z_top, z_bot, inner=None):
    top = [[x, y, z_top] for x, y in Q]
    add(layer, top, normal=[0, 0, 1])
    # With `inner`, the underside covers the room only; the part resting on the wall tops is hidden.
    bot_ring = inner if inner is not None else Q
    add(layer, [[x, y, z_bot] for x, y in bot_ring], normal=[0, 0, -1])
    n = len(Q)
    for i in range(n):
        a, b = Q[i], Q[(i + 1) % n]
        d = (b - a) / np.linalg.norm(b - a)
        add(layer, [[a[0], a[1], z_bot], [b[0], b[1], z_bot], [b[0], b[1], z_top], [a[0], a[1], z_top]],
            normal=[d[1], -d[0], 0])


def _box(add, f, s):
    cx, cy, w, dd, yaw = f["cx"] * s, f["cy"] * s, f["w"] * s, f["d"] * s, f["yaw"]
    z0, z1 = f["z0"] * s, f["z1"] * s
    c, sn = math.cos(yaw), math.sin(yaw)
    ex, ey = np.array([c, sn]) * w / 2, np.array([-sn, c]) * dd / 2
    ctr = np.array([cx, cy])
    corners = [ctr - ex - ey, ctr + ex - ey, ctr + ex + ey, ctr - ex + ey]
    add("furniture", [[x, y, z1] for x, y in corners], normal=[0, 0, 1])
    add("furniture", [[x, y, z0] for x, y in corners], normal=[0, 0, -1])
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        add("furniture", [[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1]],
            normal=[d[1], -d[0], 0])


def newell(ring) -> np.ndarray:
    P = np.asarray(ring, float)
    Qn = np.roll(P, -1, axis=0)
    n = np.array([np.sum((P[:, 1] - Qn[:, 1]) * (P[:, 2] + Qn[:, 2])),
                  np.sum((P[:, 2] - Qn[:, 2]) * (P[:, 0] + Qn[:, 0])),
                  np.sum((P[:, 0] - Qn[:, 0]) * (P[:, 1] + Qn[:, 1]))])
    ln = np.linalg.norm(n)
    return n / ln if ln > 0 else n


def _orient(outer, holes, normal):
    if newell(outer) @ normal < 0:
        outer = outer[::-1]
    holes = [h[::-1] if newell(h) @ normal > 0 else h for h in holes]  # holes run clockwise
    return outer, holes


def _edges(faces):
    seen = {}
    for fi, f in enumerate(faces):
        for ring in [f["outer"], *f["holes"]]:
            for k in range(len(ring)):
                a, b = ring[k], ring[(k + 1) % len(ring)]
                ka = tuple(round(x, 4) for x in a)
                kb = tuple(round(x, 4) for x in b)
                if ka == kb:
                    continue
                key = (ka, kb) if ka < kb else (kb, ka)
                e = seen.setdefault(key, {"a": list(key[0]), "b": list(key[1]), "layers": set(), "normals": []})
                e["layers"].add(f["layer"])
                e["normals"].append(newell(f["outer"]))
    out = []
    for e in seen.values():
        # Edge shared only by coplanar faces is not a visible crease: drop it.
        ns = e["normals"]
        if len(ns) == 2 and ns[0] @ ns[1] > 0.9999:
            continue
        out.append({"a": e["a"], "b": e["b"], "layers": sorted(e["layers"])})
    return out


def _bounds(faces):
    pts = np.array([p for f in faces for p in f["outer"]])
    return {"min": pts.min(axis=0).round(4).tolist(), "max": pts.max(axis=0).round(4).tolist()}


def triangulate(face) -> tuple[np.ndarray, np.ndarray]:
    """Earcut a planar face with holes. Returns (vertices Nx3, triangles Mx3)."""
    rings = [face["outer"], *face["holes"]]
    verts = np.array([p for r in rings for p in r], float)
    nrm = newell(face["outer"])
    a = np.array([0, 0, 1.0]) if abs(nrm[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(nrm, a)
    u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    uv = np.stack([verts @ u, verts @ v], axis=1)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = mapbox_earcut.triangulate_float64(uv, ends).reshape(-1, 3)
    # Earcut winding follows the 2D basis; make triangles face along the face normal.
    if len(tri):
        t0 = verts[tri[0]]
        if np.cross(t0[1] - t0[0], t0[2] - t0[0]) @ nrm < 0:
            tri = tri[:, ::-1]
    return verts, tri
