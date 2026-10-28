"""Clash-detection harness for Revit IFC exports.

Usage:
    python analyze.py <project_dir> [<project_dir> ...] --out viewer/data

Each project_dir holds one IFC per discipline (Architecture, Structural,
Mech/Plumb/Elec). For every project the harness:
  1. tessellates every building element in world coordinates,
  2. finds hard clashes between disciplines (AABB broad phase, FCL mesh
     narrow phase),
  3. classifies each clash in plain language with a severity and a
     recommended action,
  4. writes <project>.csv, <project>.json and one .glb per discipline for
     the 3D viewer.
"""
import argparse
import csv
import json
import multiprocessing
import re
import sys
import time
from pathlib import Path

import fcl
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as eutil
import numpy as np
import trimesh

FT = 3.28084        # metres -> feet
IN = 39.3701        # metres -> inches
TOL = 0.0125        # 1/2" overlap allowed before something counts as a clash (m)

# Never shown and never clashed.
SKIP = {"IfcOpeningElement", "IfcSpace", "IfcAnnotation", "IfcSite", "IfcGrid"}
# Shown in 3D but ignored for clash checks (loose content / finishes).
NO_CLASH = {"IfcFurnishingElement", "IfcCovering", "IfcRailing"}

STRUCT_PRIMARY = {"IfcBeam", "IfcColumn", "IfcFooting", "IfcMember"}
ASSEMBLIES = {"IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcRoof"}
MEP_CLASSES = {"IfcFlowSegment", "IfcFlowFitting", "IfcFlowTerminal",
               "IfcFlowController", "IfcFlowMovingDevice", "IfcDistributionElement"}

FRIENDLY = {
    "IfcWallStandardCase": "Wall", "IfcWall": "Wall", "IfcSlab": "Slab/Floor",
    "IfcBeam": "Beam", "IfcColumn": "Column", "IfcFooting": "Footing",
    "IfcMember": "Framing member", "IfcRoof": "Roof", "IfcDoor": "Door",
    "IfcWindow": "Window", "IfcStairFlight": "Stair", "IfcRailing": "Railing",
    "IfcCovering": "Ceiling/Covering", "IfcFlowSegment": "Pipe/Duct",
    "IfcFlowFitting": "Fitting", "IfcFlowTerminal": "Fixture/Terminal",
    "IfcFlowController": "Valve/Damper", "IfcFlowMovingDevice": "Fan/Pump",
    "IfcBuildingElementProxy": "Generic element",
    "IfcFurnishingElement": "Furniture",
}

DISCIPLINES = [("STRUCT", re.compile(r"struct", re.I)),
               ("MEP", re.compile(r"mech|plumb|elec|mep", re.I)),
               ("ARCH", re.compile(r"arch", re.I))]
DISC_NAME = {"ARCH": "Architectural", "STRUCT": "Structural", "MEP": "MEP"}


def discipline_of(path):
    for key, rx in DISCIPLINES:
        if rx.search(path.name):
            return key
    return "OTHER"


def element_meta(el, disc):
    level = ""
    container = eutil.get_container(el)
    if container is not None:
        level = container.Name or ""
    etype = eutil.get_type(el)
    tag = getattr(el, "Tag", None) or ""
    return {
        "guid": el.GlobalId, "disc": disc, "ifc": el.is_a(),
        "kind": FRIENDLY.get(el.is_a(), el.is_a().replace("Ifc", "")),
        "name": el.Name or "", "type": (etype.Name if etype else "") or "",
        "revit_id": tag if str(tag).isdigit() else "", "level": level,
    }


def load_model(path):
    """Tessellate one IFC file. Returns (elements, meshes) keyed by GUID."""
    disc = discipline_of(path)
    model = ifcopenshell.open(str(path))
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    settings.set("weld-vertices", True)
    elements, meshes = {}, {}
    products = [p for p in model.by_type("IfcProduct")
                if p.Representation and p.is_a() not in SKIP]
    it = ifcopenshell.geom.iterator(settings, model, multiprocessing.cpu_count(),
                                    include=products)
    if not it.initialize():
        return elements, meshes
    while True:
        shape = it.get()
        el = model.by_id(shape.id)
        g = shape.geometry
        v = np.array(g.verts, dtype=np.float64).reshape(-1, 3)
        f = np.array(g.faces, dtype=np.int64).reshape(-1, 3)
        if len(f):
            meta = element_meta(el, disc)
            meta["clash"] = el.is_a() not in NO_CLASH
            elements[el.GlobalId] = meta
            meshes[el.GlobalId] = trimesh.Trimesh(v, f, process=False)
        if not it.next():
            break
    return elements, meshes


def classify(a, b, depth_in):
    """Plain-language description, severity and action for a clash pair."""
    pair = {a["disc"], b["disc"]}
    s, m = ((a, b) if a["disc"] == "STRUCT" else (b, a))
    if pair == {"STRUCT", "MEP"}:
        if s["ifc"] in STRUCT_PRIMARY:
            return ("MEP through structure", "High",
                    f"{m['kind']} passes through a structural {s['kind'].lower()}.",
                    "Reroute MEP around the member, or have the structural engineer approve a penetration/sleeve.")
        return ("MEP through structure", "Medium",
                f"{m['kind']} passes through a structural {s['kind'].lower()}.",
                "Confirm a sleeve/core location with structural and show it on both models.")
    if pair == {"ARCH", "MEP"}:
        ar, mp = (a, b) if a["disc"] == "ARCH" else (b, a)
        if ar["ifc"] in {"IfcDoor", "IfcWindow", "IfcStairFlight", "IfcRoof"} or ar["ifc"] in STRUCT_PRIMARY:
            return ("MEP vs architecture", "High",
                    f"{mp['kind']} runs into a {ar['kind'].lower()}.",
                    "Reroute the MEP run; this location cannot be penetrated.")
        if ar["ifc"] in {"IfcWall", "IfcWallStandardCase", "IfcSlab"}:
            return ("MEP penetration", "Low",
                    f"{mp['kind']} penetrates a {ar['kind'].lower()} (normal routing).",
                    "Verify fire-rating/sleeve requirements; usually acceptable.")
        return ("MEP vs architecture", "Medium",
                f"{mp['kind']} overlaps a {ar['kind'].lower()}.",
                "Review location with the architect and adjust one of the elements.")
    # ARCH vs STRUCT
    ar = b if s is a else a
    if ar["ifc"] == s["ifc"] or (ar["kind"] == s["kind"]):
        return ("Duplicate element", "Info",
                f"{s['kind']} is modeled in both the architectural and structural models.",
                "Decide which model owns it and remove/hide the copy in the other.")
    if s["ifc"] in STRUCT_PRIMARY and ar["ifc"] in {"IfcDoor", "IfcWindow", "IfcStairFlight"}:
        return ("Structure vs opening", "High",
                f"Structural {s['kind'].lower()} cuts into a {ar['kind'].lower()}.",
                "Coordinate header/opening size; move the opening or resize the member.")
    if ar["ifc"] in ASSEMBLIES:
        return ("Structure in assembly", "Info",
                f"Structural {s['kind'].lower()} sits inside an architectural {ar['kind'].lower()} assembly.",
                "Expected for framing within walls/floors/roofs; check only if the assembly depth looks wrong.")
    if depth_in < 1.0:
        return ("Arch/Struct interface", "Info",
                f"Structural {s['kind'].lower()} slightly overlaps {ar['kind'].lower()} ({depth_in:.1f}\").",
                "Typically a modeling join; clean up if it affects quantities.")
    return ("Structure vs architecture", "Medium",
            f"Structural {s['kind'].lower()} overlaps architectural {ar['kind'].lower()} by {depth_in:.1f}\".",
            "Review with the architect; adjust the element or the framing.")


def merge_nearby(rows, radius=0.3):
    """Collapse MEP parts (pipe + its fittings) hitting the same element
    within ~1 ft into one clash, keeping the deepest overlap."""
    groups = {}
    for r in rows:
        key = (r["a"]["guid"], r["b"]["disc"]) if r["b"]["disc"] == "MEP" else (r["a"]["guid"], r["b"]["guid"])
        groups.setdefault(key, []).append(r)
    out = []
    for grp in groups.values():
        grp.sort(key=lambda r: -r["depth_in"])
        kept = []
        for r in grp:
            for k in kept:
                if np.linalg.norm(k["point"] - r["point"]) < radius:
                    k["parts"].append(r["b"]["guid"])
                    break
            else:
                r["parts"] = [r["b"]["guid"]]
                kept.append(r)
        out.extend(kept)
    return out


def fmt_ftin(m):
    total_in = m * IN
    sign = "-" if total_in < 0 else ""
    total_in = abs(total_in)
    ft = int(total_in // 12)
    inch = total_in - ft * 12
    return f"{sign}{ft}'-{inch:.1f}\""


def find_clashes(elements, meshes):
    guids = [g for g in meshes if elements[g]["clash"]]
    lo = np.array([meshes[g].bounds[0] for g in guids])
    hi = np.array([meshes[g].bounds[1] for g in guids])
    disc = np.array([elements[g]["disc"] for g in guids])
    objs = {}

    def fobj(g):
        if g not in objs:
            mm = meshes[g]
            bvh = fcl.BVHModel()
            bvh.beginModel(len(mm.vertices), len(mm.faces))
            bvh.addSubModel(mm.vertices, mm.faces)
            bvh.endModel()
            objs[g] = fcl.CollisionObject(bvh, fcl.Transform())
        return objs[g]

    clashes = []
    for i in range(len(guids)):
        # Broad phase: boxes that overlap by more than TOL on every axis.
        ov = (np.minimum(hi[i], hi[i + 1:]) - np.maximum(lo[i], lo[i + 1:]))
        cand = np.where((ov > TOL).all(axis=1) & (disc[i + 1:] != disc[i]))[0]
        for k in cand:
            j = i + 1 + k
            ga, gb = guids[i], guids[j]
            req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=True)
            res = fcl.CollisionResult()
            hit = fcl.collide(fobj(ga), fobj(gb), req, res)
            if not hit:
                # Meshes may be fully nested (pipe inside a thick wall): no
                # surface crossing, but one box sits inside the other.
                inner = ((lo[j] >= lo[i]) & (hi[j] <= hi[i])).all() or \
                        ((lo[i] >= lo[j]) & (hi[i] <= hi[j])).all()
                if not inner:
                    continue
                ca, cb = meshes[ga].bounds.mean(0), meshes[gb].bounds.mean(0)
                if not (meshes[ga].contains([cb])[0] or meshes[gb].contains([ca])[0]):
                    continue
            depth = float(ov[k].min())
            cmin = np.maximum(lo[i], lo[j]); cmax = np.minimum(hi[i], hi[j])
            point = res.contacts[0].pos if res.contacts else (cmin + cmax) / 2
            clashes.append((ga, gb, depth, np.asarray(point, float), cmin, cmax))
    return clashes


def write_glb(meshes, elements, disc, path):
    scene = trimesh.Scene()
    for g, mm in meshes.items():
        if elements[g]["disc"] == disc:
            # IFC is Z-up; glTF/three.js is Y-up.
            v = mm.vertices[:, [0, 2, 1]] * [1, 1, -1]
            scene.add_geometry(trimesh.Trimesh(v, mm.faces[:, [0, 2, 1]], process=False),
                               node_name=g, geom_name=g)
    if scene.geometry:
        scene.export(str(path))
        return True
    return False


def analyze_project(proj_dir, out_dir):
    proj_dir = Path(proj_dir)
    name = proj_dir.name
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    elements, meshes, models = {}, {}, []
    for ifc in sorted(proj_dir.glob("*.ifc")):
        t = time.time()
        e, m = load_model(ifc)
        d = discipline_of(ifc)
        models.append({"disc": d, "file": ifc.name, "elements": len(e)})
        elements.update(e); meshes.update(m)
        print(f"  {d:6s} {len(e):5d} elements  {time.time()-t:5.1f}s  {ifc.name}")

    t = time.time()
    raw = find_clashes(elements, meshes)
    print(f"  {len(raw)} clashes found in {time.time()-t:.1f}s")

    sev_rank = {"High": 0, "Medium": 1, "Low": 2, "Info": 3}
    rows = []
    for ga, gb, depth, p, cmin, cmax in raw:
        a, b = elements[ga], elements[gb]
        # Keep a consistent A/B order: STRUCT, then ARCH, then MEP.
        order = {"STRUCT": 0, "ARCH": 1, "MEP": 2}
        if order[a["disc"]] > order[b["disc"]]:
            a, b = b, a
        depth_in = depth * IN
        ctype, sev, desc, action = classify(a, b, depth_in)
        rows.append({"a": a, "b": b, "type": ctype, "severity": sev, "desc": desc,
                     "action": action, "depth_in": round(depth_in, 1), "point": p,
                     "box": [cmin.tolist(), cmax.tolist()]})
    rows = merge_nearby(rows)
    rows.sort(key=lambda r: (sev_rank[r["severity"]], r["type"], r["a"]["level"], -r["depth_in"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{slug}-conflicts.csv"
    cols = ["Clash ID", "Severity", "Status", "Clash Type", "What's Wrong", "Recommended Action",
            "Level", "Overlap (in)", "Location X", "Location Y", "Location Z",
            "MEP Parts Involved", "Element A Discipline", "Element A", "Element A Type", "Element A Revit ID", "Element A GUID",
            "Element B Discipline", "Element B", "Element B Type", "Element B Revit ID", "Element B GUID"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, r in enumerate(rows, 1):
            r["id"] = f"C{i:04d}"
            a, b, p = r["a"], r["b"], r["point"]
            w.writerow([r["id"], r["severity"], "New", r["type"], r["desc"], r["action"],
                        a["level"] or b["level"], r["depth_in"],
                        fmt_ftin(p[0]), fmt_ftin(p[1]), fmt_ftin(p[2]), len(r["parts"]),
                        DISC_NAME[a["disc"]], a["kind"], a["type"] or a["name"], a["revit_id"], a["guid"],
                        DISC_NAME[b["disc"]], b["kind"], b["type"] or b["name"], b["revit_id"], b["guid"]])

    glbs = {}
    for d in ("ARCH", "STRUCT", "MEP"):
        f = out_dir / f"{slug}-{d.lower()}.glb"
        if write_glb(meshes, elements, d, f):
            glbs[d] = f.name

    summary = {}
    for r in rows:
        summary[r["severity"]] = summary.get(r["severity"], 0) + 1
    data = {
        "project": name, "slug": slug, "models": models, "glb": glbs,
        "csv": csv_path.name, "summary": summary,
        "elements": {g: {k: e[k] for k in ("disc", "kind", "type", "name", "level", "revit_id")}
                     for g, e in elements.items()},
        "clashes": [{
            "id": r["id"], "severity": r["severity"], "type": r["type"], "desc": r["desc"],
            "action": r["action"], "level": r["a"]["level"] or r["b"]["level"],
            "depth_in": r["depth_in"], "a": r["a"]["guid"], "b": r["b"]["guid"], "parts": r["parts"],
            # Viewer is Y-up: (x, z, -y)
            "point": [r["point"][0], r["point"][2], -r["point"][1]],
        } for r in rows],
    }
    (out_dir / f"{slug}.json").write_text(json.dumps(data))
    return {"project": name, "slug": slug, "summary": summary, "total": len(rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projects", nargs="+", help="folders containing the IFC exports for one project")
    ap.add_argument("--out", default=str(Path(__file__).parent / "viewer" / "data"))
    args = ap.parse_args()
    out = Path(args.out)
    index = []
    for p in args.projects:
        print(f"== {Path(p).name}")
        index.append(analyze_project(p, out))
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    sys.exit(main())
