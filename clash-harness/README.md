# Clash Harness

Test harness for Revit IFC exports: runs a multi-discipline clash check,
writes a plain-language conflicts CSV, and shows everything in a 3D viewer
styled for Team Engineering.

## Run

```bash
pip install -r requirements.txt
./run.sh "path/to/Klein Judson SMP" "path/to/Smith Residence SME"
# open http://localhost:8765
```

Each project folder holds one IFC per discipline. The discipline comes from
the file name: `Struct…` → Structural, `Mech/Plumb/Elec/MEP…` → MEP,
`Arch…` → Architectural.

## What it checks

Hard clashes between disciplines (Structural ↔ MEP, Architectural ↔ MEP,
Architectural ↔ Structural). Overlaps under 1/2" are ignored. Openings,
spaces and annotations are skipped; furniture, ceilings and railings are
drawn but not clash-checked. A pipe and its fittings hitting the same member
within ~1 ft count as one clash.

| Severity | Meaning |
|---|---|
| High | MEP through a beam/column/footing, structure cutting a door/window/stair, MEP into doors/windows/roof |
| Medium | MEP through structural walls/slabs, other real overlaps needing review |
| Low | MEP penetrating architectural walls/floors (normal routing; check sleeves/fire rating) |
| Info | Duplicates modeled in two models, framing inside wall/floor/roof assemblies, small joins |

## Output (`viewer/data/`)

- `<project>-conflicts.csv`: one row per clash: ID, severity, what's wrong,
  recommended action, level, overlap, location (ft-in, shared coordinates),
  and both elements with Revit Element ID and IFC GUID. Opens directly in Excel.
- `<project>-{arch,struct,mep}.glb` + `<project>.json`: viewer data.

## Viewer

Filter by severity chips, clash type or text; toggle models. Click a clash
(or double-click an element in 3D) to zoom to it: element A is orange,
element B red, everything else ghosted. Esc or H returns to the full model.

Navigation: left-drag orbits, right- or middle-drag pans, scroll zooms toward
the cursor with inertia, trackpad two-finger swipe pans and pinch zooms, and
camera moves between clashes are eased.
