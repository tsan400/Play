"""RoomSketch web server: capture/import -> room model -> toon viewer + SketchUp exports."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .jobs import Jobs, solve_and_save
from .pipeline import export, photogrammetry
from .pipeline.lod import LAYERS, build_lod, triangulate
from .pipeline.pointcloud import detect_up_axis, load_scan, to_z_up
from .pipeline.synthetic import demo_scan

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("ROOMSKETCH_DATA", ROOT / "data"))
MAX_UPLOAD = 1024 * 1024 * 1024  # 1 GB

app = FastAPI(title="RoomSketch")
jobs = Jobs(DATA / "jobs")

SCAN_EXT = {".ply", ".obj", ".glb", ".gltf", ".stl", ".xyz", ".pts", ".txt", ".csv"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


async def _save(upload: UploadFile, dest: Path) -> Path:
    size = 0
    with open(dest, "wb") as fh:
        while chunk := await upload.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, "File too large")
            fh.write(chunk)
    return dest


def _safe_name(name: str | None, default: str) -> str:
    base = Path(name or default).name
    return "".join(c for c in base if c.isalnum() or c in "._-") or default


# ---------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------

@app.post("/api/scan")
async def upload_scan(file: UploadFile = File(...), up: str = Form("auto")):
    """LiDAR path: a scan exported from an iPhone LiDAR app."""
    name = _safe_name(file.filename, "scan.ply")
    ext = Path(name).suffix.lower()
    if ext not in SCAN_EXT:
        raise HTTPException(400, f"Unsupported scan format {ext!r}. Use PLY, OBJ, GLB, STL or XYZ.")
    if up not in {"auto", "y", "-y", "z", "-z", "x", "-x"}:
        raise HTTPException(400, "Invalid up axis")
    jid, d = jobs.new("lidar")
    path = await _save(file, d / name)

    def run(progress):
        progress(0.02, "Reading scan")
        pts, nrm = load_scan(path)
        axis = detect_up_axis(pts) if up == "auto" else up
        pts, nrm = to_z_up(pts, axis, nrm)
        meta = {"source": "lidar", "file": name, "up_axis": axis, "scaled": True,
                "scale_source": "LiDAR (metric)"}
        solve_and_save(d, pts, nrm, sparse=False, meta=meta, progress=progress, lo=0.05)

    jobs.submit(jid, run)
    return {"id": jid}


@app.post("/api/photos")
async def upload_photos(files: list[UploadFile] = File(...), marker_cm: float = Form(0.0)):
    """Photogrammetry path: captured frames, picked photos, or one walkthrough video."""
    jid, d = jobs.new("photos")
    raw = d / "raw"
    raw.mkdir()
    saved = []
    for k, f in enumerate(files):
        name = f"{k:04d}_{_safe_name(f.filename, f'frame{k}.jpg')}"
        if Path(name).suffix.lower() not in IMAGE_EXT | VIDEO_EXT:
            continue
        saved.append(await _save(f, raw / name))
    if not saved:
        shutil.rmtree(d)
        raise HTTPException(400, "No images or video received")
    marker = marker_cm / 100.0 if marker_cm and marker_cm > 0 else None

    def run(progress):
        img_dir = d / "images"
        videos = [p for p in saved if p.suffix.lower() in VIDEO_EXT]
        if videos:
            progress(0.01, "Extracting frames from video")
            names = photogrammetry.frames_from_video(videos[0], img_dir)
        else:
            progress(0.01, "Preparing images")
            names = photogrammetry.prepare_images(saved, img_dir)
        sub = lambda f, m: progress(0.02 + 0.6 * f, m)
        pts, info = photogrammetry.reconstruct(img_dir, names, d / "colmap", marker, sub)
        meta = {"source": "photogrammetry", "scaled": info["scaled"], "scale_source": info["scale_source"],
                "photogrammetry": {k: v for k, v in info.items() if k != "camera_centers"}}
        solve_and_save(d, pts, None, sparse=True, meta=meta, progress=progress, lo=0.65)
        shutil.rmtree(raw, ignore_errors=True)

    jobs.submit(jid, run)
    return {"id": jid}


@app.post("/api/demo")
async def demo():
    jid, d = jobs.new("demo")

    def run(progress):
        progress(0.02, "Generating synthetic LiDAR scan of an L-shaped room")
        pts = demo_scan()
        meta = {"source": "demo", "scaled": True, "scale_source": "synthetic LiDAR (metric)"}
        solve_and_save(d, pts, None, sparse=False, meta=meta, progress=progress, lo=0.05)

    jobs.submit(jid, run)
    return {"id": jid}


# ---------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------

def _room(jid: str) -> tuple[dict, Path]:
    try:
        d = jobs.dir(jid)
    except KeyError:
        raise HTTPException(404, "Unknown job")
    p = d / "room.json"
    if not p.exists():
        raise HTTPException(409, "Job not finished")
    return json.loads(p.read_text()), d


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    try:
        return jobs.status(jid)
    except KeyError:
        raise HTTPException(404, "Unknown job")


@app.get("/api/jobs/{jid}/model")
def job_model(jid: str, scale: float = Query(1.0, gt=0.01, lt=100), wall: float = Query(0.15, ge=0.02, le=0.6)):
    room, _ = _room(jid)
    model = build_lod(room, wall_thickness=wall, scale=scale)
    meshes = {}
    for layer in LAYERS:
        pos, idx = [], []
        for f in model["faces"]:
            if f["layer"] != layer:
                continue
            v, t = triangulate(f)
            idx.extend((t + len(pos)).ravel().tolist())
            pos.extend(np.round(v, 4).tolist())
        if pos:
            meshes[layer] = {"positions": [c for p in pos for c in p], "indices": idx}
    m = room["metrics"]
    s = scale
    return {
        "meshes": meshes,
        "edges": model["edges"],
        "bounds": model["bounds"],
        "outline": model["inner_outline"],
        "height": model["height"],
        "openings": room["openings"],
        "furniture": len(room["furniture"]),
        "metrics": {"floor_area": m["floor_area"] * s * s, "perimeter": m["perimeter"] * s,
                    "volume": m["volume"] * s ** 3, "height": room["height"] * s,
                    "edge_lengths": [x * s for x in m["edge_lengths"]]},
        "meta": room.get("meta", {}),
        "stats": room.get("stats", {}),
        "ceiling_estimated": room.get("ceiling_estimated", False),
    }


@app.get("/api/jobs/{jid}/points")
def job_points(jid: str):
    _, d = _room(jid)
    return FileResponse(d / "points.bin", media_type="application/octet-stream")


@app.get("/api/jobs/{jid}/export/{fmt}")
def job_export(jid: str, fmt: str, scale: float = Query(1.0, gt=0.01, lt=100),
               wall: float = Query(0.15, ge=0.02, le=0.6), layers: str = Query(",".join(LAYERS))):
    room, _ = _room(jid)
    model = build_lod(room, wall_thickness=wall, scale=scale)
    sel = [l for l in layers.split(",") if l in LAYERS] or LAYERS
    fname = f"room-{jid}"
    if fmt == "rb":
        body, mime = export.to_sketchup_rb(model, sel), "text/x-ruby"
    elif fmt == "dae":
        body, mime = export.to_dae(model, sel), "model/vnd.collada+xml"
    elif fmt == "dxf":
        body, mime = export.to_dxf(model, sel), "image/vnd.dxf"
    elif fmt == "obj":
        body, mime = export.to_obj(model, sel), "text/plain"
    elif fmt == "svg":
        body, mime = export.to_svg_plan(room, model, scale), "image/svg+xml"
    elif fmt == "json":
        body, mime = json.dumps({"room": room, "scale": scale}, indent=1), "application/json"
    else:
        raise HTTPException(404, "Unknown format")
    return Response(body, media_type=mime,
                    headers={"Content-Disposition": f'attachment; filename="{fname}.{fmt}"'})


@app.get("/api/marker.png")
def marker():
    return Response(photogrammetry.marker_png(0), media_type="image/png")


@app.get("/api/health")
def health():
    return {"ok": True}


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
