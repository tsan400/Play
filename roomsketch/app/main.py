"""RoomSketch web server: capture/import -> room model -> toon viewer + SketchUp exports.

The room endpoints are stateless (the browser keeps the solved room JSON and
sends it back for models/exports), so they run unchanged on serverless hosts
such as Vercel. Photogrammetry needs COLMAP + OpenCV and a long-running worker;
it is enabled only when those are installed (local / Docker server).
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .jobs import Jobs, overlay_points, solve_and_save
from .pipeline import export
from .pipeline.lod import LAYERS, build_lod, triangulate
from .pipeline.pointcloud import detect_up_axis, to_z_up, up_rotation
from .pipeline.room import extract_room
from .pipeline.synthetic import demo_scan

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("ROOMSKETCH_DATA", ROOT / "data"))
MAX_POINTS = 400_000
MAX_UPLOAD = 1024 * 1024 * 1024  # photogrammetry uploads (local server only)

PHOTOGRAMMETRY = (not os.environ.get("VERCEL") and not os.environ.get("ROOMSKETCH_NO_PHOTOGRAMMETRY")
                  and all(importlib.util.find_spec(m) for m in ("pycolmap", "cv2")))

app = FastAPI(title="RoomSketch")
jobs = Jobs(DATA / "jobs") if PHOTOGRAMMETRY else None

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype("<f4").tobytes()).decode()


@app.get("/api/config")
def config():
    return {"photogrammetry": PHOTOGRAMMETRY, "max_points": MAX_POINTS}


# ---------------------------------------------------------------------------------
# Room solving (stateless)
# ---------------------------------------------------------------------------------

@app.post("/api/room")
async def solve_room(request: Request, up: str = Query("auto"), name: str = Query("scan")):
    """LiDAR path. Body: uint32 count, count*3 float32 xyz, optional count*3 int8 normals.

    The browser parses the scan export and voxel-downsamples it first, so the
    upload stays small (a few MB) whatever the size of the original file.
    """
    if up not in {"auto", "y", "-y", "z", "-z", "x", "-x"}:
        raise HTTPException(400, "Invalid up axis")
    body = await request.body()
    if len(body) < 4:
        raise HTTPException(400, "Empty upload")
    (n,) = struct.unpack_from("<I", body)
    if not 50 <= n <= MAX_POINTS:
        raise HTTPException(400, f"Expected 50-{MAX_POINTS:,} points, got {n:,}")
    if len(body) not in (4 + 12 * n, 4 + 15 * n):
        raise HTTPException(400, "Malformed point upload")
    pts = np.frombuffer(body, "<f4", 3 * n, 4).reshape(n, 3).astype(np.float64)
    nrm = None
    if len(body) == 4 + 15 * n:
        nrm = np.frombuffer(body, np.int8, 3 * n, 4 + 12 * n).reshape(n, 3).astype(np.float64)
        length = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = np.where(length > 0, nrm / np.maximum(length, 1e-9), 0.0)
    if not np.isfinite(pts).all():
        raise HTTPException(400, "Points contain NaN/Inf")

    axis = detect_up_axis(pts) if up == "auto" else up
    zpts, znrm = to_z_up(pts, axis, nrm)
    try:
        room = extract_room(zpts, znrm)
    except ValueError as e:
        raise HTTPException(422, str(e))
    room["meta"] = {"source": "lidar", "file": name[:120], "up_axis": axis, "scaled": True,
                    "scale_source": "LiDAR (metric)"}
    # Raw file axes -> model frame, so the browser can overlay its own copy of the points.
    T = np.array(room["transform"])
    R = np.eye(4)
    R[:3, :3] = up_rotation(axis)
    room["input_transform"] = (T @ R).tolist()
    return {"room": room}


@app.post("/api/demo")
def demo():
    pts = demo_scan()
    room = extract_room(pts)
    room["meta"] = {"source": "demo", "scaled": True, "scale_source": "synthetic LiDAR (metric)"}
    return {"room": room, "points": _b64(overlay_points(pts, room["transform"]))}


def _check_room(room: dict) -> dict:
    """Light validation of a room posted back by the browser."""
    try:
        poly = room["polygon"]
        if not 3 <= len(poly) <= 500 or not 0.5 < float(room["height"]) < 50:
            raise ValueError
        if len(room.get("openings", [])) > 500 or len(room.get("furniture", [])) > 200:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "Invalid room data")
    return room


def _lod(room, scale, wall):
    try:
        return build_lod(_check_room(room), wall_thickness=wall, scale=scale)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Could not build a model from this room")


@app.post("/api/model")
async def room_model(request: Request):
    req = await request.json()
    room = req.get("room") or {}
    scale = float(np.clip(float(req.get("scale", 1.0)), 0.01, 100))
    wall = float(np.clip(float(req.get("wall", 0.15)), 0.02, 0.6))
    model = _lod(room, scale, wall)
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
        "openings": room.get("openings", []),
        "furniture": len(room.get("furniture", [])),
        "metrics": {"floor_area": m["floor_area"] * s * s, "perimeter": m["perimeter"] * s,
                    "volume": m["volume"] * s ** 3, "height": room["height"] * s,
                    "edge_lengths": [x * s for x in m["edge_lengths"]]},
        "meta": room.get("meta", {}),
        "stats": room.get("stats", {}),
        "ceiling_estimated": room.get("ceiling_estimated", False),
    }


@app.post("/api/export")
async def room_export(request: Request):
    req = await request.json()
    room = req.get("room") or {}
    fmt = req.get("fmt")
    scale = float(np.clip(float(req.get("scale", 1.0)), 0.01, 100))
    wall = float(np.clip(float(req.get("wall", 0.15)), 0.02, 0.6))
    sel = [l for l in req.get("layers", LAYERS) if l in LAYERS] or LAYERS
    model = _lod(room, scale, wall)
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
                    headers={"Content-Disposition": f'attachment; filename="room.{fmt}"'})


# ---------------------------------------------------------------------------------
# Photogrammetry (local / Docker server only)
# ---------------------------------------------------------------------------------

def _need_photogrammetry():
    if not PHOTOGRAMMETRY:
        raise HTTPException(501, "Photogrammetry runs on the local/Docker server (needs COLMAP + OpenCV).")


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


@app.post("/api/photos")
async def upload_photos(files: list[UploadFile] = File(...), marker_cm: float = Form(0.0)):
    """Captured frames, picked photos, or one walkthrough video."""
    _need_photogrammetry()
    from .pipeline import photogrammetry

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


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    _need_photogrammetry()
    try:
        return jobs.status(jid)
    except KeyError:
        raise HTTPException(404, "Unknown job")


@app.get("/api/jobs/{jid}/result")
def job_result(jid: str):
    _need_photogrammetry()
    try:
        d = jobs.dir(jid)
    except KeyError:
        raise HTTPException(404, "Unknown job")
    if not (d / "room.json").exists():
        raise HTTPException(409, "Job not finished")
    pts = np.frombuffer((d / "points.bin").read_bytes(), "<f4")
    return {"room": json.loads((d / "room.json").read_text()), "points": _b64(pts)}


@app.get("/api/health")
def health():
    return {"ok": True}


# Static front end (on Vercel the CDN serves web/ and this mount is skipped).
if (ROOT / "web").is_dir():
    app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
