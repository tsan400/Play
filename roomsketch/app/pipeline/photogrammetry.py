"""Photogrammetry: images/video -> metric, gravity-aligned point cloud.

Structure-from-motion runs on COLMAP (pycolmap, CPU). Metric scale comes from a
printed ArUco marker of known size (triangulated from every image that sees
it). Without a marker the cloud is provisionally scaled and flagged so the user
can calibrate against one known dimension in the viewer.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

MAX_DIM = 1600
ARUCO_DICT = cv2.aruco.DICT_4X4_50
ASSUMED_HEIGHT = 2.5  # metres, only used when no marker is found

Progress = Callable[[float, str], None]


def prepare_images(src_files: list[Path], out_dir: Path) -> list[str]:
    """Normalise uploads (resize, re-encode as JPEG). Returns image names in capture order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for k, f in enumerate(sorted(src_files)):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = _resize(img)
        name = f"img_{k:04d}.jpg"
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        names.append(name)
    return names


def frames_from_video(video: Path, out_dir: Path, fps: float = 2.5, max_frames: int = 150) -> list[str]:
    """Pick the sharpest frame in each 1/fps window."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("Could not read the video (try .mp4 / .mov H.264)")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    window = max(1, int(round(src_fps / fps)))
    if total and total / window > max_frames:
        window = int(math.ceil(total / max_frames))
    names, best, best_score, idx = [], None, -1.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
        score = cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if score > best_score:
            best, best_score = frame, score
        idx += 1
        if idx % window == 0 and best is not None:
            name = f"img_{len(names):04d}.jpg"
            cv2.imwrite(str(out_dir / name), _resize(best), [cv2.IMWRITE_JPEG_QUALITY, 93])
            names.append(name)
            best, best_score = None, -1.0
    cap.release()
    return names


def _resize(img):
    h, w = img.shape[:2]
    s = MAX_DIM / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def reconstruct(image_dir: Path, names: list[str], work: Path, marker_size_m: float | None,
                progress: Progress = lambda f, m: None) -> tuple[np.ndarray, dict]:
    import pycolmap

    if len(names) < 8:
        raise ValueError("Need at least 8 usable photos (40-120 recommended)")
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    sparse_dir = work / "sparse"
    shutil.rmtree(sparse_dir, ignore_errors=True)
    sparse_dir.mkdir()

    progress(0.05, f"Extracting features from {len(names)} images")
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = 8192
    ext.max_image_size = MAX_DIM
    pycolmap.extract_features(db, image_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext, device=pycolmap.Device.cpu)

    progress(0.25, "Matching features")
    if len(names) <= 70:
        pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 20
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=pairing, device=pycolmap.Device.cpu)

    progress(0.45, "Solving camera poses (structure from motion)")
    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_model_size = 5
    recs = pycolmap.incremental_mapping(db, image_dir, sparse_dir, options=opts)
    if not recs:
        raise ValueError("Structure-from-motion failed: not enough overlap/texture between photos")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    n_reg = rec.num_reg_images()
    progress(0.7, f"Registered {n_reg}/{len(names)} images, {rec.num_points3D()} points")

    pts = []
    for p in rec.points3D.values():
        if p.track.length() >= 3 and p.error < 2.0:
            pts.append(p.xyz)
    pts = np.asarray(pts, float)
    if len(pts) < 200:
        raise ValueError(f"Only {len(pts)} reliable 3D points - the room needs more texture or more photos")

    images = [rec.image(i) for i in rec.reg_image_ids()]
    poses = [_pose(img) for img in images]  # (R, t) cam_from_world

    # ---- Gravity: people hold the phone upright, so image "up" ~ world up.
    ups = np.array([R.T @ np.array([0.0, -1.0, 0.0]) for R, _ in poses])
    up = ups.mean(axis=0)
    up /= np.linalg.norm(up)
    Rg = _rotation_to_z(up)

    # ---- Metric scale from ArUco marker(s).
    progress(0.75, "Looking for scale marker")
    marker = _marker_scale(rec, images, poses, image_dir, marker_size_m) if marker_size_m else None
    if marker is not None:
        scale = marker["scale"]
        scaled = True
    else:
        z = (pts @ Rg.T)[:, 2]
        lo, hi = np.percentile(z, [2, 98])
        scale = ASSUMED_HEIGHT / max(hi - lo, 1e-9)
        scaled = False

    world = (pts @ Rg.T) * scale
    cams = np.array([-(R.T @ t) for R, t in poses]) @ Rg.T * scale
    info = {
        "images": len(names),
        "registered": n_reg,
        "sparse_points": int(len(pts)),
        "scaled": scaled,
        "scale_source": "aruco" if scaled else f"assumed {ASSUMED_HEIGHT} m tall scene",
        "marker": marker,
        "camera_centers": cams.round(4).tolist(),
        "mean_reprojection_error": float(rec.compute_mean_reprojection_error()),
    }
    return world, info


def _pose(img):
    cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    M = np.asarray(cfw.matrix())
    return M[:, :3], M[:, 3]


def _rotation_to_z(v: np.ndarray) -> np.ndarray:
    z = np.array([0.0, 0.0, 1.0])
    c = float(v @ z)
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(v, z)
    s = np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]) / s
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def detect_markers(img) -> dict[int, np.ndarray]:
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                       cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    out = {}
    if ids is None:
        return out
    for c, i in zip(corners, ids.ravel()):
        c = c.reshape(4, 2).astype(np.float32)
        cv2.cornerSubPix(gray, c, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        out[int(i)] = c.astype(float)
    return out


def _marker_scale(rec, images, poses, image_dir, size_m) -> dict | None:
    obs: dict[int, list] = {}
    for img, (R, t) in zip(images, poses):
        frame = cv2.imread(str(image_dir / img.name))
        if frame is None:
            continue
        cam = img.camera if not callable(img.camera) else img.camera()
        for mid, corners in detect_markers(frame).items():
            # Normalised (undistorted) image coordinates via the solved camera model.
            xy = np.asarray(cam.cam_from_img(corners))
            obs.setdefault(mid, []).append((R, t, xy))
    # A real printed marker triangulates to a square; spurious detections in busy
    # texture do not. Keep square ones, then trust the best-observed marker.
    cands = []
    for mid, views in obs.items():
        if len(views) < 2:
            continue
        X = [_triangulate([(R, t, xy[k]) for R, t, xy in views]) for k in range(4)]
        if any(x is None for x in X):
            continue
        sides = np.array([np.linalg.norm(X[k] - X[(k + 1) % 4]) for k in range(4)])
        diag = np.array([np.linalg.norm(X[0] - X[2]), np.linalg.norm(X[1] - X[3])])
        side = float(sides.mean())
        spread = float(sides.std() / side)
        if spread > 0.05 or np.abs(diag / (side * np.sqrt(2)) - 1).max() > 0.05:
            continue
        cands.append({"id": mid, "views": len(views), "side": side, "spread": spread})
    if not cands:
        return None
    best = max(cands, key=lambda c: c["views"])
    used = [c for c in cands if abs(c["side"] / best["side"] - 1) < 0.05]
    side = float(np.average([c["side"] for c in used], weights=[c["views"] for c in used]))
    return {"scale": size_m / side,
            "markers": [{"id": c["id"], "views": c["views"]} for c in used],
            "side_spread_pct": round(max(c["spread"] for c in used) * 100, 2),
            "rejected": len(obs) - len(used)}


def _triangulate(views) -> np.ndarray | None:
    A = []
    for R, t, (x, y) in views:
        P = np.hstack([R, t[:, None]])
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, s, vt = np.linalg.svd(np.asarray(A))
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def marker_png(marker_id: int = 0, px: int = 1200) -> bytes:
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), marker_id, px)
    img = cv2.copyMakeBorder(img, px // 8, px // 8, px // 8, px // 8, cv2.BORDER_CONSTANT, value=255)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()
