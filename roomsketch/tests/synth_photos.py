"""Render photos of a textured box room (pure OpenCV) for photogrammetry tests.

Inside a convex box every pixel sees exactly one surface, so each face can be
inverse-warped with its plane homography and masked by its clipped outline.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.photogrammetry import ARUCO_DICT

ROOM = (4.2, 3.4, 2.5)  # x, y, z (metres)
MARKER = 0.30  # black square size
PX_PER_M = 200


def _texture(w, h, rng, base):
    img = np.full((int(h * PX_PER_M), int(w * PX_PER_M), 3), base, np.uint8)
    for _ in range(int(w * h * 120)):
        c = tuple(int(x) for x in rng.integers(20, 235, 3))
        x, y = int(rng.uniform(0, img.shape[1])), int(rng.uniform(0, img.shape[0]))
        s = int(rng.uniform(3, 18))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + s, y + int(s * rng.uniform(0.3, 1.3))), c, -1)
        else:
            cv2.circle(img, (x, y), s // 2 + 1, c, -1)
    return img


def _faces(rng):
    X, Y, Z = ROOM
    faces = []  # (origin, U, V, width, height, texture)
    floor = _texture(X, Y, rng, (150, 170, 190))
    mk = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), 0, int(MARKER * PX_PER_M))
    pad = int(MARKER * PX_PER_M / 8)
    mk = cv2.copyMakeBorder(mk, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    mk = cv2.flip(mk, 0)  # texture rows run along +V, i.e. mirrored when seen from above
    cx, cy = int(1.9 * PX_PER_M), int(1.5 * PX_PER_M)
    floor[cy:cy + mk.shape[0], cx:cx + mk.shape[1]] = mk[..., None]
    faces.append((np.array([0, 0, 0.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), X, Y, floor))
    faces.append((np.array([0, 0, Z]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), X, Y,
                  _texture(X, Y, rng, (225, 225, 220))))
    walls = [((0, 0), (1, 0), X), ((X, 0), (0, 1), Y), ((X, Y), (-1, 0), X), ((0, Y), (0, -1), Y)]
    for (ox, oy), (dx, dy), L in walls:
        faces.append((np.array([ox, oy, 0.0]), np.array([dx, dy, 0.0]), np.array([0, 0, 1.0]), L, Z,
                      _texture(L, Z, rng, tuple(int(v) for v in rng.integers(170, 220, 3)))))
    return faces


def _look_at(eye, target):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)  # image down
    R = np.stack([r, d, f])  # world -> camera (OpenCV)
    return R, -R @ eye


def render_room(out_dir: Path, n: int = 66, size=(1024, 768), seed: int = 3) -> list[Path]:
    rng = np.random.default_rng(seed)
    faces = _faces(rng)
    w, h = size
    fx = 0.8 * w
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])
    X, Y, Z = ROOM
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        a = 2 * math.pi * i / n
        # Three interleaved passes: across the room, down at the floor, up at the ceiling line.
        height, pitch = [(1.5, -0.15), (1.2, -0.7), (1.5, 0.4)][i % 3]
        eye = np.array([X / 2 + 1.0 * math.cos(a), Y / 2 + 0.7 * math.sin(a), height])
        la = a + math.pi + 0.35
        tgt = eye + np.array([math.cos(la) * math.cos(pitch), math.sin(la) * math.cos(pitch), math.sin(pitch)])
        R, t = _look_at(eye, tgt)
        img = np.zeros((h, w, 3), np.uint8)
        for o, U, V, fw, fh, tex in faces:
            # Plane homography texture(px) -> image(px).
            S = np.array([[fw / tex.shape[1], 0, 0], [0, fh / tex.shape[0], 0], [0, 0, 1]])
            M = np.stack([R @ U, R @ V, R @ o + t], axis=1)
            Hm = K @ M @ S
            corners = [o, o + U * fw, o + U * fw + V * fh, o + V * fh]
            poly = _clip_project(corners, R, t, K)
            if poly is None:
                continue
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
            warped = cv2.warpPerspective(tex, Hm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            img[mask > 0] = warped[mask > 0]
        p = out_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(p)
    return paths


def _clip_project(corners, R, t, K, near=0.05):
    cam = [R @ c + t for c in corners]
    out = []
    for k in range(4):
        a, b = cam[k], cam[(k + 1) % 4]
        if a[2] > near:
            out.append(a)
        if (a[2] > near) != (b[2] > near):
            s = (near - a[2]) / (b[2] - a[2])
            out.append(a + s * (b - a))
    if len(out) < 3:
        return None
    P = np.array(out)
    uv = (K @ P.T).T
    return uv[:, :2] / uv[:, 2:]
