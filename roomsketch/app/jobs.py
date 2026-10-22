"""Tiny on-disk job queue: one background worker, status persisted as JSON."""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from .pipeline.room import extract_room

MAX_OVERLAY_POINTS = 60_000  # keeps JSON responses well under serverless body limits


class Jobs:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()

    def new(self, kind: str) -> tuple[str, Path]:
        jid = uuid.uuid4().hex[:12]
        d = self.root / jid
        d.mkdir(parents=True)
        self._write(jid, {"id": jid, "kind": kind, "status": "queued", "progress": 0.0,
                          "message": "Queued", "created": time.time()})
        return jid, d

    def dir(self, jid: str) -> Path:
        if not jid.isalnum():
            raise KeyError(jid)
        d = self.root / jid
        if not (d / "job.json").exists():
            raise KeyError(jid)
        return d

    def status(self, jid: str) -> dict:
        return json.loads((self.dir(jid) / "job.json").read_text())

    def update(self, jid: str, **fields) -> None:
        with self.lock:
            st = self.status(jid)
            st.update(fields)
            self._write(jid, st)

    def _write(self, jid, st):
        p = self.root / jid / "job.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(st))
        tmp.replace(p)

    def submit(self, jid: str, fn: Callable[[Callable[[float, str], None]], None]) -> None:
        def progress(f, m):
            self.update(jid, progress=round(float(f), 3), message=m)

        def run():
            self.update(jid, status="running", message="Starting")
            try:
                fn(progress)
                self.update(jid, status="done", progress=1.0, message="Done")
            except Exception as e:  # surfaced to the UI
                traceback.print_exc()
                self.update(jid, status="error", error=str(e) or e.__class__.__name__)

        self.pool.submit(run)


def overlay_points(pts: np.ndarray, transform, limit: int = MAX_OVERLAY_POINTS) -> np.ndarray:
    """Random subset of `pts` moved into the room's model frame (float32)."""
    T = np.asarray(transform, float)
    P = pts
    if len(P) > limit:
        P = P[np.random.default_rng(0).choice(len(P), limit, replace=False)]
    return (P @ T[:3, :3].T + T[:3, 3]).astype("<f4")


def solve_and_save(job_dir: Path, pts: np.ndarray, normals, *, sparse: bool, meta: dict,
                   progress: Callable[[float, str], None], lo: float = 0.0, hi: float = 1.0) -> dict:
    """Run room extraction and persist room.json + overlay points (model frame)."""
    sub = lambda f, m: progress(lo + (hi - lo) * f, m)
    room = extract_room(pts, normals, sparse=sparse, progress=sub)
    room["meta"] = meta
    (job_dir / "points.bin").write_bytes(overlay_points(pts, room["transform"]).tobytes())
    (job_dir / "room.json").write_text(json.dumps(room))
    return room
