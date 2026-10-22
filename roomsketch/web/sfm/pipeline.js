// In-browser structure-from-motion: images -> metric, gravity-aligned sparse point cloud.
// Runs inside a Web Worker with OpenCV.js (features, matching, PnP, ArUco) and the
// LA / G / BA helpers. All heavy data stays on the device; only points are uploaded.
(function (root) {
  'use strict';

  const DEFAULTS = {
    maxDim: 1280,         // working resolution (long side)
    features: 2500,       // per image, after grid filtering
    window: 5,            // sequential neighbours matched for each image
    loopCandidates: 2,    // extra non-sequential pairs per image (thumbnail similarity)
    exhaustiveBelow: 36,  // match all pairs for small sets
    minInliers: 25,
    reprojPx: 4,
    minAngleDeg: 2,
    markerDict: 'DICT_4X4_50',
    assumedHeight: 2.5,
  };

  // ---- 1. Per-image features (pool workers) ---------------------------------------
  function extractFeatures(cv, gray, w, h, opts) {
    const o = { ...DEFAULTS, ...opts };
    const mat = new cv.Mat(h, w, cv.CV_8UC1);
    mat.data.set(gray);
    const orb = new cv.ORB(o.features * 3, 1.2, 8, 31, 0, 2); // scoreType enum is unbound in OpenCV.js
    orb.setFastThreshold(12);
    const kv = new cv.KeyPointVector(), d = new cv.Mat(), none = new cv.Mat();
    orb.detectAndCompute(mat, none, kv, d);
    // Keep the strongest keypoints per grid cell so features cover the whole frame.
    const GX = 16, GY = 12, cap = Math.ceil(o.features / (GX * GY)) + 2;
    const cells = new Map(), all = [];
    for (let i = 0; i < kv.size(); i++) {
      const k = kv.get(i);
      all.push({ i, x: k.pt.x, y: k.pt.y, r: k.response });
    }
    all.sort((a, b) => b.r - a.r);
    const keep = [];
    for (const k of all) {
      const c = Math.floor(k.x / w * GX) + GX * Math.floor(k.y / h * GY);
      const used = cells.get(c) || 0;
      if (used < cap) { cells.set(c, used + 1); keep.push(k); }
      if (keep.length >= o.features) break;
    }
    const kps = new Float32Array(keep.length * 2);
    const desc = new Uint8Array(keep.length * 32);
    keep.forEach((k, j) => {
      kps[2 * j] = k.x; kps[2 * j + 1] = k.y;
      desc.set(d.data.subarray(k.i * 32, k.i * 32 + 32), j * 32);
    });
    orb.delete(); kv.delete(); d.delete(); none.delete();
    const markers = detectMarkers(cv, mat, o.markerDict);
    mat.delete();
    return { kps, desc, thumb: thumbnail(gray, w, h), markers };
  }

  function detectMarkers(cv, mat, dictName) {
    const out = {};
    try {
      const dict = cv.getPredefinedDictionary(cv[dictName]);
      const params = new cv.aruco_DetectorParameters();
      // AprilTag-style edge refinement: unbiased corners (SUBPIX shrinks markers ~0.3%).
      params.cornerRefinementMethod = cv.CORNER_REFINE_APRILTAG ?? 3;
      const refine = new cv.aruco_RefineParameters(10, 3, true);
      const det = new cv.aruco_ArucoDetector(dict, params, refine);
      const corners = new cv.MatVector(), ids = new cv.Mat(), rej = new cv.MatVector();
      det.detectMarkers(mat, corners, ids, rej);
      for (let i = 0; i < ids.rows; i++) {
        const c = corners.get(i);
        out[ids.data32S[i]] = Array.from(c.data32F); // 4 corners x (x, y)
        c.delete();
      }
      corners.delete(); ids.delete(); rej.delete(); det.delete(); params.delete(); refine.delete();
    } catch (e) {
      // Marker detection is optional; never fail the reconstruction for it.
    }
    return out;
  }

  // ---- 2. Pair selection + matching -------------------------------------------------
  function selectPairs(thumbs, opts) {
    const o = { ...DEFAULTS, ...opts }, n = thumbs.length;
    const pairs = new Set();
    const key = (i, j) => (i < j ? i * 100000 + j : j * 100000 + i);
    if (n <= o.exhaustiveBelow) {
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) pairs.add(key(i, j));
    } else {
      for (let i = 0; i < n; i++) for (let j = i + 1; j <= Math.min(n - 1, i + o.window); j++) pairs.add(key(i, j));
      // Loop closure: most similar thumbnails outside the sequential window.
      for (let i = 0; i < n; i++) {
        const sims = [];
        for (let j = 0; j < n; j++) if (Math.abs(i - j) > o.window) sims.push([ncc(thumbs[i], thumbs[j]), j]);
        sims.sort((a, b) => b[0] - a[0]);
        for (const [, j] of sims.slice(0, o.loopCandidates)) pairs.add(key(i, j));
      }
    }
    return [...pairs].sort((a, b) => a - b).map(k => [Math.floor(k / 100000), k % 100000]);
  }

  // feats: Map id -> { kps: Float32Array, desc: Uint8Array }
  function matchPairs(cv, pairs, feats, opts, onPair) {
    const o = { ...DEFAULTS, ...opts };
    const mats = new Map();
    const matOf = id => {
      if (!mats.has(id)) {
        const f = feats.get(id);
        const m = new cv.Mat(f.desc.length / 32, 32, cv.CV_8UC1);
        m.data.set(f.desc);
        mats.set(id, m);
      }
      return mats.get(id);
    };
    const matcher = new cv.BFMatcher(cv.NORM_HAMMING, false);
    const out = [];
    pairs.forEach(([i, j], k) => {
      const A = feats.get(i), B = feats.get(j);
      // One-directional ratio test + uniqueness; the epipolar RANSAC removes the rest.
      const fwd = knn(cv, matcher, matOf(i), matOf(j));
      const ia = [], ib = [];
      for (const [qa, [tb]] of fwd) { ia.push(qa); ib.push(tb); }
      onPair && onPair(k);
      if (ia.length < o.minInliers) return;
      const p1 = new Float64Array(ia.length * 2), p2 = new Float64Array(ia.length * 2);
      ia.forEach((a, n) => {
        p1[2 * n] = A.kps[2 * a]; p1[2 * n + 1] = A.kps[2 * a + 1];
        p2[2 * n] = B.kps[2 * ib[n]]; p2[2 * n + 1] = B.kps[2 * ib[n] + 1];
      });
      const res = G.ransacEpipolar(p1, p2, 2.0, { iters: 300 });
      if (!res || res.inliers.length < o.minInliers) return;
      out.push({ i, j, a: res.inliers.map(n => ia[n]), b: res.inliers.map(n => ib[n]) });
    });
    matcher.delete();
    for (const m of mats.values()) m.delete();
    return out;
  }

  // ---- 3+. Reconstruction (single worker) ----------------------------------------------
  class Reconstruction {
    // images: [{ kps, markers }], matches: [{ i, j, a, b }], size: [w, h]
    constructor(cv, images, matches, size, focal, opts, progress) {
      this.cv = cv;
      this.o = { ...DEFAULTS, ...opts };
      this.progress = progress || (() => {});
      this.images = images;
      this.matches = matches;
      this.size = size;
      this.focal = focal;
    }

    // ---- 3. Tracks ---------------------------------------------------------------------
    buildTracks() {
      const off = [0];
      for (const im of this.images) off.push(off[off.length - 1] + im.kps.length / 2);
      const total = off[off.length - 1];
      const parent = new Int32Array(total).map((_, k) => k);
      const used = new Uint8Array(total);
      const find = x => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
      for (const m of this.matches)
        for (let k = 0; k < m.a.length; k++) {
          const ia = off[m.i] + m.a[k], ib = off[m.j] + m.b[k];
          used[ia] = used[ib] = 1;
          const x = find(ia), y = find(ib);
          if (x !== y) parent[x] = y;
        }
      const groups = new Map();
      for (let im = 0; im < this.images.length; im++)
        for (let id = off[im]; id < off[im + 1]; id++) {
          if (!used[id]) continue;
          const r = find(id);
          if (!groups.has(r)) groups.set(r, []);
          groups.get(r).push([im, id - off[im]]);
        }
      this.tracks = [];
      this.kpTrack = this.images.map(im => new Int32Array(im.kps.length / 2).fill(-1));
      for (const obs of groups.values()) {
        if (obs.length < 2) continue;
        if (new Set(obs.map(o => o[0])).size !== obs.length) continue; // two kps in one image: ambiguous
        const t = this.tracks.length;
        this.tracks.push({ obs, X: null });
        for (const [im, kp] of obs) this.kpTrack[im][kp] = t;
      }
    }

    // ---- 4. Incremental reconstruction ---------------------------------------------------
    reconstruct() {
      const [w, h] = this.size;
      this.K = { f: this.focal, cx: w / 2, cy: h / 2, k1: 0 };
      this.cams = new Array(this.images.length).fill(null);
      this.order = [];
      if (!this._initialize()) throw new Error('Could not find a good starting pair. Move sideways more between photos.');
      let sinceGlobal = 0;
      while (true) {
        const next = this._nextImage();
        if (next < 0) break;
        if (!this._register(next)) { this.cams[next] = false; continue; }
        this._triangulateFrom(next);
        sinceGlobal++;
        const reg = this.order.length;
        this.progress(0.55 + 0.35 * reg / this.images.length, `Registered ${reg}/${this.images.length} photos`);
        if (sinceGlobal >= 10 || reg === 6) {
          this._ba([...this.order.slice(1)], reg >= 6, 10, 1e-5);
          this._filter();
          sinceGlobal = 0;
        } else {
          this._ba(this.order.slice(-5).filter(c => c !== this.order[0]), false, 5, 1e-4);
        }
      }
      this._ba(this.order.slice(1), true, 20, 1e-6);
      this._filter();
      this._ba(this.order.slice(1), true, 10, 1e-6);
    }

    _initialize() {
      const cands = this.matches.filter(m => m.a.length >= 60).sort((x, y) => y.a.length - x.a.length).slice(0, 25);
      let best = null;
      for (const m of cands) {
        const A = this.images[m.i], B = this.images[m.j];
        const x1 = [], x2 = [];
        m.a.forEach((a, k) => {
          x1.push(...G.undistort(this.K, A.kps[2 * a], A.kps[2 * a + 1]));
          x2.push(...G.undistort(this.K, B.kps[2 * m.b[k]], B.kps[2 * m.b[k] + 1]));
        });
        const ess = G.ransacEpipolar(x1, x2, 2.0 / this.K.f, { essential: true, iters: 400 });
        if (!ess) continue;
        const pose = G.poseFromE(ess.F, x1, x2, ess.inliers);
        // Score by inliers that triangulate with a usable angle (needs real baseline).
        const I = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
        const c2 = G.centre(pose.R, pose.t);
        let good = 0;
        for (const k of ess.inliers) {
          const X = G.triangulate([[I, [0, 0, 0], x1[2 * k], x1[2 * k + 1]], [pose.R, pose.t, x2[2 * k], x2[2 * k + 1]]]);
          if (X && X[2] > 0 && G.maxRayAngle([[0, 0, 0], c2], X) > this.o.minAngleDeg * Math.PI / 180) good++;
        }
        if (!best || good > best.good) best = { m, pose, good };
      }
      if (!best || best.good < 40) return false;
      const { m, pose } = best;
      this.cams[m.i] = { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: [0, 0, 0] };
      this.cams[m.j] = { R: Float64Array.from(pose.R), t: pose.t.slice() };
      this.order.push(m.i, m.j);
      this._triangulateFrom(m.j);
      this._ba([m.j], false, 10, 1e-5);
      this._filter();
      return true;
    }

    _nextImage() {
      let best = -1, bestCount = 19;
      for (let im = 0; im < this.images.length; im++) {
        if (this.cams[im] !== null) continue;
        let c = 0;
        for (const t of this.kpTrack[im]) if (t >= 0 && this.tracks[t].X) c++;
        if (c > bestCount) { bestCount = c; best = im; }
      }
      return best;
    }

    _register(im) {
      const cv = this.cv, img = this.images[im];
      const obj = [], pix = [];
      this.kpTrack[im].forEach((t, kp) => {
        if (t >= 0 && this.tracks[t].X) { obj.push(...this.tracks[t].X); pix.push(img.kps[2 * kp], img.kps[2 * kp + 1]); }
      });
      const n = obj.length / 3;
      const O = cv.matFromArray(n, 3, cv.CV_64F, obj), P = cv.matFromArray(n, 2, cv.CV_64F, pix);
      const Km = cv.matFromArray(3, 3, cv.CV_64F, [this.K.f, 0, this.K.cx, 0, this.K.f, this.K.cy, 0, 0, 1]);
      const dist = cv.matFromArray(1, 4, cv.CV_64F, [this.K.k1, 0, 0, 0]);
      const rvec = new cv.Mat(), tvec = new cv.Mat(), inl = new cv.Mat();
      let ok = false;
      try {
        ok = cv.solvePnPRansac(O, P, Km, dist, rvec, tvec, false, 300, this.o.reprojPx, 0.999, inl, cv.SOLVEPNP_EPNP);
      } catch (e) { ok = false; }
      const nin = inl.rows;
      if (ok && nin >= 15) {
        this.cams[im] = { R: LA.expSO3([rvec.data64F[0], rvec.data64F[1], rvec.data64F[2]]), t: [tvec.data64F[0], tvec.data64F[1], tvec.data64F[2]] };
        this.order.push(im);
      }
      [O, P, Km, dist, rvec, tvec, inl].forEach(m => m.delete());
      return ok && nin >= 15;
    }

    // Triangulate tracks seen by `im` and at least one other registered image.
    _triangulateFrom(im) {
      const centres = new Map();
      const centreOf = c => { if (!centres.has(c)) centres.set(c, G.centre(this.cams[c].R, this.cams[c].t)); return centres.get(c); };
      for (const t of this.kpTrack[im]) {
        if (t < 0) continue;
        const tr = this.tracks[t];
        const views = tr.obs.filter(([c]) => this.cams[c]);
        if (views.length < 2) continue;
        if (tr.X) continue; // existing points get the new observation implicitly
        const X = G.triangulate(views.map(([c, kp]) => [this.cams[c].R, this.cams[c].t, ...G.undistort(this.K, this.images[c].kps[2 * kp], this.images[c].kps[2 * kp + 1])]));
        if (!X) continue;
        if (G.maxRayAngle(views.map(([c]) => centreOf(c)), X) < this.o.minAngleDeg * Math.PI / 180) continue;
        let okAll = true;
        for (const [c, kp] of views) {
          const [u, v, z] = G.project(this.K, this.cams[c].R, this.cams[c].t, X);
          if (z <= 0 || Math.hypot(u - this.images[c].kps[2 * kp], v - this.images[c].kps[2 * kp + 1]) > this.o.reprojPx) { okAll = false; break; }
        }
        if (okAll) tr.X = X;
      }
    }

    _state() {
      const obs = [];
      this.tracks.forEach((tr, t) => {
        if (!tr.X) return;
        for (const [c, kp] of tr.obs) if (this.cams[c]) obs.push({ c, p: t, u: this.images[c].kps[2 * kp], v: this.images[c].kps[2 * kp + 1] });
      });
      return { K: this.K, cams: this.cams.map(c => c || null), pts: this.tracks.map(t => t.X), obs };
    }

    _ba(varCams, refineK, iters, tol = 1e-6) {
      if (!varCams.length) return;
      const t0 = performance.now();
      const s = this._state();
      BA.bundleAdjust(s, { varCams: new Set(varCams), refineK, iters, huber: 2.0, tol });
      this.K = s.K;
      for (const c of varCams) this.cams[c] = s.cams[c];
      s.pts.forEach((X, t) => { if (X) this.tracks[t].X = X; });
      this._baMs = (this._baMs || 0) + performance.now() - t0;
      this.timing = { ba: Math.round(this._baMs / 100) / 10 };
    }

    // Drop observations with large reprojection error and points without support.
    _filter() {
      for (const tr of this.tracks) {
        if (!tr.X) continue;
        let good = 0;
        for (const [c, kp] of tr.obs) {
          if (!this.cams[c]) continue;
          const [u, v, z] = G.project(this.K, this.cams[c].R, this.cams[c].t, tr.X);
          if (z > 0 && Math.hypot(u - this.images[c].kps[2 * kp], v - this.images[c].kps[2 * kp + 1]) <= this.o.reprojPx) good++;
        }
        if (good < 2) tr.X = null;
      }
    }

    // ---- 5. Output: gravity, scale, points ----------------------------------------------
    result(markerSizeM) {
      const regs = this.order;
      const pts = [];
      for (const tr of this.tracks) {
        if (!tr.X) continue;
        let n = 0, err = 0;
        for (const [c, kp] of tr.obs) {
          if (!this.cams[c]) continue;
          const [u, v] = G.project(this.K, this.cams[c].R, this.cams[c].t, tr.X);
          err = Math.max(err, Math.hypot(u - this.images[c].kps[2 * kp], v - this.images[c].kps[2 * kp + 1]));
          n++;
        }
        if (n >= 3 && err < 2.5) pts.push(tr.X);
      }
      if (pts.length < 200) throw new Error(`Only ${pts.length} reliable 3D points - the room needs more texture or more photos.`);

      // Gravity: phones are held upright, so the mean image-up direction is world up.
      let up = [0, 0, 0];
      for (const c of regs) { const R = this.cams[c].R; up = [up[0] - R[3], up[1] - R[4], up[2] - R[5]]; }
      const L = Math.hypot(...up); up = up.map(v => v / L);
      const Rg = rotationToZ(up);

      const marker = markerSizeM ? this._markerScale(markerSizeM) : null;
      let scale, scaled;
      if (marker) { scale = marker.scale; scaled = true; }
      else {
        const zs = pts.map(X => LA.mulv3(Rg, X)[2]).sort((a, b) => a - b);
        const lo = zs[Math.floor(zs.length * 0.02)], hi = zs[Math.floor(zs.length * 0.98)];
        scale = this.o.assumedHeight / Math.max(hi - lo, 1e-9);
        scaled = false;
      }
      const out = new Float32Array(pts.length * 3);
      pts.forEach((X, k) => { const p = LA.mulv3(Rg, X); out.set([p[0] * scale, p[1] * scale, p[2] * scale], k * 3); });
      return {
        points: out,
        info: {
          images: this.images.length, registered: regs.length, sparse_points: pts.length,
          scaled, scale_source: scaled ? 'aruco' : `assumed ${this.o.assumedHeight} m tall scene`,
          marker, focal_px: +this.K.f.toFixed(1), k1: +this.K.k1.toFixed(4), size: this.size,
        },
      };
    }

    _markerScale(sizeM) {
      const byId = new Map();
      for (const c of this.order) {
        for (const [id, corners] of Object.entries(this.images[c].markers)) {
          if (!byId.has(id)) byId.set(id, []);
          byId.get(id).push([c, corners]);
        }
      }
      const cands = [];
      for (const [id, views] of byId) {
        if (views.length < 2) continue;
        const X = [0, 1, 2, 3].map(k => G.triangulate(views.map(([c, cs]) => [this.cams[c].R, this.cams[c].t, ...G.undistort(this.K, cs[2 * k], cs[2 * k + 1])])));
        if (X.some(x => !x)) continue;
        const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        const sides = [0, 1, 2, 3].map(k => d(X[k], X[(k + 1) % 4]));
        const side = sides.reduce((s, v) => s + v, 0) / 4;
        const spread = Math.sqrt(sides.reduce((s, v) => s + (v - side) ** 2, 0) / 4) / side;
        const diag = [d(X[0], X[2]), d(X[1], X[3])].map(v => Math.abs(v / (side * Math.SQRT2) - 1));
        if (spread > 0.05 || Math.max(...diag) > 0.05) continue; // not a square: spurious detection
        cands.push({ id: +id, views: views.length, side, spread });
      }
      if (!cands.length) return null;
      const best = cands.reduce((a, b) => (b.views > a.views ? b : a));
      const used = cands.filter(c => Math.abs(c.side / best.side - 1) < 0.05);
      const wsum = used.reduce((s, c) => s + c.views, 0);
      const side = used.reduce((s, c) => s + c.side * c.views, 0) / wsum;
      return { scale: sizeM / side, markers: used.map(c => ({ id: c.id, views: c.views })),
               side_spread_pct: +(Math.max(...used.map(c => c.spread)) * 100).toFixed(2) };
    }

  }

  function knn(cv, matcher, d1, d2) {
    const mv = new cv.DMatchVectorVector();
    matcher.knnMatch(d1, d2, mv, 2);
    const out = new Map(), byTrain = new Map();
    for (let i = 0; i < mv.size(); i++) {
      const m = mv.get(i);
      let best = null;
      if (m.size() >= 2) {
        const a = m.get(0), b = m.get(1);
        if (a.distance < 0.8 * b.distance && a.distance < 64) best = a;
      } else if (m.size() === 1 && m.get(0).distance < 40) {
        best = m.get(0);
      }
      m.delete();
      if (!best) continue;
      // Uniqueness: each train keypoint keeps only its closest query.
      const prev = byTrain.get(best.trainIdx);
      if (prev !== undefined) {
        if (out.get(prev)[1] <= best.distance) continue;
        out.delete(prev);
      }
      out.set(best.queryIdx, [best.trainIdx, best.distance]);
      byTrain.set(best.trainIdx, best.queryIdx);
    }
    mv.delete();
    return out;
  }

  function thumbnail(gray, w, h) {
    const TW = 32, TH = 24, out = new Float32Array(TW * TH);
    for (let y = 0; y < TH; y++) for (let x = 0; x < TW; x++) {
      let s = 0, n = 0;
      const x0 = Math.floor(x * w / TW), x1 = Math.floor((x + 1) * w / TW), y0 = Math.floor(y * h / TH), y1 = Math.floor((y + 1) * h / TH);
      for (let yy = y0; yy < y1; yy += 2) for (let xx = x0; xx < x1; xx += 2) { s += gray[yy * w + xx]; n++; }
      out[y * TW + x] = s / Math.max(n, 1);
    }
    let m = 0; for (const v of out) m += v; m /= out.length;
    let sd = 0; for (const v of out) sd += (v - m) ** 2; sd = Math.sqrt(sd / out.length) || 1;
    for (let i = 0; i < out.length; i++) out[i] = (out[i] - m) / sd;
    return out;
  }

  function ncc(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s / a.length; }

  function rotationToZ(v) {
    const z = [0, 0, 1], c = LA.dot(v, z);
    if (c > 0.999999) return Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
    if (c < -0.999999) return Float64Array.of(1, 0, 0, 0, -1, 0, 0, 0, -1);
    const axis = LA.cross(v, z), s = LA.norm(axis);
    return LA.expSO3(axis.map(a => a / s * Math.atan2(s, c)));
  }

  const SfM = { Reconstruction, DEFAULTS, extractFeatures, selectPairs, matchPairs };
  if (typeof module !== 'undefined' && module.exports) module.exports = SfM;
  else root.SfM = SfM;
})(typeof self !== 'undefined' ? self : this);
