// Multi-view geometry for the in-browser SfM: camera model, F/E estimation,
// pose recovery, triangulation. Cameras: { R (world->cam, row-major 3x3), t: [3] }.
// Intrinsics (shared by all images): { f, cx, cy, k1 } with one radial term.
(function (root) {
  'use strict';
  const LA = root.LA || (typeof require !== 'undefined' ? require('./linalg.js') : null);

  function project(K, R, t, X) {
    const x = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
    const y = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
    const z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
    const u = x / z, v = y / z;
    const d = 1 + K.k1 * (u * u + v * v);
    return [K.f * d * u + K.cx, K.f * d * v + K.cy, z];
  }

  // Pixel -> normalised, undistorted image coordinates.
  function undistort(K, px, py) {
    const ud = (px - K.cx) / K.f, vd = (py - K.cy) / K.f;
    let u = ud, v = vd;
    for (let i = 0; i < 6; i++) {
      const d = 1 + K.k1 * (u * u + v * v);
      u = ud / d; v = vd / d;
    }
    return [u, v];
  }

  function centre(R, t) { return LA.mulTv3(R, t).map(x => -x); }

  // ---- Epipolar geometry --------------------------------------------------------
  function hartley(pts, idx) {
    let mx = 0, my = 0;
    for (const i of idx) { mx += pts[2 * i]; my += pts[2 * i + 1]; }
    mx /= idx.length; my /= idx.length;
    let d = 0;
    for (const i of idx) d += Math.hypot(pts[2 * i] - mx, pts[2 * i + 1] - my);
    const s = Math.SQRT2 / (d / idx.length || 1);
    return { s, mx, my };
  }

  // Normalised 8-point fundamental matrix from correspondences idx (pixel coords, flat xy arrays).
  function eightPoint(p1, p2, idx, essential = false) {
    const n1 = hartley(p1, idx), n2 = hartley(p2, idx);
    const A = new Float64Array(idx.length * 9);
    idx.forEach((i, r) => {
      const x1 = (p1[2 * i] - n1.mx) * n1.s, y1 = (p1[2 * i + 1] - n1.my) * n1.s;
      const x2 = (p2[2 * i] - n2.mx) * n2.s, y2 = (p2[2 * i + 1] - n2.my) * n2.s;
      A.set([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1], r * 9);
    });
    let F = LA.nullVector(A, idx.length, 9);
    const { U, S, V } = LA.svd3(F);
    const s0 = essential ? 1 : S[0], s1 = essential ? 1 : S[1];
    // Rebuild with the smallest singular value zeroed (and equalised for E).
    const D = Float64Array.of(s0, 0, 0, 0, s1, 0, 0, 0, 0);
    F = LA.mul3(LA.mul3(U, D), LA.transpose3(V));
    const T1 = Float64Array.of(n1.s, 0, -n1.s * n1.mx, 0, n1.s, -n1.s * n1.my, 0, 0, 1);
    const T2 = Float64Array.of(n2.s, 0, -n2.s * n2.mx, 0, n2.s, -n2.s * n2.my, 0, 0, 1);
    return LA.mul3(LA.mul3(LA.transpose3(T2), F), T1);
  }

  function sampson(F, x1, y1, x2, y2) {
    const Fx1 = [F[0] * x1 + F[1] * y1 + F[2], F[3] * x1 + F[4] * y1 + F[5], F[6] * x1 + F[7] * y1 + F[8]];
    const Ftx2 = [F[0] * x2 + F[3] * y2 + F[6], F[1] * x2 + F[4] * y2 + F[7], F[2] * x2 + F[5] * y2 + F[8]];
    const e = x2 * Fx1[0] + y2 * Fx1[1] + Fx1[2];
    return e * e / (Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2 + 1e-30);
  }

  // RANSAC over the 8-point solver. thresh is in the units of the points (px or normalised).
  function ransacEpipolar(p1, p2, thresh, { essential = false, iters = 400, rng = Math.random } = {}) {
    const n = p1.length / 2;
    if (n < 8) return null;
    const t2 = thresh * thresh;
    let best = null, bestCount = 0;
    const score = F => {
      const inl = [];
      for (let i = 0; i < n; i++) if (sampson(F, p1[2 * i], p1[2 * i + 1], p2[2 * i], p2[2 * i + 1]) < t2) inl.push(i);
      return inl;
    };
    let maxIt = iters;
    for (let it = 0; it < maxIt; it++) {
      const s = new Set();
      while (s.size < 8) s.add(Math.floor(rng() * n));
      const F = eightPoint(p1, p2, [...s], essential);
      const inl = score(F);
      if (inl.length > bestCount) {
        bestCount = inl.length; best = { F, inliers: inl };
        const w = inl.length / n;
        maxIt = Math.min(iters, Math.ceil(Math.log(0.01) / Math.log(1 - w ** 8 + 1e-12)) + 10);
      }
    }
    if (!best || bestCount < 8) return null;
    // Refit on all inliers, twice.
    for (let k = 0; k < 2; k++) {
      const F = eightPoint(p1, p2, best.inliers, essential);
      const inl = score(F);
      if (inl.length >= best.inliers.length) best = { F, inliers: inl };
    }
    return best;
  }

  // Four (R, t) candidates from E; pick by cheirality on normalised correspondences.
  function poseFromE(E, x1, x2, idx) {
    let { U, V } = LA.svd3(E);
    if (LA.det3(U) < 0) U = U.map((v, i) => (i % 3 === 2 ? -v : v));
    if (LA.det3(V) < 0) V = V.map((v, i) => (i % 3 === 2 ? -v : v));
    const W = Float64Array.of(0, -1, 0, 1, 0, 0, 0, 0, 1);
    const Vt = LA.transpose3(V);
    const Ra = LA.mul3(LA.mul3(U, W), Vt), Rb = LA.mul3(LA.mul3(U, LA.transpose3(W)), Vt);
    const u3 = [U[2], U[5], U[8]];
    const cands = [[Ra, u3], [Ra, u3.map(v => -v)], [Rb, u3], [Rb, u3.map(v => -v)]];
    const I = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
    let best = null;
    for (const [R, t] of cands) {
      let good = 0;
      for (const i of idx) {
        const X = triangulate([[I, [0, 0, 0], x1[2 * i], x1[2 * i + 1]], [R, t, x2[2 * i], x2[2 * i + 1]]]);
        if (!X) continue;
        const z1 = X[2], z2 = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
        if (z1 > 0 && z2 > 0) good++;
      }
      if (!best || good > best.good) best = { R, t, good };
    }
    return best;
  }

  // Linear (DLT) triangulation from views [R, t, x, y] in normalised coordinates.
  function triangulate(views) {
    const A = new Float64Array(views.length * 2 * 4);
    views.forEach(([R, t, x, y], k) => {
      const P0 = [R[0], R[1], R[2], t[0]], P1 = [R[3], R[4], R[5], t[1]], P2 = [R[6], R[7], R[8], t[2]];
      for (let c = 0; c < 4; c++) {
        A[(2 * k) * 4 + c] = x * P2[c] - P0[c];
        A[(2 * k + 1) * 4 + c] = y * P2[c] - P1[c];
      }
    });
    const X = LA.nullVector(A, views.length * 2, 4);
    if (Math.abs(X[3]) < 1e-12) return null;
    return [X[0] / X[3], X[1] / X[3], X[2] / X[3]];
  }

  // Largest angle (radians) between viewing rays from camera centres to X.
  function maxRayAngle(centres, X) {
    let best = 0;
    for (let i = 0; i < centres.length; i++) for (let j = i + 1; j < centres.length; j++) {
      const a = [X[0] - centres[i][0], X[1] - centres[i][1], X[2] - centres[i][2]];
      const b = [X[0] - centres[j][0], X[1] - centres[j][1], X[2] - centres[j][2]];
      const c = LA.dot(a, b) / (LA.norm(a) * LA.norm(b) + 1e-30);
      best = Math.max(best, Math.acos(Math.min(1, Math.max(-1, c))));
    }
    return best;
  }

  const G = { project, undistort, centre, eightPoint, sampson, ransacEpipolar, poseFromE, triangulate, maxRayAngle };
  if (typeof module !== 'undefined' && module.exports) module.exports = G;
  else root.G = G;
})(typeof self !== 'undefined' ? self : this);
