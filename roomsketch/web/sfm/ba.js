// Sparse bundle adjustment (Levenberg-Marquardt + Schur complement on points).
// Refines camera poses, 3D points and, optionally, the shared intrinsics (f, k1).
(function (root) {
  'use strict';
  const LA = root.LA || (typeof require !== 'undefined' ? require('./linalg.js') : null);

  /**
   * state: { K: {f,cx,cy,k1}, cams: [{R,t}], pts: [[x,y,z] | null], obs: [{c,p,u,v}] }
   * opts:  { varCams: Set<camIdx>, refineK: bool, iters, huber (px) }
   * Points observed by any varying camera are optimised; other cameras stay fixed.
   */
  function bundleAdjust(state, opts = {}) {
    const { varCams, refineK = false, iters = 20, huber = 2.0, tol = 1e-6 } = opts;
    const camBlock = new Map();
    [...varCams].sort((a, b) => a - b).forEach((c, i) => camBlock.set(c, i));
    const nv = camBlock.size;
    const D = 6 * nv + (refineK ? 2 : 0);
    const kOff = 6 * nv;

    // Points touched by varying cameras, with all of their observations (grouped by point).
    const ptSet = new Set();
    for (const o of state.obs) if (camBlock.has(o.c) && state.pts[o.p]) ptSet.add(o.p);
    const byPoint = new Map();
    for (const o of state.obs) {
      if (!ptSet.has(o.p)) continue;
      let l = byPoint.get(o.p);
      if (!l) byPoint.set(o.p, (l = []));
      l.push(o);
    }
    const points = [...byPoint.keys()];
    const np = points.length;
    if (!np || D === 0) return { cost: 0, iterations: 0 };
    const start = new Int32Array(np + 1);
    const obs = [];
    points.forEach((p, j) => { for (const o of byPoint.get(p)) obs.push(o); start[j + 1] = obs.length; });
    const no = obs.length;
    const ocb = new Int32Array(no);           // camera block index or -1
    obs.forEach((o, k) => { const b = camBlock.get(o.c); ocb[k] = b === undefined ? -1 : b; });
    let maxDeg = 0;
    for (let j = 0; j < np; j++) maxDeg = Math.max(maxDeg, start[j + 1] - start[j]);

    // Flat work buffers (no allocation inside the iterations).
    const Jc = new Float64Array(no * 12), Jp = new Float64Array(no * 6), Jk = new Float64Array(no * 4);
    const rr = new Float64Array(no * 2), ww = new Float64Array(no), valid = new Uint8Array(no);
    const Wc = new Float64Array(no * 18);
    const V = new Float64Array(np * 9), bp = new Float64Array(np * 3), kW = new Float64Array(np * 6);
    const U = new Float64Array(D * D), bc = new Float64Array(D);
    const S = new Float64Array(D * D), rhs = new Float64Array(D);
    const Vinv = new Float64Array(np * 9), okPt = new Uint8Array(np);
    const WV = new Float64Array((maxDeg + 1) * 18), bOff = new Int32Array(maxDeg + 1), bSize = new Int32Array(maxDeg + 1);
    const bW = new Array(maxDeg + 1);

    const evalCost = (K, cams, pts) => {
      let cost = 0;
      for (let j = 0; j < np; j++) {
        const X = pts[points[j]];
        for (let k = start[j]; k < start[j + 1]; k++) {
          const r = resid(K, cams[obs[k].c], X, obs[k]);
          if (!r) { cost += huber * huber * 4; continue; }
          const s = Math.hypot(r[0], r[1]);
          cost += s <= huber ? s * s : 2 * huber * s - huber * huber;
        }
      }
      return cost;
    };

    let cost = evalCost(state.K, state.cams, state.pts);
    let lambda = 1e-3;
    let it = 0, done = false;
    for (; it < iters && !done; it++) {
      // ---- Linearise at the current estimate.
      U.fill(0); bc.fill(0); V.fill(0); bp.fill(0); kW.fill(0); Wc.fill(0);
      for (let j = 0; j < np; j++) {
        const X = state.pts[points[j]];
        for (let k = start[j]; k < start[j + 1]; k++) {
          const o = obs[k];
          valid[k] = jacInto(state.K, state.cams[o.c], X, o, Jc, Jp, Jk, rr, k) ? 1 : 0;
          if (!valid[k]) continue;
          const s = Math.hypot(rr[2 * k], rr[2 * k + 1]);
          const w = (ww[k] = s <= huber ? 1 : huber / s);
          const r0 = rr[2 * k], r1 = rr[2 * k + 1];
          const jp = 6 * k;
          for (let a = 0; a < 3; a++) {
            const pa0 = Jp[jp + a], pa1 = Jp[jp + 3 + a];
            bp[3 * j + a] -= w * (pa0 * r0 + pa1 * r1);
            for (let b = 0; b < 3; b++) V[9 * j + 3 * a + b] += w * (pa0 * Jp[jp + b] + pa1 * Jp[jp + 3 + b]);
          }
          const cb = ocb[k];
          const jc = 12 * k, jk = 4 * k;
          if (cb >= 0) {
            const off = 6 * cb;
            for (let a = 0; a < 6; a++) {
              const ca0 = Jc[jc + a], ca1 = Jc[jc + 6 + a];
              bc[off + a] -= w * (ca0 * r0 + ca1 * r1);
              const row = (off + a) * D + off;
              for (let b = 0; b < 6; b++) U[row + b] += w * (ca0 * Jc[jc + b] + ca1 * Jc[jc + 6 + b]);
              for (let b = 0; b < 3; b++) Wc[18 * k + 3 * a + b] = w * (ca0 * Jp[jp + b] + ca1 * Jp[jp + 3 + b]);
              if (refineK) for (let b = 0; b < 2; b++) {
                const v = w * (ca0 * Jk[jk + b] + ca1 * Jk[jk + 2 + b]);
                U[(kOff + b) * D + off + a] += v;
                U[(off + a) * D + kOff + b] += v;
              }
            }
          }
          if (refineK) {
            for (let a = 0; a < 2; a++) {
              const ka0 = Jk[jk + a], ka1 = Jk[jk + 2 + a];
              bc[kOff + a] -= w * (ka0 * r0 + ka1 * r1);
              for (let b = 0; b < 2; b++) U[(kOff + a) * D + kOff + b] += w * (ka0 * Jk[jk + b] + ka1 * Jk[jk + 2 + b]);
              for (let b = 0; b < 3; b++) kW[6 * j + 3 * a + b] += w * (ka0 * Jp[jp + b] + ka1 * Jp[jp + 3 + b]);
            }
          }
        }
      }

      // ---- Damped Schur solves until the cost drops.
      let accepted = false;
      for (let tries = 0; tries < 6 && !accepted; tries++) {
        S.set(U); rhs.set(bc);
        for (let i = 0; i < D; i++) S[i * D + i] += lambda * Math.max(U[i * D + i], 1e-6) + 1e-9;
        for (let j = 0; j < np; j++) {
          const v = 9 * j;
          const Vd = [V[v], V[v + 1], V[v + 2], V[v + 3], V[v + 4], V[v + 5], V[v + 6], V[v + 7], V[v + 8]];
          for (let i = 0; i < 3; i++) Vd[i * 4] += lambda * Math.max(V[v + i * 4], 1e-9) + 1e-12;
          const Vi = LA.inv3(Vd);
          if (!Vi) { okPt[j] = 0; continue; }
          okPt[j] = 1;
          Vinv.set(Vi, v);
          // Camera-side blocks of this point: its varying cameras, plus intrinsics.
          let nb = 0;
          for (let k = start[j]; k < start[j + 1]; k++) {
            if (!valid[k] || ocb[k] < 0) continue;
            bOff[nb] = 6 * ocb[k]; bSize[nb] = 6; bW[nb] = 18 * k; nb++;
          }
          if (refineK) { bOff[nb] = kOff; bSize[nb] = 2; bW[nb] = -1; nb++; }
          for (let a = 0; a < nb; a++) {
            const src = bW[a] >= 0 ? Wc : kW, so = bW[a] >= 0 ? bW[a] : 6 * j;
            for (let i = 0; i < bSize[a]; i++) for (let c = 0; c < 3; c++)
              WV[18 * a + 3 * i + c] = src[so + 3 * i] * Vi[c] + src[so + 3 * i + 1] * Vi[3 + c] + src[so + 3 * i + 2] * Vi[6 + c];
          }
          const b0 = bp[3 * j], b1 = bp[3 * j + 1], b2 = bp[3 * j + 2];
          for (let a = 0; a < nb; a++) {
            const wa = 18 * a, offA = bOff[a];
            for (let i = 0; i < bSize[a]; i++) rhs[offA + i] -= WV[wa + 3 * i] * b0 + WV[wa + 3 * i + 1] * b1 + WV[wa + 3 * i + 2] * b2;
            // Lower triangle only (Cholesky reads it): blocks with offA >= offB.
            for (let b = 0; b < nb; b++) {
              const offB = bOff[b];
              if (offB > offA) continue;
              const srcB = bW[b] >= 0 ? Wc : kW, sb = bW[b] >= 0 ? bW[b] : 6 * j;
              for (let i = 0; i < bSize[a]; i++) {
                const row = (offA + i) * D + offB;
                const x0 = WV[wa + 3 * i], x1 = WV[wa + 3 * i + 1], x2 = WV[wa + 3 * i + 2];
                for (let q = 0; q < bSize[b]; q++)
                  S[row + q] -= x0 * srcB[sb + 3 * q] + x1 * srcB[sb + 3 * q + 1] + x2 * srcB[sb + 3 * q + 2];
              }
            }
          }
        }
        const dc = LA.cholSolve(S, rhs, D);
        if (!dc) { lambda *= 10; continue; }

        // Candidate update.
        const K2 = { ...state.K };
        if (refineK) { K2.f += dc[kOff]; K2.k1 += dc[kOff + 1]; }
        const cams2 = state.cams.slice();
        for (const [c, i] of camBlock) {
          const o = 6 * i;
          cams2[c] = {
            R: LA.mul3(LA.expSO3([dc[o], dc[o + 1], dc[o + 2]]), state.cams[c].R),
            t: [state.cams[c].t[0] + dc[o + 3], state.cams[c].t[1] + dc[o + 4], state.cams[c].t[2] + dc[o + 5]],
          };
        }
        const pts2 = state.pts.slice();
        for (let j = 0; j < np; j++) {
          if (!okPt[j]) continue;
          let g0 = bp[3 * j], g1 = bp[3 * j + 1], g2 = bp[3 * j + 2];
          for (let k = start[j]; k < start[j + 1]; k++) {
            if (!valid[k] || ocb[k] < 0) continue;
            const off = 6 * ocb[k], wo = 18 * k;
            for (let i = 0; i < 6; i++) { const d = dc[off + i]; g0 -= Wc[wo + 3 * i] * d; g1 -= Wc[wo + 3 * i + 1] * d; g2 -= Wc[wo + 3 * i + 2] * d; }
          }
          if (refineK) for (let i = 0; i < 2; i++) { const d = dc[kOff + i]; g0 -= kW[6 * j + 3 * i] * d; g1 -= kW[6 * j + 3 * i + 1] * d; g2 -= kW[6 * j + 3 * i + 2] * d; }
          const v = 9 * j, X = state.pts[points[j]];
          pts2[points[j]] = [X[0] + Vinv[v] * g0 + Vinv[v + 1] * g1 + Vinv[v + 2] * g2,
                             X[1] + Vinv[v + 3] * g0 + Vinv[v + 4] * g1 + Vinv[v + 5] * g2,
                             X[2] + Vinv[v + 6] * g0 + Vinv[v + 7] * g1 + Vinv[v + 8] * g2];
        }
        const cost2 = evalCost(K2, cams2, pts2);
        if (cost2 < cost && K2.f > 0) {
          const rel = (cost - cost2) / cost;
          state.K = K2; state.cams = cams2; state.pts = pts2;
          cost = cost2;
          lambda = Math.max(lambda * 0.3, 1e-9);
          accepted = true;
          if (rel < tol) done = true;
        } else {
          lambda *= 8;
        }
      }
      if (!accepted) break;
    }
    return { cost, iterations: it };
  }

  function resid(K, cam, X, o) {
    const R = cam.R, t = cam.t;
    const z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
    if (z <= 1e-6) return null;
    const x = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
    const y = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
    const u = x / z, v = y / z, d = 1 + K.k1 * (u * u + v * v);
    return [K.f * d * u + K.cx - o.u, K.f * d * v + K.cy - o.v];
  }

  function jac(K, cam, X, o) {
    const R = cam.R, t = cam.t;
    const rx = R[0] * X[0] + R[1] * X[1] + R[2] * X[2];
    const ry = R[3] * X[0] + R[4] * X[1] + R[5] * X[2];
    const rz = R[6] * X[0] + R[7] * X[1] + R[8] * X[2];
    const x = rx + t[0], y = ry + t[1], z = rz + t[2];
    if (z <= 1e-6) return null;
    const iz = 1 / z, u = x * iz, v = y * iz;
    const r2 = u * u + v * v, d = 1 + K.k1 * r2, f = K.f;
    const r = [f * d * u + K.cx - o.u, f * d * v + K.cy - o.v];
    const a = d + 2 * K.k1 * u * u, b = 2 * K.k1 * u * v, c = d + 2 * K.k1 * v * v;
    // d(pix)/d(p): f * [[a,b],[b,c]] * [[1/z,0,-u/z],[0,1/z,-v/z]]
    const P = [f * a * iz, f * b * iz, -f * (a * u + b * v) * iz,
               f * b * iz, f * c * iz, -f * (b * u + c * v) * iz];
    // Point: dp/dX = R.
    const Jp = new Float64Array(6);
    for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++)
      Jp[i * 3 + j] = P[i * 3] * R[j] + P[i * 3 + 1] * R[3 + j] + P[i * 3 + 2] * R[6 + j];
    // Camera: left-perturbed rotation dp/dw = -[R X]x, translation dp/dt = I.
    const Sx = [0, rz, -ry, -rz, 0, rx, ry, -rx, 0]; // -[RX]x
    const Jc = new Float64Array(12);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 3; j++)
        Jc[i * 6 + j] = P[i * 3] * Sx[j] + P[i * 3 + 1] * Sx[3 + j] + P[i * 3 + 2] * Sx[6 + j];
      for (let j = 0; j < 3; j++) Jc[i * 6 + 3 + j] = P[i * 3 + j];
    }
    const Jk = Float64Array.of(d * u, f * u * r2, d * v, f * v * r2);
    return { r, Jp, Jc, Jk };
  }

  function jacInto(K, cam, X, o, Jc, Jp, Jk, rr, k) {
    const J = jac(K, cam, X, o);
    if (!J) return false;
    Jc.set(J.Jc, 12 * k); Jp.set(J.Jp, 6 * k); Jk.set(J.Jk, 4 * k);
    rr[2 * k] = J.r[0]; rr[2 * k + 1] = J.r[1];
    return true;
  }

  const BA = { bundleAdjust, resid, _jac: jac };
  if (typeof module !== 'undefined' && module.exports) module.exports = BA;
  else root.BA = BA;
})(typeof self !== 'undefined' ? self : this);
