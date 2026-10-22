// Small dense linear algebra for the in-browser SfM (row-major Float64Arrays).
// Classic script: defines `LA` on the global scope (worker) or module.exports (Node tests).
(function (root) {
  'use strict';

  // Symmetric eigendecomposition (cyclic Jacobi). Returns ascending eigenvalues and
  // eigenvectors as columns of V (V[r*n + c] = component r of vector c).
  function eigSym(A, n) {
    const a = Float64Array.from(A);
    const V = new Float64Array(n * n);
    for (let i = 0; i < n; i++) V[i * n + i] = 1;
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
      if (off < 1e-30) break;
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          const apq = a[p * n + q];
          if (Math.abs(apq) < 1e-300) continue;
          const app = a[p * n + p], aqq = a[q * n + q];
          const theta = (aqq - app) / (2 * apq);
          const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < n; k++) {
            const akp = a[k * n + p], akq = a[k * n + q];
            a[k * n + p] = c * akp - s * akq;
            a[k * n + q] = s * akp + c * akq;
          }
          for (let k = 0; k < n; k++) {
            const apk = a[p * n + k], aqk = a[q * n + k];
            a[p * n + k] = c * apk - s * aqk;
            a[q * n + k] = s * apk + c * aqk;
          }
          for (let k = 0; k < n; k++) {
            const vkp = V[k * n + p], vkq = V[k * n + q];
            V[k * n + p] = c * vkp - s * vkq;
            V[k * n + q] = s * vkp + c * vkq;
          }
        }
      }
    }
    const order = [...Array(n).keys()].sort((i, j) => a[i * n + i] - a[j * n + j]);
    const values = new Float64Array(n), vectors = new Float64Array(n * n);
    order.forEach((src, dst) => {
      values[dst] = a[src * n + src];
      for (let r = 0; r < n; r++) vectors[r * n + dst] = V[r * n + src];
    });
    return { values, vectors };
  }

  // Unit vector x minimising |A x| for A (m x n): smallest eigenvector of AᵀA.
  function nullVector(A, m, n) {
    const AtA = new Float64Array(n * n);
    for (let r = 0; r < m; r++) {
      const o = r * n;
      for (let i = 0; i < n; i++) {
        const ai = A[o + i];
        if (ai === 0) continue;
        for (let j = i; j < n; j++) AtA[i * n + j] += ai * A[o + j];
      }
    }
    for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) AtA[i * n + j] = AtA[j * n + i];
    const { vectors } = eigSym(AtA, n);
    const x = new Float64Array(n);
    for (let r = 0; r < n; r++) x[r] = vectors[r * n];
    return x;
  }

  // 3x3 SVD via eigen(MᵀM). Returns U, S (desc), V with M = U diag(S) Vᵀ, det(U),det(V) free.
  function svd3(M) {
    const MtM = mul3(transpose3(M), M);
    const { values, vectors } = eigSym(MtM, 3);
    const V = new Float64Array(9), S = new Float64Array(3);
    for (let k = 0; k < 3; k++) {          // descending
      const src = 2 - k;
      S[k] = Math.sqrt(Math.max(values[src], 0));
      for (let r = 0; r < 3; r++) V[r * 3 + k] = vectors[r * 3 + src];
    }
    const U = new Float64Array(9);
    for (let k = 0; k < 2; k++) {
      const v = [V[k], V[3 + k], V[6 + k]];
      const u = mulv3(M, v);
      const L = Math.hypot(u[0], u[1], u[2]) || 1;
      for (let r = 0; r < 3; r++) U[r * 3 + k] = u[r] / L;
    }
    const u2 = cross([U[0], U[3], U[6]], [U[1], U[4], U[7]]);
    for (let r = 0; r < 3; r++) U[r * 3 + 2] = u2[r];
    return { U, S, V };
  }

  // In-place Cholesky solve of SPD A (n x n) x = b. Adds jitter if needed. Returns x or null.
  function cholSolve(A, b, n) {
    const L = Float64Array.from(A);
    for (let j = 0; j < n; j++) {
      let d = L[j * n + j];
      for (let k = 0; k < j; k++) d -= L[j * n + k] ** 2;
      if (!(d > 1e-12)) return null;
      d = Math.sqrt(d);
      L[j * n + j] = d;
      for (let i = j + 1; i < n; i++) {
        let s = L[i * n + j];
        const oi = i * n, oj = j * n;
        for (let k = 0; k < j; k++) s -= L[oi + k] * L[oj + k];
        L[oi + j] = s / d;
      }
    }
    const y = Float64Array.from(b);
    for (let i = 0; i < n; i++) {
      let s = y[i];
      for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
      y[i] = s / L[i * n + i];
    }
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < n; k++) s -= L[k * n + i] * y[k];
      y[i] = s / L[i * n + i];
    }
    return y;
  }

  function inv3(m) {
    const [a, b, c, d, e, f, g, h, i] = m;
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-300) return null;
    const r = 1 / det;
    return Float64Array.of(A * r, -(b * i - c * h) * r, (b * f - c * e) * r,
      B * r, (a * i - c * g) * r, -(a * f - c * d) * r,
      C * r, -(a * h - b * g) * r, (a * e - b * d) * r);
  }

  function mul3(A, B) {
    const C = new Float64Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
      C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
    return C;
  }
  const transpose3 = A => Float64Array.of(A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]);
  const mulv3 = (A, v) => [A[0] * v[0] + A[1] * v[1] + A[2] * v[2], A[3] * v[0] + A[4] * v[1] + A[5] * v[2], A[6] * v[0] + A[7] * v[1] + A[8] * v[2]];
  const mulTv3 = (A, v) => [A[0] * v[0] + A[3] * v[1] + A[6] * v[2], A[1] * v[0] + A[4] * v[1] + A[7] * v[2], A[2] * v[0] + A[5] * v[1] + A[8] * v[2]];
  const det3 = m => m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = a => Math.hypot(a[0], a[1], a[2]);

  // Rotation vector <-> matrix.
  function expSO3(w) {
    const th = norm(w);
    if (th < 1e-12) return Float64Array.of(1, -w[2], w[1], w[2], 1, -w[0], -w[1], w[0], 1);
    const k = [w[0] / th, w[1] / th, w[2] / th];
    const c = Math.cos(th), s = Math.sin(th), v = 1 - c;
    return Float64Array.of(
      c + k[0] * k[0] * v, k[0] * k[1] * v - k[2] * s, k[0] * k[2] * v + k[1] * s,
      k[1] * k[0] * v + k[2] * s, c + k[1] * k[1] * v, k[1] * k[2] * v - k[0] * s,
      k[2] * k[0] * v - k[1] * s, k[2] * k[1] * v + k[0] * s, c + k[2] * k[2] * v);
  }

  function logSO3(R) {
    const c = Math.min(1, Math.max(-1, (R[0] + R[4] + R[8] - 1) / 2));
    const th = Math.acos(c);
    if (th < 1e-9) return [0, 0, 0];
    if (Math.PI - th < 1e-6) {
      // Near 180°: axis from the largest diagonal entry.
      const i = R[0] > R[4] && R[0] > R[8] ? 0 : R[4] > R[8] ? 1 : 2;
      const a = [0, 0, 0];
      a[i] = Math.sqrt(Math.max((R[i * 4] + 1) / 2, 0));
      for (let j = 0; j < 3; j++) if (j !== i) a[j] = R[i * 3 + j] / (2 * a[i]);
      return a.map(x => x * th);
    }
    const s = th / (2 * Math.sin(th));
    return [(R[7] - R[5]) * s, (R[2] - R[6]) * s, (R[3] - R[1]) * s];
  }

  const LA = { eigSym, nullVector, svd3, cholSolve, inv3, mul3, transpose3, mulv3, mulTv3, det3, cross, dot, norm, expSO3, logSO3 };
  if (typeof module !== 'undefined' && module.exports) module.exports = LA;
  else root.LA = LA;
})(typeof self !== 'undefined' ? self : this);
