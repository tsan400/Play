// Node unit test for the in-browser SfM math: node tests/js/sfm_math.test.js
const assert = require('assert');
const LA = require('../../web/sfm/linalg.js');
const G = require('../../web/sfm/geometry.js');
const BA = require('../../web/sfm/ba.js');

let seed = 1;
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());

function lookAt(eye, target) {
  const f = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]].map((v, _, a) => v / Math.hypot(...a));
  let r = LA.cross(f, [0, 0, 1]); const rl = LA.norm(r); r = r.map(v => v / rl);
  const d = LA.cross(f, r);
  const R = Float64Array.of(...r, ...d, ...f);
  const t = LA.mulv3(R, eye).map(v => -v);
  return { R, t };
}

// Room-like scene: points on the walls of a 5 x 4 x 2.6 box, cameras walking a loop inside.
const K = { f: 900, cx: 640, cy: 480, k1: -0.05 };
const pts = [];
for (let i = 0; i < 1500; i++) {
  const w = Math.floor(rnd() * 4), a = rnd(), h = rnd() * 2.6;
  pts.push(w === 0 ? [a * 5, 0, h] : w === 1 ? [5, a * 4, h] : w === 2 ? [a * 5, 4, h] : [0, a * 4, h]);
}
const cams = [];
for (let i = 0; i < 16; i++) {
  const th = 2 * Math.PI * i / 16;
  const eye = [2.5 + Math.cos(th), 2 + 0.7 * Math.sin(th), 1.5];
  const look = th + Math.PI + 0.3;
  cams.push(lookAt(eye, [eye[0] + Math.cos(look), eye[1] + Math.sin(look), 1.3]));
}
const obs = [];
cams.forEach((c, ci) => pts.forEach((X, pi) => {
  const [u, v, z] = G.project(K, c.R, c.t, X);
  if (z > 1.0 && u >= 0 && u < 1280 && v >= 0 && v < 960) obs.push({ c: ci, p: pi, u: u + 0.5 * gauss(), v: v + 0.5 * gauss() });
}));

// 1) Two-view init from the first pair with the true intrinsics (but no k1 knowledge).
const pair = obs.filter(o => o.c === 0).map(o => o.p).filter(p => obs.some(o => o.c === 1 && o.p === p));
const get = (c, p) => obs.find(o => o.c === c && o.p === p);
const x1 = [], x2 = [];
for (const p of pair) {
  const a = G.undistort(K, get(0, p).u, get(0, p).v), b = G.undistort(K, get(1, p).u, get(1, p).v);
  x1.push(...a); x2.push(...b);
}
const ess = G.ransacEpipolar(x1, x2, 1.5 / K.f, { essential: true, rng: rnd });
assert(ess && ess.inliers.length > 0.9 * pair.length, 'essential RANSAC inliers');
const pose = G.poseFromE(ess.F, x1, x2, ess.inliers);
const Rrel = LA.mul3(cams[1].R, LA.transpose3(cams[0].R));
const angErr = LA.norm(LA.logSO3(LA.mul3(pose.R, LA.transpose3(Rrel))));
assert(angErr < 0.01, 'relative rotation error ' + angErr);
console.log('two-view: inliers', ess.inliers.length, '/', pair.length, 'rot err (rad)', angErr.toFixed(5));

// 2) Triangulation is exact on noise-free views.
const X0 = pts[pair[0]];
const views = [0, 1].map(c => { const [u, v] = G.project(K, cams[c].R, cams[c].t, X0); return [cams[c].R, cams[c].t, ...G.undistort(K, u, v)]; });
const Xt = G.triangulate(views);
assert(Math.hypot(Xt[0] - X0[0], Xt[1] - X0[1], Xt[2] - X0[2]) < 1e-6, 'triangulation');

// 3) Bundle adjustment from perturbed poses/points and a wrong focal length.
const state = {
  K: { ...K, f: 850, k1: 0 },
  cams: cams.map((c, i) => i < 2 ? c : {
    R: LA.mul3(LA.expSO3([0.01 * gauss(), 0.01 * gauss(), 0.01 * gauss()]), c.R),
    t: c.t.map(v => v + 0.03 * gauss()),
  }),
  pts: pts.map(X => X.map(v => v + 0.03 * gauss())),
  obs,
};
const rms = s => { let e = 0, n = 0; for (const o of s.obs) { const r = BA.resid(s.K, s.cams[o.c], s.pts[o.p], o); if (r) { e += r[0] ** 2 + r[1] ** 2; n++; } } return Math.sqrt(e / n); };
const before = rms(state);
const t0 = Date.now();
const res = BA.bundleAdjust(state, { varCams: new Set([...cams.keys()].filter(i => i > 1)), refineK: true, iters: 40 });
console.log(`BA: rms ${before.toFixed(1)} -> ${rms(state).toFixed(3)} px, f=${state.K.f.toFixed(1)} k1=${state.K.k1.toFixed(4)}, ${res.iterations} it, ${Date.now() - t0} ms, ${obs.length} obs`);
assert(rms(state) < 0.8, 'BA residual');
assert(Math.abs(state.K.f - 900) < 15, 'focal recovered');
console.log('OK');
