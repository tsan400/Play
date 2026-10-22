// Photogrammetry pool worker. Every worker loads OpenCV.js and serves tasks:
//   {id, type:'features', blob, focal35, options} -> {w, h, focal, kps, desc, thumb, markers}
//   {id, type:'pairs', thumbs, options}           -> [[i, j], ...]
//   {id, type:'match', pairs, feats, options}     -> {matches}
//   {id, type:'solve', images, matches, size, focal, options} -> {points, info}
// Replies: {id, ok:true, result} | {id, ok:false, error}; progress: {id, progress:[f, msg]}.
/* global cv, SfM */
importScripts('../vendor/opencv/opencv.js', 'linalg.js', 'geometry.js', 'ba.js', 'pipeline.js');

const cvReady = new Promise(resolve => {
  const c = self.cv;
  if (c && c.Mat) return resolve(c);
  if (c && typeof c.then === 'function') {
    // Emscripten's module is thenable; strip `then` so resolving doesn't recurse.
    c.then(mod => { delete mod.then; resolve(mod); });
  } else {
    c.onRuntimeInitialized = () => resolve(c);
  }
});

self.onmessage = async e => {
  const { id, type } = e.data;
  try {
    const cvm = await cvReady;
    let result, transfer = [];
    if (type === 'init') result = true;
    else if (type === 'features') ({ result, transfer } = await features(cvm, e.data));
    else if (type === 'pairs') result = SfM.selectPairs(e.data.thumbs, e.data.options);
    else if (type === 'match') result = match(cvm, e.data, id);
    else if (type === 'solve') ({ result, transfer } = solve(cvm, e.data, id));
    else throw new Error('Unknown task ' + type);
    postMessage({ id, ok: true, result }, transfer);
  } catch (err) {
    postMessage({ id, ok: false, error: (err && err.message) || String(err) });
  }
};

async function features(cvm, { blob, focal35, options }) {
  const bmp = await createImageBitmap(blob);
  const maxDim = options.maxDim || SfM.DEFAULTS.maxDim;
  const s = Math.min(1, maxDim / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s);
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close?.();
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const gray = new Uint8Array(w * h);
  for (let i = 0, j = 0; j < gray.length; i += 4, j++) gray[j] = (rgba[i] * 77 + rgba[i + 1] * 150 + rgba[i + 2] * 29) >> 8;
  // Focal length in working pixels: EXIF 35 mm equivalent if known, else a typical phone FOV.
  const focal = focal35 ? focal35 / 43.267 * Math.hypot(w, h) : (options.focalFactor || 0.75) * Math.max(w, h);
  const f = SfM.extractFeatures(cvm, gray, w, h, options);
  return { result: { w, h, focal, ...f }, transfer: [f.kps.buffer, f.desc.buffer, f.thumb.buffer] };
}

function match(cvm, { pairs, feats, options }, id) {
  const map = new Map(Object.entries(feats).map(([k, v]) => [+k, v]));
  const matches = SfM.matchPairs(cvm, pairs, map, options, k => {
    if (k % 20 === 0) postMessage({ id, progress: [k / pairs.length, ''] });
  });
  return { matches };
}

function solve(cvm, { images, matches, size, focal, options }, id) {
  const progress = (f, msg) => postMessage({ id, progress: [f, msg] });
  const t0 = performance.now();
  const r = new SfM.Reconstruction(cvm, images, matches, size, focal, options, progress);
  r.buildTracks();
  r.reconstruct();
  const { points, info } = r.result(options.markerSizeM || 0);
  info.solve_s = Math.round((performance.now() - t0) / 100) / 10;
  info.ba_s = r.timing ? r.timing.ba : undefined;
  return { result: { points, info }, transfer: [points.buffer] };
}
