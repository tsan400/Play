// Main-thread coordinator for in-browser photogrammetry. Spreads feature extraction
// and matching over a pool of workers (each with OpenCV.js), then solves on one.
const WORKER_URL = new URL('./worker.js', import.meta.url);

export function poolSize() {
  const cores = navigator.hardwareConcurrency || 4;
  const mem = navigator.deviceMemory || 4; // GB, Chrome only; Safari leaves it undefined
  // Safari may under-report cores for privacy; every current iPhone has at least 6.
  const usable = cores <= 2 ? 3 : cores - 1;
  return Math.max(1, Math.min(usable, mem < 4 ? 2 : 4));
}

class Pool {
  constructor(n) {
    this.workers = Array.from({ length: n }, () => new Worker(WORKER_URL));
    this.pending = new Map();
    this.nextId = 1;
    for (const w of this.workers) {
      w.onmessage = e => {
        const { id, ok, result, error, progress } = e.data;
        const p = this.pending.get(id);
        if (!p) return;
        if (progress) return p.onProgress && p.onProgress(...progress);
        this.pending.delete(id);
        ok ? p.resolve(result) : p.reject(new Error(error));
      };
      w.onerror = e => {
        for (const p of this.pending.values()) p.reject(new Error(e.message || 'Worker failed to start'));
        this.pending.clear();
      };
    }
  }

  call(worker, msg, { transfer = [], onProgress } = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.workers[worker].postMessage({ ...msg, id }, transfer);
    });
  }

  // Run tasks with each worker pulling the next one when free.
  async map(tasks, run) {
    const out = new Array(tasks.length);
    let next = 0;
    await Promise.all(this.workers.map(async (_, w) => {
      while (next < tasks.length) {
        const k = next++;
        out[k] = await run(w, tasks[k], k);
      }
    }));
    return out;
  }

  close() { this.workers.forEach(w => w.terminate()); }
}

/**
 * sources: [{ blob, name, focal35? }] in capture order.
 * options: { markerSizeM, focalFactor, maxDim }
 * onProgress(fraction, message)
 */
export async function reconstructPhotos(sources, options = {}, onProgress = () => {}) {
  const pool = new Pool(Math.min(poolSize(), Math.max(1, sources.length)));
  const t0 = performance.now();
  const timing = {};
  try {
    onProgress(0.01, `Loading vision engine on ${pool.workers.length} thread${pool.workers.length > 1 ? 's' : ''}…`);
    await Promise.all(pool.workers.map((_, w) => pool.call(w, { type: 'init' })));

    // 1. Features (decode + ORB + ArUco), in parallel.
    let done = 0;
    let t = performance.now();
    const feats = await pool.map(sources, async (w, s) => {
      const r = await pool.call(w, { type: 'features', blob: s.blob, focal35: s.focal35 || 0, options });
      done++;
      onProgress(0.02 + 0.28 * done / sources.length, `Analysed ${done}/${sources.length} photos`);
      return r;
    });
    timing.features_s = secs(t);

    // Single-camera model: keep the most common image size.
    const counts = new Map();
    for (const f of feats) { const k = `${f.w}x${f.h}`; counts.set(k, (counts.get(k) || 0) + 1); }
    const [bestSize] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
    const keep = feats.filter(f => `${f.w}x${f.h}` === bestSize);
    if (keep.length < 8) throw new Error(`Need at least 8 photos of the same size (got ${keep.length}).`);
    const size = [keep[0].w, keep[0].h];
    const focal = median(keep.map(f => f.focal));

    // 2. Pair selection (cheap) then matching, in parallel chunks.
    const pairs = await pool.call(0, { type: 'pairs', thumbs: keep.map(f => f.thumb), options });
    const chunkSize = Math.max(4, Math.ceil(pairs.length / (pool.workers.length * 4)));
    const chunks = [];
    for (let i = 0; i < pairs.length; i += chunkSize) chunks.push(pairs.slice(i, i + chunkSize));
    let matched = 0;
    t = performance.now();
    const results = await pool.map(chunks, async (w, chunk) => {
      const ids = new Set(chunk.flat());
      const sub = {};
      for (const id of ids) sub[id] = { kps: keep[id].kps, desc: keep[id].desc };
      const r = await pool.call(w, { type: 'match', pairs: chunk, feats: sub, options });
      matched += chunk.length;
      onProgress(0.3 + 0.3 * matched / pairs.length, `Matched ${matched}/${pairs.length} photo pairs`);
      return r.matches;
    });
    timing.matching_s = secs(t);
    const matches = results.flat();
    if (matches.length < keep.length / 2) throw new Error('Too few overlapping photos. Move slower and keep about 2/3 overlap.');

    // 3. Solve on one worker.
    t = performance.now();
    const images = keep.map(f => ({ kps: f.kps, markers: f.markers }));
    const res = await pool.call(0, { type: 'solve', images, matches, size, focal, options }, {
      onProgress: (f, msg) => onProgress(0.6 + 0.38 * Math.min(1, Math.max(0, (f - 0.55) / 0.45)), msg),
    });
    timing.solve_s = secs(t);
    res.info.timing = { ...timing, total_s: secs(t0), threads: pool.workers.length };
    res.info.pairs = pairs.length;
    res.info.matched_pairs = matches.length;
    return res;
  } finally {
    pool.close();
  }
}

const secs = t => Math.round((performance.now() - t) / 100) / 10;
const median = a => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
