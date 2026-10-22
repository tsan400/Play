// Browser-side media helpers for photogrammetry: video -> frames, EXIF focal length.

const once = (el, ev) => new Promise((res, rej) => {
  const ok = () => { cleanup(); res(); };
  const bad = () => { cleanup(); rej(new Error('Could not read this video in the browser. Try an .mp4/.mov (H.264) or upload photos.')); };
  const cleanup = () => { el.removeEventListener(ev, ok); el.removeEventListener('error', bad); };
  el.addEventListener(ev, ok, { once: true });
  el.addEventListener('error', bad, { once: true });
});

function sharpness(ctx, w, h) {
  const d = ctx.getImageData(0, 0, w, h).data;
  let s = 0, s2 = 0, n = 0;
  const g = i => 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = (y * w + x) * 4;
    const lap = 4 * g(i) - g(i - 4) - g(i + 4) - g(i - 4 * w) - g(i + 4 * w);
    s += lap; s2 += lap * lap; n++;
  }
  return s2 / n - (s / n) ** 2;
}

/** Pick the sharpest of a few candidate frames in each 1/fps window. */
export async function framesFromVideo(file, { fps = 2.5, maxFrames = 150, maxDim = 1600 } = {}, onProgress = () => {}) {
  const url = URL.createObjectURL(file);
  const v = document.createElement('video');
  v.muted = true; v.playsInline = true; v.preload = 'auto'; v.src = url;
  try {
    await once(v, 'loadeddata');
    const dur = v.duration;
    if (!Number.isFinite(dur) || dur <= 0) throw new Error('Could not read the video length.');
    let step = 1 / fps;
    if (dur / step > maxFrames) step = dur / maxFrames;
    const s = Math.min(1, maxDim / Math.max(v.videoWidth, v.videoHeight));
    const W = Math.round(v.videoWidth * s), H = Math.round(v.videoHeight * s);
    const full = document.createElement('canvas'); full.width = W; full.height = H;
    const fctx = full.getContext('2d');
    const small = document.createElement('canvas'); small.width = 160; small.height = Math.round(160 * H / W);
    const sctx = small.getContext('2d', { willReadFrequently: true });
    const frames = [];
    const windows = Math.floor(dur / step);
    for (let k = 0; k < windows; k++) {
      let best = -1, bestT = 0;
      for (const frac of [0.2, 0.6]) {
        const t = Math.min(dur - 0.01, (k + frac) * step);
        v.currentTime = t;
        await once(v, 'seeked');
        sctx.drawImage(v, 0, 0, small.width, small.height);
        const sh = sharpness(sctx, small.width, small.height);
        if (sh > best) { best = sh; bestT = t; }
      }
      if (Math.abs(v.currentTime - bestT) > 1e-3) { v.currentTime = bestT; await once(v, 'seeked'); }
      fctx.drawImage(v, 0, 0, W, H);
      const blob = await new Promise(r => full.toBlob(r, 'image/jpeg', 0.92));
      frames.push({ blob, name: `video_${String(k).padStart(4, '0')}.jpg` });
      onProgress((k + 1) / windows, `Extracting frames ${k + 1}/${windows}`);
    }
    return frames;
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** 35 mm-equivalent focal length from JPEG EXIF (tag 0xA405), or 0 if unknown. */
export async function focal35FromExif(file) {
  try {
    const v = new DataView(await file.slice(0, 256 * 1024).arrayBuffer());
    if (v.getUint16(0) !== 0xFFD8) return 0;
    let off = 2;
    while (off + 10 < v.byteLength) {
      const marker = v.getUint16(off), len = v.getUint16(off + 2);
      if ((marker & 0xFF00) !== 0xFF00) break;
      if (marker === 0xFFE1 && v.getUint32(off + 4) === 0x45786966) { // "Exif"
        const tiff = off + 10, le = v.getUint16(tiff) === 0x4949;
        const u16 = o => v.getUint16(tiff + o, le), u32 = o => v.getUint32(tiff + o, le);
        const find = (ifd, tag) => { const n = u16(ifd); for (let i = 0; i < n; i++) { const e = ifd + 2 + i * 12; if (u16(e) === tag) return e; } return -1; };
        const ptr = find(u32(4), 0x8769);
        if (ptr < 0) return 0;
        const e = find(u32(ptr + 8), 0xA405);
        return e < 0 ? 0 : u16(e + 8);
      }
      off += 2 + len;
    }
  } catch { /* unreadable EXIF: fall back to the default field of view */ }
  return 0;
}
