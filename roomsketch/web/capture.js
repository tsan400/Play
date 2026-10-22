// In-browser photogrammetry capture: auto-snaps sharp frames once the view has moved enough.
const MAX_FRAMES = 160;
const MIN_INTERVAL = 450; // ms between auto shots
const MOTION_MIN = 7;     // mean abs grey difference (0-255) vs last shot
const SMALL_W = 160;

export class Capture {
  constructor(video, { onFrame, onStatus }) {
    this.video = video;
    this.onFrame = onFrame;
    this.onStatus = onStatus;
    this.frames = [];
    this.auto = false;
    this.small = document.createElement('canvas');
    this.sctx = this.small.getContext('2d', { willReadFrequently: true });
    this.full = document.createElement('canvas');
    this.lastGray = null;
    this.lastShot = 0;
    this.sharpHist = [];
  }

  async start() {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Camera needs HTTPS (or localhost) and a browser with camera access.');
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1440 } },
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    this.timer = setInterval(() => this.tick(), 150);
  }

  stop() {
    clearInterval(this.timer);
    this.auto = false;
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = null;
  }

  reset() {
    this.frames.forEach(f => URL.revokeObjectURL(f.thumb));
    this.frames = [];
    this.lastGray = null;
  }

  measure() {
    const v = this.video;
    if (!v.videoWidth) return null;
    const h = Math.round(SMALL_W * v.videoHeight / v.videoWidth);
    this.small.width = SMALL_W; this.small.height = h;
    this.sctx.drawImage(v, 0, 0, SMALL_W, h);
    const d = this.sctx.getImageData(0, 0, SMALL_W, h).data;
    const g = new Float32Array(SMALL_W * h);
    for (let i = 0, j = 0; j < g.length; i += 4, j++) g[j] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    // Variance of the Laplacian = focus / motion-blur measure.
    let sum = 0, sum2 = 0, n = 0;
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < SMALL_W - 1; x++) {
      const k = y * SMALL_W + x;
      const lap = 4 * g[k] - g[k - 1] - g[k + 1] - g[k - SMALL_W] - g[k + SMALL_W];
      sum += lap; sum2 += lap * lap; n++;
    }
    const sharp = sum2 / n - (sum / n) ** 2;
    let motion = 255;
    if (this.lastGray && this.lastGray.length === g.length) {
      let m = 0;
      for (let k = 0; k < g.length; k++) m += Math.abs(g[k] - this.lastGray[k]);
      motion = m / g.length;
    }
    return { gray: g, sharp, motion };
  }

  tick() {
    const m = this.measure();
    if (!m) return;
    this.sharpHist.push(m.sharp);
    if (this.sharpHist.length > 40) this.sharpHist.shift();
    const sorted = [...this.sharpHist].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)] || 1;
    const sharpOk = m.sharp >= 0.6 * median;
    let status = sharpOk ? 'Sharp' : 'Hold steady';
    if (this.auto) {
      const now = performance.now();
      if (m.motion < MOTION_MIN) status = 'Keep moving';
      if (sharpOk && m.motion >= MOTION_MIN && now - this.lastShot > MIN_INTERVAL) this.snap(m.gray);
    }
    this.onStatus?.({ status, sharpOk, count: this.frames.length });
  }

  snap(gray) {
    if (this.frames.length >= MAX_FRAMES) { this.auto = false; return; }
    const v = this.video;
    if (!v.videoWidth) return;
    this.full.width = v.videoWidth; this.full.height = v.videoHeight;
    this.full.getContext('2d').drawImage(v, 0, 0);
    this.lastShot = performance.now();
    this.lastGray = gray || this.measure()?.gray || null;
    this.full.toBlob(blob => {
      if (!blob) return;
      const f = { blob, thumb: URL.createObjectURL(blob) };
      this.frames.push(f);
      this.onFrame?.(f, this.frames.length);
    }, 'image/jpeg', 0.92);
  }
}
