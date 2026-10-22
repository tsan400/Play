import { Capture } from './capture.js';
import { RoomViewer, fmtLen } from './viewer.js';
import { loadScan, packUpload } from './scanload.js';

const $ = s => document.querySelector(s);
const screens = ['home', 'capture', 'processing', 'viewer'];
let capture = null;
let viewer = null;
let config = { photogrammetry: false };
// The solved room lives in the browser; the server is stateless.
let current = { id: null, room: null, points: null, scale: 1 };

function show(name) {
  screens.forEach(s => $('#' + s).classList.toggle('hidden', s !== name));
  if (name !== 'capture' && capture) capture.stop();
}

// ---- Persistence (per-device conveniences only) ---------------------------------
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch { return false; } },
};
const newId = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);

function saveRoom(room, label, points) {
  const id = newId();
  // Rooms are a few KB; overlay points stay in memory for this session only.
  store.set('rs.room.' + id, room);
  const list = store.get('rs.recent', []).filter(r => store.get('rs.room.' + r.id, null));
  list.unshift({ id, label, t: Date.now() });
  for (const old of list.slice(10)) { try { localStorage.removeItem('rs.room.' + old.id); } catch { /* ignore */ } }
  store.set('rs.recent', list.slice(0, 10));
  current = { id, room, points, scale: 1 };
  location.hash = '#/room/' + id;
}

function renderRecent() {
  const list = store.get('rs.recent', []);
  $('#recent').innerHTML = list.length ? '<div class="muted small">Recent (this device)</div>' + list.map(r =>
    `<a href="#/room/${r.id}">${escapeHtml(r.label)} <span class="muted small">· ${new Date(r.t).toLocaleString()}</span></a>`).join('') : '';
}
const escapeHtml = s => s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// ---- Routing -----------------------------------------------------------------------
async function route() {
  const h = location.hash;
  let m = h.match(/^#\/room\/([a-z0-9]+)/);
  if (m) return openRoom(m[1]);
  m = h.match(/^#\/job\/([a-f0-9]+)/);
  if (m) return pollJob(m[1]);
  if (h === '#/capture' && config.photogrammetry) return startCapture();
  show('home');
  renderRecent();
}
window.addEventListener('hashchange', route);
document.addEventListener('click', e => {
  const go = e.target.closest('[data-go]');
  if (go && !go.classList.contains('disabled')) { e.preventDefault(); location.hash = go.dataset.go === 'home' ? '' : '#/' + go.dataset.go; }
});

// ---- Processing screen helpers ---------------------------------------------------
function busy(title, msg = '', pct = null) {
  show('processing');
  $('#proc-title').textContent = title;
  $('#proc-msg').textContent = msg;
  $('#proc-err').classList.add('hidden');
  $('#proc-back').classList.add('hidden');
  $('#proc-bar').style.width = pct == null ? '100%' : pct + '%';
  $('#proc-bar').classList.toggle('indeterminate', pct == null);
}

function fail(msg) {
  show('processing');
  $('#proc-title').textContent = 'Something went wrong';
  $('#proc-err').textContent = msg;
  $('#proc-err').classList.remove('hidden');
  $('#proc-back').classList.remove('hidden');
  $('#proc-bar').classList.remove('indeterminate');
}

function post(url, body, { onUpload, type } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    if (type) xhr.setRequestHeader('Content-Type', type);
    if (onUpload) xhr.upload.onprogress = e => e.lengthComputable && onUpload(e.loaded / e.total, e);
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new Error(data.detail || `Request failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error('Network error'));
    xhr.send(body);
  });
}

const b64ToFloats = b64 => {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
};

// ---- LiDAR import ----------------------------------------------------------------
$('#scan-file').addEventListener('change', async e => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  try {
    busy('Reading scan', f.name);
    const scan = await loadScan(f, m => { $('#proc-msg').textContent = m; });
    const body = packUpload(scan);
    busy('Uploading', `${scan.count.toLocaleString()} points · ${(body.size / 1e6).toFixed(1)} MB`, 0);
    const up = $('#up-axis').value;
    const res = await post(`/api/room?up=${up}&name=${encodeURIComponent(f.name)}`, body, {
      type: 'application/octet-stream',
      onUpload: p => {
        $('#proc-bar').style.width = (p * 100).toFixed(0) + '%';
        if (p >= 1) busy('Solving room', 'Planes → walls → openings → furniture');
      },
    });
    // Overlay: our own thinned points, moved into the solved model frame.
    const T = res.room.input_transform;
    const P = scan.pts, out = new Float32Array(P.length);
    for (let i = 0; i < P.length; i += 3) {
      const x = P[i], y = P[i + 1], z = P[i + 2];
      out[i] = T[0][0] * x + T[0][1] * y + T[0][2] * z + T[0][3];
      out[i + 1] = T[1][0] * x + T[1][1] * y + T[1][2] * z + T[1][3];
      out[i + 2] = T[2][0] * x + T[2][1] * y + T[2][2] * z + T[2][3];
    }
    saveRoom(res.room, `LiDAR · ${f.name}`, out);
  } catch (err) {
    fail(err.message || String(err));
  }
});

// ---- Demo --------------------------------------------------------------------------
document.querySelector('[data-action="demo"]').addEventListener('click', async () => {
  try {
    busy('Solving demo room', 'Synthetic LiDAR scan of an L-shaped room');
    const res = await post('/api/demo', null);
    saveRoom(res.room, 'Demo room', b64ToFloats(res.points));
  } catch (err) {
    fail(err.message);
  }
});

// ---- Photogrammetry (only when the server has COLMAP) -----------------------------
function uploadPhotos(form, label) {
  busy('Uploading…', '', 0);
  post('/api/photos', form, {
    onUpload: (p, e) => {
      $('#proc-bar').style.width = (p * 100).toFixed(1) + '%';
      $('#proc-msg').textContent = `${(e.loaded / 1e6).toFixed(1)} / ${(e.total / 1e6).toFixed(1)} MB`;
    },
  }).then(({ id }) => {
    store.set('rs.joblabel.' + id, label);
    location.hash = '#/job/' + id;
  }).catch(err => fail(err.message));
}

$('#media-file').addEventListener('change', e => {
  const files = [...e.target.files];
  e.target.value = '';
  if (!files.length) return;
  const form = new FormData();
  files.forEach(f => form.append('files', f, f.name));
  form.append('marker_cm', $('#marker-cm').value || '0');
  const video = files.some(f => f.type.startsWith('video'));
  uploadPhotos(form, video ? 'Photogrammetry · video' : `Photogrammetry · ${files.length} photos`);
});

let pollToken = 0;
async function pollJob(id) {
  const token = ++pollToken;
  busy('Processing', '', 0);
  while (token === pollToken) {
    const r = await fetch(`/api/jobs/${id}`);
    if (!r.ok) return fail('Job not found (jobs live on the server that started them)');
    const st = await r.json();
    $('#proc-bar').style.width = (st.progress * 100).toFixed(0) + '%';
    $('#proc-msg').textContent = st.message || '';
    if (st.status === 'error') return fail(st.error);
    if (st.status === 'done') break;
    await new Promise(res => setTimeout(res, 700));
  }
  if (token !== pollToken) return;
  const res = await fetch(`/api/jobs/${id}/result`).then(r => r.json());
  saveRoom(res.room, store.get('rs.joblabel.' + id, 'Photogrammetry'), b64ToFloats(res.points));
}

$('#marker-cm').value = store.get('rs.marker', '');
$('#marker-cm').addEventListener('change', e => store.set('rs.marker', e.target.value));

async function startCapture() {
  show('capture');
  if (!capture) {
    capture = new Capture($('#cam'), {
      onFrame: (f, n) => {
        $('#cap-count').textContent = n;
        $('#cap-done').disabled = n < 8;
        const img = new Image();
        img.src = f.thumb;
        $('#cap-strip').prepend(img);
        $('#cap-flash').classList.add('on');
        setTimeout(() => $('#cap-flash').classList.remove('on'), 60);
      },
      onStatus: s => { $('#cap-sharp').textContent = s.status; },
    });
  }
  try {
    await capture.start();
  } catch (err) {
    fail((err && err.message) || 'Could not open the camera');
  }
}

$('#cap-rec').addEventListener('click', () => {
  capture.auto = !capture.auto;
  $('#cap-rec').classList.toggle('on', capture.auto);
  $('#cap-tip').textContent = capture.auto
    ? 'Walk slowly along the walls. Include floor, ceiling edges and corners. Finish where you started.'
    : 'Paused. Tap the red button to keep capturing.';
});
$('#cap-shutter').addEventListener('click', () => capture.snap());
$('#cap-done').addEventListener('click', () => {
  const form = new FormData();
  capture.frames.forEach((f, i) => form.append('files', f.blob, `frame_${String(i).padStart(4, '0')}.jpg`));
  form.append('marker_cm', $('#marker-cm').value || '0');
  const n = capture.frames.length;
  capture.stop();
  capture.reset();
  $('#cap-strip').innerHTML = '';
  $('#cap-count').textContent = '0';
  $('#cap-rec').classList.remove('on');
  uploadPhotos(form, `Camera capture · ${n} photos`);
});

// ---- Viewer ------------------------------------------------------------------------
async function openRoom(id) {
  if (current.id !== id) {
    const room = store.get('rs.room.' + id, null);
    if (!room) { location.hash = ''; return; }
    current = { id, room, points: null, scale: store.get('rs.scale.' + id, 1) };
  }
  busy('Building model');
  try {
    await showModel(true);
  } catch (err) {
    fail(err.message);
  }
}

async function showModel(fresh) {
  const { room, scale, points } = current;
  const model = await post('/api/model', JSON.stringify({ room, scale }), { type: 'application/json' });
  current.model = model;
  show('viewer');
  if (!viewer) {
    viewer = new RoomViewer($('#stage'), () => {
      const r = $('.sheet').getBoundingClientRect();
      return innerWidth >= 900 ? { right: r.width + 12, bottom: 0 } : { right: 0, bottom: r.height };
    });
    bindViewerUI();
  }
  if (fresh) viewer._fitted = false;
  viewer.resize();
  viewer.load(model, points, scale);
  $('#t-points').disabled = !points;
  $('#t-points').parentElement.title = points ? '' : 'Scan points are kept only for the session that imported the scan';
  renderMeta(model);
}

function renderMeta(model) {
  const m = model.metrics, meta = model.meta || {};
  const src = { lidar: 'LiDAR scan', photogrammetry: 'Photogrammetry', demo: 'Demo (synthetic LiDAR)' }[meta.source] || 'Room';
  $('#v-title').textContent = src;
  const doors = model.openings.filter(o => o.kind !== 'window').length;
  const wins = model.openings.filter(o => o.kind === 'window').length;
  $('#v-sub').textContent = `${model.outline.length} walls · ${doors} door${doors === 1 ? '' : 's'} · ${wins} window${wins === 1 ? '' : 's'} · ${model.furniture} objects`;
  $('#v-metrics').innerHTML = `
    <div><span class="muted">Floor</span><b>${m.floor_area.toFixed(2)} m²</b><span class="muted small">${(m.floor_area * 10.7639).toFixed(0)} ft²</span></div>
    <div><span class="muted">Ceiling</span><b>${m.height.toFixed(2)} m</b><span class="muted small">${fmtLen(m.height).split('· ')[1]}</span></div>
    <div><span class="muted">Perimeter</span><b>${m.perimeter.toFixed(2)} m</b></div>
    <div><span class="muted">Volume</span><b>${m.volume.toFixed(1)} m³</b></div>`;
  const warn = [];
  if (!meta.scaled && current.scale === 1) warn.push(`Scale is approximate (${meta.scale_source}). Tap <b>Calibrate scale</b> and enter one real dimension.`);
  if (current.scale !== 1) warn.push(`Calibrated: ×${current.scale.toFixed(4)}.`);
  if (model.ceiling_estimated) warn.push('No ceiling surface found; height estimated from the top of the scan.');
  $('#v-warn').innerHTML = warn.join('<br>');
  $('#v-warn').classList.toggle('hidden', !warn.length);
}

async function download(fmt, btn) {
  const label = btn.textContent;
  btn.textContent = '…';
  try {
    const r = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ room: current.room, scale: current.scale, layers: viewer.visibleLayers(), fmt }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `Export failed (${r.status})`);
    const url = URL.createObjectURL(await r.blob());
    const a = document.createElement('a');
    a.href = url;
    a.download = `room-${current.id}.${fmt}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30_000);
  } catch (err) {
    alert(err.message);
  } finally {
    btn.textContent = label;
  }
}

function bindViewerUI() {
  for (const key of ['ceiling', 'furniture', 'dims', 'sketchy', 'shadows', 'points']) {
    const el = $('#t-' + key);
    el.checked = viewer.state[key];
    el.addEventListener('change', () => viewer.set(key, el.checked));
  }
  document.querySelectorAll('[data-fmt]').forEach(btn => {
    btn.addEventListener('click', e => { e.preventDefault(); download(btn.dataset.fmt, btn); });
  });
  $('#v-reset').addEventListener('click', () => viewer.resetView());
  $('#v-opts').open = innerWidth >= 900;
  $('#v-opts').addEventListener('toggle', () => viewer.resize());
  let picked = null;
  $('#v-calibrate').addEventListener('click', () => {
    $('#v-calib').classList.remove('hidden');
    $('#v-calib-row').classList.add('hidden');
    $('#v-calib-msg').textContent = 'Tap an edge whose real length you know (e.g. a wall you measured).';
    viewer.setPicking(true);
  });
  viewer.onPick = hit => {
    picked = hit;
    $('#v-calib-msg').textContent = 'Selected edge:';
    $('#v-calib-cur').textContent = fmtLen(hit.length) + ' →';
    $('#v-calib-row').classList.remove('hidden');
    $('#v-calib-val').focus();
  };
  const close = () => { $('#v-calib').classList.add('hidden'); viewer.setPicking(false); picked = null; };
  $('#v-calib-cancel').addEventListener('click', close);
  $('#v-calib-apply').addEventListener('click', async () => {
    const v = parseFloat($('#v-calib-val').value);
    if (!picked || !(v > 0)) return;
    const unit = { m: 1, cm: 0.01, in: 0.0254, ft: 0.3048 }[$('#v-calib-unit').value];
    current.scale = current.scale * (v * unit) / picked.length;
    store.set('rs.scale.' + current.id, current.scale);
    close();
    await showModel(false);
  });
}

// ---- Boot ----------------------------------------------------------------------------
(async () => {
  try { config = await fetch('/api/config').then(r => r.json()); } catch { /* offline: keep defaults */ }
  if (!config.photogrammetry) {
    for (const el of document.querySelectorAll('.needs-photogrammetry')) {
      el.classList.add('disabled');
      el.setAttribute('aria-disabled', 'true');
      el.querySelector('.card-k').textContent = 'Photogrammetry · local server only';
      el.querySelector('input')?.setAttribute('disabled', '');
    }
    $('#pg-note').classList.remove('hidden');
    $('.cards').prepend($('label[for="scan-file"]')); // LiDAR is the primary path here
  }
  route();
})();
