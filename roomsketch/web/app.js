import { Capture } from './capture.js';
import { RoomViewer, fmtLen } from './viewer.js';

const $ = s => document.querySelector(s);
const screens = ['home', 'capture', 'processing', 'viewer'];
let capture = null;
let viewer = null;
let current = { id: null, scale: 1 };

function show(name) {
  screens.forEach(s => $('#' + s).classList.toggle('hidden', s !== name));
  if (name !== 'capture' && capture) { capture.stop(); }
}

// ---- Persistence (per-device conveniences only) ---------------------------------
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};
function remember(id, label) {
  const list = store.get('rs.recent', []).filter(r => r.id !== id);
  list.unshift({ id, label, t: Date.now() });
  store.set('rs.recent', list.slice(0, 8));
}
function renderRecent() {
  const list = store.get('rs.recent', []);
  $('#recent').innerHTML = list.length ? '<div class="muted small">Recent</div>' + list.map(r =>
    `<a href="#/job/${r.id}">${r.label} <span class="muted small">· ${new Date(r.t).toLocaleString()}</span></a>`).join('') : '';
}

// ---- Routing -----------------------------------------------------------------------
async function route() {
  const h = location.hash;
  const m = h.match(/^#\/job\/([a-f0-9]+)/);
  if (m) return openJob(m[1]);
  if (h === '#/capture') return startCapture();
  show('home');
  renderRecent();
}
window.addEventListener('hashchange', route);
document.addEventListener('click', e => {
  const go = e.target.closest('[data-go]');
  if (go) { e.preventDefault(); location.hash = go.dataset.go === 'home' ? '' : '#/' + go.dataset.go; }
});

// ---- Uploads -----------------------------------------------------------------------
function upload(url, form, label) {
  show('processing');
  $('#proc-title').textContent = 'Uploading…';
  $('#proc-err').classList.add('hidden');
  $('#proc-back').classList.add('hidden');
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) {
        $('#proc-bar').style.width = (e.loaded / e.total * 100).toFixed(1) + '%';
        $('#proc-msg').textContent = `${(e.loaded / 1e6).toFixed(1)} / ${(e.total / 1e6).toFixed(1)} MB`;
      }
    };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) {
        remember(body.id, label);
        location.hash = '#/job/' + body.id;
        resolve(body.id);
      } else {
        fail(body.detail || `Upload failed (${xhr.status})`);
        reject();
      }
    };
    xhr.onerror = () => { fail('Network error during upload'); reject(); };
    xhr.send(form);
  });
}

function fail(msg) {
  show('processing');
  $('#proc-title').textContent = 'Something went wrong';
  $('#proc-err').textContent = msg;
  $('#proc-err').classList.remove('hidden');
  $('#proc-back').classList.remove('hidden');
}

$('#scan-file').addEventListener('change', e => {
  const f = e.target.files[0];
  if (!f) return;
  const form = new FormData();
  form.append('file', f);
  form.append('up', $('#up-axis').value);
  upload('/api/scan', form, `LiDAR · ${f.name}`);
  e.target.value = '';
});

$('#media-file').addEventListener('change', e => {
  const files = [...e.target.files];
  if (!files.length) return;
  const form = new FormData();
  files.forEach(f => form.append('files', f, f.name));
  form.append('marker_cm', $('#marker-cm').value || '0');
  const video = files.some(f => f.type.startsWith('video'));
  upload('/api/photos', form, video ? 'Photogrammetry · video' : `Photogrammetry · ${files.length} photos`);
  e.target.value = '';
});

document.querySelector('[data-action="demo"]').addEventListener('click', async () => {
  const r = await fetch('/api/demo', { method: 'POST' });
  const { id } = await r.json();
  remember(id, 'Demo room');
  location.hash = '#/job/' + id;
});

$('#marker-cm').value = store.get('rs.marker', '');
$('#marker-cm').addEventListener('change', e => store.set('rs.marker', e.target.value));

// ---- Capture -----------------------------------------------------------------------
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
  upload('/api/photos', form, `Camera capture · ${n} photos`);
});

// ---- Job: poll, then view ----------------------------------------------------------
let pollToken = 0;
async function openJob(id) {
  const token = ++pollToken;
  show('processing');
  $('#proc-err').classList.add('hidden');
  $('#proc-back').classList.add('hidden');
  $('#proc-title').textContent = 'Processing';
  while (token === pollToken) {
    const r = await fetch(`/api/jobs/${id}`);
    if (!r.ok) return fail('Job not found');
    const st = await r.json();
    $('#proc-bar').style.width = (st.progress * 100).toFixed(0) + '%';
    $('#proc-msg').textContent = st.message || '';
    if (st.status === 'error') return fail(st.error);
    if (st.status === 'done') break;
    await new Promise(res => setTimeout(res, 700));
  }
  if (token !== pollToken) return;
  current = { id, scale: store.get('rs.scale.' + id, 1) };
  await showModel(true);
}

async function showModel(fresh) {
  const { id, scale } = current;
  const [model, pts] = await Promise.all([
    fetch(`/api/jobs/${id}/model?scale=${scale}`).then(r => r.json()),
    fresh || !current.points ? fetch(`/api/jobs/${id}/points`).then(r => r.arrayBuffer()).then(b => new Float32Array(b)) : current.points,
  ]);
  current.points = pts;
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
  viewer.load(model, pts, scale);
  renderMeta(model);
  updateExportLinks();
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

function updateExportLinks() {
  const layers = viewer.visibleLayers().join(',');
  document.querySelectorAll('[data-fmt]').forEach(a => {
    a.href = `/api/jobs/${current.id}/export/${a.dataset.fmt}?scale=${current.scale}&layers=${layers}`;
  });
}

function bindViewerUI() {
  for (const key of ['ceiling', 'furniture', 'dims', 'sketchy', 'shadows', 'points']) {
    const el = $('#t-' + key);
    el.checked = viewer.state[key];
    el.addEventListener('change', () => { viewer.set(key, el.checked); updateExportLinks(); });
  }
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

route();
