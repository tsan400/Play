// Parse a LiDAR scan export in the browser and thin it to a compact point upload.
// Meshes are sampled evenly by surface area (with face normals); point clouds are
// used as-is. Everything is voxel-averaged at 2.5 cm, so a 200 MB scan uploads as
// a few MB, which also keeps requests inside serverless body limits.
import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { MeshSurfaceSampler } from 'three/addons/math/MeshSurfaceSampler.js';

const SAMPLE_SPACING = 0.015; // metres between mesh samples
const VOXEL = 0.025;          // matches the server's working resolution
const MAX_POINTS = 240_000;   // ~3.6 MB upload with normals

export const SCAN_EXTENSIONS = ['ply', 'obj', 'glb', 'gltf', 'stl', 'xyz', 'pts', 'txt', 'csv'];

export async function loadScan(file, status = () => {}) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!SCAN_EXTENSIONS.includes(ext)) throw new Error(`Unsupported file type .${ext}`);
  status(`Reading ${(file.size / 1e6).toFixed(1)} MB…`);
  const buf = await file.arrayBuffer();
  status('Parsing scan…');
  const { meshes, clouds } = await parse(ext, buf);

  let pts, nrm = null;
  if (meshes.length) {
    status('Sampling mesh surfaces…');
    ({ pts, nrm } = sampleMeshes(meshes));
  } else if (clouds.length) {
    pts = concat(clouds);
  } else {
    throw new Error('No geometry found in this file.');
  }
  let n = pts.length / 3;
  if (n < 50) throw new Error('Too few points in this file.');

  // Millimetre exports: a room is never 300+ units across in metres.
  let unitNote = '';
  const ext3 = extent(pts);
  if (Math.max(...ext3) > 300) {
    for (let i = 0; i < pts.length; i++) pts[i] *= 0.001;
    unitNote = ' (converted from mm)';
  }

  status('Thinning points…');
  let voxel = VOXEL, out;
  do {
    out = voxelDown(pts, nrm, voxel);
    voxel *= 1.25;
  } while (out.count > MAX_POINTS);
  status(`${out.count.toLocaleString()} points${unitNote}`);
  return out; // { count, pts: Float32Array, nrm: Int8Array|null }
}

export function packUpload({ count, pts, nrm }) {
  const head = new Uint32Array([count]);
  const parts = [head, pts];
  if (nrm) parts.push(nrm);
  return new Blob(parts, { type: 'application/octet-stream' });
}

async function parse(ext, buf) {
  const meshes = [], clouds = [];
  const take = root => {
    root.updateMatrixWorld(true);
    root.traverse(o => {
      if (!o.geometry || !o.geometry.attributes.position) return;
      const g = o.geometry.clone().applyMatrix4(o.matrixWorld);
      if (o.isMesh) meshes.push(g);
      else if (o.isPoints) clouds.push(g.attributes.position.array);
    });
  };
  if (ext === 'ply') {
    const g = new PLYLoader().parse(buf);
    if (g.index && g.index.count >= 3) meshes.push(g);
    else clouds.push(g.attributes.position.array);
  } else if (ext === 'stl') {
    meshes.push(new STLLoader().parse(buf));
  } else if (ext === 'obj') {
    const text = new TextDecoder().decode(buf);
    take(new OBJLoader().parse(text));
    if (!meshes.length && !clouds.length) clouds.push(objVertices(text)); // vertex-only OBJ
  } else if (ext === 'glb' || ext === 'gltf') {
    let gltf;
    try {
      gltf = await new GLTFLoader().parseAsync(buf, '');
    } catch (e) {
      throw new Error('Could not read this GLB (Draco-compressed or external files?). Export PLY or OBJ instead.');
    }
    take(gltf.scene);
  } else {
    clouds.push(textPoints(new TextDecoder().decode(buf)));
  }
  return { meshes, clouds };
}

function triArea(g) {
  const p = g.attributes.position, idx = g.index;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const n = idx ? idx.count : p.count;
  let area = 0;
  for (let i = 0; i + 2 < n; i += 3) {
    const i0 = idx ? idx.getX(i) : i, i1 = idx ? idx.getX(i + 1) : i + 1, i2 = idx ? idx.getX(i + 2) : i + 2;
    a.fromBufferAttribute(p, i0); b.fromBufferAttribute(p, i1); c.fromBufferAttribute(p, i2);
    area += b.sub(a).cross(c.sub(a)).length() / 2;
  }
  return area;
}

function sampleMeshes(geoms) {
  const areas = geoms.map(triArea);
  let total = areas.reduce((s, a) => s + a, 0);
  // Guess units before sampling: millimetre meshes would explode the sample count.
  const unit = total > 1e6 ? 1e-3 : 1;
  const want = Math.min(1_500_000, Math.max(50_000, Math.round(total * unit * unit / SAMPLE_SPACING ** 2)));
  const pts = new Float32Array(want * 3), nrm = new Float32Array(want * 3);
  const p = new THREE.Vector3(), q = new THREE.Vector3();
  let k = 0;
  geoms.forEach((g, gi) => {
    const count = gi === geoms.length - 1 ? want - k : Math.round(want * areas[gi] / (total || 1));
    if (count <= 0 || areas[gi] === 0) return;
    const sampler = new MeshSurfaceSampler(new THREE.Mesh(g)).build();
    for (let i = 0; i < count && k < want; i++, k++) {
      sampler.sample(p, q);
      pts.set([p.x, p.y, p.z], k * 3);
      nrm.set([q.x, q.y, q.z], k * 3);
    }
  });
  return { pts: pts.subarray(0, k * 3), nrm: nrm.subarray(0, k * 3) };
}

function voxelDown(pts, nrm, voxel) {
  const n = pts.length / 3;
  let mn = [Infinity, Infinity, Infinity];
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) mn[a] = Math.min(mn[a], pts[i * 3 + a]);
  const map = new Map();
  const sum = [], nsum = [], cnt = [];
  for (let i = 0; i < n; i++) {
    const x = pts[i * 3], y = pts[i * 3 + 1], z = pts[i * 3 + 2];
    if (!Number.isFinite(x + y + z)) continue;
    const kx = Math.floor((x - mn[0]) / voxel), ky = Math.floor((y - mn[1]) / voxel), kz = Math.floor((z - mn[2]) / voxel);
    const key = (kx * 131072 + ky) * 131072 + kz; // exact while each index < 2^17
    let v = map.get(key);
    if (v === undefined) {
      v = cnt.length; map.set(key, v); cnt.push(0); sum.push(0, 0, 0);
      if (nrm) nsum.push(0, 0, 0);
    }
    cnt[v]++;
    sum[v * 3] += x; sum[v * 3 + 1] += y; sum[v * 3 + 2] += z;
    if (nrm) {
      // Align to the voxel's running normal before summing (mesh normals may flip).
      const nx = nrm[i * 3], ny = nrm[i * 3 + 1], nz = nrm[i * 3 + 2];
      const s = nsum[v * 3] * nx + nsum[v * 3 + 1] * ny + nsum[v * 3 + 2] * nz < 0 ? -1 : 1;
      nsum[v * 3] += s * nx; nsum[v * 3 + 1] += s * ny; nsum[v * 3 + 2] += s * nz;
    }
  }
  const m = cnt.length;
  const out = new Float32Array(m * 3);
  const outN = nrm ? new Int8Array(m * 3) : null;
  for (let v = 0; v < m; v++) {
    for (let a = 0; a < 3; a++) out[v * 3 + a] = sum[v * 3 + a] / cnt[v];
    if (outN) {
      const L = Math.hypot(nsum[v * 3], nsum[v * 3 + 1], nsum[v * 3 + 2]) || 1;
      for (let a = 0; a < 3; a++) outN[v * 3 + a] = Math.round(nsum[v * 3 + a] / L * 127);
    }
  }
  return { count: m, pts: out, nrm: outN };
}

function extent(pts) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pts.length; i += 3) for (let a = 0; a < 3; a++) {
    lo[a] = Math.min(lo[a], pts[i + a]); hi[a] = Math.max(hi[a], pts[i + a]);
  }
  return hi.map((h, a) => h - lo[a]);
}

function concat(arrays) {
  const out = new Float32Array(arrays.reduce((s, a) => s + a.length, 0));
  let o = 0;
  for (const a of arrays) { out.set(a, o); o += a.length; }
  return out;
}

function objVertices(text) {
  const v = [];
  for (const line of text.split('\n')) {
    if (line.startsWith('v ')) {
      const p = line.trim().split(/\s+/);
      v.push(+p[1], +p[2], +p[3]);
    }
  }
  return new Float32Array(v);
}

function textPoints(text) {
  const v = [];
  for (const line of text.split('\n')) {
    const p = line.replace(/,/g, ' ').trim().split(/\s+/);
    if (p.length < 3) continue;
    const x = +p[0], y = +p[1], z = +p[2];
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) v.push(x, y, z);
  }
  return new Float32Array(v);
}
