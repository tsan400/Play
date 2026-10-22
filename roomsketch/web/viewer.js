// Toon / "SketchUp style" viewer: white cel-shaded faces, black line work, soft grey shadows.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

export function fmtLen(m) {
  const inches = m / 0.0254;
  let ft = Math.floor(inches / 12);
  let rem = Math.round(inches - ft * 12);
  if (rem === 12) { ft += 1; rem = 0; }
  return `${m.toFixed(2)} m · ${ft}'-${rem}"`;
}

function toonGradient() {
  // Three flat tones: shade, mid, lit. Nearest filtering gives hard cel bands.
  // Ambient sets the shade tone; the sun adds 0 / ~half / full on top (see lights).
  const data = new Uint8Array([0, 120, 255]);
  const tex = new THREE.DataTexture(data, data.length, 1, THREE.RedFormat);
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.needsUpdate = true;
  return tex;
}

export class RoomViewer {
  constructor(container, getInsets = () => ({ right: 0, bottom: 0 })) {
    this.container = container;
    this.getInsets = getInsets;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.labels = new CSS2DRenderer();
    Object.assign(this.labels.domElement.style, { position: 'absolute', inset: '0', pointerEvents: 'none' });
    container.appendChild(this.labels.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xffffff);
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.05, 500);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.screenSpacePanning = true;

    // White albedo: shade = ambient/pi, lit = (ambient + sun)/pi in linear space.
    // ~0.55 linear shade reads as a light SketchUp grey after sRGB encoding.
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.75));
    this.sun = new THREE.DirectionalLight(0xffffff, 1.45);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.02;
    this.scene.add(this.sun, this.sun.target);

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.gradient = toonGradient();
    this.lineMat = new LineMaterial({ color: 0x15171a, linewidth: 1.4 });
    this.pickMat = new LineMaterial({ color: 0xe02424, linewidth: 4 });
    this.state = { ceiling: false, furniture: true, dims: true, sketchy: true, shadows: true, points: false };
    this.onPick = null;
    this.picking = false;

    this._resize = () => this.resize();
    window.addEventListener('resize', this._resize);
    this._bindPick();
    this.resize();
    const loop = () => {
      this._raf = requestAnimationFrame(loop);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this.labels.render(this.scene, this.camera);
    };
    loop();
  }

  resize() {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h);
    this.labels.setSize(w, h);
    this.camera.aspect = w / h;
    // Centre the model in the area not covered by the control panel.
    const { right, bottom } = this.getInsets();
    if (right || bottom) this.camera.setViewOffset(w + right, h + bottom, right, bottom, w, h);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
    this.lineMat.resolution.set(w, h);
    this.pickMat.resolution.set(w, h);
  }

  load(model, points, scale = 1) {
    this.model = model;
    this.scale = scale;
    this.root.clear();
    this.layerMeshes = {};
    for (const [layer, m] of Object.entries(model.meshes)) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(m.positions, 3));
      g.setIndex(m.indices);
      g.computeVertexNormals();
      const glass = layer === 'openings';
      const mat = new THREE.MeshToonMaterial({
        color: glass ? 0xbfd4e2 : 0xffffff,
        gradientMap: this.gradient,
        transparent: glass, opacity: glass ? 0.55 : 1, depthWrite: !glass,
        side: THREE.DoubleSide,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      });
      const mesh = new THREE.Mesh(g, mat);
      mesh.castShadow = !glass;
      mesh.receiveShadow = true;
      mesh.userData.layer = layer;
      this.root.add(mesh);
      this.layerMeshes[layer] = mesh;
    }
    // Ground catches the shadow outside the building.
    const b = model.bounds;
    const size = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
    this.size = size;
    this.center = new THREE.Vector3((b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(size * 8, size * 8),
      new THREE.ShadowMaterial({ opacity: 0.13 }));
    ground.position.set(this.center.x, this.center.y, b.min[2] - 0.001);
    ground.receiveShadow = true;
    this.root.add(ground);

    this.edgeObj = new LineSegments2(new LineSegmentsGeometry(), this.lineMat);
    this.root.add(this.edgeObj);
    this.pickObj = new LineSegments2(new LineSegmentsGeometry(), this.pickMat);
    this.pickObj.visible = false;
    this.root.add(this.pickObj);

    this.dimGroup = new THREE.Group();
    this.root.add(this.dimGroup);
    this._buildDims();

    if (points && points.length) {
      const pg = new THREE.BufferGeometry();
      const arr = new Float32Array(points.length);
      for (let i = 0; i < points.length; i++) arr[i] = points[i] * scale;
      pg.setAttribute('position', new THREE.BufferAttribute(arr, 3));
      this.pointsObj = new THREE.Points(pg, new THREE.PointsMaterial({
        color: 0x2a62d8, size: 1.6, sizeAttenuation: false, transparent: true, opacity: 0.45, depthWrite: false }));
      this.root.add(this.pointsObj);
    } else {
      this.pointsObj = null;
    }

    const s = this.sun;
    s.position.set(this.center.x - size * 0.9, this.center.y - size * 1.4, this.center.z + size * 2.2);
    s.target.position.copy(this.center);
    const cam = s.shadow.camera;
    cam.left = cam.bottom = -size * 1.2;
    cam.right = cam.top = size * 1.2;
    cam.near = 0.1;
    cam.far = size * 6;
    cam.updateProjectionMatrix();
    this.apply();
    if (!this._fitted) { this.resetView(); this._fitted = true; }
  }

  resetView() {
    // Fit the bounding sphere inside the narrower of the two fields of view.
    const b = this.model.bounds;
    const r = 0.5 * Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
    const vfov = THREE.MathUtils.degToRad(this.camera.fov);
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * this.camera.aspect);
    const dist = r / Math.sin(Math.min(vfov, hfov) / 2);
    const dir = new THREE.Vector3(-0.85, -1.45, 1.2).normalize();
    this.camera.position.copy(this.center).addScaledVector(dir, dist);
    this.controls.target.copy(this.center);
    this.controls.update();
  }

  set(key, value) { this.state[key] = value; this.apply(); }

  visibleLayers() {
    const v = ['walls', 'floor', 'openings'];
    if (this.state.ceiling) v.push('ceiling');
    if (this.state.furniture) v.push('furniture');
    return v;
  }

  apply() {
    if (!this.model) return;
    const vis = new Set(this.visibleLayers());
    for (const [layer, mesh] of Object.entries(this.layerMeshes)) mesh.visible = vis.has(layer);
    this.visibleEdges = this.model.edges.filter(e => e.layers.some(l => vis.has(l)));
    const ext = this.state.sketchy ? 0.06 : 0;
    const pos = new Float32Array(this.visibleEdges.length * 6);
    this.visibleEdges.forEach((e, i) => {
      const a = new THREE.Vector3(...e.a), b = new THREE.Vector3(...e.b);
      if (ext) {
        const d = b.clone().sub(a).normalize().multiplyScalar(ext);
        a.sub(d); b.add(d);
      }
      pos.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6);
    });
    this.edgeObj.geometry.dispose();
    this.edgeObj.geometry = new LineSegmentsGeometry().setPositions(pos);
    this.dimGroup.visible = this.state.dims;
    this.renderer.shadowMap.enabled = this.state.shadows;
    this.sun.castShadow = this.state.shadows;
    this.scene.traverse(o => { if (o.material) o.material.needsUpdate = true; });
    if (this.pointsObj) this.pointsObj.visible = this.state.points;
  }

  _buildDims() {
    this.dimGroup.clear();
    const P = this.model.outline;
    const n = P.length;
    // Signed area tells us the winding; labels sit just inside the room.
    for (let i = 0; i < n; i++) {
      const a = P[i], b = P[(i + 1) % n];
      const dx = b[0] - a[0], dy = b[1] - a[1];
      const L = Math.hypot(dx, dy);
      const inX = -dy / L, inY = dx / L;
      const el = document.createElement('div');
      el.className = 'dim';
      el.textContent = fmtLen(L);
      const obj = new CSS2DObject(el);
      obj.position.set((a[0] + b[0]) / 2 + inX * 0.3, (a[1] + b[1]) / 2 + inY * 0.3, 0.02);
      this.dimGroup.add(obj);
    }
    const hEl = document.createElement('div');
    hEl.className = 'dim';
    hEl.textContent = 'H ' + fmtLen(this.model.height);
    const h = new CSS2DObject(hEl);
    h.position.set(P[0][0], P[0][1], this.model.height / 2);
    this.dimGroup.add(h);
  }

  // ---- Calibration picking: nearest visible edge in screen space -------------------
  _bindPick() {
    const el = this.renderer.domElement;
    let down = null;
    el.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
    el.addEventListener('pointerup', e => {
      if (!this.picking || !down) return;
      if (Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 8) return;
      const r = el.getBoundingClientRect();
      const hit = this.pickEdge(e.clientX - r.left, e.clientY - r.top);
      if (hit) {
        this.pickObj.geometry.dispose();
        this.pickObj.geometry = new LineSegmentsGeometry().setPositions([...hit.a, ...hit.b]);
        this.pickObj.visible = true;
        this.onPick && this.onPick(hit);
      }
    });
  }

  pickEdge(x, y) {
    const w = this.renderer.domElement.clientWidth, h = this.renderer.domElement.clientHeight;
    const proj = v => { const p = new THREE.Vector3(...v).project(this.camera); return [(p.x + 1) / 2 * w, (1 - p.y) / 2 * h, p.z]; };
    let best = null, bestD = 28;
    // Only measured geometry is a valid reference: wall thickness and slabs are assumptions.
    for (const e of (this.visibleEdges || []).filter(e => this._measured(e.a) && this._measured(e.b))) {
      const a = proj(e.a), b = proj(e.b);
      if (a[2] > 1 || b[2] > 1) continue;
      const vx = b[0] - a[0], vy = b[1] - a[1];
      const t = Math.max(0, Math.min(1, ((x - a[0]) * vx + (y - a[1]) * vy) / (vx * vx + vy * vy || 1)));
      const d = Math.hypot(a[0] + vx * t - x, a[1] + vy * t - y);
      if (d < bestD) { bestD = d; best = e; }
    }
    if (!best) return null;
    const L = Math.hypot(best.b[0] - best.a[0], best.b[1] - best.a[1], best.b[2] - best.a[2]);
    return { a: best.a, b: best.b, length: L };
  }

  _measured(p) {
    const H = this.model.height, eps = 1e-3;
    if (p[2] < -eps || p[2] > H + eps) return false;
    const P = this.model.outline;
    let inside = false;
    for (let i = 0, j = P.length - 1; i < P.length; j = i++) {
      const [xi, yi] = P[i], [xj, yj] = P[j];
      // On the boundary counts as inside.
      const dx = xj - xi, dy = yj - yi, L2 = dx * dx + dy * dy || 1;
      const t = Math.max(0, Math.min(1, ((p[0] - xi) * dx + (p[1] - yi) * dy) / L2));
      if (Math.hypot(xi + dx * t - p[0], yi + dy * t - p[1]) < eps) return true;
      if ((yi > p[1]) !== (yj > p[1]) && p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  setPicking(on) {
    this.picking = on;
    if (!on && this.pickObj) this.pickObj.visible = false;
  }

  snapshot() { return this.renderer.domElement.toDataURL('image/png'); }
}
