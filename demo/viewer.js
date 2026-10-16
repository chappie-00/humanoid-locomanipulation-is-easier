// three.js rendering of the MuJoCo scene. MuJoCo is Z-up; everything simulated lives
// under ``this.world``, rotated so that its Z axis points up in three's Y-up frame.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';

export const GOAL_TOLERANCE = 0.35;

// MuJoCo world (z-up) <-> three world (y-up)
export const toThree = (p) => new THREE.Vector3(p[0], p[2], -p[1]);
export const fromThree = (v) => [v.x, -v.z, v.y];

function canvasTexture(w, h, draw) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

function noise(ctx, w, h, alpha, n = 6000) {
  for (let i = 0; i < n; i++) {
    const v = Math.random() < 0.5 ? 0 : 255;
    ctx.fillStyle = `rgba(${v},${v},${v},${alpha * Math.random()})`;
    ctx.fillRect(Math.random() * w, Math.random() * h, 2, 2);
  }
}

const TEXTURES = {};
function cardboard(top) {
  const key = top ? 'cardboard_top' : 'cardboard';
  if (!TEXTURES[key]) {
    TEXTURES[key] = canvasTexture(256, 256, (ctx, w, h) => {
      ctx.fillStyle = '#c19a6b';
      ctx.fillRect(0, 0, w, h);
      noise(ctx, w, h, 0.08);
      ctx.strokeStyle = 'rgba(90,60,30,0.35)';
      ctx.lineWidth = 6;
      ctx.strokeRect(0, 0, w, h);
      if (top) {
        ctx.fillStyle = 'rgba(214,190,150,0.95)';
        ctx.fillRect(w * 0.42, 0, w * 0.16, h);
        ctx.fillStyle = 'rgba(80,50,20,0.35)';
        ctx.fillRect(0, h * 0.495, w, h * 0.01);
      }
    });
  }
  return TEXTURES[key];
}

function ballTexture(kind) {
  if (TEXTURES[kind]) return TEXTURES[kind];
  TEXTURES[kind] = canvasTexture(512, 256, (ctx, w, h) => {
    if (kind === 'basketball') {
      ctx.fillStyle = '#d4682a'; ctx.fillRect(0, 0, w, h); noise(ctx, w, h, 0.15, 12000);
      ctx.strokeStyle = '#2a1608'; ctx.lineWidth = 5;
      ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
      for (const x of [w / 4, 3 * w / 4]) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
      for (const x of [0, w / 2, w]) {
        ctx.beginPath(); ctx.ellipse(x, h / 2, w * 0.11, h * 0.5, 0, 0, 2 * Math.PI); ctx.stroke();
      }
    } else if (kind === 'football') {
      ctx.fillStyle = '#f4f4f0'; ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#1d1d1d';
      for (let r = 0; r < 4; r++) for (let k = 0; k < 6; k++) {
        const x = (k + (r % 2) * 0.5) * w / 6, y = (r + 0.5) * h / 4;
        ctx.beginPath();
        for (let i = 0; i < 5; i++) {
          const a = i * 2 * Math.PI / 5 - Math.PI / 2;
          ctx.lineTo(x + 18 * Math.cos(a), y + 15 * Math.sin(a));
        }
        ctx.fill();
      }
    } else if (kind === 'volleyball') {
      const cols = ['#f2f2ee', '#f0c419', '#2457a6'];
      for (let i = 0; i < 9; i++) { ctx.fillStyle = cols[i % 3]; ctx.fillRect(0, i * h / 9, w, h / 9 + 1); }
      ctx.strokeStyle = 'rgba(0,0,0,0.25)'; ctx.lineWidth = 2;
      for (let i = 1; i < 9; i++) { ctx.beginPath(); ctx.moveTo(0, i * h / 9); ctx.lineTo(w, i * h / 9); ctx.stroke(); }
    } else if (kind === 'medicine_ball') {
      ctx.fillStyle = '#2b2b2e'; ctx.fillRect(0, 0, w, h); noise(ctx, w, h, 0.2, 15000);
      ctx.fillStyle = '#b3282d'; ctx.fillRect(0, h * 0.44, w, h * 0.12);
    } else if (kind === 'beach_ball') {
      const cols = ['#e63946', '#f8f8f2', '#1d70b8', '#f4c430', '#2a9d58', '#f8f8f2'];
      for (let i = 0; i < 6; i++) { ctx.fillStyle = cols[i]; ctx.fillRect(i * w / 6, 0, w / 6 + 1, h); }
      ctx.fillStyle = '#f8f8f2';
      ctx.fillRect(0, 0, w, h * 0.06); ctx.fillRect(0, h * 0.94, w, h * 0.06);
    }
  });
  return TEXTURES[kind];
}

const OBJECT_COLORS = {
  plasticbox: 0x3b7dd8, suitcase: 0x3a3f4a, smallbox: 0xc19a6b, monitor: 0x1f2125, trashcan: 0x8a9399,
  whitechair: 0xe9e7e1, woodchair: 0x9a6a3c,
};

function floorTexture() {
  return canvasTexture(1024, 1024, (ctx, w, h) => {
    ctx.fillStyle = '#ebe4d6'; ctx.fillRect(0, 0, w, h);
    noise(ctx, w, h, 0.04, 30000);
    ctx.strokeStyle = 'rgba(110,90,60,0.20)'; ctx.lineWidth = 2;
    for (let i = 0; i <= 8; i++) {
      ctx.beginPath(); ctx.moveTo(i * w / 8, 0); ctx.lineTo(i * w / 8, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * h / 8); ctx.lineTo(w, i * h / 8); ctx.stroke();
    }
  });
}

export class Viewer {
  constructor(container) {
    this.container = container;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);
    this.renderer = renderer;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xebe4d6);
    scene.fog = new THREE.Fog(0xebe4d6, 9, 22);
    this.scene = scene;
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.05, 60);
    this.camera.position.set(2.6, 2.4, 4.0);
    this.controls = new OrbitControls(this.camera, renderer.domElement);
    this.controls.target.set(0.9, 0.5, 0);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minDistance = 0.8;
    this.controls.maxDistance = 14;

    scene.add(new THREE.HemisphereLight(0xfffaf0, 0x9a8f80, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(3, 6, 2);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const s = sun.shadow.camera;
    s.left = -4; s.right = 4; s.top = 4; s.bottom = -4; s.near = 0.5; s.far = 20;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    scene.add(sun, sun.target);
    this.sun = sun;

    const tex = floorTexture();
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(10, 10); // 0.5 m grid on a 40 m floor
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40),
      new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95, metalness: 0 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);
    this.floor = floor;

    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;
    scene.add(this.world);

    this.bodyGroups = [];
    this.robotMeshes = [];
    this.objectGroup = null;
    this.objectMeshes = [];
    this.raycaster = new THREE.Raycaster();
    this.makeGoal();
    this.makeRings();
    // Allowed range around the robot's start, shown while dragging.
    this.range = new THREE.Mesh(new THREE.RingGeometry(0.985, 1, 160),
      new THREE.MeshBasicMaterial({ color: 0x2b2520, transparent: true, opacity: 0.35, depthWrite: false }));
    this.range.rotation.x = -Math.PI / 2;
    this.range.position.y = 0.004;
    this.range.visible = false;
    this.scene.add(this.range);
    this.forceLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({ color: 0xd62828, depthTest: false }));
    this.forceLine.renderOrder = 10;
    this.forceLine.visible = false;
    scene.add(this.forceLine);

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  makeGoal() {
    const g = new THREE.Group();
    const disk = new THREE.Mesh(new THREE.CircleGeometry(0.16, 48),
      new THREE.MeshStandardMaterial({ color: 0xf28c28, roughness: 0.6, emissive: 0x6b3000, emissiveIntensity: 0.25 }));
    disk.rotation.x = -Math.PI / 2;
    disk.position.y = 0.002;
    disk.receiveShadow = true;
    const ring = new THREE.Mesh(new THREE.RingGeometry(GOAL_TOLERANCE - 0.012, GOAL_TOLERANCE, 64),
      new THREE.MeshBasicMaterial({ color: 0xf28c28, transparent: true, opacity: 0.55 }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.003;
    g.add(disk, ring);
    disk.userData.pick = 'goal';
    this.goal = g;
    this.goalDisk = disk;
    this.scene.add(g);
  }

  makeRings() {
    const mk = () => {
      const g = new THREE.Group();
      const ring = new THREE.Mesh(new THREE.TorusGeometry(1, 0.012, 8, 96),
        new THREE.MeshBasicMaterial({ color: 0x2b2520, transparent: true, opacity: 0.6 }));
      ring.rotation.x = -Math.PI / 2;
      const knob = new THREE.Mesh(new THREE.SphereGeometry(0.045, 20, 12), new THREE.MeshStandardMaterial({ color: 0x2b2520 }));
      const hit = new THREE.Mesh(new THREE.TorusGeometry(1, 0.06, 6, 48), new THREE.MeshBasicMaterial({ visible: false }));
      hit.rotation.x = -Math.PI / 2;
      g.add(ring, knob, hit);
      g.position.y = 0.01;
      this.scene.add(g);
      return { group: g, ring, knob, hit, radius: 1 };
    };
    this.rings = { robot: mk(), object: mk() };
    this.rings.robot.hit.userData.pick = 'rotate-robot';
    this.rings.robot.knob.userData.pick = 'rotate-robot';
    this.rings.object.hit.userData.pick = 'rotate-object';
    this.rings.object.knob.userData.pick = 'rotate-object';
  }

  setRing(which, xy, yaw, radius, visible) {
    const r = this.rings[which];
    r.group.visible = visible;
    if (!visible) return;
    r.group.position.set(xy[0], 0.01, -xy[1]);
    r.ring.scale.setScalar(radius);
    r.hit.scale.setScalar(radius);
    r.knob.position.set(radius * Math.cos(yaw), 0, -radius * Math.sin(yaw));
  }

  setRange(xy, radius, visible) {
    this.range.visible = visible;
    if (!visible) return;
    this.range.scale.set(radius, radius, 1);
    this.range.position.set(xy[0], 0.004, -xy[1]);
  }

  setGoal(xy, visible = true) {
    this.goal.visible = visible && !!xy;
    if (xy) this.goal.position.set(xy[0], 0, -xy[1]);
  }

  // -- robot ---------------------------------------------------------------------------
  async loadRobot(glbBytes, metaUrl, sim) {
    const meta = await (await fetch(metaUrl)).json();
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    const gltf = await loader.parseAsync(glbBytes.buffer.slice(glbBytes.byteOffset, glbBytes.byteOffset + glbBytes.byteLength), '');
    this.robotNodes = {};
    gltf.scene.traverse((o) => { if (o.isMesh) this.robotNodes[o.name] = o; });
    this.robotMeta = meta;
    const mats = {};
    this.robotTemplates = meta.nodes.map((n) => {
      let mesh = this.robotNodes[n.node];
      if (!mesh) gltf.scene.traverse((o) => { if (o.isMesh && o.parent && o.parent.name === n.node) mesh = o; });
      const key = n.rgba.join(',');
      if (!mats[key]) {
        const dark = n.rgba[0] < 0.4;
        mats[key] = new THREE.MeshStandardMaterial({
          color: new THREE.Color(n.rgba[0], n.rgba[1], n.rgba[2]).convertSRGBToLinear(),
          metalness: dark ? 0.1 : 0.35, roughness: dark ? 0.55 : 0.38,
        });
      }
      const geometry = toCreasedNormals(mesh.geometry, Math.PI / 6);
      mesh.updateWorldMatrix(true, false);
      return { body: n.body, geometry, material: mats[key], matrix: mesh.matrixWorld.clone() };
    });
    this.attachRobot(sim);
  }

  // (Re)attach the robot visuals to the bodies of the current model.
  attachRobot(sim) {
    for (const g of this.bodyGroups) this.world.remove(g);
    this.bodyGroups = [];
    this.robotMeshes = [];
    const m = sim.model;
    const byName = {};
    for (let b = 0; b < m.nbody; b++) {
      const g = new THREE.Group();
      g.matrixAutoUpdate = true;
      this.world.add(g);
      this.bodyGroups.push(g);
      byName[sim.mj.mj_id2name(m, sim.mj.mjtObj.mjOBJ_BODY.value, b)] = b;
    }
    for (const t of this.robotTemplates) {
      const mesh = new THREE.Mesh(t.geometry, t.material);
      mesh.matrixAutoUpdate = false;
      mesh.matrix.copy(t.matrix);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      const b = byName[t.body];
      mesh.userData.pick = 'robot';
      mesh.userData.body = b;
      this.bodyGroups[b].add(mesh);
      this.robotMeshes.push(mesh);
    }
    this.objectBody = sim.objBody;
  }

  // -- object --------------------------------------------------------------------------
  setObject(sim) {
    const group = this.bodyGroups[sim.objBody];
    if (this.objectGroup) for (const mesh of this.objectMeshes) mesh.parent.remove(mesh);
    this.objectMeshes = [];
    this.objectGroup = group;
    const spec = sim.spec;
    const axes = sim.boxAxes;
    const c = sim.boxCentre;
    const frame = new THREE.Matrix4().set(
      axes[0], axes[1], axes[2], c[0],
      axes[3], axes[4], axes[5], c[1],
      axes[6], axes[7], axes[8], c[2],
      0, 0, 0, 1);
    const add = (mesh) => {
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData.pick = 'object';
      mesh.userData.body = sim.objBody;
      group.add(mesh);
      this.objectMeshes.push(mesh);
    };
    if (spec.kind === 'box') {
      const [a, b, h] = sim.size;
      const side = new THREE.MeshStandardMaterial({ map: cardboard(false), roughness: 0.9 });
      const top = new THREE.MeshStandardMaterial({ map: cardboard(true), roughness: 0.9 });
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(a, b, h), [side, side, side, side, top, top]);
      mesh.matrixAutoUpdate = false;
      mesh.matrix.copy(frame);
      add(mesh);
      return;
    }
    const type = sim.objectType(spec.type);
    const variant = type.variants[spec.variant];
    if (type.family === 'ball') {
      const r = Math.max(...variant.box_size) / 2;
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32),
        new THREE.MeshStandardMaterial({ map: ballTexture(type.id), roughness: type.id === 'beach_ball' ? 0.35 : 0.7 }));
      mesh.position.set(c[0], c[1], c[2]);
      add(mesh);
      return;
    }
    const color = OBJECT_COLORS[type.id] ?? 0x999999;
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(color), roughness: type.id === 'monitor' ? 0.35 : 0.7, flatShading: true,
      metalness: type.id === 'monitor' ? 0.2 : 0,
    });
    for (const part of type.parts) {
      const pos = new Float32Array(part.v.length);
      for (let i = 0; i < part.v.length; i++) pos[i] = c[i % 3] + variant.ratio * (part.v[i] - c[i % 3]);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setIndex(part.f);
      geo.computeVertexNormals();
      add(new THREE.Mesh(geo.toNonIndexed(), mat));
    }
  }

  // -- per frame -----------------------------------------------------------------------
  update(sim) {
    const d = sim.data;
    const xpos = d.xpos, xquat = d.xquat;
    for (let b = 1; b < this.bodyGroups.length; b++) {
      const g = this.bodyGroups[b];
      g.position.set(xpos[3 * b], xpos[3 * b + 1], xpos[3 * b + 2]);
      g.quaternion.set(xquat[4 * b + 1], xquat[4 * b + 2], xquat[4 * b + 3], xquat[4 * b]);
    }
    const p = sim.pelvisBody;
    const pel = toThree([xpos[3 * p], xpos[3 * p + 1], 0]);
    this.sun.position.set(pel.x + 3, 6, pel.z + 2);
    this.sun.target.position.copy(pel);
  }

  pelvisThree(sim) {
    const xpos = sim.data.xpos, p = sim.pelvisBody;
    return toThree([xpos[3 * p], xpos[3 * p + 1], xpos[3 * p + 2]]);
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  // -- picking -------------------------------------------------------------------------
  ndc(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1);
  }

  pick(event, candidates) {
    this.raycaster.setFromCamera(this.ndc(event), this.camera);
    const hits = this.raycaster.intersectObjects(candidates, false);
    return hits.length ? hits[0] : null;
  }

  // Intersection of the pointer ray with the horizontal plane at height ``h`` (three y), as MuJoCo xy.
  floorPoint(event, h = 0) {
    this.raycaster.setFromCamera(this.ndc(event), this.camera);
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -h);
    const out = new THREE.Vector3();
    if (!this.raycaster.ray.intersectPlane(plane, out)) return null;
    return fromThree(out);
  }

  // Intersection with the camera-facing plane through ``point`` (three coords), as MuJoCo xyz.
  dragPoint(event, point) {
    this.raycaster.setFromCamera(this.ndc(event), this.camera);
    const normal = new THREE.Vector3();
    this.camera.getWorldDirection(normal);
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, point);
    const out = new THREE.Vector3();
    if (!this.raycaster.ray.intersectPlane(plane, out)) return null;
    return fromThree(out);
  }

  showForce(fromMj, toMj) {
    if (!fromMj) { this.forceLine.visible = false; return; }
    const pos = this.forceLine.geometry.attributes.position;
    const a = toThree(fromMj), b = toThree(toMj);
    pos.setXYZ(0, a.x, a.y, a.z);
    pos.setXYZ(1, b.x, b.y, b.z);
    pos.needsUpdate = true;
    this.forceLine.visible = true;
  }
}
