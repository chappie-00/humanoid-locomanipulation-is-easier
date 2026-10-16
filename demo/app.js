// Interactive demo: the trained policies, live, in MuJoCo compiled to WebAssembly.
import loadMujoco from '../vendor/mujoco/mujoco.js';
import { Sim, quatToMat } from './sim.js';
import { Viewer, GOAL_TOLERANCE, toThree } from './viewer.js';

const ASSETS = new URL('./assets/', import.meta.url);
const VENDOR = new URL('../vendor/', import.meta.url);
const CONTROL_DT = 0.02;
const MAX_DIST = 3.5; // object and goal stay within this distance (m) of the robot's start, for now

// ``xy`` pulled back onto the disc of radius ``r`` around ``centre``.
const clampTo = (centre, xy, r) => {
  const dx = xy[0] - centre[0], dy = xy[1] - centre[1];
  const d = Math.hypot(dx, dy);
  return d <= r ? xy : [centre[0] + dx * r / d, centre[1] + dy * r / d];
};
const MESHES = ['left_rubber_hand.STL', 'left_wrist_pitch_link.STL', 'left_wrist_roll_link.STL', 'left_wrist_yaw_link.STL',
  'right_rubber_hand.STL', 'right_wrist_pitch_link.STL', 'right_wrist_roll_link.STL', 'right_wrist_yaw_link.STL'];

// One policy per task.
const POLICY = { pick_place: 'pp_stage2_latest', pick_up: 'pu_objects' };

const FAMILIES = [
  ['Boxes', ['box']],
  ['Balls', ['football', 'basketball', 'volleyball', 'medicine_ball', 'beach_ball']],
  ['Household', ['plasticbox', 'suitcase', 'smallbox', 'monitor', 'trashcan']],
  ['Chairs', ['whitechair', 'woodchair']],
];

const $ = (s) => document.querySelector(s);
const clamp = (x, a, b) => Math.min(Math.max(x, a), b);
const rand = (a, b) => a + Math.random() * (b - a);

// Hosts that do not serve binary types (e.g. the Artifact host) get a base64 JSON copy at ``<url>.b64.json``.
async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) {
    const alt = await fetch(`${url}.b64.json`);
    if (!alt.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const s = (await alt.json()).b64;
    const out = new Uint8Array(s.length * 3 / 4 - (s.endsWith('==') ? 2 : s.endsWith('=') ? 1 : 0));
    const bin = atob(s);
    for (let i = 0; i < out.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !onProgress) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) onProgress(Math.min(got / total, 1));
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

// ``.gz`` copies are decompressed here; a host that already decoded them is fine too.
async function gunzipMaybe(bytes) {
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function fetchGz(url, onProgress) {
  if (typeof DecompressionStream !== 'undefined') {
    try { return await gunzipMaybe(await fetchBytes(`${url}.gz`, onProgress)); } catch (e) { /* fall back */ }
  }
  return fetchBytes(url, onProgress);
}

const text = (bytes) => new TextDecoder().decode(bytes);

class Demo {
  constructor(root) {
    this.root = root;
    this.policyCache = {};
    this.task = 'pick_place';
    this.policyId = POLICY.pick_place;
    this.mode = 'setup';
    this.speed = 1;
    this.follow = true;
    this.object = { kind: 'box', size: [0.27, 0.28, 0.30], mass: 1.0 };
    this.scene = { robotXY: [0, 0], robotYaw: 0, objectXY: [1.9, 0.7], objectYaw: 0.35, goalXY: [0.25, -1.3] };
    this.drag = null;
  }

  progress(frac, label) {
    $('#demo-loading .bar span').style.width = `${Math.round(frac * 100)}%`;
    if (label) $('#demo-loading p').textContent = label;
  }

  async init() {
    this.progress(0.02, 'Loading MuJoCo (WebAssembly)…');
    const wasm = await fetchGz(new URL('mujoco/mujoco.wasm', VENDOR).href, (f) => this.progress(0.02 + 0.6 * f));
    this.mj = await loadMujoco({ wasmBinary: wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.byteLength) });
    this.progress(0.65, 'Loading the robot and objects…');
    const [xml, objects, ...meshes] = await Promise.all([
      fetchBytes(new URL('robot/g1_physics.xml', ASSETS)),
      fetchGz(new URL('objects.json', ASSETS).href),
      ...MESHES.map((f) => fetchBytes(new URL(`robot/meshes/${f}`, ASSETS))),
    ]);
    this.sim = new Sim(this.mj, {
      xml: text(xml),
      objects: JSON.parse(text(objects)),
      meshes: Object.fromEntries(MESHES.map((f, i) => [f, meshes[i]])),
    });
    this.progress(0.8, 'Loading the policy…');
    await this.usePolicy(this.policyId, false);
    this.sim.build(this.object);
    this.viewer = new Viewer($('#viewer'));
    this.progress(0.9, 'Loading the robot model…');
    const glb = await fetchBytes(new URL('robot/g1_visual.glb', ASSETS));
    await this.viewer.loadRobot(glb, new URL('robot/g1_visual.json', ASSETS).href, this.sim);
    this.viewer.setObject(this.sim);
    this.prepareObjectPoints();
    this.buildPanel();
    this.applyScene();
    this.bindPointer();
    $('#demo-loading').classList.add('hidden');
    this.last = performance.now();
    requestAnimationFrame((t) => this.frame(t));
    // Start the default scene right away.
    this.run();
  }

  // -- policies and objects -------------------------------------------------------------
  async usePolicy(id, rebuild = true) {
    if (!this.policyCache[id]) {
      const [meta, bin] = await Promise.all([
        fetch(new URL(`policies/${id}.json`, ASSETS)).then((r) => r.json()),
        fetchBytes(new URL(`policies/${id}.bin`, ASSETS)),
      ]);
      this.policyCache[id] = { meta, floats: new Float32Array(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)) };
    }
    const { meta, floats } = this.policyCache[id];
    this.policyId = id;
    this.task = meta.task;
    this.sim.setPolicy(meta, floats);
    if (rebuild) this.rebuild();
  }

  rebuild() {
    this.sim.build(this.object);
    this.viewer.attachRobot(this.sim);
    this.viewer.setObject(this.sim);
    this.prepareObjectPoints();
    this.applyScene();
  }

  prepareObjectPoints() {
    const sim = this.sim;
    if (this.object.kind === 'box') { this.objectPoints = sim.cornersBody; return; }
    const type = sim.objectType(this.object.type);
    const v = type.variants[this.object.variant];
    const c = sim.boxCentre;
    const pts = [];
    for (const part of type.parts) {
      for (let i = 0; i < part.v.length; i += 3 * 3) {
        pts.push([0, 1, 2].map((k) => c[k] + v.ratio * (part.v[i + k] - c[k])));
      }
    }
    this.objectPoints = pts;
  }

  objectRadius() {
    if (this.object.kind === 'box') return Math.hypot(this.object.size[0], this.object.size[1]) / 2 + 0.12;
    const v = this.sim.objectType(this.object.type).variants[this.object.variant];
    return v.rest_radius + 0.12;
  }

  // -- scene ---------------------------------------------------------------------------
  applyScene() {
    const s = this.scene;
    this.sim.reset({ ...s, goalXY: this.task === 'pick_place' ? s.goalXY : null });
    this.mode = 'setup';
    this.tracker = new Tracker(this);
    this.viewer.update(this.sim);
    this.viewer.setGoal(s.goalXY, this.task === 'pick_place');
    this.updateRings();
    this.viewer.showForce(null);
    this.updateButtons();
    this.updateHud();
  }

  updateRings() {
    const setup = this.mode === 'setup';
    const s = this.scene;
    this.viewer.setRing('robot', s.robotXY, s.robotYaw, 0.42, setup);
    this.viewer.setRing('object', s.objectXY, s.objectYaw, this.objectRadius(), setup);
  }

  randomScene() {
    const s = this.scene;
    s.robotXY = [0, 0];
    s.robotYaw = rand(-Math.PI, Math.PI);
    for (let tries = 0; tries < 50; tries++) {
      const d = rand(0.8, MAX_DIST), b = rand(-Math.PI, Math.PI);
      s.objectXY = [d * Math.cos(s.robotYaw + b), d * Math.sin(s.robotYaw + b)];
      s.objectYaw = rand(-Math.PI, Math.PI);
      const gd = rand(0.8, 1.8), gb = rand(-2.4, 2.4);
      s.goalXY = [gd * Math.cos(s.robotYaw + gb), gd * Math.sin(s.robotYaw + gb)];
      if (Math.hypot(s.goalXY[0] - s.objectXY[0], s.goalXY[1] - s.objectXY[1]) > 0.8) break;
    }
  }

  // -- run control ---------------------------------------------------------------------
  run() {
    this.mode = 'running';
    this.updateRings();
    this.updateButtons();
  }

  pause() {
    this.mode = 'paused';
    this.updateButtons();
  }

  frame(now) {
    const dt = Math.min((now - this.last) / 1000, 0.1);
    this.last = now;
    if (this.mode === 'running') {
      this.simDebt = (this.simDebt ?? 0) + dt * this.speed;
      let budget = 12;
      const t0 = performance.now();
      while (this.simDebt >= CONTROL_DT && budget-- > 0) {
        this.sim.step((m, d) => this.applyForce(m, d));
        this.simDebt -= CONTROL_DT;
        // Runs until the visitor pauses: after a placement they can keep pushing things around.
        this.tracker.tick();
        if (performance.now() - t0 > 30) { this.simDebt = 0; break; }
      }
      this.viewer.update(this.sim);
      this.updateHud();
      if (this.follow) this.followCamera();
    } else {
      this.simDebt = 0;
    }
    if (this.drag && this.drag.kind === 'force') this.viewer.showForce(this.drag.world, this.drag.target);
    this.viewer.render();
    requestAnimationFrame((t) => this.frame(t));
  }

  followCamera() {
    const v = this.viewer;
    const p = v.pelvisThree(this.sim);
    const target = v.controls.target;
    const desired = p.clone();
    desired.y = 0.55;
    const delta = desired.sub(target).multiplyScalar(0.06);
    target.add(delta);
    v.camera.position.add(delta);
  }

  // -- pointer ---------------------------------------------------------------------------
  bindPointer() {
    const el = this.viewer.renderer.domElement;
    // Capture phase: runs before OrbitControls, which then never sees a grab.
    el.addEventListener('pointerdown', (e) => this.pointerDown(e), { capture: true });
    el.addEventListener('pointermove', (e) => this.pointerMove(e));
    window.addEventListener('pointerup', () => this.pointerUp());
    el.addEventListener('pointerleave', () => { if (!this.drag) el.style.cursor = ''; });
  }

  candidates() {
    const v = this.viewer;
    const list = [...v.robotMeshes, ...v.objectMeshes];
    if (this.task === 'pick_place') list.push(v.goalDisk);
    if (this.mode === 'setup') list.unshift(v.rings.robot.hit, v.rings.robot.knob, v.rings.object.hit, v.rings.object.knob);
    return list;
  }

  pointerDown(e) {
    if (e.button !== 0) return;
    const hit = this.viewer.pick(e, this.candidates());
    if (!hit) return;
    const what = hit.object.userData.pick;
    const s = this.scene;
    const floor = this.viewer.floorPoint(e);
    if (what === 'goal') {
      this.drag = { kind: 'goal', offset: [s.goalXY[0] - floor[0], s.goalXY[1] - floor[1]] };
    } else if (what === 'rotate-robot' || what === 'rotate-object') {
      this.drag = { kind: what };
    } else if (this.mode === 'setup' && (what === 'robot' || what === 'object')) {
      const key = what === 'robot' ? 'robotXY' : 'objectXY';
      this.drag = { kind: 'move', key, offset: [s[key][0] - floor[0], s[key][1] - floor[1]] };
    } else if (this.mode === 'running' || this.mode === 'paused') {
      if (what !== 'robot' && what !== 'object') return;
      const body = hit.object.userData.body;
      const d = this.sim.data;
      const R = Array.from(d.xmat.subarray(9 * body, 9 * body + 9));
      const p = [d.xpos[3 * body], d.xpos[3 * body + 1], d.xpos[3 * body + 2]];
      const w = [hit.point.x, -hit.point.z, hit.point.y];
      const rel = [w[0] - p[0], w[1] - p[1], w[2] - p[2]];
      const local = [0, 1, 2].map((i) => R[i] * rel[0] + R[3 + i] * rel[1] + R[6 + i] * rel[2]);
      this.drag = { kind: 'force', body, local, plane: hit.point.clone(), world: w, target: w, last: null, isObject: what === 'object' };
    } else {
      return;
    }
    this.viewer.controls.enabled = false;
    this.viewer.renderer.domElement.setPointerCapture?.(e.pointerId);
    e.preventDefault();
    e.stopImmediatePropagation();
  }

  pointerMove(e) {
    const el = this.viewer.renderer.domElement;
    if (!this.drag) {
      const hit = this.viewer.pick(e, this.candidates());
      el.style.cursor = hit ? (this.mode === 'setup' || hit.object.userData.pick === 'goal' ? 'grab' : 'pointer') : '';
      return;
    }
    el.style.cursor = 'grabbing';
    const s = this.scene;
    if (this.drag.kind === 'force') {
      const t = this.viewer.dragPoint(e, this.drag.plane);
      if (t) this.drag.target = t;
      return;
    }
    const f = this.viewer.floorPoint(e);
    if (!f) return;
    if (this.drag.kind === 'goal') {
      s.goalXY = clampTo(s.robotXY, [f[0] + this.drag.offset[0], f[1] + this.drag.offset[1]], MAX_DIST);
      this.viewer.setRange(s.robotXY, MAX_DIST, true);
      this.viewer.setGoal(s.goalXY, true);
      if (this.mode === 'setup') this.applyScene();
      else this.sim.goal = [s.goalXY[0], s.goalXY[1], 0];
      return;
    }
    if (this.drag.kind === 'move') {
      const xy = [f[0] + this.drag.offset[0], f[1] + this.drag.offset[1]];
      if (this.drag.key === 'objectXY') {
        s.objectXY = clampTo(s.robotXY, xy, MAX_DIST);
      } else {
        // The robot must stay within range of both the object and the goal: alternate projections onto both discs.
        let r = xy;
        for (let k = 0; k < 12; k++) {
          r = clampTo(s.objectXY, r, MAX_DIST);
          if (this.task === 'pick_place') r = clampTo(s.goalXY, r, MAX_DIST);
        }
        s.robotXY = r;
      }
      this.viewer.setRange(s.robotXY, MAX_DIST, true);
    } else if (this.drag.kind === 'rotate-robot') {
      s.robotYaw = Math.atan2(f[1] - s.robotXY[1], f[0] - s.robotXY[0]);
    } else if (this.drag.kind === 'rotate-object') {
      s.objectYaw = Math.atan2(f[1] - s.objectXY[1], f[0] - s.objectXY[0]);
    }
    this.applyScene();
  }

  pointerUp() {
    if (!this.drag) return;
    if (this.drag.kind === 'force') {
      const d = this.sim.data;
      d.xfrc_applied.fill(0);
      this.viewer.showForce(null);
    }
    this.drag = null;
    this.viewer.setRange(null, 0, false);
    this.viewer.controls.enabled = true;
    this.viewer.renderer.domElement.style.cursor = '';
  }

  // A spring from the grabbed point to the pointer, applied before every physics step.
  applyForce(m, d) {
    const g = this.drag;
    if (!g || g.kind !== 'force') return;
    const b = g.body;
    const R = d.xmat.subarray(9 * b, 9 * b + 9);
    const p = [0, 1, 2].map((i) => d.xpos[3 * b + i] + R[3 * i] * g.local[0] + R[3 * i + 1] * g.local[1] + R[3 * i + 2] * g.local[2]);
    const v = g.last ? p.map((x, i) => (x - g.last[i]) / 0.005) : [0, 0, 0];
    g.last = p;
    g.world = p;
    const mass = g.isObject ? m.body_mass[b] : 35;
    const k = (g.isObject ? 120 : 60) * Math.max(mass, 0.4);
    const c = 2 * Math.sqrt(k * Math.max(mass, 0.4)) * 0.6;
    let F = p.map((x, i) => k * (g.target[i] - x) - c * v[i]);
    const fmax = g.isObject ? 60 + 40 * mass : 350;
    const n = Math.hypot(...F);
    if (n > fmax) F = F.map((x) => x * fmax / n);
    const com = [d.xipos[3 * b], d.xipos[3 * b + 1], d.xipos[3 * b + 2]];
    const r = [p[0] - com[0], p[1] - com[1], p[2] - com[2]];
    const tau = [r[1] * F[2] - r[2] * F[1], r[2] * F[0] - r[0] * F[2], r[0] * F[1] - r[1] * F[0]];
    d.xfrc_applied.set([...F, ...tau], 6 * b);
  }

  // -- panel -----------------------------------------------------------------------------
  buildPanel() {
    const root = this.root;
    root.querySelectorAll('[data-task]').forEach((btn) => btn.addEventListener('click', async () => {
      if (btn.dataset.task === this.task) return;
      await this.usePolicy(POLICY[btn.dataset.task]);
      this.syncTask();
      this.updateButtons();
      this.run();
    }));
    this.syncTask();

    const chips = $('#object-chips');
    const types = this.sim.objects.types;
    for (const [family, ids] of FAMILIES) {
      const row = document.createElement('div');
      row.className = 'chip-row';
      row.innerHTML = `<span class="chip-family">${family}</span>`;
      for (const id of ids) {
        const b = document.createElement('button');
        b.className = 'chip';
        b.dataset.object = id;
        b.textContent = id === 'box' ? 'Box' : types.find((t) => t.id === id).label;
        b.addEventListener('click', () => this.chooseObject(id));
        row.appendChild(b);
      }
      chips.appendChild(row);
    }

    const box = this.sim.meta.box;
    ['w', 'd', 'h'].forEach((k, i) => {
      const el = $(`#box-${k}`);
      el.addEventListener('input', () => {
        this.object.size[i] = Number(el.value) / 100;
        this.syncObjectControls();
        this.rebuild();
      });
    });
    $('#variant').addEventListener('input', (e) => {
      this.object.variant = Number(e.target.value);
      delete this.object.mass;
      this.syncObjectControls();
      this.rebuild();
    });
    $('#mass').addEventListener('input', (e) => {
      this.object.mass = Math.round(Math.exp(Number(e.target.value)) * 100) / 100;
      this.syncObjectControls();
      this.rebuild();
    });
    void box;
    $('#btn-run').addEventListener('click', () => (this.mode === 'running' ? this.pause() : this.run()));
    $('#btn-reset').addEventListener('click', () => this.applyScene());
    const dice = $('#btn-random');
    dice.addEventListener('animationend', () => dice.classList.remove('rolling', 'nudge'));
    setTimeout(() => dice.classList.add('nudge'), 2500); // draw the eye once the demo is up
    dice.addEventListener('click', () => {
      dice.classList.remove('nudge', 'rolling');
      void dice.offsetWidth; // restart the roll on every press
      dice.classList.add('rolling');
      this.randomObject();
      this.randomScene();
      this.rebuild();
      this.syncObjectControls();
      this.run();
    });
    root.querySelectorAll('[data-speed]').forEach((btn) => btn.addEventListener('click', () => {
      this.speed = Number(btn.dataset.speed);
      root.querySelectorAll('[data-speed]').forEach((b) => b.classList.toggle('on', b === btn));
    }));
    $('#follow').addEventListener('change', (e) => { this.follow = e.target.checked; });
    this.syncObjectControls();
  }

  syncTask() {
    this.root.querySelectorAll('[data-task]').forEach((b) => b.classList.toggle('on', b.dataset.task === this.task));
    this.root.classList.toggle('task-pick-up', this.task === 'pick_up');
  }

  chooseObject(id) {
    if (id === 'box') {
      this.object = { kind: 'box', size: [0.27, 0.28, 0.30], mass: 1.0 };
    } else {
      this.object = { kind: 'object', type: id, variant: 8 };
    }
    this.syncObjectControls();
    this.rebuild();
    this.run();
  }

  randomObject() {
    const all = FAMILIES.flatMap(([, ids]) => ids);
    const id = Math.random() < 0.3 ? 'box' : all[1 + Math.floor(Math.random() * (all.length - 1))];
    if (id === 'box') {
      const b = this.sim.meta.box;
      this.object = { kind: 'box', size: b.size_min.map((lo, i) => Math.round(rand(lo, b.size_max[i]) * 100) / 100),
        mass: Math.round(Math.exp(rand(Math.log(0.2), Math.log(3))) * 10) / 10 };
    } else {
      this.object = { kind: 'object', type: id, variant: Math.floor(Math.random() * 16) };
    }
  }

  syncObjectControls() {
    const o = this.object;
    this.root.querySelectorAll('[data-object]').forEach((b) => b.classList.toggle('on', b.dataset.object === (o.kind === 'box' ? 'box' : o.type)));
    $('#box-controls').hidden = o.kind !== 'box';
    $('#variant-controls').hidden = o.kind === 'box';
    let mass, massLo, massHi, inRange = true;
    const b = this.sim.meta.box;
    if (o.kind === 'box') {
      ['w', 'd', 'h'].forEach((k, i) => {
        $(`#box-${k}`).value = Math.round(o.size[i] * 100);
        $(`#box-${k}-val`).textContent = `${Math.round(o.size[i] * 100)} cm`;
        if (o.size[i] < b.size_min[i] - 0.005 || o.size[i] > b.size_max[i] + 0.005) inRange = false;
      });
      mass = o.mass;
      [massLo, massHi] = b.mass_range;
    } else {
      const type = this.sim.objectType(o.type);
      const v = type.variants[o.variant];
      $('#variant').value = o.variant;
      $('#variant-val').textContent = `${Math.round(Math.max(...v.box_size) * 100)} cm`;
      mass = o.mass ?? v.mass;
      const masses = type.variants.map((x) => x.mass);
      massLo = Math.min(...masses) * 0.8;
      massHi = Math.max(...masses) * 1.25;
    }
    $('#mass').value = Math.log(Math.max(mass, 0.05));
    $('#mass-val').textContent = `${mass < 1 ? mass.toFixed(2) : mass.toFixed(1)} kg`;
    if (mass < massLo || mass > massHi) inRange = false;
    $('#range-note').hidden = inRange;
  }

  updateButtons() {
    const btn = $('#btn-run');
    btn.textContent = this.mode === 'running' ? 'Pause' : this.mode === 'paused' ? 'Resume' : 'Run';
    btn.classList.toggle('primary', this.mode !== 'running');
    this.root.classList.toggle('is-setup', this.mode === 'setup');
  }

  updateHud() {
    const t = this.sim.ticks * CONTROL_DT;
    $('#hud-time').textContent = `${t.toFixed(1)} s`;
    const tr = this.tracker;
    const el = $('#hud-phase');
    el.textContent = this.mode === 'setup' ? 'Ready — drag to rearrange' : tr.phase;
    el.dataset.state = tr.state;
  }
}

// Task phases for display, read from contacts and geometry (the benchmark's criteria).
class Tracker {
  constructor(demo) {
    this.demo = demo;
    const sim = demo.sim, m = sim.model, mj = sim.mj;
    this.phase = 'Approach';
    this.state = '';
    this.carriedTicks = 0;
    this.carried = false;
    this.successTicks = 0;
    this.holdTicks = 0;
    this.recent = [];
    const subtree = (name) => {
      const root = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY.value, name);
      const out = new Set();
      for (let g = 0; g < m.ngeom; g++) {
        let b = m.geom_bodyid[g];
        while (b > 0 && b !== root) b = m.body_parentid[b];
        if (b === root) out.add(g);
      }
      return out;
    };
    this.hands = [subtree('left_wrist_yaw_link'), subtree('right_wrist_yaw_link')];
    this.objGeoms = new Set();
    for (let g = 0; g < m.ngeom; g++) if (m.geom_bodyid[g] === sim.objBody) this.objGeoms.add(g);
    this.floorGeom = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM.value, 'floor');
    this.lastCentre = null;
  }

  tick() {
    const demo = this.demo, sim = demo.sim, d = sim.data;
    let left = false, right = false, floor = false;
    const contacts = d.contact;
    for (let i = 0; i < d.ncon; i++) {
      const c = contacts.get(i);
      const g1 = c.geom1, g2 = c.geom2;
      c.delete();
      const o1 = this.objGeoms.has(g1), o2 = this.objGeoms.has(g2);
      if (!o1 && !o2) continue;
      const other = o1 ? g2 : g1;
      if (other === this.floorGeom) floor = true;
      if (this.hands[0].has(other)) left = true;
      if (this.hands[1].has(other)) right = true;
    }
    contacts.delete();
    const b = sim.objBody;
    const R = d.xmat.subarray(9 * b, 9 * b + 9);
    const p = [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]];
    let low = Infinity;
    for (const v of demo.objectPoints) low = Math.min(low, p[2] + R[6] * v[0] + R[7] * v[1] + R[8] * v[2]);
    const c = sim.boxCentre;
    const centre = [0, 1, 2].map((i) => p[i] + R[3 * i] * c[0] + R[3 * i + 1] * c[1] + R[3 * i + 2] * c[2]);
    const speed = this.lastCentre ? Math.hypot(...centre.map((x, i) => x - this.lastCentre[i])) / CONTROL_DT : 0;
    this.lastCentre = centre;
    const qpos = d.qpos;
    const up = quatToMat([qpos[3], qpos[4], qpos[5], qpos[6]])[8];
    const fell = qpos[2] < 0.3 || up < Math.cos(80 * Math.PI / 180);
    const anyHand = left || right;
    this.recent.push(anyHand);
    if (this.recent.length > 7) this.recent.shift();
    const handRecent = this.recent.some(Boolean);
    const held = handRecent && low >= 0.10 && !floor;
    this.carriedTicks = held ? this.carriedTicks + 1 : 0;
    if (this.carriedTicks >= 15) this.carried = true;
    if (fell) { this.phase = 'Fell'; this.state = 'bad'; return; }
    if (sim.goal === null) {
      if (!this.carried) { this.phase = anyHand ? 'Grasp' : 'Approach'; this.state = ''; return; }
      this.holdTicks = held ? this.holdTicks + 1 : 0;
      if (this.holdTicks > 50) { this.phase = 'Holding ✓'; this.state = 'good'; } else { this.phase = held ? 'Lift' : 'Re-grasp'; this.state = ''; }
      return;
    }
    const dGoal = Math.hypot(centre[0] - sim.goal[0], centre[1] - sim.goal[1]);
    if (!this.carried) { this.phase = anyHand ? 'Grasp' : 'Approach'; this.state = ''; this.successTicks = 0; return; }
    if (held) {
      this.phase = dGoal < GOAL_TOLERANCE + 0.2 ? 'Place' : 'Carry';
      this.state = '';
      this.successTicks = 0;
      return;
    }
    const resting = low < 0.03 && speed < 0.05;
    if (anyHand) {
      this.phase = dGoal <= GOAL_TOLERANCE ? 'Release' : 'Re-grasp';
      this.state = '';
      this.successTicks = 0;
      return;
    }
    if (resting && dGoal <= GOAL_TOLERANCE) {
      this.successTicks += 1;
      this.phase = 'Placed ✓';
      this.state = 'good';
    } else {
      this.phase = 'Dropped';
      this.state = 'bad';
      this.successTicks = 0;
    }
  }
}

// Load the demo when it scrolls into view.
const root = document.getElementById('demo');
if (root) {
  let started = false;
  const start = () => {
    if (started) return;
    started = true;
    const demo = new Demo(root);
    if (new URLSearchParams(location.search).has('debug')) window.demo = demo; // for the browser tests
    demo.init().catch((err) => {
      console.error(err);
      $('#demo-loading p').textContent = `The simulator could not start in this browser (${err.message}).`;
      $('#demo-loading').classList.add('error');
    });
  };
  new IntersectionObserver((entries, obs) => {
    if (entries.some((e) => e.isIntersecting)) { obs.disconnect(); start(); }
  }, { rootMargin: '400px' }).observe(root);
}

void toThree;
