// The in-browser simulator: MuJoCo (WASM) + the trained policy, line for line the
// Python reference (tools/web_reference.py), which is checked against the paper's
// benchmark runner. Runs in the browser and in Node (for the parity tests).
//
// Control loop: perfect state, policy at 50 Hz, explicit PD torques
// kp (q* - q) - kd dq clipped at the effort limit at every 5 ms physics step.

const DECIMATION = 4;
const OBJECT_FRICTION = '0.6 0.005 0.0001';
const PALM_SITES = ['left_palm', 'right_palm'];

// -- small linear algebra (row-major 3x3, MuJoCo's quaternion conventions) -----------
export function quatToMat(q, out = new Float64Array(9)) {
  const [w, x, y, z] = q;
  const ww = w * w, xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y, wz = w * z;
  out[0] = ww + xx - yy - zz; out[1] = 2 * (xy - wz); out[2] = 2 * (xz + wy);
  out[3] = 2 * (xy + wz); out[4] = ww - xx + yy - zz; out[5] = 2 * (yz - wx);
  out[6] = 2 * (xz - wy); out[7] = 2 * (yz + wx); out[8] = ww - xx - yy + zz;
  return out;
}

// mju_mat2Quat
export function matToQuat(m) {
  const q = [0, 0, 0, 0];
  if (m[0] + m[4] + m[8] > 0) {
    q[0] = 0.5 * Math.sqrt(1 + m[0] + m[4] + m[8]);
    q[1] = 0.25 * (m[7] - m[5]) / q[0];
    q[2] = 0.25 * (m[2] - m[6]) / q[0];
    q[3] = 0.25 * (m[3] - m[1]) / q[0];
  } else if (m[0] > m[4] && m[0] > m[8]) {
    q[1] = 0.5 * Math.sqrt(1 + m[0] - m[4] - m[8]);
    q[0] = 0.25 * (m[7] - m[5]) / q[1];
    q[2] = 0.25 * (m[1] + m[3]) / q[1];
    q[3] = 0.25 * (m[2] + m[6]) / q[1];
  } else if (m[4] > m[8]) {
    q[2] = 0.5 * Math.sqrt(1 - m[0] + m[4] - m[8]);
    q[0] = 0.25 * (m[2] - m[6]) / q[2];
    q[1] = 0.25 * (m[1] + m[3]) / q[2];
    q[3] = 0.25 * (m[5] + m[7]) / q[2];
  } else {
    q[3] = 0.5 * Math.sqrt(1 - m[0] - m[4] + m[8]);
    q[0] = 0.25 * (m[3] - m[1]) / q[3];
    q[1] = 0.25 * (m[2] + m[6]) / q[3];
    q[2] = 0.25 * (m[5] + m[7]) / q[3];
  }
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  return q.map((v) => v / n);
}

// mju_mulQuat
export function quatMul(a, b) {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

export const yawQuat = (yaw) => [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];

// R^T v for row-major R
function mulTVec(R, v) {
  return [
    R[0] * v[0] + R[3] * v[1] + R[6] * v[2],
    R[1] * v[0] + R[4] * v[1] + R[7] * v[2],
    R[2] * v[0] + R[5] * v[1] + R[8] * v[2],
  ];
}
function mulVec(R, v) {
  return [
    R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
    R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
    R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
  ];
}
// A^T B for row-major 3x3
function mulTMat(A, B) {
  const out = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    out[i * 3 + j] = A[i] * B[j] + A[3 + i] * B[3 + j] + A[6 + i] * B[6 + j];
  }
  return out;
}

const fmt = (values) => Array.from(values, (v) => String(+v)).join(' ');

// -- the policy -----------------------------------------------------------------------
export class PolicyNet {
  constructor(meta, floats) {
    const n = meta.obs_dim;
    this.obsDim = n;
    this.mean = floats.subarray(0, n);
    this.std = floats.subarray(n, 2 * n);
    this.eps = Math.fround(meta.net.eps);
    this.layers = [];
    let off = 2 * n;
    for (const [outDim, inDim] of meta.net.layers) {
      const W = floats.subarray(off, off + outDim * inDim); off += outDim * inDim;
      const b = floats.subarray(off, off + outDim); off += outDim;
      this.layers.push({ W, b, outDim, inDim, out: new Float32Array(outDim) });
    }
    if (off !== floats.length || meta.net.activation !== 'elu') throw new Error('unexpected policy layout');
    this.input = new Float32Array(n);
  }

  act(obs) {
    const x = this.input;
    for (let i = 0; i < this.obsDim; i++) {
      x[i] = Math.fround(Math.fround(obs[i] - this.mean[i]) / Math.fround(this.std[i] + this.eps));
    }
    let h = x;
    const last = this.layers.length - 1;
    this.layers.forEach(({ W, b, outDim, inDim, out }, li) => {
      for (let o = 0; o < outDim; o++) {
        let acc = 0;
        const row = o * inDim;
        for (let i = 0; i < inDim; i++) acc += W[row + i] * h[i];
        let v = Math.fround(acc + b[o]);
        if (li < last && v <= 0) v = Math.fround(Math.expm1(v));
        out[o] = v;
      }
      h = out;
    });
    return h;
  }
}

// -- the simulator ---------------------------------------------------------------------
export class Sim {
  // files: { xml: string, meshes: {name: Uint8Array}, objects: parsed objects.json }
  constructor(mujoco, files) {
    this.mj = mujoco;
    this.template = files.xml;
    this.objects = files.objects;
    this.vfs = new mujoco.MjVFS();
    for (const [name, bytes] of Object.entries(files.meshes)) this.vfs.addBuffer(`meshes/${name}`, bytes);
    this.model = null;
    this.data = null;
  }

  setPolicy(meta, floats) {
    this.meta = meta;
    this.net = new PolicyNet(meta, floats);
    const j = meta.joints;
    this.defaultPos = Float64Array.from(j.default_pos);
    this.actionScale = Float64Array.from(j.action_scale);
    this.kp = Float64Array.from(j.kp);
    this.kd = Float64Array.from(j.kd);
    this.effort = Float64Array.from(j.effort_limit);
    this.boxAxes = Float64Array.from(meta.box.axes.flat());
    this.boxCentre = meta.box.centre.slice();
    this.history = null;
  }

  id(kind, name) {
    const i = this.mj.mj_name2id(this.model, this.mj.mjtObj[`mjOBJ_${kind}`].value, name);
    if (i < 0) throw new Error(`no ${kind.toLowerCase()} named ${name}`);
    return i;
  }

  objectType(id) {
    return this.objects.types.find((t) => t.id === id);
  }

  // Asset and body XML for ``spec`` and the bounding-box edges the policy sees.
  objectXml(spec) {
    const common = `friction="${OBJECT_FRICTION}" contype="4" conaffinity="3" condim="3"`;
    if (spec.kind === 'box') {
      const size = spec.size.map(Number);
      const quat = matToQuat(this.boxAxes);
      const body = `<body name="object" pos="0 0 1"><freejoint name="object"/>`
        + `<geom name="object_box" type="box" size="${fmt(size.map((s) => s / 2))}" pos="${fmt(this.boxCentre)}" `
        + `quat="${fmt(quat)}" mass="${String(+spec.mass)}" ${common}/></body>`;
      return { assets: '', body, size };
    }
    const type = this.objectType(spec.type);
    const v = type.variants[spec.variant];
    const mass = spec.mass ?? v.mass;
    const inertia = v.inertia.map((I) => I * (mass / v.mass));
    const c = this.boxCentre;
    const assets = [];
    const geoms = [];
    type.parts.forEach((part, p) => {
      const verts = new Float64Array(part.v.length);
      for (let i = 0; i < part.v.length; i++) verts[i] = c[i % 3] + v.ratio * (part.v[i] - c[i % 3]);
      assets.push(`<mesh name="object_p${p}" vertex="${fmt(verts)}" face="${part.f.join(' ')}"/>`);
      geoms.push(`<geom name="object_p${p}" type="mesh" mesh="object_p${p}" ${common}/>`);
    });
    const body = `<body name="object" pos="0 0 1"><freejoint name="object"/>`
      + `<inertial pos="${fmt(v.ipos)}" quat="${fmt(v.iquat)}" mass="${String(+mass)}" diaginertia="${fmt(inertia)}"/>`
      + geoms.join('') + '</body>';
    return { assets: assets.join(''), body, size: v.box_size.slice() };
  }

  build(spec) {
    const { assets, body, size } = this.objectXml(spec);
    const xml = this.template.replace('<!--OBJECT_ASSETS-->', assets).replace('<!--OBJECT_BODY-->', body);
    const model = this.mj.MjModel.from_xml_string(xml, this.vfs);
    if (this.data) this.data.delete();
    if (this.model) this.model.delete();
    this.model = model;
    this.data = new this.mj.MjData(model);
    this.spec = spec;
    this.size = size;
    const m = model;
    const names = this.meta.joints.names;
    const jid = (n) => this.id('JOINT', n);
    this.qadr = names.map((n) => m.jnt_qposadr[jid(n)]);
    this.vadr = names.map((n) => m.jnt_dofadr[jid(n)]);
    this.act = names.map((n) => this.id('ACTUATOR', n));
    this.objBody = this.id('BODY', 'object');
    this.pelvisBody = this.id('BODY', 'pelvis');
    const jo = jid('object');
    this.oq = m.jnt_qposadr[jo];
    this.ov = m.jnt_dofadr[jo];
    this.palms = PALM_SITES.map((s) => this.id('SITE', s));
    this.cornersBody = [];
    for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
      const s = [x * size[0] / 2, y * size[1] / 2, z * size[2] / 2];
      const A = this.boxAxes;
      this.cornersBody.push([
        this.boxCentre[0] + s[0] * A[0] + s[1] * A[1] + s[2] * A[2],
        this.boxCentre[1] + s[0] * A[3] + s[1] * A[4] + s[2] * A[5],
        this.boxCentre[2] + s[0] * A[6] + s[1] * A[7] + s[2] * A[8],
      ]);
    }
    return model;
  }

  // Object pose (body origin, quaternion) resting on the floor at ``xy`` with ``yaw``.
  restingPose(xy, yaw) {
    if (this.spec.kind === 'box') {
      // The benchmark's place_box: vertical axis up, box axis 0 at ``yaw``.
      const b = this.meta.box;
      const va = b.vertical_axis, vs = b.vertical_sign;
      const horizontal = [0, 1, 2].filter((a) => a !== va);
      const cols = [null, null, null];
      cols[va] = [0, 0, vs];
      cols[horizontal[0]] = [Math.cos(yaw), Math.sin(yaw), 0];
      const miss = horizontal[1];
      const a = cols[(miss + 1) % 3], c = cols[(miss + 2) % 3];
      cols[miss] = [a[1] * c[2] - a[2] * c[1], a[2] * c[0] - a[0] * c[2], a[0] * c[1] - a[1] * c[0]];
      const F = new Float64Array(9); // face axes, row-major with columns = cols
      for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) F[i * 3 + k] = cols[k][i];
      const t = [xy[0], xy[1], this.size[va] / 2 + 1e-4];
      // body_w = face_w @ inv(body_from_face): R = F A^T, p = t - R c
      const A = this.boxAxes;
      const R = new Float64Array(9);
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        R[i * 3 + j] = F[i * 3] * A[j * 3] + F[i * 3 + 1] * A[j * 3 + 1] + F[i * 3 + 2] * A[j * 3 + 2];
      }
      const rc = mulVec(R, this.boxCentre);
      return { pos: [t[0] - rc[0], t[1] - rc[1], t[2] - rc[2]], quat: matToQuat(R) };
    }
    const v = this.objectType(this.spec.type).variants[this.spec.variant];
    return { pos: [xy[0], xy[1], v.rest_height + 1e-4], quat: quatMul(yawQuat(yaw), v.rest_quat) };
  }

  reset({ robotXY = [0, 0], robotYaw = 0, objectXY = [1.5, 0], objectYaw = 0, goalXY = null } = {}) {
    const m = this.model, d = this.data;
    this.mj.mj_resetData(m, d);
    const qpos = d.qpos;
    qpos[0] = robotXY[0]; qpos[1] = robotXY[1]; qpos[2] = this.meta.standing_pelvis_height_m + 0.002;
    qpos.set(yawQuat(robotYaw), 3);
    this.qadr.forEach((a, i) => { qpos[a] = this.defaultPos[i]; });
    const pose = this.restingPose(objectXY, objectYaw);
    qpos.set(pose.pos, this.oq);
    qpos.set(pose.quat, this.oq + 3);
    d.qvel.fill(0, this.ov, this.ov + 6);
    this.mj.mj_forward(m, d);
    this.goal = goalXY ? [goalXY[0], goalXY[1], 0] : null;
    this.action = new Float64Array(this.defaultPos.length);
    this.history = null;
    this.ticks = 0;
  }

  // -- observation ------------------------------------------------------------------
  terms() {
    const m = this.model, d = this.data;
    const ob = this.objBody;
    const xpos = d.xpos, xmat = d.xmat;
    const boxPos = [xpos[3 * ob], xpos[3 * ob + 1], xpos[3 * ob + 2]]; // the last step's kinematics
    const boxRot = Float64Array.from(xmat.subarray(9 * ob, 9 * ob + 9));
    const qvel = d.qvel;
    const v = Array.from(qvel.subarray(this.ov, this.ov + 6));
    this.mj.mj_kinematics(m, d); // palms from the current joints
    const qpos = d.qpos;
    const pelvisPos = [qpos[0], qpos[1], qpos[2]];
    const R = quatToMat([qpos[3], qpos[4], qpos[5], qpos[6]]);
    const sx = d.site_xpos;
    const palmsB = this.palms.map((s) => mulTVec(R, [
      sx[3 * s] - pelvisPos[0], sx[3 * s + 1] - pelvisPos[1], sx[3 * s + 2] - pelvisPos[2]]));
    const boxB = mulTVec(R, [boxPos[0] - pelvisPos[0], boxPos[1] - pelvisPos[1], boxPos[2] - pelvisPos[2]]);
    const rot = mulTMat(R, boxRot);
    const corners = [];
    for (const c of this.cornersBody) {
      const r = mulVec(rot, c);
      corners.push(r[0] + boxB[0], r[1] + boxB[1], r[2] + boxB[2]);
    }
    const fields = {
      object_position: boxB,
      object_orientation: [rot[0], rot[1], rot[3], rot[4], rot[6], rot[7]],
      object_velocity: mulTVec(R, v.slice(0, 3)),
      object_angular_velocity: mulTVec(R, mulVec(boxRot, v.slice(3, 6))),
      palm_to_object: [boxB[0] - palmsB[0][0], boxB[1] - palmsB[0][1], boxB[2] - palmsB[0][2],
        boxB[0] - palmsB[1][0], boxB[1] - palmsB[1][1], boxB[2] - palmsB[1][2]],
      box_corners: corners,
    };
    if (this.goal) {
      const g = mulTVec(R, [this.goal[0] - pelvisPos[0], this.goal[1] - pelvisPos[1], this.goal[2] - pelvisPos[2]]);
      fields.goal_position = g;
      fields.box_to_goal = [g[0] - boxB[0], g[1] - boxB[1], g[2] - boxB[2]];
    }
    const values = {};
    for (const term of this.meta.terms) {
      let val;
      switch (term.func) {
        case 'builtin_sensor': val = [qvel[3], qvel[4], qvel[5]]; break;
        case 'projected_gravity': val = mulTVec(R, [0, 0, -1]); break;
        case 'joint_pos_rel': val = this.qadr.map((a, i) => qpos[a] - this.defaultPos[i]); break;
        case 'joint_vel_rel': val = this.vadr.map((a) => qvel[a]); break;
        case 'last_action': val = Array.from(this.action); break;
        case 'root_height': val = [qpos[2]]; break;
        case 'hand_positions_b': val = [...palmsB[0], ...palmsB[1]]; break;
        default: val = fields[term.field];
      }
      if (!val || val.length !== term.dim) throw new Error(`term ${term.name}: got ${val && val.length}, want ${term.dim}`);
      values[term.name] = val;
    }
    return values;
  }

  observation() {
    const values = this.terms();
    const terms = this.meta.terms;
    if (!this.history) {
      this.history = {};
      for (const t of terms) {
        const h = [];
        for (let k = 0; k < t.history; k++) h.push(values[t.name].slice());
        this.history[t.name] = h;
      }
    } else {
      for (const t of terms) {
        const h = this.history[t.name];
        h.shift();
        h.push(values[t.name].slice());
      }
    }
    const obs = new Float32Array(this.net.obsDim);
    let off = 0;
    for (const t of terms) for (const row of this.history[t.name]) for (const x of row) obs[off++] = x;
    return obs;
  }

  // -- control ----------------------------------------------------------------------
  // One 50 Hz control step. ``beforePhysics`` runs before every 5 ms physics step
  // (the page applies mouse forces there).
  step(beforePhysics = null) {
    const m = this.model, d = this.data;
    const obs = this.observation();
    const a = this.net.act(obs);
    for (let i = 0; i < a.length; i++) this.action[i] = a[i];
    const n = this.defaultPos.length;
    const target = new Float64Array(n);
    for (let i = 0; i < n; i++) target[i] = this.defaultPos[i] + this.actionScale[i] * this.action[i];
    for (let k = 0; k < DECIMATION; k++) {
      const qpos = d.qpos, qvel = d.qvel, ctrl = d.ctrl;
      for (let i = 0; i < n; i++) {
        const tau = this.kp[i] * (target[i] - qpos[this.qadr[i]]) - this.kd[i] * qvel[this.vadr[i]];
        ctrl[this.act[i]] = Math.min(Math.max(tau, -this.effort[i]), this.effort[i]);
      }
      if (beforePhysics) beforePhysics(m, d);
      this.mj.mj_step(m, d);
    }
    this.ticks += 1;
    return obs;
  }
}
