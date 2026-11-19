/* Robot view: data decoding and the mapping from dataset state vectors to MJCF joints. Pure functions, no DOM/WebGL;
 * shared by app.js and tests/robot-view.test.js. */
(function (root) {
  'use strict';

  // ALOHA simulation (gym-aloha constants.py): the state's gripper value is the finger position normalised between
  // closed and open, (x - CLOSE) / (OPEN - CLOSE); fingers are two slide joints, the right one mirrored.
  const ALOHA_GRIPPER = Object.freeze({ OPEN: 0.05800, CLOSE: 0.01844, MIN: 0.021, MAX: 0.057 });
  const ALOHA_ARM = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate'];
  const ALOHA_LINKS = ['shoulder_link', 'upper_arm_link', 'upper_forearm_link', 'lower_forearm_link', 'wrist_link',
    'gripper_link'];
  const PANDA_FINGER_OPEN = 0.04;   // REASSEMBLE's state has no gripper dimension: fingers drawn open (fixed)

  const ROBOTS = Object.freeze({
    aloha: { model: 'aloha', ee: [['vx300s_left/gripper_link', [0.12, 0, 0]], ['vx300s_right/gripper_link', [0.12, 0, 0]]],
      target: [0, 0.45, 0.2], dist: 1.55 },
    reassemble: { model: 'panda', ee: [['hand', [0, 0, 0.105]]], target: [0.15, 0, 0.42], dist: 1.25 },
  });

  /** state vector q (dataset order, physical units) -> {mjcf joint name: position} */
  function jointTargets(ds, q) {
    const out = {};
    if (ds === 'aloha') {
      ['left', 'right'].forEach((side, s) => {
        const o = 7 * s;
        for (let i = 0; i < 6; i++) out[`vx300s_${side}/${ALOHA_ARM[i]}`] = q[o + i];
        const g = ALOHA_GRIPPER;
        const f = Math.min(g.MAX, Math.max(g.MIN, g.CLOSE + q[o + 6] * (g.OPEN - g.CLOSE)));
        out[`vx300s_${side}/left_finger`] = f;
        out[`vx300s_${side}/right_finger`] = -f;
      });
    } else if (ds === 'reassemble') {
      for (let i = 0; i < 7; i++) out[`joint${i + 1}`] = q[i];
      out.finger_joint1 = PANDA_FINGER_OPEN; out.finger_joint2 = PANDA_FINGER_OPEN;
    } else throw new Error('unknown dataset ' + ds);
    return out;
  }

  /** bodies moved by state dimension j (highlighted when the joint is selected) */
  function jointBodies(ds, j) {
    if (ds === 'aloha') {
      const side = j < 7 ? 'left' : 'right', i = j % 7;
      return i < 6 ? [`vx300s_${side}/${ALOHA_LINKS[i]}`]
        : [`vx300s_${side}/left_finger_link`, `vx300s_${side}/right_finger_link`];
    }
    return [`link${j + 1}`];
  }

  function windowLabel(label) {
    if (label === 'worst') return 'worst';
    const p = parseInt(label.slice(1), 10);
    return p === 0 ? 'best' : p === 50 ? 'median' : `${p}th pct`;
  }

  // ------------------------------------------------------------------ decoding
  function bytesFromB64(b64) {
    if (typeof Buffer !== 'undefined' && typeof atob === 'undefined') return new Uint8Array(Buffer.from(b64, 'base64'));
    const s = atob(b64), out = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out;
  }
  function halfToFloat(h) {
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  /** {dtype: 'f4'|'f2'|'i4', shape, b64} -> {data: typed array, shape} (little endian) */
  function decode(a) {
    const bytes = bytesFromB64(a.b64), buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    let data;
    if (a.dtype === 'f4') data = new Float32Array(buf);
    else if (a.dtype === 'i4') data = new Int32Array(buf);
    else if (a.dtype === 'f2') {
      const h = new Uint16Array(buf);
      data = new Float32Array(h.length);
      for (let i = 0; i < h.length; i++) data[i] = halfToFloat(h[i]);
    } else throw new Error('dtype ' + a.dtype);
    const n = a.shape.reduce((x, y) => x * y, 1);
    if (data.length !== n) throw new Error('shape mismatch');
    return { data, shape: a.shape };
  }
  function decodeAll(bundle) {
    const out = { meta: bundle.meta };
    for (const k in bundle.arrays) out[k] = decode(bundle.arrays[k]);
    return out;
  }
  /** row of the last axis: arr[i0, i1, ..., :] */
  function row(arr, ...idx) {
    const sh = arr.shape, D = sh[sh.length - 1];
    let off = 0;
    for (let a = 0; a < idx.length; a++) off = off * sh[a] + idx[a];
    return arr.data.subarray(off * D, off * D + D);
  }

  const RV = { ALOHA_GRIPPER, PANDA_FINGER_OPEN, ROBOTS, jointTargets, jointBodies, windowLabel, halfToFloat, decode,
    decodeAll, row };
  if (typeof module !== 'undefined' && module.exports) module.exports = RV;
  else root.RV = RV;
})(typeof self !== 'undefined' ? self : this);
