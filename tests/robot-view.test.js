// Checks for interactive/robot-view. Run: node --test tests/*.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const RV = require('../interactive/robot-view/model.js');

const dir = path.join(__dirname, '..', 'interactive', 'robot-view');
function loadGlobal(file) {
  const ctx = { window: {} };
  vm.runInNewContext(fs.readFileSync(path.join(dir, file), 'utf8'), ctx);
  return ctx.window;
}
const DATA = {}, MODELS = {};
for (const ds of ['aloha', 'reassemble']) DATA[ds] = RV.decodeAll(loadGlobal(`data/${ds}.js`).ROBOT_VIEW_DATA[ds]);
for (const m of ['aloha', 'panda']) MODELS[m] = loadGlobal(`models/${m}.js`).ROBOT_MODELS[m];

test('decoded arrays reproduce the exported values exactly (readout shows these numbers)', () => {
  for (const ds in DATA) {
    const D = DATA[ds], c = D.meta.check, T = D.meta.T;
    assert.deepEqual(Array.from(RV.row(D.truth, 0, 0)), [...c['truth[0,0]']]);
    assert.deepEqual(Array.from(RV.row(D.q50, 3, 10)), [...c['q50[3,10]']]);
    assert.deepEqual(Array.from(RV.row(D.q95, 10, T - 1)), [...c['q95[10,-1]']]);
    assert.deepEqual(Array.from(RV.row(D.draws, 1, 2, 5)), [...c['draws[1,2,5]']]);
  }
});

test('bundle shapes and window rule', () => {
  for (const ds in DATA) {
    const D = DATA[ds], M = D.meta;
    assert.deepEqual([...D.truth.shape], [11, M.T, M.D]);
    assert.deepEqual([...D.draws.shape], [11, M.draws_exported, M.T, M.D]);
    assert.equal(M.labels.at(-1), 'worst');
    const r = Array.from(D.rmse.data);
    for (let i = 1; i < 10; i++) assert.ok(r[i] >= r[i - 1], 'quantile windows sorted by RMSE');
    assert.ok(r[10] >= r[9]);
    for (let w = 0; w < 11; w++) for (let t = 0; t < M.T; t += 17) {
      const lo = RV.row(D.q05, w, t), hi = RV.row(D.q95, w, t), md = RV.row(D.q50, w, t);
      for (let j = 0; j < M.D; j++) assert.ok(lo[j] <= md[j] && md[j] <= hi[j]);
    }
  }
});

test('every state dimension drives a joint that exists in the robot model', () => {
  for (const ds of ['aloha', 'reassemble']) {
    const model = MODELS[RV.ROBOTS[ds].model];
    const joints = new Set(model.bodies.filter(b => b.joint).map(b => b.joint.name));
    const bodies = new Set(model.bodies.map(b => b.name));
    const q = Array.from(RV.row(DATA[ds].truth, 0, 0));
    for (const name of Object.keys(RV.jointTargets(ds, q))) assert.ok(joints.has(name), name);
    for (let j = 0; j < q.length; j++) for (const b of RV.jointBodies(ds, j)) assert.ok(bodies.has(b), b);
    for (const [b] of RV.ROBOTS[ds].ee) assert.ok(bodies.has(b), b);
  }
});

test('ALOHA gripper: normalised value maps back to the simulator finger position', () => {
  const g = RV.ALOHA_GRIPPER, q = new Array(14).fill(0);
  q[6] = 0.5; q[13] = 1;
  const t = RV.jointTargets('aloha', q);
  assert.ok(Math.abs(t['vx300s_left/left_finger'] - (g.CLOSE + 0.5 * (g.OPEN - g.CLOSE))) < 1e-12);
  assert.equal(t['vx300s_left/right_finger'], -t['vx300s_left/left_finger']);
  assert.equal(t['vx300s_right/left_finger'], g.MAX);   // clipped to the joint range
});

test('float16 decoding', () => {
  assert.equal(RV.halfToFloat(0x3c00), 1);
  assert.equal(RV.halfToFloat(0xc000), -2);
  assert.equal(RV.halfToFloat(0x3555), 0.333251953125);
});

test('model files stay within the size budget (<= 4 MB per robot)', () => {
  for (const m of ['aloha', 'panda']) assert.ok(fs.statSync(path.join(dir, 'models', m + '.js')).size <= 4e6);
});
