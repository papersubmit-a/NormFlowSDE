// Checks for interactive/overview (moving observation window of Figure 1). Run: node --test tests/*.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const OV = require('../interactive/overview/model.js');

test('window holds the last o measurements and ends at t (causal)', () => {
  for (let k = OV.KMIN; k <= OV.KMAX; k++) {
    const w = OV.windowCols(k);
    assert.equal(w.length, OV.O);
    assert.equal(w[w.length - 1], k);
    assert.ok(w[0] >= 0);
    for (let j = 0; j < OV.N; j++) {
      const r = OV.role(j, k);
      assert.equal(r === 'window', w.includes(j));
      assert.equal(r === 'future', j > k);
    }
  }
});

test('window moves forward one step with t', () => {
  for (let k = OV.KMIN; k < OV.KMAX; k++) assert.deepEqual(OV.windowCols(k + 1), OV.windowCols(k).map(j => j + 1));
});

test('time axis of panel (c) spans the paper figure, and the default box sits near the paper position', () => {
  assert.equal(OV.timeX(0), 1040);
  assert.equal(OV.timeX(OV.N), 1570);
  assert.ok(Math.abs(OV.timeX(OV.DEFAULT_K) - 1314.5) < OV.DX);
});

test('box (b) encloses the 90 % band over [t, t + dt] (paper SVG)', () => {
  const svg = fs.readFileSync(path.join(__dirname, '..', 'paper', 'figures', 'NormFlowSDE_figure.svg'), 'utf8');
  const c = svg.slice(svg.indexOf('id="panel-c-posterior"'));
  const band = OV.parsePoints(/<polygon points="([^"]*)"/.exec(c)[1]);
  const half = band.length / 2, upper = band.slice(0, half), lower = band.slice(half).reverse();
  for (let k = OV.KMIN; k <= OV.KMAX; k++) {
    const b = OV.boxB(k, upper, lower);
    for (let i = 0; i <= 4; i++) {
      const x = b.x + b.width * i / 4;
      assert.ok(OV.interp(upper, x) >= b.y && OV.interp(lower, x) <= b.y + b.height, `k=${k}`);
    }
  }
});

test('context shades and measurements are deterministic', () => {
  assert.deepEqual(OV.contextShades(5), OV.contextShades(5));
  assert.notDeepEqual(OV.contextShades(5), OV.contextShades(6));
  assert.deepEqual(OV.measurements(), OV.measurements());
});

test('embedded SVG is the paper figure, unchanged', () => {
  const js = fs.readFileSync(path.join(__dirname, '..', 'interactive', 'overview', 'figure-svg.js'), 'utf8');
  const svg = JSON.parse(js.slice(js.indexOf('= ') + 2, js.lastIndexOf(';')));
  const paper = fs.readFileSync(path.join(__dirname, '..', 'paper', 'figures', 'NormFlowSDE_figure.svg'), 'utf8').trim();
  assert.equal(svg, paper, 'run python scripts/sync_overview_figure.py');
});

test('panel (b) at the default step reproduces the paper drawing', () => {
  const r = OV.panelB(OV.stepParams(OV.DEFAULT_K, 1));
  const want = [[0.6, 255, 16], [0.4, 345, 14]];
  r.mixture.forEach((c, i) => [c.w, c.mu, c.sd].forEach((v, j) => assert.ok(Math.abs(v - want[i][j]) < 1e-9, `${i},${j}: ${v}`)));
  assert.ok(Math.abs(r.drift - 312) < 1e-9);
  assert.equal(r.paths.length, 16);   // as many sample paths as the paper drawing
  for (const p of r.paths) { assert.deepEqual(p[0], [520, 392]); assert.equal(p.at(-1)[0], 790); }
});

test('panel (b) changes with t, continuously, and stays inside the drawing', () => {
  const a = OV.panelB(OV.stepParams(4, 0.8)), b = OV.panelB(OV.stepParams(15, 1.3));
  assert.notDeepEqual(a.ends, b.ends);
  for (let k = OV.KMIN; k < OV.KMAX; k += 0.25) {
    const p = OV.stepParams(k, 1), q = OV.stepParams(k + 0.25, 1);
    p.forEach((v, i) => assert.ok(Math.abs(v - q[i]) < 0.2 * Math.abs(v) + 6, `param ${i} jumps at k=${k}`));
    const r = OV.panelB(p);
    for (const c of r.mixture) assert.ok(c.mu >= OV.B.ymin && c.mu <= OV.B.ymax && c.w > 0 && c.sd > 0);
  }
});

test('panel (b) has one sample path and end point per path in the paper SVG', () => {
  const svg = fs.readFileSync(path.join(__dirname, '..', 'paper', 'figures', 'NormFlowSDE_figure.svg'), 'utf8');
  const b = svg.slice(svg.indexOf('id="panel-b-one-step"'), svg.indexOf('id="panel-c-posterior"'));
  assert.equal((b.match(/<polyline points="520\.0,392\.0[^>]*stroke-width="0\.9"/g) || []).length, OV.B.nSamples);
  assert.equal((b.match(/<circle [^>]*r="2\.3"/g) || []).length, OV.B.nSamples);
});
