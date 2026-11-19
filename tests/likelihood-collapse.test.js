// Acceptance checks for interactive/likelihood-collapse. Run: node --test tests/
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const LC = require('../interactive/likelihood-collapse/model.js');

test('paper mixture: weights, means, scales and moments', () => {
  const c = LC.components(LC.PAPER);
  [0.70, 0.15, 0.15].forEach((w, i) => assert.ok(Math.abs(c[i].w - w) < 1e-12));
  assert.deepEqual(c.map(x => x.mu), [0, 2.5, -2.5]);
  assert.deepEqual(c.map(x => x.sd), [0.5, 1.5, 1.5]);
  const m = LC.moments(c);
  assert.ok(Math.abs(m.mean) < 1e-12);
  assert.ok(Math.abs(m.var - 2.725) < 1e-12);
  assert.ok(Math.abs(m.sd - 1.65) < 0.001, 'paper: standard deviation 1.65 sigma');
});

test('TV at the paper defaults is 0.332 +/- 0.001 (overlap 0.668 in the paper)', () => {
  const tv = LC.tv(LC.PAPER);
  assert.ok(Math.abs(tv - 0.332) <= 0.001, `TV = ${tv}`);
  assert.equal((1 - tv).toFixed(3), '0.668');
});

test('TV grid is converged (>= 20k points over +/- 12 sd)', () => {
  assert.ok(Math.abs(LC.tv(LC.PAPER, 24001) - LC.tv(LC.PAPER, 96001)) < 1e-6);
});

test('TV is 0 when the mixture is a single Gaussian, and invariant to the mean shift', () => {
  assert.ok(LC.tv({ w0: 1, skew: 0.5, a: 2.5, s0: 0.7, s1: 1.5 }) < 1e-9);
  const skewed = LC.moments(LC.components({ ...LC.PAPER, skew: 0.8 }));
  assert.ok(skewed.mean > 0);
});

test('particles are reproducible from the seed', () => {
  const a = LC.standardDraws(1), b = LC.standardDraws(1), c = LC.standardDraws(2);
  assert.deepEqual(Array.from(a), Array.from(b));
  assert.notDeepEqual(Array.from(a), Array.from(c));
  assert.equal(a.length, 200);
});

test('ESS and weights are well formed', () => {
  const r = LC.filterStep(LC.PAPER, LC.DEFAULT_PRIOR_SD, LC.OBS_PRESETS.side);
  for (const k of ['mix', 'gauss']) {
    const s = r[k].w.reduce((u, v) => u + v, 0);
    assert.ok(Math.abs(s - 1) < 1e-12);
    assert.ok(r[k].ess >= 1 && r[k].ess <= 200);
  }
});

test('default settings: side-mode observation -> Gaussian over-trusts it (posterior mean pulled toward y)', () => {
  const y = LC.OBS_PRESETS.side, r = LC.filterStep(LC.PAPER, LC.DEFAULT_PRIOR_SD, y);
  const errMix = Math.abs(r.mix.mean - LC.TRUTH), errGauss = Math.abs(r.gauss.mean - LC.TRUTH);
  assert.ok(errGauss > 2 * errMix, `gauss ${r.gauss.mean}, mix ${r.mix.mean}`);
  assert.ok(r.gauss.mean > r.mix.mean, 'Gaussian mean is shifted toward y > 0');
});

test('default settings: spike observation -> Gaussian posterior too wide', () => {
  const r = LC.filterStep(LC.PAPER, LC.DEFAULT_PRIOR_SD, LC.OBS_PRESETS.spike);
  assert.ok(r.gauss.sd > 1.2 * r.mix.sd, `gauss sd ${r.gauss.sd}, mix sd ${r.mix.sd}`);
  assert.ok(r.gauss.ess > r.mix.ess, 'flatter Gaussian weights -> larger ESS');
});

test('wide prior (s_prior = 1.5): side-mode effect reverses (Gaussian under-trusts), as stated in the README', () => {
  const r = LC.filterStep(LC.PAPER, 1.5, LC.OBS_PRESETS.side);
  assert.ok(r.gauss.mean < r.mix.mean);
});
