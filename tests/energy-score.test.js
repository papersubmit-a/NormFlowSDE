// Acceptance checks for interactive/energy-score. Run: node --test tests/
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const ES = require('../interactive/energy-score/model.js');
const IX = require('../interactive/_shared/ix.js');

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ''} ${a} vs ${b}`);

test('d = 1 closed forms: E|z-x| = sqrt(2(1+c^2)/pi), E|z-z\'| = 2c/sqrt(pi)', () => {
  for (const c of [0.2, 0.5, 1, 1.7, 3]) {
    const t = ES.terms(c, 1);
    close(t.acc, Math.sqrt(2 * (1 + c * c) / Math.PI), 1e-12);
    close(t.spread, 2 * c / Math.sqrt(Math.PI), 1e-12);
    close(t.half, 0.5 * t.spread, 1e-12);
  }
});

test('kappa_d matches sqrt(2) Gamma((d+1)/2)/Gamma(d/2) (known values)', () => {
  close(ES.kappa(1), Math.sqrt(2 / Math.PI), 1e-14);
  close(ES.kappa(2), Math.sqrt(Math.PI / 2), 1e-14);
  close(ES.kappa(3), 2 * Math.sqrt(2 / Math.PI), 1e-14);
});

test('d = 1, beta = 0.5: argmin c = 1.000 (+/- 0.005), closed form and grid', () => {
  close(ES.argminC(), 1, 1e-12);
  close(ES.argminGrid(c => ES.terms(c, 1).es, 0.2, 3), 1, 0.005);
});

test('argmin c is independent of d (reported in the README)', () => {
  for (const d of [1, 2, 7, 12, 14]) close(ES.argminGrid(c => ES.terms(c, d).es, 0.2, 3), 1, 0.005, `d=${d}`);
});

test('1/M^2 estimator: expected spread is (M-1)/M times smaller and the optimum shrinks', () => {
  const o = { estimator: 'vstat', M: 4 };
  close(ES.terms(1, 7, o).half, 0.75 * ES.terms(1, 7).half, 1e-12);
  const g = ES.argminGrid(c => ES.terms(c, 7, o).es, 0.2, 3);
  close(g, ES.argminC(o), 0.005);
  assert.ok(ES.argminC(o) < 1);
});

test('entropy term is increasing in c (no interior minimum)', () => {
  for (const d of [1, 7, 14]) {
    let prev = -Infinity;
    for (let c = 0.2; c <= 3; c += 0.01) { const h = ES.entropy(c, d); assert.ok(h > prev); prev = h; }
  }
  close(ES.entropy(1, 1), 0.5 * Math.log(2 * Math.PI * Math.E), 1e-12);
});

test('learned scale: ES of s * c0 is minimised at s = 1/c0', () => {
  for (const c0 of ES.C0S) {
    close(ES.argminS(c0), 1 / c0, 1e-12);
    close(ES.argminGrid(s => ES.terms(s * c0, 7).es, 0.2, 3), 1 / c0, 0.005, `c0=${c0}`);
  }
});

test('closed form agrees with seeded Monte Carlo (d = 7, 20k samples)', () => {
  const rng = IX.mulberry32(7), d = 7, n = 20000, c = 1.3;
  let acc = 0, spr = 0;
  for (let i = 0; i < n; i++) {
    let a = 0, b = 0;
    for (let k = 0; k < d; k++) {
      const x = IX.randn(rng), z = c * IX.randn(rng), z2 = c * IX.randn(rng);
      a += (z - x) ** 2; b += (z - z2) ** 2;
    }
    acc += Math.sqrt(a); spr += Math.sqrt(b);
  }
  const t = ES.terms(c, d);
  close(acc / n, t.acc, 0.03 * t.acc, 'E||z-x||');
  close(spr / n, t.spread, 0.03 * t.spread, "E||z-z'||");
});
