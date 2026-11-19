// Prints the default state of an interactive figure as JSON, computed by the same model code the page runs.
// Used by scripts/make_fallbacks.py so that the static fallbacks show exactly the page's default numbers.
// Usage: node scripts/dump_figure_state.js <slug>
'use strict';
const path = require('path');
const slug = process.argv[2];
const root = path.join(__dirname, '..', 'interactive');
const out = {};

if (slug === 'likelihood-collapse') {
  const LC = require(path.join(root, slug, 'model.js'));
  const comps = LC.components(LC.PAPER), mo = LC.moments(comps);
  const xs = [];
  for (let i = 0; i <= 800; i++) xs.push(-8 + 16 * i / 800);
  const y = LC.OBS_PRESETS.side, r = LC.filterStep(LC.PAPER, LC.DEFAULT_PRIOR_SD, y, LC.DEFAULT_SEED);
  const L = Math.max(4.5, 4 * LC.DEFAULT_PRIOR_SD), xl = [];
  for (let i = 0; i <= 600; i++) xl.push(-L + 2 * L * i / 600);
  Object.assign(out, {
    a: { e: xs, p: xs.map(x => LC.mixPdf(comps, x)), q: xs.map(x => LC.normPdf(x, mo.mean, mo.sd)),
      tv: LC.tv(LC.PAPER), mean: mo.mean, var: mo.var },
    b: { y, truth: LC.TRUTH, priorSd: LC.DEFAULT_PRIOR_SD, seed: LC.DEFAULT_SEED, L, x: r.x,
      wMix: Array.from(r.mix.w), wGauss: Array.from(r.gauss.w),
      mix: { mean: r.mix.mean, sd: r.mix.sd, ess: r.mix.ess }, gauss: { mean: r.gauss.mean, sd: r.gauss.sd, ess: r.gauss.ess },
      xl, likMix: xl.map(x => LC.mixPdf(comps, y - x)), likGauss: xl.map(x => LC.normPdf(y - x, mo.mean, mo.sd)) },
  });
} else if (slug === 'energy-score') {
  const ES = require(path.join(root, slug, 'model.js'));
  const D = ES.DEFAULTS, cs = [];
  for (let i = 0; i <= 300; i++) cs.push(ES.C_RANGE[0] * Math.pow(ES.C_RANGE[1] / ES.C_RANGE[0], i / 300));
  const T = cs.map(c => ES.terms(c, D.d, D));
  Object.assign(out, { d: D.d, M: D.M, c: cs, acc: T.map(t => t.acc), half: T.map(t => t.half), es: T.map(t => t.es),
    entropy: T.map(t => t.entropy), cstar: ES.argminC(D), esStar: ES.terms(ES.argminC(D), D.d, D).es });
} else {
  console.error('unknown slug ' + slug);
  process.exit(1);
}
process.stdout.write(JSON.stringify(out));
