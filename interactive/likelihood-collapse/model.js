/* Model for the likelihood-collapse figure. Pure functions, no DOM; shared by the page, the Node tests and the
 * static-fallback export. All noise quantities are in units of the noise scale sigma (the paper's Equation obsnoise). */
(function (root) {
  'use strict';
  const IX = (typeof module !== 'undefined' && module.exports) ? require('../_shared/ix.js') : root.IX;

  /** Paper values: V ~ 0.70 N(0,(0.5s)^2) + 0.15 N(2.5s,(1.5s)^2) + 0.15 N(-2.5s,(1.5s)^2), per observed coordinate.
   *  w0: centre weight; skew: share of the side weight at +a (0.5 = symmetric, as in the paper);
   *  a: side-mode offset; s0: centre scale; s1: side scale. */
  const PAPER = Object.freeze({ w0: 0.70, skew: 0.5, a: 2.5, s0: 0.5, s1: 1.5 });

  function components(p) {
    const side = 1 - p.w0;
    return [
      { w: p.w0, mu: 0, sd: p.s0 },
      { w: side * p.skew, mu: p.a, sd: p.s1 },
      { w: side * (1 - p.skew), mu: -p.a, sd: p.s1 },
    ].filter(c => c.w > 0);
  }
  const SQ2PI = Math.sqrt(2 * Math.PI);
  function normPdf(x, mu, sd) { const z = (x - mu) / sd; return Math.exp(-0.5 * z * z) / (sd * SQ2PI); }
  function mixPdf(comps, x) { let s = 0; for (const c of comps) s += c.w * normPdf(x, c.mu, c.sd); return s; }

  /** Analytic mean and variance of the mixture (= parameters of the moment-matched Gaussian). */
  function moments(comps) {
    let m = 0, m2 = 0;
    for (const c of comps) { m += c.w * c.mu; m2 += c.w * (c.sd * c.sd + c.mu * c.mu); }
    return { mean: m, var: m2 - m * m, sd: Math.sqrt(m2 - m * m) };
  }

  /** TV(p, q) = 0.5 * int |p - q|, trapezoid rule on n points over mean +/- 12 total std (n >= 20k). */
  function tv(p, n) {
    n = n || 24001;
    const comps = components(p), mo = moments(comps);
    const xs = IX.linspace(mo.mean - 12 * mo.sd, mo.mean + 12 * mo.sd, n), d = new Float64Array(n);
    for (let i = 0; i < n; i++) d[i] = Math.abs(mixPdf(comps, xs[i]) - normPdf(xs[i], mo.mean, mo.sd));
    return 0.5 * IX.trapz(d, xs);
  }

  // ------------------------------------------------------------------ Panel B: one weighting step
  const TRUTH = 0;                 // true state x* (the prior is centred on it)
  const N_PARTICLES = 200;
  const DEFAULT_PRIOR_SD = 0.5;    // see README: side-mode over-trust is visible for prior sd <= 0.75
  const DEFAULT_SEED = 1;
  const OBS_PRESETS = Object.freeze({ spike: 0.3, side: 2.5 });  // y = x* + e with e in the spike / a side mode

  /** N particles x_i = s_prior * eps_i, eps_i ~ N(0,1) from mulberry32(seed). Standardised draws are returned so
   *  that changing the prior width rescales the same cloud. */
  function standardDraws(seed, n) {
    const rng = IX.mulberry32(seed), out = new Float64Array(n || N_PARTICLES);
    for (let i = 0; i < out.length; i++) out[i] = IX.randn(rng);
    return out;
  }

  /** Normalised bootstrap weights w_i ∝ lik(y - x_i), with summaries. */
  function weigh(particles, lik) {
    const n = particles.length, w = new Float64Array(n);
    let s = 0;
    for (let i = 0; i < n; i++) { w[i] = lik(particles[i]); s += w[i]; }
    let mean = 0, m2 = 0, ss = 0;
    for (let i = 0; i < n; i++) { w[i] /= s; mean += w[i] * particles[i]; ss += w[i] * w[i]; }
    for (let i = 0; i < n; i++) m2 += w[i] * (particles[i] - mean) ** 2;
    return { w, mean, sd: Math.sqrt(m2), ess: 1 / ss };
  }

  /** Full Panel-B state: particles and weights under the mixture and under the moment-matched Gaussian. */
  function filterStep(p, priorSd, y, seed) {
    const comps = components(p), mo = moments(comps);
    const eps = standardDraws(seed === undefined ? DEFAULT_SEED : seed);
    const x = Array.from(eps, e => TRUTH + priorSd * e);
    const mix = weigh(x, xi => mixPdf(comps, y - xi));
    const gauss = weigh(x, xi => normPdf(y - xi, mo.mean, mo.sd));
    return { x, mix, gauss, comps, moments: mo };
  }

  const LC = { PAPER, components, normPdf, mixPdf, moments, tv, TRUTH, N_PARTICLES, DEFAULT_PRIOR_SD, DEFAULT_SEED,
    OBS_PRESETS, standardDraws, weigh, filterStep };
  if (typeof module !== 'undefined' && module.exports) module.exports = LC;
  else root.LC = LC;
})(typeof self !== 'undefined' ? self : this);
