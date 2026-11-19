/* Model for the energy-score figure. Pure functions, no DOM.
 *
 * Toy setting: reference x ~ N(0, I_d); ensemble members z_m = c * eps_m, eps_m ~ N(0, I_d) independent of x.
 * Then z - x ~ N(0, (1 + c^2) I_d) and z - z' ~ N(0, 2 c^2 I_d), and every expectation below is CLOSED FORM via the
 * chi mean  E||N(0, s^2 I_d)|| = s * kappa_d,  kappa_d = sqrt(2) * Gamma((d+1)/2) / Gamma(d/2).
 * No Monte Carlo is used on the page; the tests cross-check the closed form against seeded Monte Carlo. */
(function (root) {
  'use strict';

  const BETA = 0.5;  // proper energy score, Equation (eq:flow_sde_rollout_loss): E||z-x|| - 1/2 E||z-z'||

  /** kappa_d = E||N(0, I_d)|| by the exact recursion r(d+1) = (d/2) / r(d), r(d) = Gamma((d+1)/2)/Gamma(d/2). */
  function kappa(d) {
    let r = 1 / Math.sqrt(Math.PI);   // r(1) = Gamma(1)/Gamma(1/2)
    for (let k = 1; k < d; k++) r = (k / 2) / r;
    return Math.SQRT2 * r;
  }

  /** Factor of the expected spread estimate relative to E||z-z'||.
   *  'unbiased': 1/(M(M-1)) sum_{i != j} (the paper's Equation eq:energy-score-estimator) -> 1.
   *  'vstat':    1/M^2 sum_{i,j} (includes the zero self-pairs)                          -> (M-1)/M. */
  function spreadFactor(estimator, M) { return estimator === 'vstat' ? (M - 1) / M : 1; }

  /** All curves at spread c. Returns accuracy E||z-x||, spread E||z-z'||, the weighted spread term
   *  1/2 * factor * E||z-z'||, the expected score ES = accuracy - spread term, and the entropy of N(0, c^2 I_d). */
  function terms(c, d, opts) {
    const o = opts || {};
    const k = kappa(d), f = spreadFactor(o.estimator, o.M || 16);
    const acc = k * Math.sqrt(1 + c * c);
    const spread = k * Math.SQRT2 * c;
    const half = BETA * f * spread;
    return { acc, spread, half, es: acc - half, entropy: entropy(c, d) };
  }
  /** Differential entropy of N(0, c^2 I_d): d log c + (d/2) log(2 pi e). Increasing in c: on its own it is
   *  minimised by shrinking the spread (the paper: "this term alone could shrink the noise almost to zero"). */
  function entropy(c, d) { return d * Math.log(c) + 0.5 * d * Math.log(2 * Math.PI * Math.E); }

  /** argmin_c ES. d/dc ES = kappa_d (c/sqrt(1+c^2) - sqrt(2) beta f) = 0  =>  c* = k/sqrt(1-k^2), k = sqrt(2) beta f.
   *  Independent of d: kappa_d only scales the curve. */
  function argminC(opts) {
    const o = opts || {};
    const k = Math.SQRT2 * BETA * spreadFactor(o.estimator, o.M || 16);
    return k / Math.sqrt(1 - k * k);
  }

  /** Learned scale: a miscalibrated spread c0 multiplied by s. ES(s * c0) is minimised at s* = c* / c0. */
  function argminS(c0, opts) { return argminC(opts) / c0; }

  /** Numerical argmin on a log grid (used by the tests to check the closed forms). */
  function argminGrid(fn, lo, hi, n) {
    n = n || 200001;
    let best = Infinity, arg = lo;
    const a = Math.log(lo), b = Math.log(hi);
    for (let i = 0; i < n; i++) {
      const c = Math.exp(a + (b - a) * i / (n - 1)), v = fn(c);
      if (v < best) { best = v; arg = c; }
    }
    return arg;
  }

  const C_RANGE = Object.freeze([0.2, 3]);
  const DIMS = Object.freeze([
    { d: 1, label: '1', aria: 'dimension 1' },
    { d: 2, label: '2 <span class="dl">pendulum</span>', aria: 'dimension 2, pendulum' },
    { d: 7, label: '7 <span class="dl">REASSEMBLE</span>', aria: 'dimension 7, REASSEMBLE' },
    { d: 12, label: '12 <span class="dl">Lorenz-96</span>', aria: 'dimension 12, Lorenz-96' },
    { d: 14, label: '14 <span class="dl">ALOHA</span>', aria: 'dimension 14, ALOHA' },
  ]);
  const MS = Object.freeze([4, 16, 50, 200]);   // 4: stage-1 rollout draws, 16: stage-2 rollouts (paper, App. C)
  const C0S = Object.freeze([0.6, 1.6]);
  const DEFAULTS = Object.freeze({ c: 1, d: 7, M: 16, estimator: 'unbiased', c0: 0.6, s: 1 });

  const ES = { BETA, kappa, spreadFactor, terms, entropy, argminC, argminS, argminGrid, C_RANGE, DIMS, MS, C0S, DEFAULTS };
  if (typeof module !== 'undefined' && module.exports) module.exports = ES;
  else root.ES = ES;
})(typeof self !== 'undefined' ? self : this);
