/* Geometry of the interactive overview (paper Figure 1). Pure functions, no DOM; used by app.js and the tests.
 * Coordinates are in the paper SVG's user units (viewBox 0 0 1600 606). The schematic uses a window of o = 4
 * measurements and N = 20 steps; the trained models use o = 10 (Appendix C). */
(function (root) {
  'use strict';
  const IX = (typeof module !== 'undefined' && module.exports) ? require('../_shared/ix.js') : root.IX;

  const N = 20;            // time steps shown in panel (c); measurements y_0 ... y_{N-1}
  const O = 4;             // measurements per window in the schematic (boxed in panel a)
  const X0 = 1040, X1 = 1570, DX = (X1 - X0) / N;   // panel (c) time axis
  const STRIP = { x0: 62, x1: 298, rows: [122, 150, 178] };  // panel (a) measurement strip
  const KMIN = O - 1, KMAX = N - 1;
  const DEFAULT_K = 10;    // box (b) at x = 1305 in (c), close to the paper figure (1314.5)

  /** time position of step j in panel (c) */
  const timeX = j => X0 + DX * j;
  /** x position of measurement column j in panel (a) */
  const stripX = j => STRIP.x0 + (STRIP.x1 - STRIP.x0) * j / (N - 1);
  /** columns of the trailing window y_{t-o+1:t} for time index k */
  function windowCols(k) { const out = []; for (let j = k - O + 1; j <= k; j++) out.push(j); return out; }
  /** 'window' (used now), 'past' (used at earlier steps), or 'future' (not yet measured at time t) */
  function role(j, k) { return j > k ? 'future' : j >= k - O + 1 ? 'window' : 'past'; }

  /** Schematic measurements: per row and column, observed or missing, with a small vertical jitter (seeded). */
  function measurements(seed) {
    const rng = IX.mulberry32(seed === undefined ? 3 : seed), out = [];
    for (let r = 0; r < STRIP.rows.length; r++) {
      const row = [];
      for (let j = 0; j < N; j++) row.push({ observed: rng() < 0.72, dy: (rng() - 0.5) * 9 });
      out.push(row);
    }
    return out;
  }

  /** Shades of the context vector c_t (9 cells, palette of the paper figure), a deterministic function of the
   *  window, so that the same t always gives the same c_t. */
  const PALETTE = ['#EED0BE', '#D17D4F', '#DEA383', '#D9956F', '#D38458', '#E8BDA5', '#D07A4B', '#E4B499', '#D5885F'];
  function contextShades(k) {
    const rng = IX.mulberry32(7919 + 31 * k), out = [];
    for (let i = 0; i < 9; i++) out.push(PALETTE[Math.floor(rng() * PALETTE.length)]);
    return out;
  }

  /** Linear interpolation of a polyline given as [[x, y], ...] with increasing x. */
  function interp(pts, x) {
    if (x <= pts[0][0]) return pts[0][1];
    for (let i = 1; i < pts.length; i++) {
      if (x <= pts[i][0]) {
        const [xa, ya] = pts[i - 1], [xb, yb] = pts[i];
        return ya + (yb - ya) * (x - xa) / (xb - xa);
      }
    }
    return pts[pts.length - 1][1];
  }
  function parsePoints(s) { return s.trim().split(/\s+/).map(p => p.split(',').map(Number)); }

  /** Box (b) in panel (c) for step k: spans [t, t + dt], vertically the 90 % band over that interval plus a margin. */
  function boxB(k, upper, lower) {
    const x = timeX(k), w = DX;
    let top = Infinity, bot = -Infinity;
    for (let i = 0; i <= 10; i++) {
      const xi = x + w * i / 10;
      top = Math.min(top, interp(upper, xi)); bot = Math.max(bot, interp(lower, xi));
    }
    return { x, y: top - 8, width: w, height: bot - top + 16 };
  }

  // ------------------------------------------------------------------ panel (b): the one-step distribution at step k
  // Schematic, like the paper figure: a two-component mixture in panel-(b) pixel coordinates (z grows downwards).
  // At DEFAULT_K it reproduces the drawing (modes at 255 and 345, drift end at 312); for other k, centre, separation,
  // weights and widths vary smoothly with k, and the widths scale with the 90 % band of panel (c) at t + dt.
  const B = Object.freeze({ x0: 520, x1: 790, zt: 392, ymin: 170, ymax: 420, width: 110, nSamples: 16, pathPts: 50 });
  /** mixture parameters [mid, sep, w1, sd1, sd2] for continuous step k and band-width ratio s (1 at DEFAULT_K) */
  function stepParams(k, s) {
    const u = k - DEFAULT_K, g = Math.pow(Math.min(2, Math.max(0.5, s === undefined ? 1 : s)), 0.6);  // softened
    return [300 + 28 * Math.sin(0.37 * u), 90 * (1 + 0.3 * Math.sin(0.53 * u)) * Math.sqrt(g),
      0.6 + 0.2 * Math.sin(0.41 * u), Math.max(9, 16 * g * (1 + 0.15 * Math.sin(0.7 * u))),
      Math.max(9, 14 * g * (1 + 0.15 * Math.sin(0.6 * u + 1.2) - 0.15 * Math.sin(1.2)))];
  }
  function mixture(P) {
    const [mid, sep, w1, s1, s2] = P;
    const lo = Math.max(B.ymin + 2.5 * s1, mid - sep / 2), hi = Math.min(B.ymax - 2.5 * s2, mid + sep / 2);
    return [{ w: w1, mu: lo, sd: s1 }, { w: 1 - w1, mu: hi, sd: s2 }];
  }
  function mixPdf(m, y) {
    let p = 0;
    for (const c of m) { const z = (y - c.mu) / c.sd; p += c.w * Math.exp(-0.5 * z * z) / (c.sd * Math.sqrt(2 * Math.PI)); }
    return p;
  }
  /** density outline x(y) on [ymin, ymax], scaled to the drawing's maximum width */
  function densityCurve(m, n) {
    n = n || 220;
    const ys = [], ps = [];
    for (let i = 0; i < n; i++) { const y = B.ymin + (B.ymax - B.ymin) * i / (n - 1); ys.push(y); ps.push(mixPdf(m, y)); }
    const pmax = Math.max(...ps);
    return ys.map((y, i) => [B.x1 + B.width * ps[i] / pmax, y]);
  }
  /** fixed quantile levels of the samples (seeded), so that the samples move continuously with the parameters */
  const LEVELS = (() => { const r = IX.mulberry32(11), out = []; for (let i = 0; i < B.nSamples; i++) out.push(0.04 + 0.92 * r()); return out; })();
  function quantiles(m, levels) {
    const n = 800, ys = [], cdf = [];
    let acc = 0;
    for (let i = 0; i < n; i++) { const y = B.ymin - 60 + (B.ymax - B.ymin + 120) * i / (n - 1); ys.push(y); acc += mixPdf(m, y); cdf.push(acc); }
    return levels.map(q => { const t = q * acc; let i = 0; while (i < n - 1 && cdf[i] < t) i++; return ys[i]; });
  }
  /** Brownian-bridge offsets per sample path (seeded, fixed), scaled by the spread at drawing time */
  const BRIDGES = (() => {
    const r = IX.mulberry32(23), out = [];
    for (let i = 0; i < B.nSamples; i++) {
      const w = [0];
      for (let j = 1; j < B.pathPts; j++) w.push(w[j - 1] + IX.randn(r));
      const last = w[B.pathPts - 1];
      out.push(w.map((v, j) => (v - last * j / (B.pathPts - 1)) / Math.sqrt(B.pathPts)));
    }
    return out;
  })();
  /** everything panel (b) draws for parameters P: density outline, sample end points, paths, drift end, highlight */
  function panelB(P) {
    const m = mixture(P), ends = quantiles(m, LEVELS);
    const mean = m[0].w * m[0].mu + m[1].w * m[1].mu, drift = mean + 21;   // drift end 21 px below the mixture mean, as drawn in the paper figure
    const amp = 0.9 * (m[0].sd + m[1].sd);
    const paths = ends.map((y, i) => BRIDGES[i].map((b, j) => {
      const f = j / (B.pathPts - 1);
      return [B.x0 + (B.x1 - B.x0) * f, B.zt + (y - B.zt) * f + amp * b];
    }));
    const hi = ends.reduce((best, y, i) => (Math.abs(LEVELS[i] - 0.25) < Math.abs(LEVELS[best] - 0.25) ? i : best), 0);
    return { mixture: m, curve: densityCurve(m), ends, paths, drift, highlight: ends[hi] };
  }

  const OV = { N, O, X0, X1, DX, STRIP, KMIN, KMAX, DEFAULT_K, timeX, stripX, windowCols, role, measurements,
    PALETTE, contextShades, interp, parsePoints, boxB, B, stepParams, mixture, mixPdf, quantiles, panelB, LEVELS };
  if (typeof module !== 'undefined' && module.exports) module.exports = OV;
  else root.OV = OV;
})(typeof self !== 'undefined' ? self : this);
