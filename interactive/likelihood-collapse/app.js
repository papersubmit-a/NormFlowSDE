/* Likelihood-collapse figure: page logic. Model in model.js, helpers in ../_shared/ix.js. */
(function () {
  'use strict';
  const { el, h, linear, ticks, fmt, axes, path, slider, seg, button } = IX;
  const C = { mix: 'var(--c-mix)', gauss: 'var(--c-gauss)', diff: 'var(--c-diff)' };
  const st = Object.assign({}, LC.PAPER, { priorSd: LC.DEFAULT_PRIOR_SD, y: LC.OBS_PRESETS.side, seed: LC.DEFAULT_SEED });

  // ------------------------------------------------------------------ controls
  const ac = document.getElementById('a-controls');
  const sl = {};
  const sdef = [
    ['w0', 'centre weight <i>w</i><sub>0</sub>', 0, 1, 0.01, 'centre component weight'],
    ['skew', 'share at +<i>a</i>', 0, 1, 0.01, 'share of the side weight in the positive side mode'],
    ['a', 'side offset <i>a</i>', 0, 5, 0.05, 'side mode offset in units of sigma'],
    ['s0', 'centre scale', 0.1, 2, 0.01, 'centre component standard deviation in units of sigma'],
    ['s1', 'side scale', 0.2, 3, 0.01, 'side component standard deviation in units of sigma'],
  ];
  for (const [k, label, min, max, step, aria] of sdef) {
    sl[k] = slider(ac, { label, min, max, step, value: st[k], aria, onInput: v => { st[k] = v; renderAll(); } });
  }
  button(ac, 'Reset to paper', () => {
    Object.assign(st, LC.PAPER);
    for (const k in LC.PAPER) sl[k].set(st[k], true);
    renderAll();
  }, 'Reset the mixture to the values used in the paper');

  const bc = document.getElementById('b-controls');
  const slPrior = slider(bc, { label: 'prior sd <i>s</i><sub>prior</sub>', min: 0.2, max: 3, step: 0.01, value: st.priorSd,
    aria: 'prior standard deviation of the particles in units of sigma', onInput: v => { st.priorSd = v; renderB(); } });
  const slObs = slider(bc, { label: 'observation <i>y</i>', min: -4.5, max: 4.5, step: 0.05, value: st.y,
    aria: 'observation in units of sigma', onInput: v => { st.y = v; renderB(); } });
  button(bc, 'Observation in the spike', () => { st.y = LC.OBS_PRESETS.spike; slObs.set(st.y, true); renderB(); },
    'Set the observation to the true state plus a noise draw from the central spike');
  button(bc, 'Observation in the side mode', () => { st.y = LC.OBS_PRESETS.side; slObs.set(st.y, true); renderB(); },
    'Set the observation to the true state plus a noise draw from the positive side mode');
  button(bc, 'Resample particles', () => { st.seed += 1; renderB(); }, 'Draw a new particle cloud');

  // ------------------------------------------------------------------ Panel A
  const aRead = document.getElementById('a-read');
  function renderA() {
    const host = document.getElementById('a-plot');
    host.textContent = '';
    const W = host.clientWidth || 800, H = 250, m = { l: 46, r: 12, t: 8, b: 40 };
    const w = W - m.l - m.r, hh = H - m.t - m.b;
    const comps = LC.components(st), mo = LC.moments(comps);
    const xs = IX.linspace(-8, 8, 801), p = xs.map(x => LC.mixPdf(comps, x)), q = xs.map(x => LC.normPdf(x, mo.mean, mo.sd));
    const ymax = Math.max(...p, ...q) * 1.08;
    const sx = linear(-8, 8, 0, w), sy = linear(0, ymax, hh, 0);
    const svg = el('svg', { width: W, height: H, 'aria-hidden': 'true' }, host);
    const g = el('g', { transform: `translate(${m.l},${m.t})` }, svg);
    axes(g, { sx, sy, w, h: hh, xt: ticks(-8, 8, 8), yt: ticks(0, ymax, 4), xLabel: 'noise <tspan font-style="italic">e</tspan> / σ',
      yLabel: 'density', yLabelOffset: 38 });
    // shaded |p - q|: polygon between the two curves
    let d = path(xs, p, sx, sy);
    for (let i = xs.length - 1; i >= 0; i--) d += 'L' + sx(xs[i]).toFixed(2) + ',' + sy(q[i]).toFixed(2);
    el('path', { d: d + 'Z', fill: C.diff, 'fill-opacity': 0.3, stroke: 'none' }, g);
    el('path', { d: path(xs, p, sx, sy), fill: 'none', stroke: C.mix, 'stroke-width': 2.2 }, g);
    el('path', { d: path(xs, q, sx, sy), fill: 'none', stroke: C.gauss, 'stroke-width': 2.2, 'stroke-dasharray': '7 4' }, g);
    // direct labels
    const lab = (x, y, t, c, anchor) => { const e = el('text', { x: sx(x), y: sy(y), fill: c, 'text-anchor': anchor || 'start' }, g); e.innerHTML = t; };
    lab(0.35, LC.mixPdf(comps, 0.35) , '<tspan font-style="italic">p</tspan>', C.mix);
    const xq = mo.mean + 1.6 * mo.sd;
    if (xq < 7.5) lab(xq + 0.2, LC.normPdf(xq, mo.mean, mo.sd) + ymax * 0.03, '<tspan font-style="italic">q</tspan>', C.gauss);
    const tvv = LC.tv(st);
    aRead.innerHTML = `<span><b>TV(<i>p</i>, <i>q</i>) = ${fmt(tvv, 3)}</b></span><span>overlap 1 − TV = ${fmt(1 - tvv, 3)}</span>` +
      `<span>mean = ${fmt(mo.mean, 3)} σ</span><span>variance = ${fmt(mo.var, 3)} σ² (sd ${fmt(mo.sd, 3)} σ)</span>`;
    host.setAttribute('aria-label', `Mixture density and moment-matched Gaussian; total variation distance ${fmt(tvv, 3)}, ` +
      `mean ${fmt(mo.mean, 2)}, variance ${fmt(mo.var, 2)}`);
  }

  // ------------------------------------------------------------------ Panel B
  const bTable = document.getElementById('b-table'), bNote = document.getElementById('b-note');
  let bScale = null;
  function renderB() {
    const host = document.getElementById('b-plot');
    host.textContent = '';
    const L = Math.max(4.5, 4 * st.priorSd);
    slObs.range(-L, L);
    st.y = slObs.get();
    const W = host.clientWidth || 600, H = 300, m = { l: 14, r: 14, t: 26, b: 40 };
    const w = W - m.l - m.r, hh = H - m.t - m.b, mid = hh / 2;
    const r = LC.filterStep(st, st.priorSd, st.y, st.seed);
    const wmax = Math.max(...r.mix.w, ...r.gauss.w);
    const sx = linear(-L, L, 0, w);
    const amp = mid - 12, sUp = v => mid - v / wmax * amp, sDn = v => mid + v / wmax * amp;
    bScale = { sx, m, L };
    const svg = el('svg', { width: W, height: H, role: 'img',
      'aria-label': `Particle weights for observation ${fmt(st.y, 2)}: mixture posterior mean ${fmt(r.mix.mean, 2)}, ` +
        `Gaussian posterior mean ${fmt(r.gauss.mean, 2)}` }, host);
    const g = el('g', { transform: `translate(${m.l},${m.t})` }, svg);
    axes(g, { sx, sy: linear(0, 1, hh, 0), w, h: hh, xt: ticks(-L, L, 9), yt: [], noYAxis: true,
      xLabel: 'state <tspan font-style="italic">x</tspan> / σ' });
    el('line', { x1: 0, x2: w, y1: mid, y2: mid, stroke: 'var(--axis)' }, g);
    // scaled likelihoods as functions of x (faint), mixture up, Gaussian down
    const xs = IX.linspace(-L, L, 600);
    const lm = xs.map(x => LC.mixPdf(r.comps, st.y - x)), lg = xs.map(x => LC.normPdf(st.y - x, r.moments.mean, r.moments.sd));
    const lmax = Math.max(...lm, ...lg);
    el('path', { d: path(xs, lm.map(v => v / lmax * wmax), sx, sUp), fill: 'none', stroke: C.mix, 'stroke-opacity': 0.35, 'stroke-width': 1.5 }, g);
    el('path', { d: path(xs, lg.map(v => v / lmax * wmax), sx, sDn), fill: 'none', stroke: C.gauss, 'stroke-opacity': 0.35, 'stroke-width': 1.5, 'stroke-dasharray': '6 4' }, g);
    // stems
    const gs = el('g', {}, g);
    for (let i = 0; i < r.x.length; i++) {
      const x = sx(r.x[i]);
      if (x < 0 || x > w) continue;
      el('line', { x1: x, x2: x, y1: mid, y2: sUp(r.mix.w[i]), stroke: C.mix, 'stroke-width': 1, 'stroke-opacity': 0.75 }, gs);
      el('circle', { cx: x, cy: sUp(r.mix.w[i]), r: 2.2, fill: C.mix }, gs);
      el('line', { x1: x, x2: x, y1: mid, y2: sDn(r.gauss.w[i]), stroke: C.gauss, 'stroke-width': 1, 'stroke-opacity': 0.75 }, gs);
      el('rect', { x: x - 2, y: sDn(r.gauss.w[i]) - 2, width: 4, height: 4, fill: C.gauss }, gs);
    }
    // posterior means
    // posterior means: label on the side away from the observation marker
    const meanMark = (v, y0, y1, c, dash, label, below) => {
      const x = sx(v), left = v < st.y;
      el('line', { x1: x, x2: x, y1: y0, y2: y1, stroke: c, 'stroke-width': 2.5, 'stroke-dasharray': dash || null }, g);
      el('text', { x: left ? x - 5 : x + 5, y: below ? y1 - 4 : y1 + 10, fill: c, 'font-size': 11.5,
        'text-anchor': left ? 'end' : 'start' }, g).textContent = label;
    };
    meanMark(r.mix.mean, mid - 2, 2, C.mix, null, 'posterior mean (mixture)');
    meanMark(r.gauss.mean, mid + 2, hh - 2, C.gauss, '6 3', 'posterior mean (Gaussian)', true);
    el('text', { x: 2, y: mid - 6, fill: 'var(--muted)' }, g).textContent = '● mixture likelihood ↑';
    el('text', { x: 2, y: mid + 15, fill: 'var(--muted)' }, g).textContent = '■ moment-matched Gaussian ↓';
    // truth marker (triangle below the axis line)
    const xt = sx(LC.TRUTH);
    el('path', { d: `M${xt - 6},${hh + 1}L${xt + 6},${hh + 1}L${xt},${hh - 9}Z`, fill: 'var(--fg)' }, g);
    // observation marker: draggable, keyboard-operable diamond above the plot
    const xo = sx(st.y);
    const og = el('g', { class: 'focusable obs-handle', tabindex: 0, role: 'slider', 'aria-label': 'observation y',
      'aria-valuemin': (-L).toFixed(2), 'aria-valuemax': L.toFixed(2), 'aria-valuenow': st.y.toFixed(2),
      'aria-valuetext': `observation ${fmt(st.y, 2)} sigma` }, g);
    el('line', { x1: xo, x2: xo, y1: -12, y2: hh, stroke: 'var(--fg)', 'stroke-width': 1, 'stroke-dasharray': '2 3' }, og);
    el('path', { class: 'focus-ring', d: `M${xo},-22L${xo + 8},-12L${xo},-2L${xo - 8},-12Z`, fill: 'var(--fg)', stroke: 'none' }, og);
    el('rect', { x: xo - 14, y: -24, width: 28, height: hh + 24, fill: 'transparent' }, og);
    el('text', { x: xo + 11, y: -8 }, og).innerHTML = '<tspan font-style="italic">y</tspan> = ' + fmt(st.y, 2);
    og.addEventListener('keydown', e => {
      const step = e.shiftKey ? 0.5 : 0.05;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') st.y -= step;
      else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') st.y += step;
      else return;
      e.preventDefault(); slObs.set(st.y, true); st.y = slObs.get(); renderB();
      const n = document.querySelector('#b-plot .obs-handle'); if (n) n.focus();
    });
    og.addEventListener('pointerdown', e => {
      e.preventDefault();
      const move = ev => {
        const rect = host.getBoundingClientRect();  // host persists across re-renders; the svg does not
        st.y = bScale.sx.inv(ev.clientX - rect.left - bScale.m.l);
        slObs.set(st.y, true); st.y = slObs.get(); renderB();
      };
      const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
      window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
    });
    // readout
    bTable.innerHTML = '<caption class="ix-note" style="text-align:left;caption-side:top">weighted posterior, true state <i>x</i>* = 0</caption>' +
      '<tr><th></th><th style="color:var(--c-mix)">● mixture</th><th style="color:var(--c-gauss)">■ Gaussian</th></tr>' +
      `<tr><td>posterior mean</td><td>${fmt(r.mix.mean, 3)}</td><td>${fmt(r.gauss.mean, 3)}</td></tr>` +
      `<tr><td>posterior sd</td><td>${fmt(r.mix.sd, 3)}</td><td>${fmt(r.gauss.sd, 3)}</td></tr>` +
      `<tr><td>ESS = 1/Σ<i>w</i>²</td><td>${fmt(r.mix.ess, 1)}</td><td>${fmt(r.gauss.ess, 1)}</td></tr>`;
    bNote.innerHTML = `Particles <i>x</i><sub><i>i</i></sub> ~ 𝒩(0, ${fmt(st.priorSd, 2)}²), seed ${st.seed}. ` +
      'Weights <i>w</i><sub><i>i</i></sub> ∝ likelihood of <i>y</i> − <i>x</i><sub><i>i</i></sub>; both rows share one vertical scale.';
    IX.reportHeight();
  }

  function renderAll() { renderA(); renderB(); }
  IX.onResize(renderAll);
})();
