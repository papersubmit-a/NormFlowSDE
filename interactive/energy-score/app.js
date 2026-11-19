/* Energy-score figure: page logic. Model in model.js (closed form), helpers in ../_shared/ix.js. */
(function () {
  'use strict';
  const { el, linear, log, ticks, fmt, axes, path, slider, seg } = IX;
  const C = { acc: 'var(--c-mix)', half: 'var(--c-gauss)', es: 'var(--fg)', ent: 'var(--c-third)' };
  const st = Object.assign({}, ES.DEFAULTS);
  const [CLO, CHI] = ES.C_RANGE;

  const ac = document.getElementById('a-controls'), ac2 = document.getElementById('a-controls2');
  const slC = slider(ac, { label: 'spread <i>c</i>', min: CLO, max: CHI, log: true, value: st.c, fmt: v => fmt(v, 3),
    aria: 'ensemble spread c', onInput: v => { st.c = v; renderA(); } });
  seg(ac, { label: 'dimension <i>d</i>', aria: 'state dimension', value: st.d,
    options: ES.DIMS.map(o => ({ key: o.d, label: o.label, aria: o.aria })), onChange: v => { st.d = +v; renderAll(); } });
  seg(ac2, { label: 'members <i>M</i>', aria: 'number of ensemble members', value: st.M,
    options: ES.MS.map(m => ({ key: m, label: String(m), aria: m + ' members' })), onChange: v => { st.M = +v; renderAll(); } });
  seg(ac2, { label: 'spread estimator', aria: 'spread term estimator', value: st.estimator, options: [
    { key: 'unbiased', label: '1/(<i>M</i>(<i>M</i>−1)) Σ<sub><i>i</i>≠<i>j</i></sub> (paper)', aria: 'unbiased, excludes self pairs, as in the paper' },
    { key: 'vstat', label: '1/<i>M</i>² Σ<sub><i>i</i>,<i>j</i></sub>', aria: 'one over M squared, includes self pairs' },
  ], onChange: v => { st.estimator = v; renderAll(); } });

  const bc = document.getElementById('b-controls');
  seg(bc, { label: 'spread before scaling <i>c</i><sub>0</sub>', aria: 'miscalibrated spread c0', value: st.c0, options: [
    { key: 0.6, label: '0.6 under-dispersed' }, { key: 1.6, label: '1.6 over-dispersed' }],
  onChange: v => { st.c0 = +v; renderB(); } });
  const slS = slider(bc, { label: 'scale <i>s</i>', min: CLO, max: CHI, log: true, value: st.s, fmt: v => fmt(v, 3),
    aria: 'learned scale s', onInput: v => { st.s = v; renderB(); } });

  const opts = () => ({ M: st.M, estimator: st.estimator });
  const cs = (() => { const out = []; for (let i = 0; i <= 300; i++) out.push(CLO * Math.pow(CHI / CLO, i / 300)); return out; })();
  const lt = [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3];

  function clickToSet(g, sx, w, hgt, setter) {
    const hit = el('rect', { class: 'hit', x: 0, y: 0, width: w, height: hgt, fill: 'transparent' }, g);
    const go = e => { const r = hit.getBoundingClientRect(); setter(sx.inv(Math.min(w, Math.max(0, e.clientX - r.left)))); };
    hit.addEventListener('pointerdown', e => {
      go(e);
      const up = () => { window.removeEventListener('pointermove', go); window.removeEventListener('pointerup', up); };
      window.addEventListener('pointermove', go); window.addEventListener('pointerup', up);
    });
  }
  function endLabel(g, x, y, html, color) {
    const t = el('text', { x: x + 6, y: y + 4, fill: color }, g); t.innerHTML = html;
  }

  // ------------------------------------------------------------------ Panel A
  function renderA() {
    const host = document.getElementById('a-plot');
    host.textContent = '';
    const o = opts(), d = st.d;
    const W = host.clientWidth || 800, m = { l: 52, r: 120, t: 10, b: 40 }, h1 = 210, h2 = 110, gap = 18;
    const w = W - m.l - m.r;
    const T = cs.map(c => ES.terms(c, d, o));
    const ymax = Math.max(...T.map(t => t.acc)) * 1.05;
    const sx = log(CLO, CHI, 0, w), sy = linear(0, ymax, h1, 0);
    const ent = T.map(t => t.entropy), emin = Math.min(...ent), emax = Math.max(...ent), epad = (emax - emin) * 0.08;
    const sy2 = linear(emin - epad, emax + epad, h2, 0);
    const svg = el('svg', { width: W, height: m.t + h1 + gap + h2 + m.b, role: 'img' }, host);
    const g1 = el('g', { transform: `translate(${m.l},${m.t})` }, svg);
    const g2 = el('g', { transform: `translate(${m.l},${m.t + h1 + gap})` }, svg);
    axes(g1, { sx, sy, w, h: h1, xt: lt, yt: ticks(0, ymax, 5), noXLabels: true, yLabel: 'expected value', yLabelOffset: 42 });
    axes(g2, { sx, sy: sy2, w, h: h2, xt: lt, yt: ticks(emin - epad, emax + epad, 3), xLabel: 'ensemble spread <tspan font-style="italic">c</tspan> (log scale)',
      yLabel: 'entropy (nats)', yLabelOffset: 42 });
    const line = (g, ys, syy, color, wdt, dash) =>
      el('path', { d: path(cs, ys, sx, syy), fill: 'none', stroke: color, 'stroke-width': wdt, 'stroke-dasharray': dash || null }, g);
    line(g1, T.map(t => t.acc), sy, C.acc, 2.2);
    line(g1, T.map(t => t.half), sy, C.half, 2.2, '7 4');
    line(g1, T.map(t => t.es), sy, C.es, 3.2);
    line(g2, ent, sy2, C.ent, 2.6, '2 3');
    const last = T[T.length - 1];
    endLabel(g1, w, sy(last.acc), 'E‖<tspan font-style="italic">z</tspan>−<tspan font-style="italic">x</tspan>‖', C.acc);
    endLabel(g1, w, sy(last.half), '½E‖<tspan font-style="italic">z</tspan>−<tspan font-style="italic">z</tspan>′‖', C.half);
    endLabel(g1, w, sy(last.es), 'energy score', C.es);
    endLabel(g2, w, sy2(last.entropy), '<tspan font-style="italic">H</tspan>, Stage 1', C.ent);
    // argmin of the energy score (closed form, independent of d)
    const cstar = ES.argminC(o), xs = sx(cstar), esStar = ES.terms(cstar, d, o).es;
    el('line', { x1: xs, x2: xs, y1: 0, y2: h1, stroke: C.es, 'stroke-dasharray': '1 3', 'stroke-width': 1.5 }, g1);
    el('circle', { cx: xs, cy: sy(esStar), r: 5, fill: '#fff', stroke: C.es, 'stroke-width': 2 }, g1);
    el('text', { x: xs + 7, y: sy(esStar) + 18, 'font-size': 11.5 }, g1).innerHTML = 'min at <tspan font-style="italic">c</tspan> = ' + fmt(cstar, 3);
    el('text', { x: 6, y: 14, fill: 'var(--muted)', 'font-size': 11.5 }, g2).textContent = 'no interior minimum: smallest at the smallest spread';
    // cursor at current c
    const xc = sx(st.c);
    for (const [g, hh] of [[g1, h1], [g2, h2]]) el('line', { x1: xc, x2: xc, y1: 0, y2: hh, stroke: 'var(--focus)', 'stroke-width': 1.5 }, g);
    const tc = ES.terms(st.c, d, o);
    for (const [v, s, c] of [[tc.acc, sy, C.acc], [tc.half, sy, C.half], [tc.es, sy, C.es]]) el('circle', { cx: xc, cy: s(v), r: 3.5, fill: c }, g1);
    el('circle', { cx: xc, cy: sy2(tc.entropy), r: 3.5, fill: C.ent }, g2);
    clickToSet(g1, sx, w, h1, v => { st.c = v; slC.set(v, true); renderA(); });
    clickToSet(g2, sx, w, h2, v => { st.c = v; slC.set(v, true); renderA(); });
    svg.setAttribute('aria-label', `Energy score terms against ensemble spread for d = ${d}; minimum at c = ${fmt(cstar, 3)}; ` +
      `at c = ${fmt(st.c, 2)} the score is ${fmt(tc.es, 3)} and the entropy ${fmt(tc.entropy, 2)}`);
    document.getElementById('a-read').innerHTML =
      `<span>at <i>c</i> = ${fmt(st.c, 3)}:</span><span>E‖<i>z</i>−<i>x</i>‖ = ${fmt(tc.acc, 3)}</span>` +
      `<span>½E‖<i>z</i>−<i>z</i>′‖ = ${fmt(tc.half, 3)}</span><span><b>energy score = ${fmt(tc.es, 3)}</b></span>` +
      `<span><i>H</i> = ${fmt(tc.entropy, 3)}</span><span>argmin energy score: <i>c</i>* = ${fmt(cstar, 3)}</span>`;
    IX.reportHeight();
  }

  // ------------------------------------------------------------------ Panel B
  function renderB() {
    const host = document.getElementById('b-plot');
    host.textContent = '';
    const o = opts(), d = st.d;
    const W = host.clientWidth || 800, m = { l: 52, r: 120, t: 10, b: 40 }, hh = 150, w = W - m.l - m.r;
    const vals = cs.map(s => ES.terms(s * st.c0, d, o).es);
    const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.08;
    const sx = log(CLO, CHI, 0, w), sy = linear(lo - pad, hi + pad, hh, 0);
    const svg = el('svg', { width: W, height: m.t + hh + m.b, role: 'img' }, host);
    const g = el('g', { transform: `translate(${m.l},${m.t})` }, svg);
    axes(g, { sx, sy, w, h: hh, xt: lt, yt: ticks(lo - pad, hi + pad, 4), xLabel: 'scale <tspan font-style="italic">s</tspan> (log scale)',
      yLabel: 'energy score', yLabelOffset: 42 });
    el('path', { d: path(cs, vals, sx, sy), fill: 'none', stroke: C.es, 'stroke-width': 3 }, g);
    const sStar = ES.argminS(st.c0, o), xs = sx(sStar), ys = sy(ES.terms(sStar * st.c0, d, o).es);
    el('line', { x1: xs, x2: xs, y1: 0, y2: hh, stroke: C.es, 'stroke-dasharray': '1 3', 'stroke-width': 1.5 }, g);
    el('circle', { cx: xs, cy: ys, r: 5, fill: '#fff', stroke: C.es, 'stroke-width': 2 }, g);
    el('text', { x: xs, y: ys - 24, 'font-size': 11.5, 'text-anchor': 'middle' }, g).innerHTML =
      `min at <tspan font-style="italic">s</tspan> = ${fmt(sStar, 3)}` + (st.estimator === 'unbiased' ? ' = 1/<tspan font-style="italic">c</tspan>₀' : '');
    const xc = sx(st.s), es = ES.terms(st.s * st.c0, d, o).es;
    el('line', { x1: xc, x2: xc, y1: 0, y2: hh, stroke: 'var(--focus)', 'stroke-width': 1.5 }, g);
    el('circle', { cx: xc, cy: sy(es), r: 4, fill: 'var(--focus)' }, g);
    clickToSet(g, sx, w, hh, v => { st.s = v; slS.set(v, true); renderB(); });
    svg.setAttribute('aria-label', `Energy score of the scaled spread s times ${st.c0}; minimum at s = ${fmt(sStar, 3)}`);
    document.getElementById('b-read').innerHTML =
      `<span>effective spread <i>s</i>·<i>c</i><sub>0</sub> = ${fmt(st.s * st.c0, 3)}</span>` +
      `<span>energy score = ${fmt(es, 3)}</span><span>optimum <i>s</i>* = ${fmt(sStar, 3)} (spread ${fmt(sStar * st.c0, 3)})</span>`;
    IX.reportHeight();
  }

  function renderAll() { renderA(); renderB(); }
  IX.onResize(renderAll);
})();
