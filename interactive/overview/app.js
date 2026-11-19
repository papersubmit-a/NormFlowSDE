/* Interactive overview (paper Figure 1): the time point t moves, and with it the trailing observation window. The
 * paper's SVG (figure-svg.js) is inserted unchanged; only the time-dependent elements are moved or redrawn. */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const ORANGE = '#C8642E', GREY = '#D5DEE3', DARK = '#3A4756';
  const st = { k: OV.DEFAULT_K, playing: false, timer: null };
  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ------------------------------------------------------------------ insert the paper figure
  const host = document.getElementById('fig');
  host.innerHTML = window.OVERVIEW_SVG;
  const svg = host.querySelector('svg');
  svg.removeAttribute('width'); svg.removeAttribute('height');
  svg.setAttribute('role', 'img');
  svg.classList.add('ix-raw');
  svg.style.width = '100%'; svg.style.height = 'auto';
  const el = (tag, attrs, parent, before) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.insertBefore(e, before || null);
    return e;
  };

  // ------------------------------------------------------------------ panel (a): strip, window box, label, c_t
  const ga = svg.querySelector('#panel-a-encoder');
  ga.querySelectorAll('circle').forEach(c => c.remove());           // redrawn below with N columns
  const boxA = Array.from(ga.querySelectorAll('rect')).find(r => r.getAttribute('fill-opacity') === '0.08');
  const labelA = Array.from(ga.querySelectorAll('text')).find(t => t.querySelector('tspan[y="80.0"]'));
  const cells = Array.from(ga.querySelectorAll('rect[y="386"]'));
  const dotsA = el('g', { 'aria-hidden': 'true' }, ga, boxA.nextSibling);
  const meas = OV.measurements();
  const dotEls = meas.map((row, r) => row.map((m, j) => el('circle', {
    cx: OV.stripX(j).toFixed(1), cy: (OV.STRIP.rows[r] + (m.observed ? m.dy : 0)).toFixed(1),
    r: m.observed ? 3.4 : 3.1, fill: m.observed ? ORANGE : 'none', stroke: m.observed ? 'none' : GREY, 'stroke-width': 1.2,
  }, dotsA)));
  const boxX0 = +boxA.getAttribute('x');

  // ------------------------------------------------------------------ panel (c): window shade, ticks, box (b), z_t
  const gc = svg.querySelector('#panel-c-posterior');
  const polys = Array.from(gc.querySelectorAll('polygon'));
  const band = OV.parsePoints(polys[0].getAttribute('points'));
  const half = band.length / 2, upper = band.slice(0, half), lower = band.slice(half).reverse();
  const median = OV.parsePoints(Array.from(gc.querySelectorAll('polyline')).find(p => p.getAttribute('stroke-width') === '2.8')
    .getAttribute('points'));
  const boxC = Array.from(gc.querySelectorAll('rect')).find(r => r.getAttribute('fill') === 'none');
  const labelC = Array.from(gc.querySelectorAll('text')).find(t => t.textContent.trim() === '(b)');
  const zoom = Array.from(svg.querySelectorAll('path')).filter(p => /^M1314\.5,/.test(p.getAttribute('d')));
  const firstPlot = polys[0];
  const shade = el('rect', { y: 96, height: 404, fill: ORANGE, 'fill-opacity': 0.09, stroke: ORANGE, 'stroke-opacity': 0.35,
    'stroke-dasharray': '3 3', rx: 3 }, gc, firstPlot);
  const shadeLabel = el('text', { y: 489, 'font-size': 17, 'font-style': 'italic', fill: ORANGE, 'text-anchor': 'middle',
    'font-family': 'DejaVu Sans, Helvetica, Arial, sans-serif' }, gc, firstPlot);
  shadeLabel.textContent = 'window';
  const ticks = [];
  for (let j = 0; j < OV.N; j++) ticks.push(el('circle', { cx: OV.timeX(j), cy: 509, r: 3.3 }, gc));
  const zt = el('circle', { r: 6, fill: DARK, stroke: '#fff', 'stroke-width': 1.2 }, gc);
  const hit = el('rect', { x: OV.X0, y: 96, width: OV.X1 - OV.X0, height: 420, fill: 'transparent', style: 'cursor:pointer' }, gc);

  // ------------------------------------------------------------------ panel (b): the one-step distribution follows t
  const gb = svg.querySelector('#panel-b-one-step');
  const startsWith = (e, s) => (e.getAttribute('points') || '').trim().startsWith(s);
  const dens = Array.from(gb.querySelectorAll('polygon, polyline')).filter(e => startsWith(e, '790.0,170.0'));
  const bPaths = Array.from(gb.querySelectorAll('polyline')).filter(e => startsWith(e, '520.0,392.0') && e.getAttribute('stroke-width') === '0.9');
  const bDots = Array.from(gb.querySelectorAll('circle[r="2.3"]'));
  const arrow = p => Array.from(gb.querySelectorAll('path')).find(e => e.getAttribute('d').startsWith(p));
  const driftArrow = arrow('M527,390'), noiseArrow = arrow('M790,306'), dzArrow = arrow('M526,387');
  const driftDot = gb.querySelector('circle[cy="312"]'), nextDot = gb.querySelector('circle[cy="246"]');
  const nextLabel = [gb.querySelector('rect[y="198.4"]'), Array.from(gb.querySelectorAll('text')).find(t => t.querySelector('tspan[y="224.0"]'))];
  const pts = a => a.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ');
  const bandWidth = x => OV.interp(lower, x) - OV.interp(upper, x);
  const width0 = bandWidth(OV.timeX(OV.DEFAULT_K + 1));
  function drawB(P) {
    const r = OV.panelB(P);
    dens.forEach(e => e.setAttribute('points', e.tagName === 'polyline' ? pts(r.curve)
      : pts([[OV.B.x1, OV.B.ymin], ...r.curve, [OV.B.x1, OV.B.ymax]])));
    bPaths.forEach((e, i) => r.paths[i] && e.setAttribute('points', pts(r.paths[i])));
    bDots.forEach((e, i) => r.ends[i] !== undefined && e.setAttribute('cy', r.ends[i].toFixed(1)));
    const d = r.drift, h = r.highlight;
    driftArrow.setAttribute('d', `M527,390 L787,${(d + 1).toFixed(1)}`);
    driftDot.setAttribute('cy', d.toFixed(1));
    nextDot.setAttribute('cy', h.toFixed(1));
    const dir = h < d ? 1 : -1;   // noise arrow from the drift end towards the sample
    noiseArrow.setAttribute('d', `M790,${(d - dir * 6).toFixed(1)} L790,${(h + dir * 7).toFixed(1)}`);
    noiseArrow.style.display = Math.abs(h - d) < 16 ? 'none' : '';
    dzArrow.setAttribute('d', `M526,387 L784,${(h + 3).toFixed(1)}`);
    nextLabel.forEach(e => e && e.setAttribute('transform', `translate(0,${(h - 246).toFixed(1)})`));
  }
  let bCur = null, bAnim = 0;
  function animateB(k) {
    const target = OV.stepParams(k, bandWidth(OV.timeX(k + 1)) / width0);
    if (!bCur || reduceMotion) { bCur = target; drawB(bCur); return; }
    const from = bCur.slice(), t0 = performance.now(), dur = 380, id = ++bAnim;
    const tick = now => {
      if (id !== bAnim) return;
      const f = Math.min(1, (now - t0) / dur), e = f * f * (3 - 2 * f);
      bCur = from.map((v, i) => v + (target[i] - v) * e);
      drawB(bCur);
      if (f < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // ------------------------------------------------------------------ controls
  const ctl = document.getElementById('controls');
  const play = IX.button(ctl, 'Play', () => setPlaying(!st.playing), 'Play or pause the time point');
  IX.button(ctl, '◀ step', () => { setPlaying(false); set(st.k - 1); }, 'One step back');
  IX.button(ctl, 'step ▶', () => { setPlaying(false); set(st.k + 1); }, 'One step forward');
  const slider = IX.slider(ctl, { label: 'time point <i>t</i>', min: OV.KMIN, max: OV.KMAX, step: 1, value: st.k,
    fmt: v => String(Math.round(v)), aria: 'time step t', onInput: v => { setPlaying(false); set(Math.round(v)); } });
  const readout = document.getElementById('readout');

  function setPlaying(on) {
    st.playing = on; play.textContent = on ? 'Pause' : 'Play';
    clearInterval(st.timer);
    if (on) st.timer = setInterval(() => set(st.k >= OV.KMAX ? OV.KMIN : st.k + 1), 800);
  }
  hit.addEventListener('pointerdown', e => {
    const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
    const u = p.matrixTransform(svg.getScreenCTM().inverse());
    setPlaying(false); set(Math.floor((u.x - OV.X0) / OV.DX));
  });

  // ------------------------------------------------------------------ render
  function set(k) {
    st.k = Math.max(OV.KMIN, Math.min(OV.KMAX, k));
    slider.set(st.k, true);
    const k0 = st.k - OV.O + 1;
    // (a) dots: window full strength, earlier measurements muted, later ones not yet available
    meas.forEach((row, r) => row.forEach((m, j) => {
      const role = OV.role(j, st.k);
      dotEls[r][j].setAttribute('opacity', role === 'window' ? 1 : role === 'past' ? 0.45 : 0.15);
    }));
    const xa0 = OV.stripX(k0) - 9, xa1 = OV.stripX(st.k) + 9;
    boxA.setAttribute('x', xa0.toFixed(1)); boxA.setAttribute('width', (xa1 - xa0).toFixed(1));
    labelA.setAttribute('transform', `translate(${(xa1 - 306 + 0).toFixed(1)},0)`);  // label right-aligned with the box
    OV.contextShades(st.k).forEach((c, i) => cells[i].setAttribute('fill', c));
    // (c) window shade over the measurement times of y_{t-o+1:t}; box (b) over [t, t + dt]
    const xs0 = OV.timeX(k0) - 7, xs1 = OV.timeX(st.k) + 7;
    shade.setAttribute('x', xs0.toFixed(1)); shade.setAttribute('width', (xs1 - xs0).toFixed(1));
    shadeLabel.setAttribute('x', ((xs0 + xs1) / 2).toFixed(1));
    ticks.forEach((t, j) => {
      const role = OV.role(j, st.k);
      t.setAttribute('fill', role === 'future' ? 'none' : ORANGE);
      t.setAttribute('stroke', role === 'future' ? GREY : 'none');
      t.setAttribute('opacity', role === 'past' ? 0.45 : 1);
    });
    const b = OV.boxB(st.k, upper, lower);
    boxC.setAttribute('x', b.x.toFixed(1)); boxC.setAttribute('y', b.y.toFixed(1));
    boxC.setAttribute('width', b.width.toFixed(1)); boxC.setAttribute('height', b.height.toFixed(1));
    labelC.setAttribute('x', (b.x + b.width / 2).toFixed(1)); labelC.setAttribute('y', (b.y - 9).toFixed(1));
    if (zoom[0]) zoom[0].setAttribute('d', `M${b.x.toFixed(1)},${b.y.toFixed(1)} L988,74`);
    if (zoom[1]) zoom[1].setAttribute('d', `M${b.x.toFixed(1)},${(b.y + b.height).toFixed(1)} L988,540`);
    zt.setAttribute('cx', b.x.toFixed(1)); zt.setAttribute('cy', OV.interp(median, b.x).toFixed(1));
    animateB(st.k);
    readout.innerHTML = `step <i>t</i> = ${st.k}: the encoder reads <i>y</i><sub>${k0}</sub>, …, <i>y</i><sub>${st.k}</sub> ` +
      `and conditions the update from <i>z</i><sub>${st.k}</sub> to <i>z</i><sub>${st.k + 1}</sub>; ` +
      (st.k === OV.KMAX ? 'this is the last step of the window.'
        : `${OV.N - 1 - st.k} later measurement${OV.N - 1 - st.k === 1 ? ' is' : 's are'} not yet used.`);
    svg.setAttribute('aria-label', `NormFlowSDE overview at time step ${st.k}: observation window from step ${k0} to ` +
      `${st.k}, update to step ${st.k + 1}`);
  }

  const q = /[?&#]t=(\d+)/.exec(location.search + location.hash);   // e.g. index.html?t=5 links to a time point
  set(q ? +q[1] : st.k);
  IX.onResize(() => {});
  if (!reduceMotion && !IX.embedded && /[?&]autoplay/.test(location.search)) setPlaying(true);
})();
