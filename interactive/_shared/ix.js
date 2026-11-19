/* Shared helpers for the interactive figures: seeded RNG, scales, SVG axes, accessible controls.
 * Classic script (no ES module) so that pages also open from file:// in every browser.
 * Exposes `IX` as a global in the browser and as module.exports in Node (tests, fallback export). */
(function (root) {
  'use strict';

  // ---------------------------------------------------------------- numerics (also used in Node)
  /** mulberry32: small seeded PRNG returning floats in [0, 1). */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  /** Standard normal draw by Box-Muller (one draw per two uniforms; simple and reproducible). */
  function randn(rng) {
    let u = rng();
    while (u <= 0) u = rng();
    const v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function linspace(a, b, n) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = a + (b - a) * i / (n - 1);
    return out;
  }
  function trapz(y, x) {
    let s = 0;
    for (let i = 1; i < y.length; i++) s += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
    return s;
  }

  const IX = { mulberry32, randn, linspace, trapz };

  // ---------------------------------------------------------------- browser-only helpers
  if (typeof document !== 'undefined') {
    const SVGNS = 'http://www.w3.org/2000/svg';
    const embedded = (function () { try { return window.self !== window.top; } catch (e) { return true; } })();
    if (embedded) document.documentElement.classList.add('embedded');

    function el(tag, attrs, parent) {
      const e = document.createElementNS(SVGNS, tag);
      for (const k in attrs || {}) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
      if (parent) parent.appendChild(e);
      return e;
    }
    function h(tag, attrs, parent, text) {
      const e = document.createElement(tag);
      for (const k in attrs || {}) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
      if (text !== undefined) e.textContent = text;
      if (parent) parent.appendChild(e);
      return e;
    }
    function linear(d0, d1, r0, r1) {
      const f = v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
      f.inv = p => d0 + (p - r0) / (r1 - r0) * (d1 - d0);
      f.domain = [d0, d1];
      return f;
    }
    function log(d0, d1, r0, r1) {
      const l0 = Math.log(d0), l1 = Math.log(d1);
      const f = v => r0 + (Math.log(v) - l0) / (l1 - l0) * (r1 - r0);
      f.inv = p => Math.exp(l0 + (p - r0) / (r1 - r0) * (l1 - l0));
      f.domain = [d0, d1];
      return f;
    }
    function ticks(a, b, n) {
      const span = b - a, step0 = span / Math.max(1, n);
      const mag = Math.pow(10, Math.floor(Math.log10(step0)));
      const err = step0 / mag;
      const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
      const out = [];
      for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
      return out;
    }
    function fmt(v, d) { return (Math.abs(v) < 1e-12 ? 0 : v).toFixed(d === undefined ? 2 : d).replace('-', '−'); }
    function tickFmt(v) {
      const s = Math.abs(v) >= 100 || Number.isInteger(+v.toFixed(6)) ? String(Math.round(v)) : String(+v.toPrecision(3));
      return s.replace('-', '−');
    }
    /** Draw grid and axes into group g. o: {sx, sy, w, h, xt, yt, xLabel, yLabel, xFmt, yFmt, noYAxis} */
    function axes(g, o) {
      const grid = el('g', { class: 'grid' }, g), ax = el('g', { class: 'axis' }, g);
      for (const t of o.yt || []) {
        const y = o.sy(t);
        el('line', { x1: 0, x2: o.w, y1: y, y2: y }, grid);
        if (!o.noYAxis) el('text', { x: -6, y: y + 4, 'text-anchor': 'end' }, ax).textContent = (o.yFmt || tickFmt)(t);
      }
      for (const t of o.xt || []) {
        const x = o.sx(t);
        el('line', { x1: x, x2: x, y1: o.h, y2: o.h + 4 }, ax);
        if (!o.noXLabels) el('text', { x: x, y: o.h + 16, 'text-anchor': 'middle' }, ax).textContent = (o.xFmt || tickFmt)(t);
      }
      el('line', { x1: 0, x2: o.w, y1: o.h, y2: o.h }, ax);
      if (!o.noYAxis) el('line', { x1: 0, x2: 0, y1: 0, y2: o.h }, ax);
      if (o.xLabel) {
        const t = el('text', { class: 'label', x: o.w / 2, y: o.h + 33, 'text-anchor': 'middle' }, ax);
        t.innerHTML = o.xLabel;
      }
      if (o.yLabel) {
        const t = el('text', { class: 'label', transform: `translate(${-o.yLabelOffset || -40},${o.h / 2}) rotate(-90)`,
          'text-anchor': 'middle' }, ax);
        t.innerHTML = o.yLabel;
      }
      return ax;
    }
    function path(xs, ys, sx, sy) {
      let d = '';
      for (let i = 0; i < xs.length; i++) {
        if (!isFinite(ys[i])) continue;
        d += (d ? 'L' : 'M') + sx(xs[i]).toFixed(2) + ',' + sy(ys[i]).toFixed(2);
      }
      return d;
    }
    let uid = 0;
    /** Accessible range slider. o: {label, min, max, step, value, fmt, log, aria, onInput} */
    function slider(parent, o) {
      const id = 'ix-s' + (++uid);
      const wrap = h('span', { class: 'ix-slider' }, parent);
      const lab = h('label', { for: id }, wrap); lab.innerHTML = o.label;
      const toPos = v => o.log ? Math.log(v) : v, fromPos = p => o.log ? Math.exp(p) : p;
      const inp = h('input', { type: 'range', id: id, min: toPos(o.min), max: toPos(o.max),
        step: o.log ? (toPos(o.max) - toPos(o.min)) / 400 : o.step, 'aria-label': o.aria || o.label }, wrap);
      const out = h('output', { for: id }, wrap);
      const f = o.fmt || (v => fmt(v, 2));
      let value = o.value;
      function show() { out.textContent = f(value); inp.setAttribute('aria-valuetext', (o.aria || '') + ' ' + f(value)); }
      function set(v, silent) {
        value = Math.min(o.max, Math.max(o.min, v)); inp.value = toPos(value); show();
        if (!silent && o.onInput) o.onInput(value);
      }
      inp.addEventListener('input', () => { value = fromPos(+inp.value); if (!o.log && o.step) value = Math.round(value / o.step) * o.step; show(); if (o.onInput) o.onInput(value); });
      set(o.value, true);
      return { set, get: () => value, input: inp,
        range(min, max) { o.min = min; o.max = max; inp.min = toPos(min); inp.max = toPos(max); set(value, true); } };
    }
    /** Segmented choice (buttons with aria-pressed). o: {label, options:[{key,label,aria}], value, onChange} */
    function seg(parent, o) {
      const wrap = h('span', { class: 'ix-seg', role: 'group', 'aria-label': o.aria || o.label }, parent);
      if (o.label) h('span', { class: 'lbl' }, wrap).innerHTML = o.label;
      const opts = h('span', { class: 'opts' }, wrap), btns = {};
      let value = o.value;
      for (const it of o.options) {
        const b = h('button', { type: 'button', 'aria-label': it.aria || null }, opts); b.innerHTML = it.label;
        b.addEventListener('click', () => { set(it.key); });
        btns[it.key] = b;
      }
      function set(v, silent) {
        value = v;
        for (const k in btns) btns[k].setAttribute('aria-pressed', String(String(k) === String(v)));
        if (!silent && o.onChange) o.onChange(v);
      }
      set(value, true);
      return { set, get: () => value };
    }
    function button(parent, label, onClick, aria) {
      const b = h('button', { type: 'button', class: 'ix-btn', 'aria-label': aria || null }, parent, label);
      b.addEventListener('click', onClick);
      return b;
    }
    /** Re-render on width change; report height to an embedding page (TMLR web version). */
    function onResize(fn) {
      let w = 0, pending = false;
      const run = () => {
        pending = false;
        const nw = document.body.clientWidth;
        if (nw !== w) { w = nw; fn(); }
        reportHeight();
      };
      window.addEventListener('resize', () => { if (!pending) { pending = true; requestAnimationFrame(run); } });
      run();
    }
    function reportHeight() {
      if (!embedded) return;
      try {
        const main = document.querySelector('main') || document.body;
        const hgt = Math.ceil(main.getBoundingClientRect().height + 8);
        window.parent.postMessage({ nfFrame: window.name || document.title, height: hgt }, '*');
      } catch (e) { /* ignore */ }
    }
    Object.assign(IX, { embedded, el, h, linear, log, ticks, fmt, tickFmt, axes, path, slider, seg, button, onResize,
      reportHeight });
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = IX;
  else root.IX = IX;
})(typeof self !== 'undefined' ? self : this);
