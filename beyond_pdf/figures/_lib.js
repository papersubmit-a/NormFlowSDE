/* Minimal dependency-free SVG plotting helpers shared by the interactive figures. */
const NF = (() => {
  const SVGNS = 'http://www.w3.org/2000/svg';
  const METHODS = {
    normflowsde:    { label: 'NormFlowSDE',    color: '#d9480f', route: 'direct' },
    latent_nsf:     { label: 'Latent NSF',     color: '#1c7ed6', route: 'direct' },
    latent_sde:     { label: 'Latent SDE',     color: '#2b8a3e', route: 'direct' },
    sigker_sde:     { label: 'SigKer SDE',     color: '#7048e8', route: 'direct' },
    nf_dpf:         { label: 'NF-DPF',         color: '#0b7285', route: 'pf' },
    nsf:            { label: 'NSF-PF',         color: '#c2255c', route: 'pf' },
    flowsde:        { label: 'FlowSDE-PF',     color: '#5c940d', route: 'pf' },
    normflowsde_pf: { label: 'NormFlowSDE-PF', color: '#f08c00', route: 'pf' },
  };

  function el(tag, attrs = {}, parent = null, text = null) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    if (text !== null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  function html(tag, attrs = {}, parent = null, text = null) {
    const e = document.createElement(tag);
    for (const k in attrs) {
      if (k === 'class') e.className = attrs[k];
      else if (k === 'style') e.style.cssText = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    if (text !== null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  function linear(d0, d1, r0, r1) {
    const f = v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
    f.inv = p => d0 + (p - r0) / (r1 - r0) * (d1 - d0);
    f.domain = [d0, d1]; f.range = [r0, r1]; f.isLog = false;
    return f;
  }

  function log(d0, d1, r0, r1) {
    const l0 = Math.log10(d0), l1 = Math.log10(d1);
    const f = v => r0 + (Math.log10(Math.max(v, 1e-300)) - l0) / (l1 - l0) * (r1 - r0);
    f.inv = p => Math.pow(10, l0 + (p - r0) / (r1 - r0) * (l1 - l0));
    f.domain = [d0, d1]; f.range = [r0, r1]; f.isLog = true;
    return f;
  }

  function niceStep(span, n) {
    const raw = span / Math.max(n, 1);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / p;
    return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  }

  function ticks(a, b, n = 5) {
    const s = niceStep(b - a, n), out = [];
    for (let v = Math.ceil(a / s - 1e-9) * s; v <= b + s * 1e-9; v += s) out.push(Math.abs(v) < s * 1e-9 ? 0 : v);
    return out;
  }

  function logTicks(a, b) {
    const out = [];
    const span = Math.log10(b) - Math.log10(a);
    const mults = span > 2.5 ? [1] : span > 1.2 ? [1, 2, 5] : [1, 2, 3, 5, 7];
    for (let e = Math.floor(Math.log10(a)); e <= Math.ceil(Math.log10(b)); e++)
      for (const m of mults) { const v = m * Math.pow(10, e); if (v >= a * 0.999 && v <= b * 1.001) out.push(v); }
    return out;
  }

  function fmt(v, digits = 3) {
    if (v === null || v === undefined || !isFinite(v)) return '–';
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1);
    return Number(v.toPrecision(digits)).toString();
  }

  function tickFmt(v) {
    const a = Math.abs(v);
    if (a !== 0 && a < 1e-3) return v.toExponential(0);
    return Number(v.toPrecision(3)).toString();
  }

  function axes(g, x, y, o = {}) {
    const [x0, x1] = x.range, [y0, y1] = y.range;
    const xt = o.xticks || (x.isLog ? logTicks(...x.domain) : ticks(...x.domain, o.nx || 6));
    const yt = o.yticks || (y.isLog ? logTicks(...y.domain) : ticks(...y.domain, o.ny || 5));
    const grid = el('g', { class: 'grid' }, g);
    for (const v of yt) el('line', { x1: x0, x2: x1, y1: y(v), y2: y(v) }, grid);
    for (const v of xt) el('line', { x1: x(v), x2: x(v), y1: y0, y2: y1 }, grid);
    const ax = el('g', { class: 'axis' }, g);
    el('line', { x1: x0, x2: x1, y1: y0, y2: y0 }, ax);
    el('line', { x1: x0, x2: x0, y1: y0, y2: y1 }, ax);
    const xf = o.xfmt || tickFmt, yf = o.yfmt || tickFmt;
    for (const v of xt) el('text', { x: x(v), y: y0 + 15, 'text-anchor': 'middle' }, ax, xf(v));
    for (const v of yt) el('text', { x: x0 - 6, y: y(v) + 4, 'text-anchor': 'end' }, ax, yf(v));
    if (o.xlabel) el('text', { x: (x0 + x1) / 2, y: y0 + 33, 'text-anchor': 'middle', class: 'label' }, ax, o.xlabel);
    if (o.ylabel) el('text', { x: 0, y: 0, 'text-anchor': 'middle', class: 'label',
      transform: `translate(${x0 - (o.ylabelOffset || 44)},${(y0 + y1) / 2}) rotate(-90)` }, ax, o.ylabel);
  }

  function path(xs, ys, x, y) {
    let d = '';
    for (let i = 0; i < xs.length; i++) {
      if (ys[i] === null || !isFinite(ys[i])) continue;
      d += (d ? 'L' : 'M') + x(xs[i]).toFixed(1) + ',' + y(ys[i]).toFixed(1);
    }
    return d;
  }

  function area(xs, lo, hi, x, y) {
    let d = '';
    for (let i = 0; i < xs.length; i++) d += (i ? 'L' : 'M') + x(xs[i]).toFixed(1) + ',' + y(hi[i]).toFixed(1);
    for (let i = xs.length - 1; i >= 0; i--) d += 'L' + x(xs[i]).toFixed(1) + ',' + y(lo[i]).toFixed(1);
    return d + 'Z';
  }

  function tooltip(container) {
    const t = html('div', { class: 'tip' }, container);
    return {
      show(px, py, content) {
        t.innerHTML = content; t.style.display = 'block';
        const cw = container.clientWidth, tw = t.offsetWidth;
        t.style.left = Math.min(Math.max(px + 14, 4), cw - tw - 4) + 'px';
        t.style.top = Math.max(py - 10, 0) + 'px';
      },
      hide() { t.style.display = 'none'; },
    };
  }

  function chips(parent, items, opts = {}) {
    /* items: [{key, label, color?, dashed?, title?}]; returns {set(keys)} */
    const box = html('div', { class: 'chips' + (opts.cls ? ' ' + opts.cls : '') }, parent);
    const btns = {};
    for (const it of items) {
      const b = html('button', { class: 'chip', type: 'button', title: it.title || '' }, box);
      if (it.color) {
        const sw = html('span', { class: 'sw' + (it.dashed ? ' dashed' : '') }, b);
        sw.style.setProperty('--c', it.color);
      }
      html('span', {}, b, it.label);
      b.addEventListener('click', () => opts.onClick && opts.onClick(it.key));
      btns[it.key] = b;
    }
    return {
      set(active) { for (const k in btns) btns[k].classList.toggle('on', active.has ? active.has(k) : active === k); },
      buttons: btns,
    };
  }

  function onResize(fn) {
    let w = 0;
    const run = () => { const nw = document.body.clientWidth; if (nw !== w) { w = nw; fn(); } reportHeight(); };
    window.addEventListener('resize', run);
    run();
  }

  function reportHeight() {
    try {
      const app = document.getElementById('app');
      const h = Math.ceil(app ? app.getBoundingClientRect().height + 6 : document.body.scrollHeight);
      window.parent.postMessage({ nfFrame: window.name || document.title, height: h }, '*');
    } catch (e) { /* ignore */ }
  }

  return { METHODS, el, html, linear, log, ticks, logTicks, fmt, axes, path, area, tooltip, chips, onResize, reportHeight };
})();
