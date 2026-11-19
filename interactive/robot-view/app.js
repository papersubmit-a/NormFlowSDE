/* Robot view: posterior draws of NormFlowSDE replayed on the robots' kinematic models (three.js r147, classic script).
 * Kinematic replay only: the joint values come from the frozen runs; no physics is simulated. */
(function () {
  'use strict';
  const ASSETS = Object.assign({ model: n => `models/${n}.js`, data: d => `data/${d}.js` }, window.RV_ASSETS || {});
  const C = { method: '#d9480f', ref: '#9aa3ad', highlight: '#f2b705', trace: '#1f2328' };
  // mode: 'ghosts' = reference + posterior draws as ghost arms; 'median' = reference + posterior median
  const st = { ds: 'reassemble', w: 5, k: 0, joint: 1, K: 15, playing: false, speed: 1, mode: 'ghosts', showTrace: true,
    proj: 'top', reducedNote: '' };
  const cache = {};          // ds -> {data, model, insts, ee}
  let cur = null;

  // ------------------------------------------------------------------ DOM
  const $ = id => document.getElementById(id);
  const host = $('view'), ctl = $('controls'), ctl2 = $('controls2'), note = $('note');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  host.appendChild(renderer.domElement);
  renderer.domElement.setAttribute('tabindex', '0');
  renderer.domElement.setAttribute('aria-label', '3D view of the robot; drag to orbit, wheel to zoom, arrow keys to orbit');
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#f4f6f8');
  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 20);
  camera.up.set(0, 0, 1);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8b97a3, 0.95));
  const sun = new THREE.DirectionalLight(0xffffff, 0.75); sun.position.set(1.2, -1.6, 2.4); scene.add(sun);
  const grid = new THREE.GridHelper(2, 20, 0xc5ccd3, 0xdfe3e8); grid.rotation.x = Math.PI / 2; scene.add(grid);

  // ------------------------------------------------------------------ orbit camera (z up)
  const orbit = { target: new THREE.Vector3(), r: 1.6, az: -0.6, el: 0.45 };
  const VIEWS = { front: [-Math.PI / 2, 0.25], side: [0, 0.25], top: [-Math.PI / 2, 1.45], persp: [-0.6, 0.45] };
  function placeCamera() {
    const { r, az, el } = orbit, t = orbit.target;
    camera.position.set(t.x + r * Math.cos(el) * Math.cos(az), t.y + r * Math.cos(el) * Math.sin(az), t.z + r * Math.sin(el));
    camera.lookAt(t);
    requestRender();
  }
  function setView(name) {
    const R = RV.ROBOTS[st.ds];
    orbit.target.set(...R.target); orbit.r = R.dist; [orbit.az, orbit.el] = VIEWS[name]; placeCamera();
  }
  (function () {
    const el = renderer.domElement;
    let drag = null;
    el.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 }; el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointermove', e => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) {
        const right = new THREE.Vector3().subVectors(camera.position, orbit.target).cross(camera.up).normalize();
        orbit.target.addScaledVector(right, dx * orbit.r * 0.0015).addScaledVector(camera.up, dy * orbit.r * 0.0015);
      } else { orbit.az -= dx * 0.008; orbit.el = Math.max(-0.2, Math.min(1.55, orbit.el + dy * 0.008)); }
      placeCamera();
    });
    el.addEventListener('pointerup', () => { drag = null; });
    el.addEventListener('contextmenu', e => e.preventDefault());
    el.addEventListener('wheel', e => { e.preventDefault(); orbit.r = Math.max(0.3, Math.min(5, orbit.r * Math.exp(e.deltaY * 0.001))); placeCamera(); }, { passive: false });
    el.addEventListener('keydown', e => {
      const m = { ArrowLeft: [0.08, 0], ArrowRight: [-0.08, 0], ArrowUp: [0, 0.06], ArrowDown: [0, -0.06] }[e.key];
      if (!m) return;
      e.preventDefault(); orbit.az += m[0]; orbit.el = Math.max(-0.2, Math.min(1.55, orbit.el + m[1])); placeCamera();
    });
  })();

  // ------------------------------------------------------------------ robots
  function b64bytes(s) { const t = atob(s), u = new Uint8Array(t.length); for (let i = 0; i < t.length; i++) u[i] = t.charCodeAt(i); return u.buffer; }
  function partGeometry(p) {
    const q = new Uint16Array(b64bytes(p.pos)), pos = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) { const a = i % 3; pos[i] = p.lo[a] + q[i] / 65535 * p.span[a]; }
    const idx = p.wide ? new Uint32Array(b64bytes(p.idx)) : new Uint16Array(b64bytes(p.idx));
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();
    return g;
  }
  const matCache = {};
  function solid(color) {
    return matCache['s' + color] || (matCache['s' + color] = new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.05 }));
  }
  const MAT = {
    ref: solid(C.ref),
    refT: new THREE.MeshStandardMaterial({ color: '#5f6b77', roughness: 0.6, transparent: true, opacity: 0.38, depthWrite: false }),
    med: new THREE.MeshStandardMaterial({ color: C.method, roughness: 0.5, metalness: 0.05 }),
    ghost: new THREE.MeshLambertMaterial({ color: C.method, transparent: true, opacity: 0.07, depthWrite: false }),
    hi: new THREE.MeshStandardMaterial({ color: C.highlight, roughness: 0.5, emissive: '#5a4300' }),
  };
  /** one robot instance: Object3D tree from the MJCF bodies; mat = material for all parts, null = skeleton only */
  function instantiate(model, mat) {
    const root = new THREE.Group(), bodies = {}, joints = {};
    const jo = [];
    model.bodies.forEach((b, i) => {
      const bo = new THREE.Object3D();
      bo.position.set(...b.pos);
      bo.quaternion.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]);   // MJCF (w, x, y, z)
      const j = new THREE.Object3D(); bo.add(j); jo.push(j);
      (b.parent < 0 ? root : jo[b.parent]).add(bo);
      bodies[b.name] = { frame: j, meshes: [] };
      if (b.joint) joints[b.joint.name] = { obj: j, type: b.joint.type, axis: new THREE.Vector3(...b.joint.axis).normalize() };
      if (mat) for (const p of b.parts) {
        const m = new THREE.Mesh(p.geom, mat); m.userData.base = mat; j.add(m); bodies[b.name].meshes.push(m);
      }
    });
    return { root, bodies, joints };
  }
  const tmpQ = new THREE.Quaternion();
  function pose(inst, q) {
    const tg = RV.jointTargets(st.ds, q);
    for (const name in tg) {
      const J = inst.joints[name]; if (!J) continue;
      if (J.type === 'slide') J.obj.position.copy(J.axis).multiplyScalar(tg[name]);
      else J.obj.quaternion.copy(tmpQ.setFromAxisAngle(J.axis, tg[name]));
    }
  }

  // ------------------------------------------------------------------ data loading (lazy, script tags work from file://)
  function loadScript(src) {
    return new Promise((res, rej) => { const s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = () => rej(new Error(src)); document.head.appendChild(s); });
  }
  async function ensure(ds) {
    if (cache[ds]) return cache[ds];
    note.textContent = 'loading ' + ds + ' …';
    const mname = RV.ROBOTS[ds].model;
    if (!(window.ROBOT_MODELS && window.ROBOT_MODELS[mname])) await loadScript(ASSETS.model(mname));
    if (!(window.ROBOT_VIEW_DATA && window.ROBOT_VIEW_DATA[ds])) await loadScript(ASSETS.data(ds));
    const model = window.ROBOT_MODELS[mname];
    for (const b of model.bodies) for (const p of b.parts) if (!p.geom) p.geom = partGeometry(p);
    const data = RV.decodeAll(window.ROBOT_VIEW_DATA[ds]);
    const group = new THREE.Group();
    const insts = { ref: instantiate(model, MAT.ref), med: instantiate(model, MAT.med), ghosts: [] };
    for (let i = 0; i < data.meta.draws_exported; i++) { const g = instantiate(model, MAT.ghost); g.root.renderOrder = 2; insts.ghosts.push(g); group.add(g.root); }
    insts.med.root.renderOrder = 3;
    group.add(insts.ref.root, insts.med.root);
    const probe = instantiate(model, null);
    cache[ds] = { data, model, insts, group, probe, ee: {} };
    note.textContent = '';
    return cache[ds];
  }

  // end-effector positions for all steps of a window: [instance][t][arm] (instance 0 = reference, 1 = median, 2.. draws)
  function eePaths(c, w) {
    if (c.ee[w]) return c.ee[w];
    const D = c.data, T = D.meta.T, R = RV.ROBOTS[st.ds], n = 2 + D.meta.draws_exported, arms = R.ee.length;
    const out = new Float32Array(n * T * arms * 3), v = new THREE.Vector3();
    for (let i = 0; i < n; i++) for (let t = 0; t < T; t++) {
      const q = i === 0 ? RV.row(D.truth, w, t) : i === 1 ? RV.row(D.q50, w, t) : RV.row(D.draws, w, i - 2, t);
      pose(c.probe, q); c.probe.root.updateMatrixWorld(true);
      R.ee.forEach(([body, off], a) => {
        v.set(...off).applyMatrix4(c.probe.bodies[body].frame.matrixWorld);
        out.set([v.x, v.y, v.z], ((i * T + t) * arms + a) * 3);
      });
    }
    return (c.ee[w] = out);
  }
  const TRACE_LEN = 20;
  const traceGroup = new THREE.Group(); scene.add(traceGroup);
  function drawTraces(c) {
    traceGroup.clear();
    if (!st.showTrace) return;
    const D = c.data, T = D.meta.T, arms = RV.ROBOTS[st.ds].ee.length, P = eePaths(c, st.w);
    const t0 = Math.max(0, st.k - TRACE_LEN + 1), len = st.k - t0 + 1;
    if (len < 2) return;
    const line = (i, mat) => {
      for (let a = 0; a < arms; a++) {
        const pts = new Float32Array(len * 3);
        for (let t = t0; t <= st.k; t++) pts.set(P.subarray(((i * T + t) * arms + a) * 3, ((i * T + t) * arms + a) * 3 + 3), (t - t0) * 3);
        const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pts, 3));
        traceGroup.add(new THREE.Line(g, mat));
      }
    };
    if (st.mode === 'ghosts') for (let i = 0; i < st.K; i++) line(2 + i, TRACE.ghost);
    else line(1, TRACE.med);
    line(0, TRACE.ref);
  }
  const TRACE = { ref: new THREE.LineBasicMaterial({ color: C.trace }), med: new THREE.LineBasicMaterial({ color: C.method }),
    ghost: new THREE.LineBasicMaterial({ color: C.method, transparent: true, opacity: 0.35 }) };

  // ------------------------------------------------------------------ controls
  const segDs = IX.seg(ctl, { label: 'Robot', value: st.ds, options: [
    { key: 'reassemble', label: 'REASSEMBLE (Franka, real robot)' }, { key: 'aloha', label: 'ALOHA insertion (2 × ViperX, simulation)' }],
  onChange: v => select(v, 5, 0) });
  const winWrap = IX.h('span', { class: 'ix-slider' }, ctl);
  IX.h('label', { for: 'rv-win' }, winWrap, 'test window');
  const winSel = IX.h('select', { id: 'rv-win', 'aria-label': 'test window, by RMSE quantile' }, winWrap);
  winSel.addEventListener('change', () => select(st.ds, +winSel.value, 0));
  const jWrap = IX.h('span', { class: 'ix-slider' }, ctl);
  IX.h('label', { for: 'rv-joint' }, jWrap, 'joint');
  const jSel = IX.h('select', { id: 'rv-joint', 'aria-label': 'joint shown in the timeline and highlighted' }, jWrap);
  jSel.addEventListener('change', () => { st.joint = +jSel.value; highlight(); drawTimeline(); readout(); });

  const playBtn = IX.button(ctl2, 'Play', () => setPlaying(!st.playing), 'Play or pause');
  IX.button(ctl2, '◀', () => { setPlaying(false); setK(st.k - 1); }, 'one step back');
  IX.button(ctl2, '▶', () => { setPlaying(false); setK(st.k + 1); }, 'one step forward');
  IX.seg(ctl2, { label: 'speed', value: 1, options: [0.25, 0.5, 1, 2].map(s => ({ key: s, label: s + '×', aria: 'speed ' + s })),
    onChange: v => { st.speed = +v; } });
  const slK = IX.slider(ctl2, { label: 'ghost arms <i>K</i>', min: 0, max: 30, step: 1, value: st.K, fmt: v => String(Math.round(v)),
    aria: 'number of posterior draws shown as ghost arms', onInput: v => { st.K = Math.round(v); st.reducedNote = ''; applyVisibility(); } });
  const segMode = IX.seg(ctl, { label: 'show', aria: 'which posterior summary is drawn with the reference', value: st.mode, options: [
    { key: 'ghosts', label: 'reference + posterior draws', aria: 'reference and posterior draws as ghost arms' },
    { key: 'median', label: 'reference + posterior median', aria: 'reference and posterior median' }],
  onChange: v => { st.mode = v; applyMode(); writeHash(); } });
  {
    const l = IX.h('label', { class: 'ix-slider' }, ctl2);
    const cb = IX.h('input', { type: 'checkbox' }, l); cb.checked = st.showTrace; l.append(' 3D trace (last 20 steps)');
    cb.addEventListener('change', () => { st.showTrace = cb.checked; applyVisibility(); });
  }
  const vw = IX.h('span', { class: 'ix-seg' }, ctl2); IX.h('span', { class: 'lbl' }, vw, 'view');
  for (const v of ['front', 'side', 'top']) IX.button(vw, v, () => setView(v), v + ' view');
  IX.button(vw, 'reset', () => setView('persp'), 'reset the camera');

  // ------------------------------------------------------------------ state changes
  async function select(ds, w, k) {
    setPlaying(false);
    const c = await ensure(ds);
    if (cur && cur.group.parent) scene.remove(cur.group);
    const dsChanged = st.ds !== ds || !cur;
    st.ds = ds; st.w = w; cur = c; segDs.set(ds, true);
    scene.add(c.group);
    const M = c.data.meta;
    winSel.innerHTML = '';
    M.labels.forEach((lab, i) => { const o = IX.h('option', { value: i }, winSel, `${RV.windowLabel(lab)} · RMSE ${M.rmse[i].toFixed(3)} · #${M.windows[i]}`); if (i === w) o.selected = true; });
    if (dsChanged) {
      jSel.innerHTML = '';
      M.joints.forEach((n, i) => IX.h('option', { value: i }, jSel, n));
      st.joint = Math.min(st.joint, M.D - 1); jSel.value = st.joint;
      slK.range(0, M.draws_exported);
      setView('persp');
    }
    eeKey = ''; applyMode(); setK(k);
    writeHash();
  }
  /** switch between the ghost version and the median version (materials, visibility, legend) */
  function applyMode() {
    if (!cur) return;
    const med = st.mode === 'median', base = med ? MAT.refT : MAT.ref;
    for (const n in cur.insts.ref.bodies) for (const m of cur.insts.ref.bodies[n].meshes) m.userData.base = base;
    cur.insts.ref.root.renderOrder = med ? 4 : 0;
    slK.input.disabled = med;
    document.documentElement.classList.toggle('mode-median', med);
    segMode.set(st.mode, true);
    eeKey = ''; highlight(); applyVisibility(); drawEE(); readout();
  }
  function applyVisibility() {
    if (!cur) return;
    const med = st.mode === 'median';
    cur.insts.ref.root.visible = true;
    cur.insts.med.root.visible = med;
    cur.insts.ghosts.forEach((g, i) => { g.root.visible = !med && i < st.K; });
    eeKey = ''; drawTraces(cur); drawEE(); requestRender();
  }
  function highlight() {
    if (!cur) return;
    const names = new Set(RV.jointBodies(st.ds, st.joint));
    for (const n in cur.insts.ref.bodies) for (const m of cur.insts.ref.bodies[n].meshes) m.material = names.has(n) ? MAT.hi : m.userData.base;
    requestRender();
  }
  function setK(k) {
    if (!cur) return;
    st.k = Math.max(0, Math.min(cur.data.meta.T - 1, k));
    const D = cur.data, w = st.w;
    pose(cur.insts.ref, RV.row(D.truth, w, st.k));
    pose(cur.insts.med, RV.row(D.q50, w, st.k));
    cur.insts.ghosts.forEach((g, i) => pose(g, RV.row(D.draws, w, i, st.k)));
    drawTraces(cur); drawTimeline(); drawEE(); readout(); requestRender();
  }
  function setPlaying(on) { st.playing = on; playBtn.textContent = on ? 'Pause' : 'Play'; if (on) { last = performance.now(); frames = []; requestRender(); } }
  function writeHash() { try { history.replaceState(null, '', `#ds=${st.ds}&w=${st.w}` + (st.mode === 'median' ? '&m=median' : '')); } catch (e) { /* file:// in some browsers */ } }

  // ------------------------------------------------------------------ end-effector trajectories (2D projection, full window)
  const PROJ = { top: [0, 1, 'x', 'y'], front: [0, 2, 'x', 'z'], side: [1, 2, 'y', 'z'] };
  const eeHost = $('ee'), eeRead = $('ee-read');
  IX.seg($('ee-controls'), { label: 'projection', value: st.proj, options: [
    { key: 'top', label: 'top (x–y)' }, { key: 'front', label: 'front (x–z)' }, { key: 'side', label: 'side (y–z)' }],
  onChange: v => { st.proj = v; eeKey = ''; drawEE(); } });
  let eeKey = '', eeLayer = null;
  function drawEE() {
    if (!cur) return;
    const D = cur.data, T = D.meta.T, arms = RV.ROBOTS[st.ds].ee.length, P = eePaths(cur, st.w);
    const [ax, ay, nx, ny] = PROJ[st.proj], med = st.mode === 'median';
    const at = (i, t, a) => ((i * T + t) * arms + a) * 3;
    const insts = med ? [1] : Array.from({ length: st.K }, (_, i) => 2 + i);
    const W = eeHost.clientWidth || 360, H = Math.round(Math.max(300, Math.min(470, W * 0.95))), m = { l: 48, r: 10, t: 10, b: 36 };
    const key = [st.ds, st.w, st.mode, st.K, st.proj, W].join('|');
    if (key !== eeKey) {
      eeKey = key;
      eeHost.textContent = '';
      // equal scales on both axes; bounds over the reference, median and all exported draws (stable across modes)
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let i = 0; i < 2 + D.meta.draws_exported; i++) for (let t = 0; t < T; t++) for (let a = 0; a < arms; a++) {
        const o = at(i, t, a), u = P[o + ax], v = P[o + ay];
        if (u < x0) x0 = u; if (u > x1) x1 = u; if (v < y0) y0 = v; if (v > y1) y1 = v;
      }
      const w = W - m.l - m.r, h = H - m.t - m.b, span = Math.max((x1 - x0) / w, (y1 - y0) / h) * 1.08;
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      const sx = IX.linear(cx - span * w / 2, cx + span * w / 2, 0, w), sy = IX.linear(cy - span * h / 2, cy + span * h / 2, h, 0);
      const svg = IX.el('svg', { width: W, height: H, role: 'img' }, eeHost);
      const g = IX.el('g', { transform: `translate(${m.l},${m.t})` }, svg);
      const tk = (a, b) => IX.ticks(a, b, 4);
      IX.axes(g, { sx, sy, w, h, xt: tk(...sx.domain), yt: tk(...sy.domain), xLabel: `${nx} [m]`, yLabel: `${ny} [m]`, yLabelOffset: 38 });
      const poly = (i, a, t1) => { let d = ''; for (let t = 0; t <= t1; t++) { const o = at(i, t, a); d += (t ? 'L' : 'M') + sx(P[o + ax]).toFixed(1) + ',' + sy(P[o + ay]).toFixed(1); } return d; };
      for (let a = 0; a < arms; a++) {
        for (const i of insts) IX.el('path', { d: poly(i, a, T - 1), fill: 'none', stroke: C.method, 'stroke-width': med ? 2.2 : 0.9,
          'stroke-opacity': med ? 0.95 : 0.3 }, g);
        IX.el('path', { d: poly(0, a, T - 1), fill: 'none', stroke: C.trace, 'stroke-width': 1.6, 'stroke-dasharray': med ? '5 3' : null }, g);
        const o = at(0, 0, a);
        IX.el('rect', { x: sx(P[o + ax]) - 4, y: sy(P[o + ay]) - 4, width: 8, height: 8, fill: '#fff', stroke: C.trace, 'stroke-width': 1.5 }, g);
        if (arms > 1) IX.el('text', { x: sx(P[o + ax]) + 7, y: sy(P[o + ay]) - 6, 'font-size': 11, fill: 'var(--muted)' }, g).textContent = a ? 'right arm' : 'left arm';
      }
      eeLayer = { g: IX.el('g', {}, g), sx, sy };
      svg.setAttribute('aria-label', `End-effector paths over the whole window, ${st.proj} projection; square: start`);
    }
    // current step: travelled part of the reference highlighted, markers at step k
    const L = eeLayer; L.g.textContent = '';
    let err = 0, spread = 0;
    for (let a = 0; a < arms; a++) {
      let d = '';
      for (let t = 0; t <= st.k; t++) { const o = at(0, t, a); d += (t ? 'L' : 'M') + L.sx(P[o + ax]).toFixed(1) + ',' + L.sy(P[o + ay]).toFixed(1); }
      IX.el('path', { d, fill: 'none', stroke: C.trace, 'stroke-width': 2.6 }, L.g);
      if (!med) for (const i of insts) { const o = at(i, st.k, a); IX.el('circle', { cx: L.sx(P[o + ax]), cy: L.sy(P[o + ay]), r: 2.4, fill: C.method, 'fill-opacity': 0.6 }, L.g); }
      const om = at(1, st.k, a), or = at(0, st.k, a);
      if (med) IX.el('circle', { cx: L.sx(P[om + ax]), cy: L.sy(P[om + ay]), r: 5, fill: C.method, stroke: '#fff', 'stroke-width': 1.2 }, L.g);
      IX.el('circle', { cx: L.sx(P[or + ax]), cy: L.sy(P[or + ay]), r: 4.5, fill: C.trace, stroke: '#fff', 'stroke-width': 1.2 }, L.g);
      const dist = (o1, o2) => Math.hypot(P[o1] - P[o2], P[o1 + 1] - P[o2 + 1], P[o1 + 2] - P[o2 + 2]);
      err += dist(om, or) / arms;
      let s = 0; for (let i = 2; i < 2 + D.meta.draws_exported; i++) s += dist(at(i, st.k, a), om);
      spread += s / D.meta.draws_exported / arms;
    }
    eeRead.innerHTML = `step ${st.k + 1}: median end effector ${(err * 1000).toFixed(1)} mm from the reference; ` +
      `posterior draws on average ${(spread * 1000).toFixed(1)} mm from the median (3D distances${arms > 1 ? ', mean of both arms' : ''}).`;
  }

  // ------------------------------------------------------------------ timeline (SVG): selected joint and band width
  const tl = $('timeline');
  function drawTimeline() {
    if (!cur) return;
    tl.textContent = '';
    const D = cur.data, M = D.meta, T = M.T, j = st.joint, w = st.w;
    const W = tl.clientWidth || 800, m = { l: 56, r: 12, t: 8, b: 30 }, h1 = 120, h2 = 54, gap = 14, H = m.t + h1 + gap + h2 + m.b;
    const svg = IX.el('svg', { width: W, height: H, role: 'img', 'aria-label': `${M.joints[j]} over the window, with the 90% band and its width` }, tl);
    const col = (arr, t) => RV.row(arr, w, t)[j];
    const ts = Array.from({ length: T }, (_, t) => t);
    const ref = ts.map(t => col(D.truth, t)), med = ts.map(t => col(D.q50, t)), lo = ts.map(t => col(D.q05, t)), hi = ts.map(t => col(D.q95, t));
    const obs = ts.map(t => col(D.obs, t)), wid = ts.map(t => hi[t] - lo[t]);
    const ymin = Math.min(...lo, ...ref, ...obs), ymax = Math.max(...hi, ...ref, ...obs), pad = (ymax - ymin) * 0.06 || 0.01;
    const sx = IX.linear(0, T - 1, 0, W - m.l - m.r), sy = IX.linear(ymin - pad, ymax + pad, h1, 0);
    const g1 = IX.el('g', { transform: `translate(${m.l},${m.t})` }, svg);
    IX.axes(g1, { sx, sy, w: W - m.l - m.r, h: h1, xt: IX.ticks(0, T - 1, 8), yt: IX.ticks(ymin - pad, ymax + pad, 4), noXLabels: true,
      yLabel: `${M.joints[j]} [${M.units[j]}]`, yLabelOffset: 44 });
    let d = IX.path(ts, hi, sx, sy); for (let t = T - 1; t >= 0; t--) d += `L${sx(t).toFixed(1)},${sy(lo[t]).toFixed(1)}`;
    IX.el('path', { d: d + 'Z', fill: C.method, 'fill-opacity': 0.18 }, g1);
    for (let t = 0; t < T; t += 1) IX.el('circle', { cx: sx(t), cy: sy(obs[t]), r: 1.4, fill: '#6e7781', 'fill-opacity': 0.6 }, g1);
    IX.el('path', { d: IX.path(ts, med, sx, sy), fill: 'none', stroke: C.method, 'stroke-width': 1.8 }, g1);
    IX.el('path', { d: IX.path(ts, ref, sx, sy), fill: 'none', stroke: '#1f2328', 'stroke-width': 1.4 }, g1);
    const g2 = IX.el('g', { transform: `translate(${m.l},${m.t + h1 + gap})` }, svg);
    const wmax = Math.max(...wid) * 1.1 || 1, sy2 = IX.linear(0, wmax, h2, 0);
    IX.axes(g2, { sx, sy: sy2, w: W - m.l - m.r, h: h2, xt: IX.ticks(0, T - 1, 8), yt: IX.ticks(0, wmax, 2), xLabel: `step k (Δt = ${M.dt.toFixed(3)} s)`,
      yLabel: '90 % width', yLabelOffset: 44 });
    IX.el('path', { d: IX.path(ts, wid, sx, sy2), fill: 'none', stroke: C.method, 'stroke-width': 1.6, 'stroke-dasharray': '5 3' }, g2);
    for (const [g, hh] of [[g1, h1], [g2, h2]]) IX.el('line', { x1: sx(st.k), x2: sx(st.k), y1: 0, y2: hh, stroke: '#0b5cad', 'stroke-width': 1.5 }, g);
    const hit = IX.el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: h1 + gap + h2, fill: 'transparent', style: 'cursor:pointer' }, svg);
    const go = e => { const r = hit.getBoundingClientRect(); setPlaying(false); setK(Math.round(sx.inv(e.clientX - r.left))); };
    hit.addEventListener('pointerdown', e => { go(e); const mv = ev => go(ev); const up = () => { window.removeEventListener('pointermove', mv); window.removeEventListener('pointerup', up); };
      window.addEventListener('pointermove', mv); window.addEventListener('pointerup', up); });
    IX.reportHeight();
  }

  // ------------------------------------------------------------------ exact readout of the exported values at step k
  function readout() {
    if (!cur) return;
    const D = cur.data, M = D.meta, w = st.w, k = st.k;
    const r = a => RV.row(a, w, k), f = v => (+v).toFixed(4);
    const ref = r(D.truth), med = r(D.q50), lo = r(D.q05), hi = r(D.q95), ob = r(D.obs);
    let h = `<caption>step k = ${k + 1} of ${M.T} (t = ${((k + 1) * M.dt).toFixed(2)} s) · test window #${M.windows[w]}</caption>` +
      '<tr><th>joint</th><th>reference</th><th>median</th><th>5 %</th><th>95 %</th><th>measurement</th></tr>';
    M.joints.forEach((n, j) => {
      h += `<tr${j === st.joint ? ' class="sel"' : ''} data-j="${j}"><td>${n} [${M.units[j]}]</td><td>${f(ref[j])}</td><td>${f(med[j])}</td>` +
        `<td>${f(lo[j])}</td><td>${f(hi[j])}</td><td>${f(ob[j])}</td></tr>`;
    });
    const tab = $('readout'); tab.innerHTML = h;
    tab.querySelectorAll('tr[data-j]').forEach(tr => tr.addEventListener('click', () => { st.joint = +tr.dataset.j; jSel.value = st.joint; highlight(); drawTimeline(); readout(); }));
    $('status').textContent = `${M.label}: test window #${M.windows[w]} of ${M.n_windows} (${RV.windowLabel(M.labels[w])}, RMSE ${M.rmse[w].toFixed(3)} normalized), ` +
      `${M.draws_total} draws per step, ` + (st.mode === 'median' ? 'posterior median shown (median of all draws, per joint).' : `${st.K} shown as ghost arms${st.reducedNote}.`);
  }

  // ------------------------------------------------------------------ render loop, playback, frame-rate guard
  let needRender = true, last = 0, acc = 0, frames = [];
  function requestRender() { needRender = true; }
  function resize() {
    const W = host.clientWidth || 800, H = Math.round(Math.max(340, Math.min(520, W * 0.68)));
    renderer.setSize(W, H); camera.aspect = W / H; camera.updateProjectionMatrix(); requestRender(); drawTimeline(); drawEE();
  }
  function loop(now) {
    requestAnimationFrame(loop);
    if (st.playing && cur) {
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      frames.push(dt); if (frames.length > 90) frames.shift();
      acc += dt * st.speed;
      const step = cur.data.meta.dt;
      if (acc >= step) { const n = Math.floor(acc / step); acc -= n * step; setK(st.k + n >= cur.data.meta.T ? 0 : st.k + n); }
      if (frames.length === 90 && st.K > 3) {
        const fps = 90 / frames.reduce((a, b) => a + b, 0);
        if (fps < 30) { st.K = Math.max(3, Math.floor(st.K * 0.7)); slK.set(st.K, true); st.reducedNote = ` (reduced automatically to keep ≥ 30 fps; measured ${fps.toFixed(0)} fps)`; frames = []; applyVisibility(); readout(); }
      }
    }
    if (needRender) { renderer.render(scene, camera); needRender = false; }
  }

  window.addEventListener('resize', resize);
  // #ds=aloha&w=10&k=120 opens a robot, a window (0-10) and a step; &still hides the controls (static fallback)
  const h = /ds=(aloha|reassemble)(?:&w=(\d+))?(?:&k=(\d+))?/.exec(location.hash);
  if (/still/.test(location.hash)) document.documentElement.classList.add('still');
  if (/m=median/.test(location.hash)) st.mode = 'median';
  { const pr = /proj=(top|front|side)/.exec(location.hash); if (pr) st.proj = pr[1]; }
  resize();
  select(h ? h[1] : st.ds, h && h[2] ? Math.min(10, +h[2]) : 5, h && h[3] ? +h[3] : 0).then(() => { resize(); requestAnimationFrame(loop); })
    .catch(e => { note.textContent = 'could not load data: ' + e.message; });
})();
