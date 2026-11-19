# Interactive figures (Beyond PDF)

Each figure lives in `interactive/<slug>/` and opens directly from `file://` or from any static server; no build step,
no network requests at view time. `beyond_pdf/build_beyond_pdf.py` flattens every figure into one self-contained HTML
file for the TMLR submission, because the TMLR kit copies only one directory level below `assets/html/`.

```
interactive/
  _shared/ix.js, ix.css      helpers (seeded RNG, scales, SVG axes, accessible sliders/buttons), shared styles
  _shared/vendor/            third-party libraries (none used yet; see vendor/README.md)
  <slug>/index.html          page + caption block (caption hidden when embedded in the web version)
  <slug>/model.js            pure math, shared by the page, the tests and the fallback export
  <slug>/app.js              page logic
  <slug>/fallback.{svg,png}        static fallback, double column (6.75 in), PNG at 300 dpi
  <slug>/fallback-1col.{svg,png}   static fallback, single column (3.25 in), PNG at 300 dpi
```

Commands (from the repository root):

```
node --test tests/*.test.js                         # numeric acceptance checks (Node >= 18, no packages)
python scripts/make_fallbacks.py                    # regenerate all fallbacks (matplotlib + Node)
python beyond_pdf/build_beyond_pdf.py               # embed the figures in the web version
```

Conventions: classic scripts instead of ES modules (Chrome refuses module scripts from `file://`); Okabe-Ito colours,
every series also distinguished by line style or marker shape; keyboard-operable controls with visible focus and
`aria-label`s; mathematical notation as HTML italics/sub-/superscripts (KaTeX was not needed).

## overview (paper Figure 1)

The paper's Figure 1 with a moving time point. `figure-svg.js` is the unchanged paper SVG
(`paper/figures/NormFlowSDE_figure.svg`), wrapped as a string by `scripts/sync_overview_figure.py` (rerun after changing
the paper figure; a test fails if they differ). `app.js` moves only the time-dependent parts:

- (a) the measurement strip is redrawn with N = 20 columns; the box `y_{t−o+1:t}` and its label slide with t;
  measurements before the window are muted, those after t are faded (not yet available, causal); the context cells
  `c_t` change shade with t (deterministic per t, schematic).
- (c) the measurement times are marked on the time axis, the window is shaded, box (b) covers [t, t + Δt] and encloses
  the 90 % band, the zoom lines to panel (b) follow it, and z_t is marked on the median.
- (b) the one-step distribution follows t: a two-component mixture (16 sample paths, density, drift end, sampled
  z_{t+dt}, arrows and label) whose centre, mode separation, weights and widths vary smoothly with t and whose widths
  scale with the 90 % band of (c) at t + dt; at the default t it reproduces the paper drawing (modes 255/345 px, weights
  0.6/0.4, drift end 312 px). Samples are fixed quantiles and the paths fixed Brownian bridges, so playback is
  continuous (380 ms tween; none with reduced motion). Schematic, like the paper figure.
- Controls: play/pause, step buttons, slider, click in panel (c); `index.html?t=5` opens at a given step.
- The schematic uses o = 4 (as drawn in the paper figure); the trained models use o = 10 (Appendix C).

Fallback: the default state (t = 10, box (b) near its position in the paper figure), from headless Chrome
(`make_fallbacks.py` needs Chrome or Edge; set `CHROME=` otherwise). The paper's own Figure 1 remains the PDF figure.

## robot-view

NormFlowSDE's joint-space posteriors on ALOHA and REASSEMBLE replayed on the robots' kinematic models (three.js r147,
vendored in `_shared/vendor/three-0.147.0/` with its MIT licence). Kinematic replay only; no physics is simulated.

- Robot models (`models/*.js`, built by `scripts/build_robot_models.py`):
  - ALOHA: the two ViperX 300 arms of the ALOHA simulation (gym-aloha MJCF and meshes, Apache-2.0; licence copied),
    i.e. the simulator the dataset was recorded in. State order verified in its `get_qpos`: left arm (waist, shoulder,
    elbow, forearm roll, wrist angle, wrist rotate), left gripper, right arm, right gripper. The gripper value is the
    finger position normalised between closed (0.01844 m) and open (0.058 m), mapped back to the two finger slides.
  - REASSEMBLE: Franka Panda from MuJoCo Menagerie (Apache-2.0; licence copied), joints 1-7 in rad. The state has no
    gripper coordinate, so the fingers are drawn open.
  - Visual meshes merged per link and colour, decimated and quantised: 0.33 MB (ALOHA, 27k triangles), 0.44 MB (Franka,
    36k triangles).
- Data (`data/*.js`, from `scripts/export_robot_view.py` on the server via `scripts/slurm/export_robot_view.sbatch`,
  then `scripts/make_robot_view_data.py` locally): NormFlowSDE seed 0; 11 test windows per robot (RMSE of the posterior
  mean in normalized units; quantiles 0, 0.1, …, 0.9 plus the worst window); per window reference, measurements, mean,
  median and 5/95 % quantiles from all draws (float32, physical units) and the first 30 draws (float16). The windows'
  RMSE reproduces the run metric (ALOHA 0.0523, REASSEMBLE 0.0314). 4.3 MB (ALOHA) and 2.2 MB (REASSEMBLE), loaded
  only when that robot is selected.
- Two versions (`show`): reference arm (opaque grey) with up to 30 ghost arms from the draws (translucent, shared
  material, no depth writes), or the posterior median (opaque orange) with the reference drawn translucent over it.
  3D end-effector traces of the last 20 steps.
- End-effector panel: the trajectories of the whole window in a top, front or side projection (equal axis scales), the
  reference bold up to step k, start and current markers, and the 3D distance median-reference and the mean distance
  of the draws to the median at step k (mm); orbit camera with front/side/top presets; play/pause, step, speed 0.25-2x; timeline of the selected joint
  (band, median, reference, measurements) and of the 90 % band width, click to jump; exact readout of all joints at the
  current step (the test checks the decoded values bit for bit against the exported arrays). Below 30 fps the number
  of ghost arms is reduced automatically and the status line says so. `#ds=aloha&w=10&k=120` opens a given view.

Not yet included (needs decisions): the contact signal r_k (defined in the paper only for ALOHA; REASSEMBLE has no
commanded joint positions) and contact-phase shading; until then the figure makes no claim about contact.

Fallback: four frames (k = 0, 80, 160, 240) of the median REASSEMBLE window, 3D view and timeline, from headless Chrome.

## likelihood-collapse

**Not in the web version** (removed from `build_beyond_pdf.py` at the author's request; the source is kept here).

Why the learned measurement densities of the particle filters collapse to the moment-matched Gaussian (paper
Section 4.3, "Shape of the observation noise"; Figure "noise recovery").

- Noise model (paper Eq. `obsnoise`): per observed coordinate, 1-D,
  `0.70 N(0, (0.5σ)²) + 0.15 N(2.5σ, (1.5σ)²) + 0.15 N(−2.5σ, (1.5σ)²)`; sd 1.65σ. All quantities in units of σ.
- The paper reports the overlap `OVL = ∫ min(p, q) = 1 − TV` = 0.668 for the moment-matched Gaussian.
- Panel A sliders: centre weight `w0`, share of the side weight at `+a` (0.5 = symmetric, as in the paper), side offset
  `a`, centre scale, side scale; "Reset to paper". TV = ½∫|p − q| by the trapezoid rule on 24 001 points over
  mean ± 12 sd.
- Panel B: one bootstrap weighting step, N = 200 particles `x_i = s_prior · ε_i` (mulberry32 seed 1, Box–Muller),
  true state x* = 0, observation y; weights ∝ likelihood of `y − x_i`; posterior mean, sd and ESS = 1/Σw².

Acceptance (`tests/likelihood-collapse.test.js`): TV at the paper values = **0.3320** (overlap 0.668).

Panel-B effects at the defaults (s_prior = 0.5σ, seed 1):

| observation | mixture: mean / sd / ESS | Gaussian: mean / sd / ESS | effect |
|---|---|---|---|
| side mode, y = 2.5 | 0.083 / 0.502 / 198.7 | 0.271 / 0.467 / 170.8 | Gaussian over-trusts y (mean pulled 3× further from x*) |
| spike, y = 0.3 | 0.196 / 0.374 / 169.0 | 0.087 / 0.486 / 198.9 | Gaussian posterior too wide (sd +30 %) |

Dependence on the prior width (checked, not tuned silently): the over-trust in the side mode is visible for
s_prior ≤ 0.75σ. From s_prior ≈ 1σ the exact posterior becomes bimodal (it puts mass on "y came from the spike"), and
the Gaussian then *under*-trusts the observation (s_prior = 1.5: mixture mean 1.58, Gaussian 1.27). The spike effect
holds for all prior widths tested (0.5–2σ). The default 0.5σ is the width at which the side-mode effect is clearest; the
slider lets readers see both regimes.

Fallback: Panel A at the defaults and Panel B with the side-mode observation (side by side; stacked in the 1-col version).

## energy-score

**Not in the web version** (removed from `build_beyond_pdf.py` at the author's request; the source is kept here).

The two terms of the energy score, their difference, and the entropy term, for a toy ensemble: reference
`x ~ N(0, I_d)`, members `z = c ε`, `ε ~ N(0, I_d)`.

- Paper: rollout score `E‖z − x‖ − ½ E‖z − z′‖` (Eq. `flow_sde_rollout_loss`) with the unbiased estimator
  `1/(M(M−1)) Σ_{i≠j}` (Eq. `energy-score-estimator`); stage 2 uses it alone, with M = 16 rollouts and the learned scale
  `s_φ` (Eq. `learned-noise-scale`); stage 1 adds it with weight λ_rollout to the reconstruction and entropy terms
  (Eq. `normflowsde_loss`, entropy weight 1).
- All curves are closed form: `E‖N(0, s² I_d)‖ = s κ_d`, `κ_d = √2 Γ((d+1)/2)/Γ(d/2)` (exact recursion in `model.js`),
  so `E‖z − x‖ = κ_d √(1 + c²)`, `E‖z − z′‖ = κ_d √2 c`. Entropy term: `H(N(0, c² I_d)) = d log c + (d/2) log(2πe)`.
  The tests cross-check the closed form against seeded Monte Carlo (d = 7, 20 000 samples).
- Controls: spread c (log, 0.2–3), d ∈ {1, 2, 7, 12, 14} (pendulum, REASSEMBLE, Lorenz-96, ALOHA), M ∈ {4, 16, 50, 200}
  (4 = stage-1 rollout draws, 16 = stage-2 rollouts), spread estimator (paper's unbiased form vs. 1/M² with self-pairs).
  Panel B: c0 ∈ {0.6, 1.6}, scale s (log, 0.2–3).

Acceptance (`tests/energy-score.test.js`): d = 1 closed forms; argmin c = 1.000 for β = ½; ES(s·c0) minimised at
s = 1/c0; entropy strictly increasing in c.

**argmin c does not depend on d.** κ_d only scales every curve, and the minimiser solves `c/√(1+c²) = √2 β f`, with
f = 1 (unbiased) or (M−1)/M (1/M² form). For β = ½: c* = 1.000 at d = 1, 2, 7, 12 and 14. With the 1/M² estimator the
optimum moves below 1 (M = 4: c* = 0.626; M = 16: 0.885; M = 50: 0.961; M = 200: 0.990), still independent of d.

Fallback: the main plot (both panels of A) at d = 7, β = ½.

## Deviations from the task specifications

- likelihood-collapse: the paper's TV claim is in Section 4.3 (there is no Section 4.4) and concerns all **four**
  particle filters (NSF-PF, FlowSDE-PF, NormFlowSDE-PF, NF-DPF), not three; the captions say so. The mixture is
  parameterised symmetrically as in the paper plus a "share at +a" slider for asymmetric side weights, instead of three
  free weights.
- energy-score: per the author's decision the spread weight 0.52 (used in code for REASSEMBLE stage 2, not stated in
  the paper) is not shown; β is fixed at ½ and the β slider, the 0.5/0.52 snap buttons and the 0.52 overlay in the
  fallback are dropped. The figure shows the four curves requested (E‖z−x‖, ½E‖z−z′‖, their sum = energy score,
  entropy term) against a spread slider; the entropy is drawn in its own panel (different units, nats) with a shared
  x axis, and without a stage-1 weighting λ_rollout (combining it with the score would also need the reconstruction
  term to be meaningful). d = 12 (Lorenz-96) added; M set {4, 16, 50, 200} instead of {10, 50, 200} to include the
  paper's values.
- Fallback file names: `fallback.{svg,png}` is the double-column version, `fallback-1col.{svg,png}` the single-column one.
- Numeric checks are Node tests (`tests/<slug>.test.js`); there is no separate `scripts/check_<slug>.py`.
- Dark mode is not implemented (the pages declare a white background, like the TMLR page they are embedded in).
