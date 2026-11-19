"""Build the TMLR Beyond PDF submission from the LaTeX manuscript and the frozen results.

Writes into tmlr-beyond-pdf-author-kit/submission_folder:
  submission.md                         converted manuscript (Distill/kramdown markdown)
  assets/bibliography/submission.bib    cleaned copy of paper/references.bib
  assets/img/submission/*.svg           static figures
  assets/html/submission/*.html         self-contained interactive figures (data inlined)

Usage:  python beyond_pdf/build_beyond_pdf.py
"""
import json
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPER = ROOT / 'paper'
RES = ROOT / 'results'
SUB = ROOT / 'tmlr-beyond-pdf-author-kit' / 'submission_folder'
FIG_SRC = HERE / 'figures'

# --------------------------------------------------------------------------------------
# Generic LaTeX helpers
# --------------------------------------------------------------------------------------


def match_brace(s, i):
    """s[i] == '{' -> index of the matching '}'."""
    assert s[i] == '{', s[i:i + 40]
    depth, j = 0, i
    while j < len(s):
        ch = s[j]
        if ch == '\\':
            j += 2
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return j
        j += 1
    raise ValueError('unbalanced brace at: ' + s[i:i + 80])


def replace_cmd(s, name, fn, nargs=1, opt=False):
    """Replace \\name[opt]{a1}...{an} by fn([opt,] a1, ..., an)."""
    pat = re.compile(r'\\' + name + r'(?![a-zA-Z])')
    out, pos = [], 0
    while True:
        m = pat.search(s, pos)
        if not m:
            out.append(s[pos:])
            break
        out.append(s[pos:m.start()])
        k = m.end()
        optarg = None
        if opt and k < len(s) and s[k] == '[':
            e = s.index(']', k)
            optarg, k = s[k + 1:e], e + 1
        args = []
        for _ in range(nargs):
            while k < len(s) and s[k] in ' \t\n':
                k += 1
            if k < len(s) and s[k] == '{':
                e = match_brace(s, k)
                args.append(s[k + 1:e])
                k = e + 1
                continue
            tm = re.match(r'\\[a-zA-Z]+|\\.|[^\s{}\\]', s[k:])  # single-token argument
            if not tm:
                raise ValueError(f'\\{name} without argument: ' + s[m.start():m.start() + 60])
            args.append(tm.group(0))
            k += tm.end()
        out.append(fn(optarg, *args) if opt else fn(*args))
        pos = k
    return ''.join(out)


def strip_comments(t):
    return '\n'.join(re.sub(r'(?<!\\)%.*$', '', ln) for ln in t.split('\n'))


def read_tex(p):
    return strip_comments(Path(p).read_text(encoding='utf-8'))


def slugify(s):
    """Jekyll's default slugify."""
    out, prev_dash = [], False
    for ch in s.lower():
        cat = unicodedata.category(ch)
        if cat[0] in 'LM' or cat == 'Nd':
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append('-')
            prev_dash = True
    return ''.join(out).strip('-')


ACCENTS = {
    "\\'": {'a': 'á', 'e': 'é', 'i': 'í', 'o': 'ó', 'u': 'ú', 'E': 'É'},
    '\\"': {'a': 'ä', 'o': 'ö', 'u': 'ü', 'A': 'Ä', 'O': 'Ö', 'U': 'Ü'},
    '\\^': {'o': 'ô', 'e': 'ê', 'a': 'â', 'i': 'î'},
    '\\`': {'a': 'à', 'e': 'è'},
    '\\v': {'s': 'š', 'c': 'č', 'z': 'ž', 'S': 'Š'},
    '\\~': {'n': 'ñ', 'a': 'ã'},
}


def accents(t):
    for a, u in (('O', 'Ø'), ('o', 'ø'), ('ss', 'ß'), ('aa', 'å'), ('AA', 'Å'), ('l', 'ł'), ('L', 'Ł')):
        t = re.sub(r'\{\\' + a + r'\}', u, t)
        t = re.sub(r'\\' + a + r'(?![a-zA-Z])\s?', u, t) if len(a) > 1 or a in 'OL' else t
    for cmd, table in ACCENTS.items():
        c = re.escape(cmd)
        for a, u in table.items():
            t = re.sub(r'\{' + c + r'\{?' + a + r'\}?\}', u, t)
            t = re.sub(c + r'\{' + a + r'\}', u, t)
            if cmd[1] in '\'"^`~':
                t = re.sub(c + a, u, t)
            else:
                t = re.sub(c + ' ' + a, u, t)
    return t


# --------------------------------------------------------------------------------------
# Paper-specific content edits for the web version
# --------------------------------------------------------------------------------------

WEB_EDITS = [
    # Multiple-shooting partition: one line in the PDF, two lines on the web
    (' S=\\left\\lceil\\frac{N}{L_S-1}\\right\\rceil,\\qquad\n'
     ' \\sum_{s=1}^S(L^{(s)}-1)=N,\\qquad\n'
     ' L^{(s)}=1+\\min\\{L_S-1,\\,N-(s-1)(L_S-1)\\},\\qquad s=1,\\ldots,S.\n',
     ' \\begin{gathered}\n'
     ' S=\\left\\lceil\\frac{N}{L_S-1}\\right\\rceil,\\qquad\n'
     ' \\sum_{s=1}^S(L^{(s)}-1)=N,\\\\\n'
     ' L^{(s)}=1+\\min\\{L_S-1,\\,N-(s-1)(L_S-1)\\},\\qquad s=1,\\ldots,S.\n'
     ' \\end{gathered}\n'),
    # The static grid figures are interactive here.
    ('drawn in blue in panels (a) to (d), explain this difference',
     'drawn in blue in Figure~\\ref{fig:noise-recovery} when a particle filter is selected, explain this difference'),
    ('for all eight learned methods, with the direct estimators in the top row and the particle filters in the bottom row.',
     'for all eight learned methods; the direct estimators and the particle filters can be selected as groups.'),
]

INJECT = [
    # (anchor text, position, latex to insert)
    ('measured for every method on the same device. ', 'after',
     ' Figure~\\ref{fig:benchmark-explorer} plots the same results as accuracy against inference cost.\n\n'
     '\\begin{figure}\\INTERACTIVE{benchmark}{560}\\caption{CAPTION}\\label{fig:benchmark-explorer}\\end{figure}\n'),
    ('only 6 out of 12 observable states. ', 'after',
     ' Figure~\\ref{fig:posterior-explorer} shows individual filtering distributions of \\method on Lorenz--96, '
     'including the six coordinates that are never observed, and on the pendulum.\n\n'
     '\\begin{figure}\\INTERACTIVE{posterior}{680}\\caption{CAPTION}\\label{fig:posterior-explorer}\\end{figure}\n'),
    # comparison of the eight methods (web only)
    ('Throughout, $x_k$ denotes the state, $y_k$ the measurement, and $u_k$ the control at step $k$.', 'after',
     ' Table~\\ref{tab:method-comparison} summarizes the eight methods and Figure~\\ref{fig:methods} draws one filtering '
     'step of each.\n\n'
     '\\begin{table}\\centering\n\\begin{tabular}{lllll}\n\\toprule\n'
     'Method & Measurements enter through & Stochastic part of the update & Transition density & Training \\\\\n'
     '\\midrule\n'
     '\\textbf{NormFlowSDE} & encoded window, into drift and flow & conditional flow on the noise & exact & '
     'reconstruction, transition, continuity, rollout score \\\\\n'
     'Latent NSF & encoder output: Gaussian posterior & Gaussian posterior (flow in the prior) & Gaussian & '
     '$\\beta$-ELBO + flow consistency \\\\\n'
     'Latent SDE & encoding in the posterior drift & Gaussian increments & Gaussian & ELBO with path KL \\\\\n'
     'SigKer SDE & encoding in drift and diffusion & Gaussian increments & none (sample-based) & '
     'signature-kernel score \\\\\n'
     '\\cmidrule(l){1-5}\n'
     'NF-DPF & proposal and likelihood & flows for transition and proposal & exact & '
     'particle likelihood or state error \\\\\n'
     'NSF-PF & likelihood only; bootstrap proposal & NSF: Gaussian candidate + flow & exact & NLL + consistency \\\\\n'
     'FlowSDE-PF & likelihood only; bootstrap proposal & Gaussian, fixed diffusion $D$ & Gaussian & flow matching \\\\\n'
     'NormFlowSDE-PF & likelihood only; bootstrap proposal & NormFlowSDE transition & exact & recipe of NSF-PF \\\\\n'
     '\\bottomrule\n\\end{tabular}\n'
     '\\caption{The eight learned estimators of the benchmark. The four direct estimators (top) sample the next state '
     'from encoded measurements; the four particle filters (bottom) propagate particles with a learned transition and '
     'weight them with a learned measurement density. NSF-PF, FlowSDE-PF, and NormFlowSDE-PF are bootstrap filters: '
     'their proposal is the learned transition itself, which does not see the new measurement, and what they share is '
     'the conditional-flow measurement density. Training refers to the transition or update model.}'
     '\\label{tab:method-comparison}\\end{table}\n\n'
     '\\begin{figure}\\INTERACTIVE{methods}{330}\\caption{CAPTION}\\label{fig:methods}\\end{figure}\n'),
    # explainers (web only)
    ('time $t_k$, control $u_k$ when available, and window feature $c_k$.', 'after',
     ' Figure~\\ref{fig:update-explainer} illustrates one such update in one dimension.\n\n'
     '\\begin{figure}\\INTERACTIVE{update}{330}\\caption{CAPTION}\\label{fig:update-explainer}\\end{figure}\n'),
    ('which serves as a reference value for a Gaussian noise model.', 'after',
     ' Figure~\\ref{fig:noise-explainer} shows the mixture and what measurements drawn from it look like.\n\n'
     '\\begin{figure}\\INTERACTIVE{noise_explainer}{330}\\caption{CAPTION}\\label{fig:noise-explainer}\\end{figure}\n'),
    # interactive/ figures (multi-file sources, flattened into one HTML file each by write_interactive)
    ('The energy score helps maintain diversity among generated samples while encouraging agreement with the reference '
     'trajectory.', 'after',
     ' Figure~\\ref{fig:energy-score} compares the two terms of the energy score with the entropy term of '
     'Equation~\\eqref{eq:transition-training-loss} for a Gaussian toy ensemble.\n\n'
     '\\begin{figure}\\INTERACTIVE{energy-score}{760}\\caption{CAPTION}\\label{fig:energy-score}\\end{figure}\n'),
    ('latent NSF has the narrowest intervals on both datasets. The signature-kernel SDE is strongly under-dispersed.',
     'after',
     ' Figure~\\ref{fig:robot-view} replays the joint-space filtering distributions of \\method on the kinematic models '
     'of both robots.\n\n'
     '\\begin{figure}\\INTERACTIVE{robot-view}{1100}\\caption{CAPTION}\\label{fig:robot-view}\\end{figure}\n'),
    ('0.5 to 1.1\\,ms slower than them on the other three datasets.', 'after',
     ' Figure~\\ref{fig:race} replays these latencies as the filtering of one test window.\n\n'
     '\\begin{figure}\\INTERACTIVE{race}{420}\\caption{CAPTION}\\label{fig:race}\\end{figure}\n'),
    ('Centering the stochastic increment and compensating the drift then avoid a divergent mean contribution as $h\\to0$.',
     'after',
     ' Figure~\\ref{fig:centering-explainer} shows this effect on the data-generating chain of the numerical illustration '
     'in Appendix~\\ref{sec:limit-scope}.\n\n'
     '\\begin{figure}\\INTERACTIVE{centering}{360}\\caption{CAPTION}\\label{fig:centering-explainer}\\end{figure}\n'),
]

# One-sentence summaries shown under the main section headings (web only).
KEY_IDEAS = {
    '1-introduction': 'Keep the step-by-step structure of a neural SDE, but replace its Gaussian noise with a conditional '
                      'normalizing flow that reads the recent measurements. One trained network then filters new sequences '
                      'without per-sequence optimization.',
    '2-related-work': 'Learned estimators come in two families. <b>Direct estimators</b> sample the next state from encoded '
                      'measurements; <b>particle filters</b> propagate particles with a learned transition and weight them '
                      'with a learned likelihood. NormFlowSDE is a direct estimator, and NormFlowSDE-PF puts its transition '
                      'into a particle filter.',
    '3-normflowsde-for-amortized-bayesian-state-estimation':
        'Each update adds a learned drift and flow-transformed noise, both conditioned on a window of measurements. Training '
        'combines reconstruction, the exact transition likelihood, continuity between segments, and an energy score on full '
        'rollouts.',
    '4-empirical-evaluation': 'Four datasets, eight learned methods, three seeds each. Particle filters win on the smooth '
                              'simulated systems; on the robot data the direct route wins, and NormFlowSDE is the most '
                              'accurate, calibrated, and about four times faster than every particle filter.',
    'a-weak-convergence-of-the-centered-normflowsde-inference-scheme':
        'With the trained networks fixed, a centered version of the update converges to an Itô SDE as the step shrinks. '
        'Without centering, the nonzero mean of the flow output becomes a drift that grows like \\(1/\\sqrt{h}\\).',
}

# Long derivations that are collapsed on screen (expanded for printing).
COLLAPSE = {
    'a-3-proof-of-theorem': 'Show the full proof (five steps)',
    'b-derivation-of-the-reconstruction-and-transition-losses': 'Show the derivation',
    'c-implementation-and-reproducibility-details':
        'Show data generation, evaluation protocol, hyperparameters, baselines, and software',
}

CAPTIONS = {
    'fig:overview': (
        'NormFlowSDE estimates states conditional on observations. (a) At each step, the encoder $C_\\psi$ maps the '
        'trailing window of the last $o$ measurements $y_{t-o+1:t}$ (boxed), which ends before the state being estimated, '
        'to a context vector $c_t$ (orange), which conditions both the drift and the noise model. (b) Each state update '
        'combines a deterministic drift increment (green) with flow-transformed Gaussian noise (blue), yielding a flexible '
        'conditional distribution for the next state $z_{t+\\Delta t}$ (purple). (c) Repeated updates from the supplied '
        'initial state generate $K$ sampled state trajectories, shown with their pointwise median, 90\\% uncertainty '
        'bands, and marginal state densities at selected times; the boxed interval is expanded in (b). Move the time point '
        '$t$ (slider, step buttons, or a click in panel (c)) or press play: the observation window (box in (a), shaded in '
        '(c)) moves forward with $t$, the encoder produces a new context $c_t$, the one-step distribution in (b) changes with '
        'it and widens or narrows with the band in (c), and measurements after $t$ (faded) are not '
        'used. The drawing uses $o=4$ (the trained models use $o=10$); all curves and densities are schematic.'),
    'fig:robot-view': (
        'Joint-space filtering distributions of \\method (seed 0) on ALOHA and REASSEMBLE, replayed on the kinematic '
        'models of the robots: the two ViperX arms of the ALOHA simulation and a Franka Panda for REASSEMBLE. No physics '
        'is simulated; every pose is a state vector from the frozen runs. Grey is the reference state, the translucent '
        'orange arms are posterior draws (the posterior median can be switched on), and the line traces the end effector '
        'over the last 20 steps. Select a robot, one of eleven test windows (the 0th, 10th, \\ldots, 90th percentiles of '
        'the per-window RMSE and the worst window) and a joint; the timeline shows that joint with its 5--95\\% band and '
        'the measurements, and the band width below it. Press play, or click the timeline to jump. The REASSEMBLE state '
        'has no gripper coordinate, so the Franka fingers are drawn open.'),
    'fig:energy-score': (
        'Energy score and entropy term for a toy ensemble: reference $x\\sim\\N(0,I_d)$, members $z=c\\,\\epsilon$ with '
        '$\\epsilon\\sim\\N(0,I_d)$; all curves are closed-form expectations. Move the spread $c$: the energy score of '
        'Equation~\\eqref{eq:flow_sde_rollout_loss} is minimised at the calibrated spread $c=1$ in every dimension $d$, '
        'whereas the entropy term of Equation~\\eqref{eq:transition-training-loss} grows with $c$ and on its own favours '
        'contraction (Appendix~\\ref{app:training-loss}). The lower panel multiplies a miscalibrated spread $c_0$ by a '
        'scale $s$, as the learned scale $s_\\phi$ of Equation~\\eqref{eq:learned-noise-scale} does in stage 2; the energy '
        'score is minimised at $s=1/c_0$.'),
    'fig:methods': (
        'One filtering step of each method. Select a method to see where the measurements enter (yellow): direct '
        'estimators read an encoded window of past measurements and sample the next state in one pass; particle filters '
        'propagate $M$ particles and use the new measurement only to weight and resample them (NF-DPF also uses it in its '
        'proposal). The boxes below the diagram list the stochastic part, the transition density, and the training '
        'objective, as described in this section and Appendix~\\ref{app:reproducibility}.'),
    'fig:update-explainer': (
        'One \\method update in one dimension. A Gaussian input $\\epsilon_k$ (left) is mapped by the conditional flow '
        '$g_\\phi(\\epsilon_k;\\cdot)$ (middle) and added to the drift step, giving the distribution of $z_{k+1}$ (right). '
        'Where the map is steep, probability mass is spread out; where it is flat, mass concentrates. The measurement '
        'feature $c_k$ moves both the drift and the shape of the flow. Euler--Maruyama (dashed) can only produce a Gaussian '
        'with the same mean and variance. Move the sliders, or draw samples to follow them through the map. The functions '
        'are hand-chosen for illustration and are not a trained model.'),
    'fig:noise-explainer': (
        'The observation noise of Equation~\\eqref{eq:obsnoise}. Left: the mixture density, with a sharp core and side modes '
        'at $\\pm2.5\\sigma$, and the Gaussian with the same mean and variance. Right: an illustrative signal observed '
        'through this noise; switch to Gaussian noise of the same variance to compare, and draw new measurements. The '
        'overlap of 0.668 between the two densities does not depend on $\\sigma$.'),
    'fig:race': (
        'Filtering one test window, replayed from the measured latencies of Tables~\\ref{tab:results-sim} '
        'and~\\ref{tab:results-robot} (median over 30 calls, seed 0, one NVIDIA L40S; slowed down for display). A direct '
        'estimator evaluates one network per step, whereas a particle filter propagates and weights every particle and '
        'resamples. Select a dataset and press play.'),
    'fig:centering-explainer': (
        'Why the noise is centered before refining the step. The chains follow the data-generating process of '
        'Figure~\\ref{fig:weak-convergence-ou}, with skewed, non-Gaussian increments; the noise output is given an '
        'illustrative nonzero mean $m$ that the drift absorbs at the training step $h_0$. With centering (blue), the paths '
        'stay in the band of the exact SDE law, and the distribution of $z_1(T)$ approaches it as $h$ decreases. Without '
        'centering (orange), the mean $m$ enters as a drift $m(1/\\sqrt h-1/\\sqrt{h_0})$, and the paths leave the band. '
        'Everything is simulated in the browser.'),
    'fig:benchmark-explorer': (
        'Accuracy against inference cost for the eight learned methods, with the numbers of '
        'Tables~\\ref{tab:results-sim} and~\\ref{tab:results-robot}. Select a dataset and a metric. Circles are direct '
        'estimators, diamonds are particle filters, and gray squares are the model-based reference filters with the true '
        'dynamics, which are not competitors. Error bars are $\\pm$ one standard deviation over three training seeds, and the '
        'dashed line is the Pareto front of the learned methods: no learned method is both faster and better than a point on '
        'it. Hover over a marker for all metrics.'),
    'fig:posterior-explorer': (
        'Filtering distributions of \\method on Lorenz--96 and the pendulum (seed 0; $M=200$ and $M=128$ posterior draws per '
        'step, in physical units). Select the system; for each, the five test windows are those at the 10th, 30th, 50th, '
        '70th, and 90th percentiles of the per-window RMSE, so they are representative rather than selected. Click a '
        'coordinate in the grid to enlarge it; on Lorenz--96 only every second coordinate is observed, and on the pendulum '
        'both the angle $\\theta$ and the angular velocity $\\omega$ are observed. Shaded '
        'bands are central 50\\% and 90\\% intervals, the orange line is the posterior mean, black is the reference state, and '
        'dots are the noisy measurements. Hover over the plot to see the marginal distribution of the draws at that time '
        '(right), and enable the sampled paths to see individual trajectories.'),
    'fig:calibration': (
        'Calibration of central marginal intervals at nominal levels from 10\\% to 95\\%, pooled over test windows, steps, '
        'and coordinates. Lines show the mean over three seeds and shaded bands the range over seeds; solid lines are direct '
        'estimators and dashed lines particle filters. Select a dataset, toggle methods in the legend, and switch to the '
        'deviation from the nominal level to magnify small differences. On the simulated systems, the gray curves are the '
        'reference filters with the true model, and on the pendulum the NormFlowSDE-PF ablation without the learned scale can '
        'be shown. On REASSEMBLE, the curves of NSF-PF and NormFlowSDE-PF coincide. The diagonal indicates perfect '
        'calibration.'),
    'fig:noise-recovery': (
        'Observation noise on REASSEMBLE as represented by the eight methods, for seed 0 in normalized units. Gray: '
        'histogram of the data residuals $y-x$; black: the mixture of Equation~\\eqref{eq:obsnoise}; dotted: the '
        'moment-matched Gaussian; dashed, in the color of the method: the implied residuals $y-\\E[x\\mid y]$ of the '
        'posterior mean on all test windows; blue (particle filters only): the learned observation model $q(y\\mid x)$ as a '
        'density of the residual $y-x$, averaged over test states, with the 10\\% to 90\\% range over states. The numbers '
        'below the plot are overlap coefficients of Equation~\\eqref{eq:overlap} with the mixture. Switch to the log scale to '
        'see that no learned measurement model reproduces the side modes, and compare the implied residuals of all methods.'),
    'fig:weak-convergence-ou': (
        'Centered refinement of a trained \\method transition recovers a standard SDE. The reference process is the '
        'two-dimensional Ornstein--Uhlenbeck SDE $dZ=-KZ\\,dt+S\\,dW$ with '
        '$K=\\bigl(\\begin{smallmatrix}1&0.8\\\\-0.8&1\\end{smallmatrix}\\bigr)$, '
        '$S=\\bigl(\\begin{smallmatrix}0.6&0\\\\0.3&0.4\\end{smallmatrix}\\bigr)$, and $Z_0=(2,-1)$. A \\method model (drift '
        'network and conditional RealNVP noise flow, $\\alpha=1.5$) was fitted by its exact one-step likelihood to transitions '
        '$z\'=z-h_0Kz+\\sqrt{h_0}\\,S\\eta$ at $h_0=0.1$, where $\\eta$ has independent, skewed two-component-mixture '
        'coordinates with zero mean, unit variance and skewness $-0.38$. The finite-step increments are therefore '
        'non-Gaussian, while the only diffusion with the same first two moments is the reference process. With the fitted '
        'networks fixed, the plot shows the error at $T=2$, relative to the exact Ornstein--Uhlenbeck law, of the selected '
        'statistic, as the inference step decreases from $h_0$ to $h_0/32$. Blue: the centered scheme of '
        'Theorem~\\ref{thm:weak}, with $m_\\phi$ and $A$ computed by $24\\times24$ Gauss--Hermite quadrature. Orange: the '
        'uncentered refinement $z+hf_\\varphi+\\sqrt h\\,g_\\phi$. Dotted: the limit SDE of the fitted model, simulated by '
        'Euler--Maruyama at $h_0/128$; its distance to the reference process is the learning error that remains as $h\\to0$. '
        'The centered error of the mean of $z_1$ decreases from $0.056$ at $h_0$, the bias of the coarse training chain, '
        'towards this floor, whereas most other statistics are already near the floor at $h_0$. The fitted noise flow has a '
        'nonzero mean ($\\norm{m_\\phi}$ up to $0.83$), which is not identifiable from the drift in the trained update. '
        'Without centering, it acts as a drift of order $1/\\sqrt h$, and the uncentered errors grow as $h$ decreases.'),
}

FIGURE_ASSETS = {
    # label: ('img', file, alt) or ('html', file, height)
    'fig:overview': ('html', 'overview.html', 520),
    'fig:update-explainer': ('html', 'update.html', 330),
    'fig:methods': ('html', 'methods.html', 330),
    'fig:noise-explainer': ('html', 'noise_explainer.html', 330),
    'fig:race': ('html', 'race.html', 420),
    'fig:centering-explainer': ('html', 'centering.html', 360),
    'fig:benchmark-explorer': ('html', 'benchmark.html', 560),
    'fig:posterior-explorer': ('html', 'posterior.html', 680),
    'fig:calibration': ('html', 'calibration.html', 560),
    'fig:noise-recovery': ('html', 'noise_recovery.html', 540),
    'fig:weak-convergence-ou': ('html', 'weak_convergence.html', 500),
    'fig:energy-score': ('html', 'energy-score.html', 760),
    'fig:robot-view': ('html', 'robot-view.html', 1100),
}
INTERACTIVE_DIR = ROOT / 'interactive'
INTERACTIVE_FIGURES = ['overview', 'energy-score', 'robot-view']
# files a figure loads lazily (not inlined): copied flat next to the page, paths passed in window.RV_ASSETS
INTERACTIVE_LAZY = {'robot-view': ['models/aloha.js', 'models/panda.js', 'data/aloha.js', 'data/reassemble.js']}

MACROS_MATH = [
    (r'\\R(?![a-zA-Z])', r'\\mathbb{R}'),
    (r'\\E(?![a-zA-Z])', r'\\mathbb{E}'),
    (r'\\U(?![a-zA-Z])', r'\\mathcal{U}'),
    (r'\\N(?![a-zA-Z])', r'\\mathcal{N}'),
    (r'\\KL(?![a-zA-Z])', r'\\operatorname{KL}'),
    (r'\\Cov(?![a-zA-Z])', r'\\operatorname{Cov}'),
    (r'\\TV(?![a-zA-Z])', r'\\operatorname{TV}'),
    (r'\\method(?![a-zA-Z])(\\xspace)?', r'\\text{NormFlowSDE}'),
    (r'\\textnormal(?![a-zA-Z])', r'\\text'),
]

THEOREM_ENVS = {'theorem': 'Theorem', 'proposition': 'Proposition', 'corollary': 'Corollary',
                'assumption': 'Assumption', 'remark': 'Remark'}
MATH_ENVS = {'equation', 'equation*', 'align', 'align*', 'gather', 'gather*', 'multline', 'multline*'}


# --------------------------------------------------------------------------------------
# Bibliography
# --------------------------------------------------------------------------------------


def parse_bib(text):
    entries = {}
    for m in re.finditer(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', text):
        start = text.index('{', m.start())
        end = match_brace(text, start)
        body = text[m.end():end]
        fields = {}
        for fm in re.finditer(r'(\w+)\s*=\s*', body):
            k = fm.end()
            if k < len(body) and body[k] == '{':
                e = match_brace(body, k)
                fields[fm.group(1).lower()] = body[k + 1:e]
        entries[m.group(2)] = fields
    return entries


def short_authors(author_field):
    names = [a.strip() for a in re.split(r'\s+and\s+', author_field) if a.strip()]
    last = []
    for n in names:
        n = accents(n).replace('{', '').replace('}', '')
        last.append(n.split(',')[0].strip() if ',' in n else n.split()[-1])
    if len(last) == 1:
        return last[0]
    if len(last) == 2:
        return f'{last[0]} and {last[1]}'
    return f'{last[0]} et al.'


def clean_bib(text):
    """Distill's BibTeX parser does not understand LaTeX accents or protective braces in values."""
    text = accents(text).replace('\\&', '&')
    out, pos = [], 0
    for fm in re.finditer(r'\b(\w+)\s*=\s*\{', text):
        if fm.start() < pos:
            continue
        s = fm.end() - 1
        e = match_brace(text, s)
        val = text[s + 1:e].replace('{', '').replace('}', '')  # drop protective braces
        val = re.sub(r'\s+', ' ', val).replace('--', '–')
        out.append(text[pos:s + 1] + val + '}')
        pos = e + 1
    out.append(text[pos:])
    return ''.join(out)


# --------------------------------------------------------------------------------------
# Numbering of sections, floats, theorems and enumerate items
# --------------------------------------------------------------------------------------


def number_document(body):
    labels = {}
    headings = []  # (level, number, title, id)
    sec, sub, subsub, appendix = 0, 0, 0, False
    thm = tab = fig = 0
    items = []
    current = None
    pat = re.compile(r'\\appendix\b|\\(section|subsection|subsubsection)(\*?)\{'
                     r'|\\begin\{(table|figure|' + '|'.join(THEOREM_ENVS) + r'|enumerate|itemize)\}'
                     r'|\\end\{(enumerate|itemize)\}|\\item\b|\\label\{([^}]*)\}')
    used_ids = set()
    for m in pat.finditer(body):
        tok = m.group(0)
        if tok == '\\appendix':
            appendix, sec = True, 0
            continue
        if m.group(1):
            level, star = m.group(1), m.group(2)
            title_raw = body[m.end() - 1:match_brace(body, m.end() - 1) + 1][1:-1]
            title = inline_text(title_raw, plain=True)
            if level == 'section':
                sec, sub, subsub, thm = sec + 1, 0, 0, 0
                num = chr(64 + sec) if appendix else str(sec)
                lvl = 2
            elif level == 'subsection':
                sub, subsub = sub + 1, 0
                num = f'{chr(64 + sec) if appendix else sec}.{sub}'
                lvl = 3
            else:
                subsub += 1
                num = f'{chr(64 + sec) if appendix else sec}.{sub}.{subsub}'
                lvl = 4
            name = f'{num}. {title}' if lvl == 2 else f'{num} {title}'
            hid = slugify(name)
            while hid in used_ids:
                hid += '-x'
            used_ids.add(hid)
            headings.append({'level': lvl, 'num': num, 'title': title, 'id': hid, 'appendix': appendix, 'pos': m.start()})
            current = ('sec', num, hid)
            continue
        if m.group(3):
            env = m.group(3)
            if env == 'table':
                tab += 1
                current = ('tab', str(tab), None)
            elif env == 'figure':
                fig += 1
                current = ('fig', str(fig), None)
            elif env in THEOREM_ENVS:
                thm += 1
                current = ('thm', f'{chr(64 + sec) if appendix else sec}.{thm}', None)
            else:
                items.append(0)
            continue
        if m.group(4):
            items.pop()
            continue
        if tok == '\\item':
            if items:
                items[-1] += 1
                current = ('item', str(items[-1]), None)
            continue
        lab = m.group(5)
        if lab is not None:
            if lab.startswith('eq:'):
                continue
            kind, num, hid = current
            if hid is None:
                hid = re.sub(r'[^a-zA-Z0-9]+', '-', lab)
            labels[lab] = {'kind': kind, 'num': num, 'id': hid}
    return labels, headings


# --------------------------------------------------------------------------------------
# Inline conversion
# --------------------------------------------------------------------------------------

CTX = {'labels': {}, 'bib': {}}


def fix_math(m):
    for a, b in MACROS_MATH:
        m = re.sub(a, b, m)
    m = replace_cmd(m, 'norm', lambda a: r'\left\lVert ' + a + r'\right\rVert ')
    # Distill re-parses article HTML at runtime, so a literal '<' in math can be read as a tag
    m = m.replace('<', r'\lt ').replace('>', r'\gt ')
    # kramdown (GFM) reads a paragraph containing '|' as a table row, even inside math
    m = m.replace(r'\|', r'\Vert ').replace('|', r'\vert ')
    m = re.sub(r'\{\{', '{ {', m)
    m = re.sub(r'\}\}', '} }', m)
    m = m.replace('{%', '{ %')
    return m


def ref_link(lab, html):
    info = CTX['labels'].get(lab)
    if info is None:
        raise KeyError('unknown label ' + lab)
    return f'<a href="#{info["id"]}">{info["num"]}</a>'


def inline_text(t, html=False, plain=False):
    """Convert a LaTeX text fragment. html=True for content inside raw HTML blocks."""
    store = []

    def keep(s):
        store.append(s)
        return f'\x00{len(store) - 1}\x00'

    def math(s):
        s = fix_math(s.strip())
        if plain:
            return s
        return keep(('\\(' + s + '\\)') if html else ('$$' + s + '$$'))

    # references to equations become MathJax references
    t = re.sub(r'\\eqref\{([^}]*)\}', lambda m: math(r'\eqref{' + m.group(1) + '}'), t)
    t = re.sub(r'\\ref\{(eq:[^}]*)\}', lambda m: math(r'\ref{' + m.group(1) + '}'), t)
    # inline math
    t = re.sub(r'(?<!\\)\$(.+?)(?<!\\)\$', lambda m: math(m.group(1)), t, flags=re.S)
    t = re.sub(r'\\\((.+?)\\\)', lambda m: math(m.group(1)), t, flags=re.S)
    # citations
    if plain:
        t = re.sub(r'\\ref\{([^}]*)\}', lambda m: CTX['labels'].get(m.group(1), {'num': '?'})['num'], t)
    else:
        def note(o):
            return f' ({inline_text(o, html=html)})' if o else ''

        def citep(o, k):
            return keep(f'<d-cite key="{",".join(x.strip() for x in k.split(","))}"></d-cite>' + note(o))
        t = replace_cmd(t, 'citep', citep, opt=True)
        t = replace_cmd(t, 'cite', citep, opt=True)

        def citet(o, k):
            parts = []
            for key in [x.strip() for x in k.split(',')]:
                who = short_authors(CTX['bib'][key].get('author', key))
                parts.append(f'{who} ' + keep(f'<d-cite key="{key}"></d-cite>'))
            return ', '.join(parts) + note(o)
        t = replace_cmd(t, 'citet', citet, opt=True)
        t = re.sub(r'\\ref\{([^}]*)\}', lambda m: keep(ref_link(m.group(1), html)), t)
    t = re.sub(r'\\label\{[^}]*\}', '', t)
    # formatting
    t = re.sub(r'\\method(\\xspace)?(?![a-zA-Z])', 'NormFlowSDE', t)
    t = accents(t)
    if plain:
        for c in ('emph', 'textbf', 'textit', 'texttt', 'textnormal', 'mbox'):
            t = replace_cmd(t, c, lambda a: a)
    elif html:
        t = replace_cmd(t, 'emph', lambda a: f'<em>{a}</em>')
        t = replace_cmd(t, 'textit', lambda a: f'<em>{a}</em>')
        t = replace_cmd(t, 'textbf', lambda a: f'<strong>{a}</strong>')
        t = replace_cmd(t, 'texttt', lambda a: f'<code>{a}</code>')
    else:
        t = replace_cmd(t, 'emph', lambda a: f'*{a}*')
        t = replace_cmd(t, 'textit', lambda a: f'*{a}*')
        t = replace_cmd(t, 'textbf', lambda a: f'**{a}**')
        t = replace_cmd(t, 'texttt', lambda a: keep(f'<code>{a}</code>'))
    t = replace_cmd(t, 'textnormal', lambda a: a)
    t = replace_cmd(t, 'mbox', lambda a: a)
    t = replace_cmd(t, 'url', lambda a: keep(f'<a href="{a}">{a}</a>'))
    t = replace_cmd(t, 'path', lambda a: keep(f'<code>{a}</code>'))
    t = replace_cmd(t, 'footnote', lambda a: keep(f'<d-footnote>{inline_text(a, html=True)}</d-footnote>'))
    t = replace_cmd(t, 'shortstack', lambda a: a.replace('\\\\', '<br>' if html else ' '))
    for cmd in ('centering', 'footnotesize', 'small', 'scriptsize', 'noindent', 'hfill', 'xspace', 'bfseries', 'normalsize'):
        t = re.sub(r'\\' + cmd + r'(?![a-zA-Z])\s*', '', t)
    t = t.replace('---', '\u2014').replace('--', '\u2013')
    t = t.replace('``', '\u201c').replace("''", '\u201d')
    t = re.sub(r'(?<!\\)~', ' ', t)
    t = t.replace('\\,', '\u2009').replace('\\ ', ' ').replace('\\@', '')
    t = t.replace('\\ldots', '\u2026').replace('\\dots', '\u2026')
    t = t.replace('\\%', '%').replace('\\&', '&amp;' if html else '&').replace('\\#', '#')
    t = t.replace('\\_', '_' if (html or plain) else '\\_')
    t = t.replace('\\\\', ' ')
    if not (html or plain):
        t = t.replace('*', '\\*') if False else t
    t = re.sub(r'[ \t]+', ' ', t)
    leftover = re.findall(r'\\[a-zA-Z]+', t)
    if leftover:
        print('  WARNING leftover commands:', set(leftover), '|', t[:100].replace('\n', ' '))
    t = re.sub('\x00(\\d+)\x00', lambda m: store[int(m.group(1))], t)
    t = re.sub('\x00(\\d+)\x00', lambda m: store[int(m.group(1))], t)
    return t.strip() if plain else t


# --------------------------------------------------------------------------------------
# Block conversion
# --------------------------------------------------------------------------------------


def split_envs(text):
    """Split text into ('text', s) and ('env', name, inner, opt) at top level."""
    out, pos = [], 0
    pat = re.compile(r'\\begin\{([a-zA-Z*]+)\}')
    while True:
        m = pat.search(text, pos)
        if not m:
            out.append(('text', text[pos:]))
            return out
        out.append(('text', text[pos:m.start()]))
        name = m.group(1)
        depth, k = 1, m.end()
        bp = re.compile(r'\\(begin|end)\{' + re.escape(name) + r'\}')
        while depth:
            n = bp.search(text, k)
            if not n:
                raise ValueError('unclosed env ' + name)
            depth += 1 if n.group(1) == 'begin' else -1
            k = n.end()
        inner = text[m.end():n.start()]
        out.append(('env', name, inner))
        pos = k


def extract_cmd(inner, name):
    m = re.search(r'\\' + name + r'(?![a-zA-Z])\s*(\[[^\]]*\])?\s*\{', inner)
    if not m:
        return None, inner
    s = m.end() - 1
    e = match_brace(inner, s)
    return inner[s + 1:e], inner[:m.start()] + inner[e + 1:]


def heading_html(level_cmd, title_raw):
    want = {'section': 2, 'subsection': 3, 'subsubsection': 4}[level_cmd]
    for h in CTX['headings']:  # headings of one level are converted in document order
        if h['level'] == want and not h.get('used'):
            h['used'] = True
            lvl = h['level']
            label = f'{h["num"]}.' if lvl == 2 else h['num']
            title = inline_text(title_raw, html=True)
            h['title'] = inline_text(title_raw, plain=True)  # labels are known now
            return f'\n\n<h{lvl} id="{h["id"]}">{label} {title}</h{lvl}>\n\n'
    raise KeyError('heading not numbered: ' + title_raw)


def convert_text(t):
    # headings
    def sect(level):
        def f(title):
            return heading_html(level, title)
        return f
    t = re.sub(r'\\appendix\b', '', t)
    for lvl in ('subsubsection', 'subsection', 'section'):
        t = replace_cmd(t, lvl, sect(lvl))
    # display math \[ \]
    t = re.sub(r'\\\[(.+?)\\\]', lambda m: '\n\n$$\n' + fix_math(m.group(1).strip()) + '\n$$\n\n', t, flags=re.S)
    # run-in paragraph headings
    t = replace_cmd(t, 'paragraph', lambda a: '\n\n\x01' + a.strip() + '\x01 ')
    paras = re.split(r'\n\s*\n', t)
    out = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if p.startswith('<h') or p.startswith('$$'):
            out.append(p)
            continue
        p = re.sub(r'\s*\n\s*', ' ', p)
        head = ''
        if p.startswith('\x01'):
            e = p.index('\x01', 1)
            head = '**' + inline_text(p[1:e]) + '**'
            p = p[e + 1:]
        body = inline_text(p).strip()
        s = (head + ' ' + body).strip() if head else body
        if s.startswith('$$'):
            s = '\\' + s  # kramdown: inline math at paragraph start
        out.append(s)
    return '\n\n'.join(out)


def move_labels_off_notag_rows(inner):
    """LaTeX attaches a \\label on an unnumbered align row to the next numbered row; MathJax drops it."""
    rows = split_top(inner, '\\\\')
    carry = []
    for i, row in enumerate(rows):
        if carry and '\\notag' not in row and '\\nonumber' not in row:
            rows[i] = row + ''.join(carry)
            carry = []
            row = rows[i]
        if '\\notag' in row or '\\nonumber' in row:
            labs = re.findall(r'\\label\{[^}]*\}', row)
            if labs:
                carry += labs
                rows[i] = re.sub(r'\\label\{[^}]*\}', '', row)
    return '\\\\'.join(rows)


def convert_math_env(name, inner):
    if name in ('align', 'gather'):
        inner = move_labels_off_notag_rows(inner)
    body = fix_math(f'\\begin{{{name}}}{inner}\\end{{{name}}}'.strip())
    return f'\n\n$$\n{body}\n$$\n\n'


def convert_list(name, inner, indent=''):
    items = re.split(r'\\item(?![a-zA-Z])', inner)[1:]
    out = []
    for i, it in enumerate(items, 1):
        bullet = f'{i}. ' if name == 'enumerate' else '- '
        md = convert_blocks(it).strip()
        pad = ' ' * len(bullet)
        lines = md.split('\n')
        first = indent + bullet + lines[0]
        rest = [(indent + pad + ln) if ln.strip() else '' for ln in lines[1:]]
        out.append('\n'.join([first] + rest))
    return '\n\n' + '\n\n'.join(out) + '\n\n'


def convert_theorem(name, inner):
    opt = None
    m = re.match(r'\s*\[([^\]]*)\]', inner)
    if m:
        opt, inner = m.group(1), inner[m.end():]
    lab = re.search(r'\\label\{([^}]*)\}', inner)
    info = CTX['labels'][lab.group(1)] if lab else None
    num = info['num'] if info else '?'
    hid = info['id'] if info else ''
    title = f'**{THEOREM_ENVS[name]} {num}**' + (f' ({inline_text(opt)}).' if opt else '.')
    body = convert_blocks(inner).strip()
    return f'\n\n<div class="theorem" id="{hid}" markdown="1">\n\n{title} {body}\n\n</div>\n\n'


def split_top(s, sep):
    """Split s on sep at brace depth 0 (sep is a literal string)."""
    parts, depth, cur, i = [], 0, [], 0
    while i < len(s):
        if s[i] == '\\' and s.startswith(sep, i) and depth == 0 and sep.startswith('\\'):
            parts.append(''.join(cur)); cur = []; i += len(sep); continue
        ch = s[i]
        if ch == '\\':
            cur.append(s[i:i + 2]); i += 2; continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        if depth == 0 and s.startswith(sep, i) and not sep.startswith('\\'):
            parts.append(''.join(cur)); cur = []; i += len(sep); continue
        cur.append(ch); i += 1
    parts.append(''.join(cur))
    return parts


def tabular_html(tab):
    m = re.search(r'\\begin\{tabular\}\s*\{', tab)
    s = m.end() - 1
    e = match_brace(tab, s)
    spec = tab[s + 1:e]
    body = tab[e + 1:tab.index('\\end{tabular}')]
    aligns = [c for c in re.sub(r'\{[^}]*\}', '', spec) if c in 'lcrp']
    rows = split_top(body, '\\\\')
    html_rows, head_done, pending_cls = [], False, ''
    header = None
    for r in rows:
        cls = []
        while True:
            r = r.strip()
            mm = re.match(r'\\(toprule|midrule|bottomrule|hline|cmidrule)(\([^)]*\))?(\{[^}]*\})?', r)
            if not mm:
                break
            if mm.group(1) in ('midrule', 'hline'):
                cls.append('mid')
            elif mm.group(1) == 'cmidrule':
                cls.append('cmid')
            r = r[mm.end():]
        if not r.strip():
            pending_cls = ' '.join(cls) or pending_cls
            continue
        cells = split_top(r, '&')
        cells_html = []
        for j, c in enumerate(cells):
            a = aligns[j] if j < len(aligns) else 'l'
            al = {'l': 'left', 'c': 'center', 'r': 'right', 'p': 'left'}[a]
            cells_html.append((al, inline_text(c.strip(), html=True)))
        rc = ' '.join(cls) or pending_cls
        pending_cls = ''
        if header is None:
            header = cells_html
            continue
        tds = ''.join(f'<td style="text-align:{al}">{c}</td>' for al, c in cells_html)
        html_rows.append(f'<tr class="{rc}">{tds}</tr>' if rc else f'<tr>{tds}</tr>')
    ths = ''.join(f'<th style="text-align:{al}">{c}</th>' for al, c in header)
    return f'<table><thead><tr>{ths}</tr></thead><tbody>{"".join(html_rows)}</tbody></table>'


def convert_table(inner):
    cap, inner2 = extract_cmd(inner, 'caption')
    lab = re.search(r'\\label\{([^}]*)\}', inner)
    info = CTX['labels'][lab.group(1)]
    inner2 = replace_cmd(inner2, 'resizebox', lambda a, b, c: c, nargs=3)
    tab = tabular_html(inner2)
    capt = inline_text(cap, html=True)
    wide = ' l-page' if tab.count('<th') > 6 else ''
    return (f'\n\n<figure class="nf-table{wide}" id="{info["id"]}">\n<div class="table-wrap">{tab}</div>\n'
            f'<figcaption><strong>Table {info["num"]}.</strong> {capt}</figcaption>\n</figure>\n\n')


def convert_figure(inner):
    lab = re.search(r'\\label\{([^}]*)\}', inner).group(1)
    info = CTX['labels'][lab]
    cap, _ = extract_cmd(inner, 'caption')
    if lab in CAPTIONS:
        cap = CAPTIONS[lab]
    capt = inline_text(cap, html=True)
    kind, fname, extra = FIGURE_ASSETS[lab]
    if kind == 'img':
        media = (f'<img src="{{{{ \'assets/img/submission/{fname}\' | relative_url }}}}" alt="{extra}" '
                 f'style="width:100%;height:auto;">')
        tag = ''
    else:
        name = fname[:-5]
        media = (f'<iframe name="{name}" class="nf-frame" src="{{{{ \'assets/html/submission/{fname}\' | relative_url }}}}" '
                 f'width="100%" height="{extra}" scrolling="no" title="Interactive figure {info["num"]}"></iframe>')
        tag = ' <span class="nf-badge">interactive</span>'
    layout = 'l-body' if kind == 'img' else 'l-page'
    return (f'\n\n<figure class="nf-figure {layout}" id="{info["id"]}">\n{media}\n'
            f'<figcaption><strong>Figure {info["num"]}.</strong>{tag} {capt}</figcaption>\n</figure>\n\n')


def convert_blocks(text):
    parts = []
    for seg in split_envs(text):
        if seg[0] == 'text':
            parts.append(convert_text(seg[1]))
            continue
        _, name, inner = seg
        if name in MATH_ENVS:
            parts.append(convert_math_env(name, inner))
        elif name == 'table':
            parts.append(convert_table(inner))
        elif name == 'figure':
            parts.append(convert_figure(inner))
        elif name in THEOREM_ENVS:
            parts.append(convert_theorem(name, inner))
        elif name in ('enumerate', 'itemize'):
            parts.append(convert_list(name, inner))
        elif name == 'center':
            parts.append(convert_blocks(inner))
        else:
            raise ValueError('unhandled environment ' + name)
    md = '\n\n'.join(p.strip('\n') for p in parts if p.strip())
    return md


# --------------------------------------------------------------------------------------
# Data for interactive figures
# --------------------------------------------------------------------------------------


def r(a, d=4):
    return np.round(np.asarray(a, dtype=float), d).tolist()


def data_calibration():
    c = json.loads((RES / 'analyses/calibration/calibration_shared_trunk.json').read_text())
    curves = {k: {s: r(v, 4) for s, v in seeds.items()} for k, seeds in c['curves'].items()}
    return {'levels': c['levels'], 'curves': curves}


def data_noise():
    n = json.loads((RES / 'analyses/noise_recovery_grid/reassemble_shared_trunk_noise_recovery_grid.json').read_text())
    grid = np.array(n['grid'])
    methods = {}
    for k in ['normflowsde', 'latent_nsf', 'latent_sde', 'sigker_sde', 'nf_dpf', 'nsf', 'flowsde', 'normflowsde_pf']:
        d = n[k]
        e = {'implied_density': r(d['implied_density'], 4), 'implied_overlap_with_gmm': d['implied_overlap_with_gmm'],
             'implied_resid_std': d['implied_resid_std']}
        if 'slice_mean' in d:
            e.update(slice_mean=r(d['slice_mean'], 4), slice_lo=r(d['slice_lo'], 4), slice_hi=r(d['slice_hi'], 4),
                     overlap_with_gmm=d['overlap_with_gmm'])
        methods[k] = e
    h = n['data_residual']
    bins = np.array(h['bins'])
    counts = np.array(h['hist'], dtype=float)
    dens = counts / (counts.sum() * np.diff(bins)) if counts.sum() > 1.5 else counts
    sd = 0.1 * np.sqrt(0.7 * 0.5 ** 2 + 0.3 * (1.5 ** 2 + 2.5 ** 2))
    gauss = np.exp(-0.5 * (grid / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    return {'grid': r(grid, 3), 'gmm': r(n['gmm'], 4), 'gauss': r(gauss, 4), 'hist': {'bins': r(bins, 4), 'density': r(dens, 4)},
            'data_resid_std': n['data_resid_std'], 'methods': methods}


def data_weak(thm_num):
    w = json.loads((RES / 'analyses/weak_convergence/ou_seed0/results.json').read_text())
    labels = {
        'z1': ('mean of z₁', None), 'z2': ('mean of z₂', None), 'z1^2': ('E[z₁²]', None), 'z1*z2': ('E[z₁z₂]', None),
        'cos(z1+z2)': ('E[cos(z₁+z₂)]', None), 'exp(-|z|^2)': ('E[exp(−|z|²)]', None),
        'E[max_t |Z_t|_inf]': ('E[maxₜ‖Zₜ‖∞]', 'Path maximum on the grid t ∈ h₀ℕ.'),
        'skewness z1': ('skewness z₁', 'Higher moments: the limit is Gaussian, so skewness and excess kurtosis vanish.'),
        'excess kurtosis z1': ('excess kurtosis z₁', 'Higher moments: the limit is Gaussian, so skewness and excess kurtosis vanish.'),
        'skewness z2': ('skewness z₂', None), 'excess kurtosis z2': ('excess kurtosis z₂', None),
    }
    keys = list(w['rows'][0]['errors'].keys())
    stats = [{'key': k, 'label': labels.get(k, (k, None))[0], 'note': labels.get(k, (k, None))[1]} for k in keys]
    rows = [{'scheme': row['scheme'], 'h': row['h'],
             'errors': {k: {'error': v['error'], 'se': v['se']} for k, v in row['errors'].items()}} for row in w['rows']]
    truth = {k: v['mean'] for k, v in w['truth'].items()}
    return {'stats': stats, 'rows': rows, 'truth': truth, 'thm': thm_num}


def parse_bench_table(path):
    t = read_tex(path)
    body = t[t.index('\\midrule') + len('\\midrule'):t.index('\\bottomrule')]
    keymap = {'NormFlowSDE': 'normflowsde', 'Latent NSF': 'latent_nsf', 'Latent SDE': 'latent_sde', 'SigKer SDE': 'sigker_sde',
              'NF-DPF': 'nf_dpf', 'NSF-PF': 'nsf', 'FlowSDE-PF': 'flowsde', 'NormFlowSDE-PF': 'normflowsde_pf',
              'EKF / UKF': 'ref_kf', 'bootstrap PF': 'ref_pf'}
    dsmap = {'Pendulum': ('pendulum', 'Pendulum'), 'Lorenz--96': ('lorenz96', 'Lorenz–96'), 'ALOHA': ('aloha', 'ALOHA'),
             'REASSEMBLE': ('reassemble', 'REASSEMBLE')}
    out, cur, route = [], None, None
    for line in body.split('\\\\'):
        line = re.sub(r'\\(midrule|cmidrule\([^)]*\)\{[^}]*\})', '', line).strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split('&')]
        if cells[0]:
            k, lab = dsmap[cells[0]]
            cur = {'key': k, 'label': lab, 'rows': []}
            out.append(cur)
        if cells[1]:
            route = {'direct': 'direct', 'PF': 'pf', 'ref.': 'ref'}[cells[1]]

        def pm(s):
            s = s.replace('\\textbf{', '').replace('}', '')
            a, b = s.split('$\\pm$')
            return [float(a), float(b)]
        lat = cells[7].replace('\\textbf{', '').replace('}', '')
        label = cells[2]
        row = {'key': keymap[label], 'label': {'EKF / UKF': 'EKF / UKF', 'bootstrap PF': 'Bootstrap PF'}.get(label, label),
               'route': route, 'm': {'rmse': pm(cells[3]), 'crps': pm(cells[4]), 'cov': pm(cells[5]), 'width': pm(cells[6])},
               'latency': float(lat.split('/')[0]), 'latencyLabel': lat.strip()}
        cur['rows'].append(row)
    return out


def data_benchmark():
    return {'datasets': parse_bench_table(PAPER / 'tables/benchmark_sim.tex') + parse_bench_table(PAPER / 'tables/benchmark_robot.tex')}


POSTERIOR_DATASETS = {
    # key: (label, dt, coordinate names, units); both are reported in physical units in the paper
    'lorenz96': ('Lorenz–96', 0.02, [f'x{i}' for i in range(12)], None),
    'pendulum': ('Pendulum', 0.05, ['θ', 'ω'], ['rad', 'rad/s']),
}


def data_posterior():
    return {'datasets': [data_posterior_ds(ds) for ds in POSTERIOR_DATASETS]}


def data_posterior_ds(ds):
    label, dt, names, units = POSTERIOR_DATASETS[ds]
    z = np.load(RES / f'shared_trunk_20261114/normflowsde/{ds}/seed0/s2/posterior.npz')
    te = np.load(RES / f'datasets/{ds}/test.npz')
    S = z['samples_normalized'].astype(np.float64)  # (win, draws, T, d)
    truth_all = te['states'].astype(np.float64)  # (win, T+1, d)
    assert np.abs(z['truth_normalized'] - truth_all[:, 1:]).max() < 1e-4
    obs = te['observations'].astype(np.float64)  # (win, T, observed)
    nwin, M, T, d = S.shape
    # arrays are normalized with training statistics; the paper reports both systems in physical units
    meta = json.loads((RES / f'datasets/{ds}/cache.json').read_text())['metadata']
    assert meta['arrays_normalized']
    mu_s, sd_s = np.array(meta['state_mean']), np.array(meta['state_std'])
    mu_o, sd_o = np.array(meta['observation_mean']), np.array(meta['observation_std'])
    S = S * sd_s + mu_s
    truth_all = truth_all * sd_s + mu_s
    obs = obs * sd_o + mu_o
    observed = list(meta['observation_indices'])
    assert obs.shape[-1] == len(observed)
    run_rmse = json.loads((RES / f'shared_trunk_20261114/normflowsde/{ds}/seed0/s2/metrics.json').read_text())['metrics']['rmse']
    x0 = truth_all[:, :1]  # known initial state
    mean = S.mean(1)
    rmse = np.sqrt(((mean - truth_all[:, 1:]) ** 2).mean(axis=(1, 2)))
    assert abs(rmse.mean() - run_rmse) < 2e-3, (rmse.mean(), run_rmse)
    print(f'  posterior {ds}: mean per-window RMSE {rmse.mean():.4f} (run metric {run_rmse:.4f})')
    order = np.argsort(rmse)
    pcts = [10, 30, 50, 70, 90]
    rng = np.random.default_rng(0)
    nd = 32
    wins = []
    for p in pcts:
        w = int(order[int(round(p / 100 * (nwin - 1)))])
        full = np.concatenate([np.repeat(x0[w][None], M, 0), S[w]], axis=1)  # (M, T+1, d)
        q = np.quantile(full, [0.05, 0.25, 0.75, 0.95], axis=0)  # (4, T+1, d)
        mu = full.mean(0)
        di = rng.choice(M, nd, replace=False)
        nb = 28
        lo = full.min(axis=(0, 1)) - 1e-6
        hi = full.max(axis=(0, 1)) + 1e-6
        counts = np.zeros((d, T + 1, nb), dtype=int)
        for c in range(d):
            idx = np.clip(((full[:, :, c] - lo[c]) / (hi[c] - lo[c]) * nb).astype(int), 0, nb - 1)
            for t in range(T + 1):
                counts[c, t] = np.bincount(idx[:, t], minlength=nb)
        ob = {}
        for j, c in enumerate(observed):
            ob[c] = [None] + r(obs[w, :, j], 2)
        wins.append({
            'index': w, 'pct': p, 'rmse': float(rmse[w]),
            'truth': r(truth_all[w].T, 3), 'mean': r(mu.T, 3),
            'q05': r(q[0].T, 3), 'q25': r(q[1].T, 3), 'q75': r(q[2].T, 3), 'q95': r(q[3].T, 3),
            'draws': [r(full[di, :, c], 2) for c in range(d)],
            'obs': ob,
            'hist': {'lo': r(lo, 4), 'hi': r(hi, 4), 'counts': counts.tolist()},
        })
    return {'key': ds, 'label': label, 'names': names, 'units': units, 'nwin': int(nwin),
            'dt': dt, 'T': T + 1, 'd': d, 'M': M, 'observed': observed, 'windows': wins}


def data_methods():
    """Per-step pipelines of the eight methods, as described in Section 2 and Appendix C."""
    S = lambda label, sub, kind, many=False: {'label': label, 'sub': sub, 'kind': kind, 'many': many}
    win = {'label': 'measurement window', 'sub': 'yₖ₋ₒ₊₁ … yₖ → encoder → cₖ'}
    meas = {'label': 'new measurement yₖ₊₁', 'sub': 'enters only through the likelihood'}
    pf_tail = [S('weights', 'flow likelihood q(yₖ₊₁ | x) of each particle', 'weight', True),
               S('resample', 'M particles with replacement', 'weight', True),
               S('particles xₖ₊₁', 'weighted posterior draws', 'state', True)]
    return {'methods': {
        'normflowsde': {'chain': [S('state xₖ', 'one sample path', 'state'),
                                  S('shared features', 'χₖ = T(xₖ, tₖ, uₖ, cₖ)', 'dyn'),
                                  S('drift + flow noise', 'Δt f(χₖ) + √Δt g(εₖ; χₖ)', 'noise'),
                                  S('state xₖ₊₁', 'sample from a non-Gaussian update', 'state')],
                        'obs': dict(win, into=[1]),
                        'facts': {'obs': 'encoded window conditions both the drift and the flow',
                                  'noise': 'conditional normalizing flow on the noise of each update (non-Gaussian)',
                                  'density': 'exact, by change of variables',
                                  'train': 'reconstruction + transition likelihood + segment continuity + rollout energy score'}},
        'latent_nsf': {'chain': [S('state xₖ', 'previous sampled state', 'state'),
                                 S('GRU posterior', 'q(xₖ₊₁ | y≤ₖ, xₖ)', 'dyn'),
                                 S('Gaussian sample', 'N(m, diag s²)', 'noise'),
                                 S('state xₖ₊₁', 'sample', 'state')],
                       'obs': dict(win, into=[1]),
                       'facts': {'obs': 'recurrent encoder of past measurements outputs the posterior',
                                 'noise': 'Gaussian posterior; the NSF flow sits in the prior transition',
                                 'density': 'Gaussian posterior; NSF prior density exact',
                                 'train': 'β-weighted ELBO + flow consistency penalty'}},
        'latent_sde': {'chain': [S('state xₖ', 'one sample path', 'state'),
                                 S('posterior drift', 'h(x, cₜ, t)', 'dyn'),
                                 S('Gaussian increment', 'learned diffusion g(x, t)', 'noise'),
                                 S('state xₖ₊₁', 'sample', 'state')],
                       'obs': dict(win, into=[1]),
                       'facts': {'obs': 'encoding enters the posterior drift',
                                 'noise': 'Gaussian increments given the state',
                                 'density': 'Gaussian given the state',
                                 'train': 'ELBO with path KL (Girsanov) between posterior and prior SDE'}},
        'sigker_sde': {'chain': [S('state xₖ', 'one sample path', 'state'),
                                 S('drift', 'μ(x, c, t)', 'dyn'),
                                 S('Gaussian increment', 'diffusion σ(x, c, t)', 'noise'),
                                 S('state xₖ₊₁', 'sample', 'state')],
                       'obs': dict(win, into=[1, 2]),
                       'facts': {'obs': 'encoding enters drift and diffusion',
                                 'noise': 'Gaussian increments given the state',
                                 'density': 'none: the score is evaluated from samples only',
                                 'train': 'signature-kernel score on paths'}},
        'nf_dpf': {'chain': [S('particles xₖ', 'M weighted particles', 'state', True),
                             S('proposal flow', 'q(xₖ₊₁ | xₖ, yₖ₊₁)', 'noise', True),
                             S('weights', 'p(y | x) p(x | xₖ) / q', 'weight', True),
                             S('OT resampling', 'entropy-regularized, differentiable', 'weight', True),
                             S('particles xₖ₊₁', 'weighted posterior draws', 'state', True)],
                   'obs': {'label': 'new measurement yₖ₊₁', 'sub': 'enters the proposal and the likelihood', 'into': [1, 2]},
                   'facts': {'obs': 'proposal and measurement likelihood',
                             'noise': 'flows for transition, proposal, and measurement',
                             'density': 'exact for all three flows',
                             'train': 'particle likelihood or supervised state error'}},
        'nsf': {'chain': [S('particles xₖ', 'M weighted particles', 'state', True),
                          S('NSF transition', 'Gaussian candidate → flow', 'noise', True)] + pf_tail,
                'obs': dict(meas, into=[2]),
                'facts': {'obs': 'only through the shared measurement flow (bootstrap filter)',
                          'noise': 'NSF: Gaussian candidate transformed by a conditional flow',
                          'density': 'exact NSF transition density',
                          'train': 'transition NLL + Chapman–Kolmogorov consistency; measurement flow by NLL'}},
        'flowsde': {'chain': [S('particles xₖ', 'M weighted particles', 'state', True),
                              S('Euler–Maruyama', 'flow-matched drift, noise √(2D)', 'noise', True)] + pf_tail,
                    'obs': dict(meas, into=[2]),
                    'facts': {'obs': 'only through the shared measurement flow (bootstrap filter)',
                              'noise': 'Gaussian with prescribed scalar diffusion D',
                              'density': 'Gaussian per Euler–Maruyama step',
                              'train': 'conditional flow matching + potential regularizers'}},
        'normflowsde_pf': {'chain': [S('particles xₖ', 'M weighted particles', 'state', True),
                                     S('NormFlowSDE step', 'drift + flow noise, no measurements', 'noise', True)] + pf_tail,
                           'obs': dict(meas, into=[2]),
                           'facts': {'obs': 'only through the shared measurement flow (bootstrap filter)',
                                     'noise': 'NormFlowSDE transition, conditioned on control and time',
                                     'density': 'exact, by change of variables',
                                     'train': 'training and selection recipe of NSF-PF'}},
    }}


def data_noise_explainer():
    # Equation (obsnoise): 0.70 N(0,(0.5s)^2) + 0.15 N(2.5s,(1.5s)^2) + 0.15 N(-2.5s,(1.5s)^2)
    src = read_tex(PAPER / 'sections/experiments.tex')
    assert '0.70\\,\\N\\,\\big(0,(0.5\\sigma)^2\\big)+0.15\\,\\N\\,\\big(2.5\\sigma,(1.5\\sigma)^2\\big)' in src
    return {'mixture': [{'w': 0.70, 'mu': 0.0, 'sd': 0.5}, {'w': 0.15, 'mu': 2.5, 'sd': 1.5}, {'w': 0.15, 'mu': -2.5, 'sd': 1.5}]}


def table1_row(name):
    src = read_tex(PAPER / 'sections/experiments.tex')
    m = re.search(re.escape(name) + r'\s*&([^\\]*)\\\\', src)
    return [c.strip() for c in m.group(1).split('&')]


def data_race(bench):
    steps = [int(v) for v in table1_row('Window length (steps)')]
    M = [int(v) for v in table1_row('Samples / particles $M$')]
    out = []
    for d, n, mm in zip(bench['datasets'], steps, M):
        out.append({'key': d['key'], 'label': d['label'], 'steps': n, 'M': mm,
                    'rows': [r for r in d['rows'] if r['route'] != 'ref']})
    return {'datasets': out}


def data_centering(labels):
    w = json.loads((RES / 'analyses/weak_convergence/ou_seed0/results.json').read_text())
    ou, inc = w['ou'], w['increment_noise']
    wgt, a = inc['w'], inc['a']
    mix = {'w': wgt, 'mu1': a, 'mu2': -wgt * a / (1 - wgt), 's': inc['s']}
    mean = mix['w'] * mix['mu1'] + (1 - mix['w']) * mix['mu2']
    var = mix['s'] ** 2 + mix['w'] * mix['mu1'] ** 2 + (1 - mix['w']) * mix['mu2'] ** 2
    assert abs(mean) < 1e-9 and abs(var - 1) < 1e-6, (mean, var)  # zero mean, unit variance as stated
    return {'K': ou['K'], 'S': ou['S'], 'z0': ou['z0'], 'h0': 0.1, 'T': 2.0, 'mix': mix,
            'm': [0.6, -0.57], 'thm': labels['thm:weak']['num'], 'fig': labels['fig:weak-convergence-ou']['num']}


def summary_cards(bench, labels):
    ds = {d['key']: {r['key']: r for r in d['rows']} for d in bench['datasets']}
    pf = ['nf_dpf', 'nsf', 'flowsde', 'normflowsde_pf']
    red = []
    for k in ('aloha', 'reassemble'):
        best = min(ds[k][p]['m']['rmse'][0] for p in pf)
        red.append(100 * (1 - ds[k]['normflowsde']['m']['rmse'][0] / best))
    cov = max(abs(ds[k]['normflowsde']['m']['cov'][0] - 0.9) for k in ds)
    speed = [min(ds[k][p]['latency'] for p in pf) / ds[k]['normflowsde']['latency'] for k in ds]
    f_bench, f_post = labels['fig:benchmark-explorer']['num'], labels['fig:posterior-explorer']['num']
    f_cal, f_race = labels['fig:calibration']['num'], labels['fig:race']['num']
    cards = [
        (f'−{red[0]:.0f}% / −{red[1]:.0f}%', 'RMSE vs. the best particle filter on the ALOHA / REASSEMBLE robot data',
         f'#fig-benchmark-explorer', f'Figure {f_bench}'),
        (f'±{cov:.2f}', 'largest deviation of NormFlowSDE\'s 90% coverage from nominal, over all four datasets',
         '#fig-calibration', f'Figure {f_cal}'),
        (f'{min(speed):.1f}–{max(speed):.1f}×', 'faster than the fastest particle filter, on every dataset',
         '#fig-race', f'Figure {f_race}'),
        ('6 of 12', 'Lorenz–96 coordinates are observed; NormFlowSDE infers the other six through its learned dynamics',
         '#fig-posterior-explorer', f'Figure {f_post}'),
    ]
    items = ''.join(f'<a class="nf-card" href="{href}"><span class="nf-card-num">{num}</span>'
                    f'<span class="nf-card-txt">{txt}</span><span class="nf-card-link">{lk} →</span></a>'
                    for num, txt, href, lk in cards)
    return f'<div class="nf-cards">{items}</div>'


def add_key_ideas(md):
    for hid, txt in KEY_IDEAS.items():
        m = re.search(r'<h2 id="' + re.escape(hid) + r'">.*?</h2>\n', md)
        assert m, 'key-idea heading not found: ' + hid
        box = f'\n<div class="nf-key"><span class="nf-key-lbl">Key idea</span> {txt}</div>\n'
        md = md[:m.end()] + box + md[m.end():]
    return md


def collapse_sections(md):
    for hid, summary in COLLAPSE.items():
        m = re.search(r'<h([23]) id="' + re.escape(hid) + r'[^"]*">.*?</h\1>\n', md)
        assert m, 'collapse heading not found: ' + hid
        stop = r'\n<h2 id=' if m.group(1) == '2' else r'\n<h[23] id='  # a section ends at the next section
        nxt = re.search(stop, md[m.end():])
        end = m.end() + nxt.start() if nxt else len(md)
        body = md[m.end():end]
        md = (md[:m.end()] + f'\n<details class="nf-details"><summary>{summary}</summary>\n<div markdown="1">\n\n'
              + body.strip() + '\n\n</div>\n</details>\n' + md[end:])
    return md


def write_figure(name, data):
    tpl = (FIG_SRC / f'{name}.html').read_text(encoding='utf-8')
    lib = (FIG_SRC / '_lib.js').read_text(encoding='utf-8')
    css = (FIG_SRC / '_lib.css').read_text(encoding='utf-8')
    js = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    out = tpl.replace('/*__CSS__*/', css, 1).replace('/*__LIB__*/', lib, 1).replace('/*__DATA__*/', js, 1)
    dst = SUB / 'assets/html/submission' / f'{name}.html'
    dst.write_text(out, encoding='utf-8')
    print(f'  {dst.name}: {dst.stat().st_size / 1e3:.0f} kB')


def write_interactive(slug):
    """Flatten interactive/<slug>/index.html into one self-contained file: the TMLR kit copies only one directory
    level below assets/html, so local <link rel=stylesheet> and <script src> references are inlined."""
    src = INTERACTIVE_DIR / slug
    page = (src / 'index.html').read_text(encoding='utf-8')

    def css(m):
        return '<style>\n' + (src / m.group(1)).read_text(encoding='utf-8') + '</style>'

    def js(m):
        code = (src / m.group(1)).read_text(encoding='utf-8')
        if '</script' in code.lower():
            raise ValueError(f'{m.group(1)} contains </script')
        return '<script>\n' + code + '</script>'

    page = re.sub(r'<link rel="stylesheet" href="([^":]+)">', css, page)
    page = re.sub(r'<script src="([^":]+)"></script>', js, page)
    lazy = INTERACTIVE_LAZY.get(slug, [])
    if lazy:
        for f in lazy:
            shutil.copy(src / f, SUB / 'assets/html/submission' / (f'{slug}-' + f.replace('/', '-')))
        amap = ('<script>window.RV_ASSETS = {model: n => ' + json.dumps(f'{slug}-models-') + " + n + '.js', "
                'data: d => ' + json.dumps(f'{slug}-data-') + " + d + '.js'};</script>\n")
        page = page.replace('<script>', amap + '<script>', 1)
        size = sum((src / f).stat().st_size for f in lazy) / 1e6
        print(f'  {slug}: {len(lazy)} lazily loaded files, {size:.1f} MB')
    if re.search(r'(src|href)="(https?:)?//', page):
        raise ValueError(f'{slug}: external reference left in page')
    dst = SUB / 'assets/html/submission' / f'{slug}.html'
    dst.write_text(page, encoding='utf-8')
    print(f'  {dst.name}: {dst.stat().st_size / 1e3:.0f} kB')


# --------------------------------------------------------------------------------------
# Page assembly
# --------------------------------------------------------------------------------------

STYLES = r"""
  d-article .nf-figure { margin: 1.6em 0; }
  d-article .nf-figure img { display: block; margin: 0 auto; }
  d-article .nf-figure iframe { border: 1px solid rgba(0,0,0,.1); border-radius: 8px; display: block; width: 100%; background: #fff; }
  d-article figcaption { font-size: 0.85em; line-height: 1.5; color: rgba(0,0,0,.7); margin-top: .6em; text-align: left; }
  d-article .nf-badge { display: inline-block; font-size: .78em; font-weight: 600; color: #d9480f; border: 1px solid #d9480f;
    border-radius: 999px; padding: 0 .55em; margin-right: .2em; line-height: 1.5; vertical-align: 1px; }
  d-article .nf-table { margin: 1.6em 0; }
  d-article .nf-table .table-wrap { overflow-x: auto; }
  d-article .nf-table table { border-collapse: collapse; font-size: 0.78em; width: 100%; border-top: 1.5px solid #333; border-bottom: 1.5px solid #333; }
  d-article .nf-table th { border-bottom: 1px solid #333; padding: 4px 6px; font-weight: 600; white-space: nowrap; }
  d-article .nf-table td { padding: 3px 6px; border: 0; white-space: nowrap; }
  d-article .nf-table tr.mid td { border-top: 1px solid #333; }
  d-article .nf-table tr.cmid td:not(:first-child) { border-top: 1px solid #bbb; }
  d-article .theorem { border-left: 3px solid #d9480f; background: rgba(217,72,15,.04); padding: .6em 1em; margin: 1.2em 0; }
  d-article .theorem > p:last-child { margin-bottom: 0; }
  d-article h2[id], d-article h3[id], d-article h4[id] { scroll-margin-top: 70px; }
  d-article > *:not(d-contents) { grid-column: text-start / page-end !important; }
  /* The kit's TOC spans only 4 grid rows, so its height is spread over them and stretches the intro note, the cards,
     the first heading and the key-idea box. Spanning more rows leaves those rows at their content height. */
  @media (min-width: 1025px) { d-article d-contents { grid-row: auto / span 30; } }
  d-article .nf-key, d-article .nf-cards { align-self: start; }
  d-title p, d-article p, d-article li, d-article figcaption, d-article .nf-key { text-align: justify; hyphens: auto;
    -webkit-hyphens: auto; text-justify: inter-word; }
  d-article .nf-table td, d-article .nf-table th { text-align: left; }
  d-article .nf-table td, d-article .nf-table th { hyphens: manual; }
  d-article #tab-method-comparison td { white-space: normal; min-width: 7em; }
  d-article .nf-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: .5em 0 1.5em; }
  d-article a.nf-card { display: flex; flex-direction: column; gap: 4px; padding: 14px 14px 10px; border: 1px solid rgba(0,0,0,.1);
    border-radius: 10px; text-decoration: none; border-bottom: 1px solid rgba(0,0,0,.1); color: inherit; background: #fff;
    transition: box-shadow .15s, transform .15s; }
  d-article a.nf-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,.1); transform: translateY(-1px); border-bottom: 1px solid rgba(0,0,0,.1); }
  d-article .nf-card-num { font-size: 1.45em; white-space: nowrap; font-weight: 700; color: #d9480f; line-height: 1.1; font-variant-numeric: tabular-nums; }
  d-article .nf-card-txt { font-size: .82em; line-height: 1.4; color: rgba(0,0,0,.75); flex: 1; }
  d-article .nf-card-link { font-size: .75em; color: #1c7ed6; }
  d-article .nf-key { background: #f8f9fa; border-radius: 8px; padding: .7em 1em; margin: 0 0 1.4em; font-size: .95em; line-height: 1.55; }
  d-article .nf-key-lbl { display: inline-block; font-size: .72em; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
    color: #fff; background: #d9480f; border-radius: 4px; padding: 1px 6px; margin-right: 6px; vertical-align: 2px; }
  d-article details.nf-details { border: 1px solid rgba(0,0,0,.1); border-radius: 8px; padding: .4em 1em; margin: 0 0 1.4em; }
  d-article details.nf-details > summary { cursor: pointer; font-weight: 600; color: #1c7ed6; padding: .3em 0; }
  d-article details.nf-details[open] > summary { margin-bottom: .6em; border-bottom: 1px solid rgba(0,0,0,.08); }
  d-article .nf-table.l-page table { font-size: 0.74em; }
  d-article .nf-table.l-page td, d-article .nf-table.l-page th { padding: 3px 5px; }
  @media print {
    d-article .nf-table .table-wrap { overflow: visible; }
    d-article .nf-table table { font-size: 0.6em; }
    d-article .nf-table td, d-article .nf-table th { white-space: normal; }
    d-article .nf-figure, d-article .nf-table, d-article .theorem { break-inside: avoid; }
  }
"""

RESIZE_SCRIPT = """<script>
window.addEventListener('message', function (e) {
  if (!e.data || !e.data.nfFrame || !e.data.height) return;
  Array.prototype.forEach.call(document.querySelectorAll('iframe.nf-frame'), function (f) {
    if (f.contentWindow === e.source) f.style.height = (e.data.height + 4) + 'px';
  });
});
</script>"""


FIT_SCRIPT = """<script>
(function () {
  document.documentElement.lang = document.documentElement.lang || 'en';  // enables hyphenation
  // Shrink display equations that are wider than the text column; scroll only below 65 %.
  function fit() {
    Array.prototype.forEach.call(document.querySelectorAll('d-article mjx-container[display="true"]'), function (m) {
      m.style.fontSize = ''; m.style.minWidth = ''; m.style.maxWidth = ''; m.style.overflowX = '';
      var inner = m.firstElementChild;
      if (!inner) return;
      // MathJax gives the container a min-width equal to the equation, so measure the column instead
      var host = m.parentElement, ref = host;
      if (host.tagName === 'D-ARTICLE') ref = host.querySelector(':scope > p') || host;
      var cs = getComputedStyle(ref);
      var cw = ref.getBoundingClientRect().width - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight), k = 1;
      m.style.minWidth = '0';
      m.style.maxWidth = cw + 'px';
      for (var i = 0; i !== 6; i++) {  // parts of the layout do not scale, so refine a few times
        var w = Math.max(m.scrollWidth, inner.scrollWidth);
        if (w - cw <= 1 || k === 0.65) break;
        k = Math.max(0.65, k * cw / w * 0.99);
        m.style.fontSize = (k * 100).toFixed(1) + '%';
      }
      if (Math.max(m.scrollWidth, inner.scrollWidth) - cw > 1) { m.style.overflowX = 'auto'; m.style.overflowY = 'hidden'; }
    });
  }
  function start() {
    if (window.MathJax && MathJax.startup && MathJax.startup.promise) MathJax.startup.promise.then(fit);
    else setTimeout(start, 300);
  }
  start();
  var t = null;
  window.addEventListener('resize', function () { clearTimeout(t); t = setTimeout(fit, 150); });
  window.addEventListener('beforeprint', fit);
})();
</script>"""

PRINT_SCRIPT = """<script>
window.addEventListener('beforeprint', function () {
  Array.prototype.forEach.call(document.querySelectorAll('details.nf-details'), function (d) { d.open = true; });
});
// A link into a collapsed appendix opens it before scrolling.
function nfOpenFor(id) {
  var el = id && document.getElementById(id);
  for (var p = el; p; p = p.parentElement) if (p.tagName === 'DETAILS') p.open = true;
  if (el) el.scrollIntoView();
}
document.addEventListener('click', function (e) {
  var a = e.target.closest ? e.target.closest('a[href^="#"]') : null;
  if (!a) return;
  var id = decodeURIComponent(a.getAttribute('href').slice(1)), el = document.getElementById(id);
  if (el && el.closest('details')) { e.preventDefault(); nfOpenFor(id); history.replaceState(null, '', '#' + id); }
});
window.addEventListener('load', function () { if (location.hash) nfOpenFor(decodeURIComponent(location.hash.slice(1))); });
</script>"""


def yaml_str(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def main():
    main_tex = read_tex(PAPER / 'main.tex')
    title = re.search(r'\\title\{(.*?)\}\n', main_tex, re.S).group(1).replace('\\\\', ' ')
    title = re.sub(r'\s+', ' ', title)
    abstract = main_tex[main_tex.index('\\begin{abstract}') + 16:main_tex.index('\\end{abstract}')]
    doc = main_tex[main_tex.index('\\end{abstract}'):main_tex.index('\\end{document}')]
    parts = []
    for m in re.finditer(r'\\input\{(sections/[^}]*)\}|\\appendix', doc):
        if m.group(0) == '\\appendix':
            parts.append('\n\\appendix\n')
        else:
            parts.append(read_tex(PAPER / (m.group(1) + '.tex')))
    body = '\n\n'.join(parts)
    body = '\n'.join(ln for ln in body.split('\n') if not ln.lstrip().startswith('\\providecommand'))
    body = re.sub(r'\\input\{(tables/[^}]*)\}', lambda m: read_tex(PAPER / (m.group(1) + '.tex')), body)
    body = re.sub(r'\\textnormal\{\\bfseries\s*([^{}]*)\}', r'\\textbf{\1}', body)
    for a, b in WEB_EDITS:
        assert a in body, 'web edit anchor missing: ' + a
        body = body.replace(a, b)
    for anchor, where, ins in INJECT:
        assert body.count(anchor) == 1, 'inject anchor not unique: ' + anchor
        body = body.replace(anchor, anchor + ins)
    # figure includes: replace \includegraphics / \INTERACTIVE by nothing (assets are mapped by label)
    body = re.sub(r'\\includegraphics(\[[^\]]*\])?\{[^}]*\}', '', body)
    body = re.sub(r'\\INTERACTIVE\{[^}]*\}\{[^}]*\}', '', body)

    bib_text = (PAPER / 'references.bib').read_text(encoding='utf-8')
    CTX['bib'] = parse_bib(bib_text)
    labels, headings = number_document(body)
    CTX['labels'], CTX['headings'] = labels, headings
    md = convert_blocks(body)

    # front matter
    toc = []
    for h in headings:
        if h['level'] == 2:
            toc.append({'name': f'{h["num"]}. {h["title"]}', 'subs': []})
        elif h['level'] == 3 and not h['appendix']:
            toc[-1]['subs'].append(f'{h["num"]} {h["title"]}')
    toc_yaml = []
    for t in toc:
        assert slugify(t['name']) in {h['id'] for h in headings}, t['name']
        toc_yaml.append(f'  - name: {yaml_str(t["name"])}')
        if t['subs']:
            toc_yaml.append('    subsections:')
            for s in t['subs']:
                toc_yaml.append(f'      - name: {yaml_str(s)}')
    desc = inline_text(re.sub(r'\s+', ' ', abstract).strip(), plain=True)
    front = '\n'.join([
        '---',
        'layout: distill',
        f'title: {yaml_str(title)}',
        f'description: {yaml_str(desc)}',
        'htmlwidgets: true',
        '',
        '# Anonymize when submitting',
        'authors:',
        '  - name: Anonymous',
        '    affiliations:',
        '      name: Anonymous',
        '',
        '# Only add author names for camera-ready',
        '# authors:',
        '#   - name: Author Name',
        '#     url: "https://[url_of_author]"',
        '#     affiliations:',
        '#       name: Research Center, University Name',
        '',
        '# Must be the same name as your submission. Do not change this name, just use "submission.bib".',
        'bibliography: submission.bib',
        '',
        'toc:',
        *toc_yaml,
        '',
        '_styles: >',
        *STYLES.strip('\n').split('\n'),
        '---',
        '',
    ])
    intro = ('<p class="nf-reading-note"><em>This is the web version of the manuscript. Figures marked '
             '<span class="nf-badge">interactive</span> respond to clicks and hovering; all numbers are computed from the same '
             'frozen run directories as the tables.</em></p>')
    bench = data_benchmark()
    md = add_key_ideas(md)
    md = collapse_sections(md)
    page = (front + '\n' + intro + '\n\n' + summary_cards(bench, labels) + '\n\n' + md + '\n\n' + RESIZE_SCRIPT + '\n'
            + PRINT_SCRIPT + '\n' + FIT_SCRIPT + '\n')
    page = re.sub(r'\n{3,}', '\n\n', page)
    for bad in ('{%', '{{'):
        for m in re.finditer(re.escape(bad), page):
            ctx = page[m.start():m.start() + 120]
            if 'relative_url' not in ctx:
                raise ValueError('Liquid-like sequence in content: ' + ctx)

    # --- write submission folder
    for sub in ('assets/img/submission', 'assets/gif/submission', 'assets/html/submission', 'assets/bibliography'):
        d = SUB / sub
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
        d.mkdir(parents=True, exist_ok=True)
    (SUB / 'assets/gif/submission/.gitkeep').write_text('')
    (SUB / 'submission.md').write_text(page, encoding='utf-8')
    (SUB / 'assets/bibliography/submission.bib').write_text(clean_bib(bib_text), encoding='utf-8')
    shutil.copy(PAPER / 'figures/NormFlowSDE_figure.svg', SUB / 'assets/img/submission/normflowsde_overview.svg')
    print('figures:')
    write_figure('calibration', data_calibration())
    write_figure('noise_recovery', data_noise())
    write_figure('weak_convergence', data_weak(labels['thm:weak']['num']))
    write_figure('benchmark', bench)
    write_figure('posterior', data_posterior())
    write_figure('update', {})
    write_figure('methods', data_methods())
    write_figure('noise_explainer', data_noise_explainer())
    race = data_race(bench)
    race['tables'] = labels['tab:results-sim']['num'] + '–' + labels['tab:results-robot']['num']
    write_figure('race', race)
    write_figure('centering', data_centering(labels))
    for slug in INTERACTIVE_FIGURES:
        write_interactive(slug)
    print(f'submission.md: {len(page) / 1e3:.0f} kB, {len(headings)} headings, '
          f'{sum(1 for v in labels.values() if v["kind"] == "fig")} figures, {sum(1 for v in labels.values() if v["kind"] == "tab")} tables')
    missing = sorted({k for m in re.finditer(r'd-cite key="([^"]*)"', page) for k in m.group(1).split(',')} - set(CTX['bib']))
    if missing:
        print('WARNING missing bib keys:', missing)


if __name__ == '__main__':
    main()
