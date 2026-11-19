"""Static fallbacks (default state) of the interactive figures, for the PDF version.

Writes, per figure, into interactive/<slug>/:
  fallback.svg, fallback.png        double column, 6.75 in wide, PNG at 300 dpi
  fallback-1col.svg, fallback-1col.png  single column, 3.25 in wide, PNG at 300 dpi
The numbers come from the page's own model code (scripts/dump_figure_state.js, run with Node), so the fallback shows
exactly the default state of the page. Metadata that would identify the machine or date is stripped.

Usage: python scripts/make_fallbacks.py [slug ...]     (default: all figures)
"""
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INTER = ROOT / 'interactive'
# Okabe-Ito, as in interactive/_shared/ix.css
MIX, GAUSS, THIRD, DIFF, FG = '#0072B2', '#D55E00', '#009E73', '#6e7781', '#1f2328'
plt.rcParams.update({
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'legend.fontsize': 6.2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.3, 'svg.hashsalt': 'ix-fallback',
    'svg.fonttype': 'path', 'mathtext.fontset': 'dejavusans',
})
SVG_META = {'Date': None, 'Creator': None, 'Contributor': None, 'Publisher': None, 'Rights': None}
PNG_META = {'Software': None}


def state(slug):
    out = subprocess.run(['node', str(ROOT / 'scripts' / 'dump_figure_state.js'), slug], check=True,
                         capture_output=True, text=True).stdout
    return json.loads(out)


def save(fig, slug, suffix):
    d = INTER / slug
    fig.savefig(d / f'fallback{suffix}.svg', metadata=SVG_META)
    fig.savefig(d / f'fallback{suffix}.png', dpi=300, metadata=PNG_META)
    plt.close(fig)


# ------------------------------------------------------------------------------ likelihood-collapse
def lc_panel_a(ax, s):
    a = s['a']
    ax.fill_between(a['e'], a['p'], a['q'], color=DIFF, alpha=0.3, lw=0, label=r'$|p-q|$ (area $=2\,$TV)')
    ax.plot(a['e'], a['p'], color=MIX, label=r'mixture $p(e)$')
    ax.plot(a['e'], a['q'], color=GAUSS, ls=(0, (5, 3)), label=r'moment-matched $q(e)$')
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'noise $e/\sigma$')
    ax.set_ylabel('density')
    ax.set_title(f"A  Mixture vs. matched Gaussian, TV = {a['tv']:.3f}", loc='left')
    ax.legend(frameon=False, loc='upper right', handlelength=2.2)


def lc_panel_b(ax, s):
    b = s['b']
    wmax = max(max(b['wMix']), max(b['wGauss']))
    lmax = max(max(b['likMix']), max(b['likGauss']))
    ax.axhline(0, color=DIFF, lw=0.6)
    ax.plot(b['xl'], [v / lmax * wmax for v in b['likMix']], color=MIX, alpha=0.35, lw=0.9)
    ax.plot(b['xl'], [-v / lmax * wmax for v in b['likGauss']], color=GAUSS, alpha=0.35, lw=0.9, ls=(0, (5, 3)))
    ax.vlines(b['x'], 0, b['wMix'], color=MIX, lw=0.5, alpha=0.8)
    ax.plot(b['x'], b['wMix'], 'o', ms=1.6, color=MIX, label='weights, mixture (up)')
    ax.vlines(b['x'], 0, [-w for w in b['wGauss']], color=GAUSS, lw=0.5, alpha=0.8)
    ax.plot(b['x'], [-w for w in b['wGauss']], 's', ms=1.5, color=GAUSS, label='weights, Gaussian (down)')
    ax.vlines(b['mix']['mean'], 0, wmax * 1.05, color=MIX, lw=1.6)
    ax.vlines(b['gauss']['mean'], -wmax * 1.05, 0, color=GAUSS, lw=1.6, ls=(0, (3, 2)))
    ax.axvline(b['y'], color=FG, lw=0.6, ls=':')
    ax.plot([b['y']], [wmax * 1.15], 'D', ms=4, color=FG, label=r'observation $y$')
    ax.plot([b['truth']], [-wmax * 1.15], '^', ms=4, color=FG, label=r'true state $x^*$')
    ax.set_xlim(-b['L'], b['L'])
    ax.set_ylim(-wmax * 1.75, wmax * 1.25)  # room for the stats box below the stems
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(r'state $x/\sigma$')
    ax.set_title(r'B  Weights, observation in the side mode ($y=%.1f\sigma$)' % b['y'], loc='left')
    m, g = b['mix'], b['gauss']
    ax.text(0.01, 0.03, f"posterior mean / sd / ESS\nmixture   {m['mean']:.2f} / {m['sd']:.2f} / {m['ess']:.0f}\n"
            f"Gaussian  {g['mean']:.2f} / {g['sd']:.2f} / {g['ess']:.0f}",
            transform=ax.transAxes, ha='left', va='bottom', fontsize=5.6, family='monospace')
    ax.legend(frameon=False, loc='upper left', fontsize=5.6, handlelength=1.2, borderaxespad=0.1)


def likelihood_collapse():
    s = state('likelihood-collapse')
    fig, axs = plt.subplots(1, 2, figsize=(6.75, 2.5), layout='constrained')
    lc_panel_a(axs[0], s)
    lc_panel_b(axs[1], s)
    save(fig, 'likelihood-collapse', '')
    fig, axs = plt.subplots(2, 1, figsize=(3.25, 4.6), layout='constrained')
    lc_panel_a(axs[0], s)
    lc_panel_b(axs[1], s)
    save(fig, 'likelihood-collapse', '-1col')
    return s['a']['tv']


# ------------------------------------------------------------------------------ energy-score
def energy_score_axes(ax1, ax2, s):
    ax1.plot(s['c'], s['acc'], color=MIX, label=r'$\mathbb{E}\Vert z-x\Vert$')
    ax1.plot(s['c'], s['half'], color=GAUSS, ls=(0, (5, 3)), label=r"$\frac{1}{2}\mathbb{E}\Vert z-z'\Vert$")
    ax1.plot(s['c'], s['es'], color=FG, lw=2, label='energy score (sum)')
    ax1.axvline(s['cstar'], color=FG, lw=0.6, ls=':')
    ax1.plot([s['cstar']], [s['esStar']], 'o', mfc='white', mec=FG, ms=4)
    ax1.annotate(f"min at $c={s['cstar']:.3f}$", (s['cstar'], s['esStar']), xytext=(4, -10),
                 textcoords='offset points', fontsize=6.2)
    ax1.set_ylabel('expected value')
    ax1.set_ylim(0, None)
    ax1.legend(frameon=False, loc='upper left', handlelength=2.2)
    ax1.set_title(f"Energy score vs. entropy term, $d={s['d']}$ (toy: $x\\sim\\mathcal{{N}}(0,I_d)$, $z=c\\,\\epsilon$)",
                  loc='left')
    ax2.plot(s['c'], s['entropy'], color=THIRD, lw=1.6, ls=(0, (1, 1.5)), label=r'entropy $H(\mathcal{N}(0,c^2I_d))$')
    ax2.set_ylabel('nats')
    ax2.set_xlabel(r'ensemble spread $c$ (log scale)')
    ax2.legend(frameon=False, loc='upper left', handlelength=2.2)
    for ax in (ax1, ax2):
        ax.set_xscale('log')
        ax.set_xlim(s['c'][0], s['c'][-1])
        ax.set_xticks([0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3])
        ax.set_xticklabels(['0.2', '0.3', '0.5', '0.7', '1', '1.5', '2', '3'])
        ax.minorticks_off()
    ax1.tick_params(labelbottom=False)


def energy_score():
    s = state('energy-score')
    for suffix, size in (('', (6.75, 3.3)), ('-1col', (3.25, 3.3))):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=size, sharex=True, height_ratios=(2, 1), layout='constrained')
        energy_score_axes(a1, a2, s)
        if suffix:
            a1.set_title(f"Energy score vs. entropy, $d={s['d']}$", loc='left')
        save(fig, 'energy-score', suffix)
    return s['cstar']


# ------------------------------------------------------------------------------ overview (paper Figure 1)
def chrome():
    for c in (os.environ.get('CHROME'), shutil.which('chrome'), shutil.which('google-chrome'), shutil.which('chromium'),
              r'C:\Program Files\Google\Chrome\Application\chrome.exe',
              r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'):
        if c and Path(c).exists():
            return c
    raise SystemExit('overview fallback needs Chrome or Edge (set CHROME=/path/to/browser)')


def overview():
    """The overview animates the paper's SVG in the browser, so its default state is taken from the rendered DOM
    (headless Chrome) and rasterised by the same browser."""
    d = INTER / 'overview'
    with tempfile.TemporaryDirectory() as tmp:
        dom = subprocess.run([chrome(), '--headless=new', '--disable-gpu', '--virtual-time-budget=3000', '--dump-dom',
                              (d / 'index.html').as_uri()], check=True, capture_output=True, text=True,
                             encoding='utf-8').stdout
        svg = re.search(r'<svg[\s\S]*?</svg>', dom[dom.index('id="fig"'):]).group(0)
        svg = re.sub(r'\s(role|aria-label|class|style)="[^"]*"', '', svg, count=4)
        svg = svg.replace('<svg ', '<svg width="1600" height="606" ', 1)
        (d / 'fallback.svg').write_text(svg, encoding='utf-8')
        (d / 'fallback-1col.svg').write_text(svg, encoding='utf-8')
        for suffix, inches in (('', 6.75), ('-1col', 3.25)):
            w = round(inches * 300)
            h = round(w * 606 / 1600)
            page = Path(tmp) / 'page.html'
            page.write_text('<!doctype html><style>html,body{margin:0;background:#fff}svg{display:block;width:100vw;'
                            'height:auto}</style>' + svg, encoding='utf-8')
            subprocess.run([chrome(), '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--window-size={w},{h}',
                            f'--screenshot={d / f"fallback{suffix}.png"}', page.as_uri()], check=True,
                           capture_output=True)
    return 'default t = ' + re.search(r'time step (\d+)', dom).group(1)


def robot_view():
    """Four frames of the median REASSEMBLE test window (3D view + timeline), screenshots of the page in headless
    Chrome (WebGL via SwiftShader), tiled 2 x 2. The SVG variants embed the PNG (the 3D view is raster content)."""
    from PIL import Image
    d = INTER / 'robot-view'
    steps = [0, 80, 160, 240]
    with tempfile.TemporaryDirectory() as tmp:
        tiles = []
        for k in steps:
            out = Path(tmp) / f'k{k}.png'
            url = (d / 'index.html').as_uri() + f'#ds=reassemble&w=5&k={k}&still'
            subprocess.run([chrome(), '--headless=new', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                            '--hide-scrollbars', '--force-device-scale-factor=2', '--virtual-time-budget=15000',
                            '--window-size=900,820', f'--screenshot={out}', url], check=True, capture_output=True)
            im = Image.open(out).convert('RGB')
            bbox = Image.eval(im, lambda v: 255 - v).getbbox()           # crop the white margin
            tiles.append(im.crop(bbox) if bbox else im)
        tw = max(t.width for t in tiles)
        th = max(t.height for t in tiles)
        grid = Image.new('RGB', (2 * tw, 2 * th), 'white')
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % 2) * tw, (i // 2) * th))
    for suffix, inches in (('', 6.75), ('-1col', 3.25)):
        w = round(inches * 300)
        im = grid.resize((w, round(grid.height * w / grid.width)), Image.LANCZOS)
        png = d / f'fallback{suffix}.png'
        im.save(png, dpi=(300, 300), optimize=True)
        b64 = base64.b64encode(png.read_bytes()).decode('ascii')
        (d / f'fallback{suffix}.svg').write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{inches}in" height="{inches * im.height / im.width:.3f}in" '
            f'viewBox="0 0 {im.width} {im.height}"><image width="{im.width}" height="{im.height}" '
            f'href="data:image/png;base64,{b64}"/></svg>\n', encoding='utf-8')
    return f'REASSEMBLE median window, steps {steps}'


FIGURES = {'overview': overview, 'likelihood-collapse': likelihood_collapse, 'energy-score': energy_score,
           'robot-view': robot_view}

if __name__ == '__main__':
    for slug in sys.argv[1:] or FIGURES:
        print(slug, FIGURES[slug]())
