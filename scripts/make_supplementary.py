"""Anonymous supplementary material for the TMLR submission (OpenReview "Supplementary Material" zip).

  python scripts/make_supplementary.py          -> supplementary/NormFlowSDE_supplementary.zip (lean, default)
  python scripts/make_supplementary.py --full   -> supplementary/NormFlowSDE_supplementary_full.zip (all tracked files)
  (supplementary/ is git-ignored)

Contents: the tracked files of this repository (code, experiment scripts, trained checkpoints, frozen light results,
manuscript sources, interactive-figure sources, tests), without
  - the TMLR Beyond PDF author kit (uploaded separately as the Beyond PDF zip),
  - paper/vendor/ (leftovers of the TMLR template) and paper/main.pdf (the submission PDF is uploaded separately),
  - anything untracked or git-ignored (large posterior/data arrays, the arXiv package with the author block),
  - lean version only: the per-run training logs (history.jsonl, tracking-events.jsonl; 427 MB) and the trained
    checkpoints (model.eqx; 156 MB), which stay in the repository cited in the paper. The tables are built from the
    metrics, summary and analysis files, which are included.
A guide (SUPPLEMENTARY.md) is added at the top. Every text file is scanned for identifying strings before writing;
the build fails on a hit.
"""
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'supplementary'
TOP = 'NormFlowSDE'
EXCLUDE = ('tmlr-beyond-pdf-author-kit/', 'paper/vendor/', 'paper/main.pdf')
LEAN_EXCLUDE = re.compile(r'(^|/)(history\.jsonl|tracking-events\.jsonl|model\.eqx)$')
FULL = '--full' in sys.argv
# identifying strings that must not appear (case-insensitive); third-party licence/template text is allowed explicitly
# Generic patterns here; names, institutions and host names go in supplementary/identifying.txt (one regex per line,
# git-ignored, so that this public script does not itself contain them).
GENERIC = [r'C:\\Users\\(?!anon\b)\w+', r'/home/(?!anon\b|u/anon\b)\w+', r'/Users/(?!anon\b)\w+']
_local = OUT / 'identifying.txt'
_extra = [ln.strip() for ln in _local.read_text(encoding='utf-8').splitlines()
          if ln.strip() and not ln.startswith('#')] if _local.exists() else []
if not _extra:
    print('WARNING: supplementary/identifying.txt missing or empty; only generic patterns are checked')
FORBIDDEN = re.compile('|'.join(GENERIC + _extra), re.IGNORECASE)
TEXT = {'.py', '.sh', '.sbatch', '.md', '.txt', '.tex', '.bib', '.bst', '.sty', '.json', '.jsonl', '.csv', '.js', '.html',
        '.css', '.svg', '.yaml', '.yml', '.toml', '.cfg', '.ps1', '.out', '.log', ''}

GUIDE = """# Supplementary material: NormFlowSDE

Anonymous code and results archive for the submission "NormFlowSDE: Normalizing Flows for Amortized Bayesian State
Estimation". It is a snapshot of the code repository cited in Appendix C (Software, hardware, and code availability).

Start with `README.md`: layout, environment (`requirements.txt`), the commands that rebuild every table and figure
from the frozen results, retraining, and what is not included.

| folder | contents |
|---|---|
| `bayesinf/`, `experiments/` | library code and experiment scripts (entry points in `experiments/normflowsde_comparison/`) |
| `results/` | frozen run directories (configs, metrics, summaries{FULL_RESULTS}), data-cache manifests, reference filters, analyses, table builders, provenance |
| `paper/` | manuscript sources (LaTeX, tables, figures, bibliography, figure scripts) |
| `interactive/`, `beyond_pdf/`, `scripts/`, `tests/` | sources of the interactive figures of the web version, its build script, export scripts, and their tests (`node --test tests/*.test.js`) |

Not included, because of their size and redistribution restrictions (as stated in the paper): the `.npz` arrays
(posterior draws, data caches, reference-filter outputs; 28 GB) and the raw robot recordings. `README.md`, section
"What is not included", lists them. All metrics, summaries and analysis rows behind the tables are included.{LEAN_NOTE}
"""
LEAN_NOTE = ('\n\nTo stay within the upload limit, this archive also leaves out the per-run training logs '
             '(`history.jsonl`, `tracking-events.jsonl`) and the trained checkpoints (`model.eqx`). Both are in the '
             'code repository cited in Appendix C, which has the same layout.')


def tracked():
    out = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, check=True).stdout.decode('utf-8')
    keep = [f for f in out.split('\0') if f and not f.startswith(EXCLUDE) and (ROOT / f).is_file()]
    return keep if FULL else [f for f in keep if not LEAN_EXCLUDE.search(f)]


def main():
    files = tracked()
    hits = []
    for f in files:
        p = ROOT / f
        if p.suffix.lower() in TEXT or p.name in ('LICENSE', 'Makefile'):
            t = p.read_text(encoding='utf-8', errors='ignore')
        else:                                         # binaries: scan the raw bytes as latin-1
            t = p.read_bytes().decode('latin-1')
        for m in FORBIDDEN.finditer(t):
            hits.append(f'{f}: {t[max(0, m.start() - 40):m.end() + 40]!r}')
    if hits:
        print('\n'.join(hits[:30]))
        raise SystemExit(f'{len(hits)} identifying strings found; nothing written')
    OUT.mkdir(exist_ok=True)
    zpath = OUT / ('NormFlowSDE_supplementary_full.zip' if FULL else 'NormFlowSDE_supplementary.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        guide = GUIDE.replace('{FULL_RESULTS}', ', training logs, trained checkpoints `model.eqx`' if FULL else '')
        z.writestr(f'{TOP}/SUPPLEMENTARY.md', guide.replace('{LEAN_NOTE}', '' if FULL else LEAN_NOTE))
        for f in sorted(files):
            z.write(ROOT / f, f'{TOP}/{f}')
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        n = len(z.namelist())
    size = sum((ROOT / f).stat().st_size for f in files) / 1e6
    print(f'{zpath.relative_to(ROOT)}: {n} files, {size:.1f} MB unpacked, {zpath.stat().st_size / 1e6:.1f} MB zipped; '
          'anonymity scan clean')


if __name__ == '__main__':
    main()
