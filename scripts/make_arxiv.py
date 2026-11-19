"""Build the arXiv source package of the manuscript (paper/) and check that it compiles on its own.

  python scripts/make_arxiv.py          -> arxiv/normflowsde_arxiv.zip, arxiv/build/ (unpacked), arxiv/check.pdf

What it does
  - copies only the files the LaTeX build reads (paper/main.fls) into arxiv/build/;
  - switches tmlr to the `preprint` option (authors shown, no "Under review" header) and replaces the anonymous
    author block by arxiv/authors.tex (kept out of git: the repository stays anonymous during review);
  - removes all % comments (arXiv publishes the source), keeping a bare % where it joins lines;
  - adds \\pdfoutput=1 so that arXiv uses pdfLaTeX, and bundles a freshly generated main.bbl (arXiv does not run BibTeX);
  - compiles the package in a clean directory (pdflatex, bibtex, pdflatex x2) and fails on errors or undefined
    references, then zips it.
arxiv/ is listed in .gitignore.
"""
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
OUT = ROOT / 'arxiv'
BUILD = OUT / 'build'
AUTHORS = OUT / 'authors.tex'
TEXBIN = Path(os.environ['TEXBIN']) if os.environ.get('TEXBIN') else None   # optional TeX Live bin directory, else PATH
AUTHORS_TEMPLATE = r"""% Author block for the arXiv version (tmlr format). Not in git. Example:
% \author{\name First Author \email first@example.org \\
%         \addr Affiliation, City, Country
%         \AND
%         \name Second Author \email second@example.org \\
%         \addr Affiliation, City, Country}
\author{\name AUTHOR NAMES TO BE FILLED IN \email email@example.org \\ \addr Affiliation}
"""


def tool(name):
    if TEXBIN:
        for cand in (TEXBIN / name, TEXBIN / (name + '.exe')):
            if cand.exists():
                return str(cand)
    return shutil.which(name) or name


def used_files():
    """Files under paper/ that the last LaTeX run read (main.fls), except build products."""
    fls = (PAPER / 'main.fls').read_text(encoding='utf-8', errors='ignore')
    out = set()
    for line in fls.splitlines():
        if not line.startswith('INPUT '):
            continue
        p = Path(line[6:].strip())
        p = p if p.is_absolute() else (PAPER / p)
        try:
            rel = p.resolve().relative_to(PAPER.resolve())
        except ValueError:
            continue                                   # TeX Live files
        if rel.suffix in ('.aux', '.out', '.toc', '.bbl', '.log') or rel.name.startswith('main.'):
            continue
        out.add(rel)
    return sorted(out)


def strip_comments(text):
    """Remove % comments; whole-line comments disappear, inline ones keep a bare % (it may join lines)."""
    lines = []
    for line in text.splitlines():
        m = re.search(r'(?<!\\)%', line)
        if m is None:
            lines.append(line)
            continue
        head = line[:m.start()]
        if head.strip() == '':
            continue
        lines.append(head + '%')
    return '\n'.join(lines) + '\n'


def prepare_main(text):
    text = text.replace(r'\usepackage{tmlr}', r'\usepackage[preprint]{tmlr}', 1)
    assert r'\usepackage[preprint]{tmlr}' in text
    authors = AUTHORS.read_text(encoding='utf-8')
    authors = strip_comments(authors).strip()
    text, n = re.subn(r'\\author\{\\name Anonymous authors\}', lambda _: authors, text)
    assert n == 1, 'anonymous author block not found in main.tex'
    return '\\pdfoutput=1\n' + text


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise SystemExit(f'failed: {" ".join(cmd)}')
    return r.stdout


def main():
    OUT.mkdir(exist_ok=True)
    if not AUTHORS.exists():
        AUTHORS.write_text(AUTHORS_TEMPLATE, encoding='utf-8')
        print(f'wrote a template {AUTHORS.relative_to(ROOT)}: fill in the authors and run again')
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    files = used_files()
    for rel in files:
        dst = BUILD / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel.suffix in ('.tex',):
            dst.write_text(strip_comments((PAPER / rel).read_text(encoding='utf-8')), encoding='utf-8')
        else:
            shutil.copy(PAPER / rel, dst)
    (BUILD / 'main.tex').write_text(prepare_main(strip_comments((PAPER / 'main.tex').read_text(encoding='utf-8'))),
                                    encoding='utf-8')
    # bibliography: generate main.bbl in a scratch copy (with references.bib and tmlr.bst), ship only the .bbl
    work = OUT / 'work'
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(BUILD, work)
    shutil.copy(PAPER / 'references.bib', work)
    shutil.copy(PAPER / 'tmlr.bst', work)
    tex = [tool('pdflatex'), '-interaction=nonstopmode', '-halt-on-error', 'main.tex']
    run(tex, work)
    run([tool('bibtex'), 'main'], work)
    shutil.copy(work / 'main.bbl', BUILD / 'main.bbl')
    # check: compile the package exactly as shipped (no .bib, no .bst), as arXiv will
    check = OUT / 'check'
    if check.exists():
        shutil.rmtree(check)
    shutil.copytree(BUILD, check)
    for _ in range(3):
        log = run(tex, check)
    full = (check / 'main.log').read_text(encoding='utf-8', errors='ignore')
    problems = [l for l in full.splitlines() if re.search(r'undefined|Citation .* undefined|Reference .* undefined', l)]
    if problems:
        print('\n'.join(problems[:20]))
        raise SystemExit('undefined references or citations')
    shutil.copy(check / 'main.pdf', OUT / 'check.pdf')
    pages = re.search(r'Output written on main\.pdf \((\d+) pages', log)
    # zip (flat paths as in the build directory)
    zpath = OUT / 'normflowsde_arxiv.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(BUILD.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(BUILD).as_posix())
    shutil.rmtree(work)
    shutil.rmtree(check)
    print(f'{len(files) + 2} files ({", ".join(sorted({p.suffix for p in BUILD.rglob("*") if p.is_file()}))}); '
          f'check.pdf: {pages.group(1) if pages else "?"} pages; {zpath.relative_to(ROOT)}: {zpath.stat().st_size / 1e6:.2f} MB')
    if 'AUTHOR NAMES TO BE FILLED IN' in (BUILD / 'main.tex').read_text(encoding='utf-8'):
        print('NOTE: arxiv/authors.tex still holds the placeholder author block')


if __name__ == '__main__':
    main()
