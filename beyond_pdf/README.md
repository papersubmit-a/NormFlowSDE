# TMLR Beyond PDF version

Builds the web (Beyond PDF) submission from the LaTeX manuscript in `paper/` and the frozen results in `results/`.

```
python beyond_pdf/build_beyond_pdf.py   # writes tmlr-beyond-pdf-author-kit/submission_folder/
python beyond_pdf/preview.py            # optional local check -> beyond_pdf/_preview/index.html, preview.pdf
```

- `build_beyond_pdf.py` converts `paper/main.tex` and its sections to `submission.md` (Distill layout, kramdown + MathJax),
  numbers sections, tables, figures and theorems like LaTeX, copies a cleaned `references.bib`, and writes the
  interactive figures with their data inlined. Text changes specific to the web version are listed in `WEB_EDITS`,
  `INJECT` and `CAPTIONS`.
- `figures/*.html` are the figure templates (vanilla JS/SVG, no external scripts); `_lib.js` and `_lib.css` are
  inlined into each.
- `preview.py` reproduces the parts of the TMLR pipeline that affect the content (math-delimiter conversion, kramdown
  with Jekyll's options, the Distill layout with the kit's own assets) without Docker. It needs WSL Ubuntu with
  `gem install --user-install kramdown kramdown-parser-gfm rexml`. The authoritative check is the kit's
  `python compile_submission.py` (Docker) or the online editor at https://tmlr-beyond-pdf.org/editor.

Interactive figures: four explainers (one update, observation noise, latency race, centering), accuracy vs. latency (Tables 2–3), Lorenz–96 posterior explorer (seed 0, from
`results/shared_trunk_20261114/normflowsde/lorenz96/seed0/s2/posterior.npz`), calibration curves, REASSEMBLE noise
recovery, and the Ornstein–Uhlenbeck weak-convergence study.

Newer figures (likelihood collapse, energy score vs. entropy) are developed as multi-file pages in `interactive/<slug>/`
(see `interactive/README.md`, tests in `tests/`, fallbacks by `scripts/make_fallbacks.py`). `write_interactive` inlines
their local scripts and styles into one file each, because the TMLR kit copies only one directory level below
`assets/html/`.

Upload: `submission_folder.zip` (zip of `submission_folder/`) in the "Beyond PDF" section on OpenReview, plus a PDF made
with the browser's Print to PDF from the rendered page.
