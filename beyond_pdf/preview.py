"""Approximate local preview of the Beyond PDF page without Docker/Jekyll.

Reproduces the parts of the TMLR pipeline that affect our content:
  1. compile_submission.py's convert_math_delimiters ($ -> $$),
  2. Liquid's relative_url filter,
  3. kramdown (GFM input, MathJax engine, Jekyll's default options), run in WSL,
  4. the Distill layout with the kit's own JS/CSS and the same MathJax configuration.
Then prints the page to PDF with headless Edge (like the required browser "Print to PDF").

Usage:  python beyond_pdf/preview.py   -> beyond_pdf/_preview/index.html and preview.pdf
Requires: WSL Ubuntu with `gem install --user-install kramdown kramdown-parser-gfm rexml`.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent / 'tmlr-beyond-pdf-author-kit'
SUB = KIT / 'submission_folder'
TMLR = KIT / 'tmlr_do_not_modify'
OUT = HERE / '_preview'
EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'

RUBY = r'''
require "kramdown"
require "kramdown-parser-gfm"
src = File.read(ARGV[0], encoding: "UTF-8")
opts = { input: "GFM", math_engine: "mathjax", auto_ids: true, hard_wrap: false, entity_output: "as_char",
         smart_quotes: "lsquo,rsquo,ldquo,rdquo", syntax_highlighter: nil, show_warnings: true }
doc = Kramdown::Document.new(src, opts)
File.write(ARGV[1], doc.to_html)
doc.warnings.each { |w| STDERR.puts "kramdown: #{w}" }
'''


def wsl_path(p):
    p = str(p.resolve()).replace('\\', '/')
    return '/mnt/' + p[0].lower() + p[2:]


def convert_math_delimiters(content):  # copied from compile_submission.py
    m = re.compile(r'^(---\s*\n.*?\n---\s*\n)(.*)', re.DOTALL).match(content)
    fm, body = (m.group(1), m.group(2)) if m else ('', content)
    return fm, re.sub(r'(?<![\\\$])\$(?!\$)', '$$', body)


def main():
    OUT.mkdir(exist_ok=True)
    for p in OUT.iterdir():  # keep the folder itself so a running preview server keeps working
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    (OUT / 'assets').mkdir()
    shutil.copytree(TMLR / 'assets', OUT / 'assets', dirs_exist_ok=True)
    shutil.copytree(SUB / 'assets', OUT / 'assets', dirs_exist_ok=True)
    fm, body = convert_math_delimiters((SUB / 'submission.md').read_text(encoding='utf-8'))
    body = re.sub(r"\{\{\s*'/?([^']+)'\s*\|\s*relative_url\s*\}\}", r'\1', body)
    assert '{{' not in body and '{%' not in body
    (OUT / 'body.md').write_text(body, encoding='utf-8')
    (OUT / 'kd.rb').write_text(RUBY, encoding='utf-8')
    res = subprocess.run(['wsl', '-d', 'Ubuntu-24.04', '--', 'ruby', wsl_path(OUT / 'kd.rb'),
                          wsl_path(OUT / 'body.md'), wsl_path(OUT / 'body.html')], capture_output=True, text=True)
    if res.returncode:
        raise SystemExit(res.stderr)
    for ln in res.stderr.splitlines():
        print(ln)
    html_body = (OUT / 'body.html').read_text(encoding='utf-8')
    title = re.search(r'^title: "(.*)"$', fm, re.M).group(1)
    desc = re.search(r'^description: "(.*)"$', fm, re.M).group(1)
    styles = fm.split('_styles: >', 1)[1].rsplit('---', 1)[0]
    toc = re.findall(r'^\s*- name: "(.*)"$', fm, re.M)
    toc_html = ''.join(f'<div><a href="#{re.sub("-+", "-", re.sub(r"[^0-9a-z]+", "-", t.lower())).strip("-")}">{t}</a></div>' for t in toc)
    head_html = (TMLR / '_includes/head.html').read_text(encoding='utf-8')
    inline_styles = '\n'.join(re.findall(r'<style[^>]*>.*?</style>', head_html, re.S))
    inline_styles = re.sub(r'\{%.*?%\}', '', re.sub(r'\{\{.*?\}\}', '', inline_styles, flags=re.S), flags=re.S)
    front = json.dumps({'title': title, 'description': desc, 'authors': [{'author': 'Anonymous', 'authorURL': '',
                        'affiliations': [{'name': 'Anonymous', 'url': ''}]}]})
    page = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.1/dist/css/bootstrap.min.css" rel="stylesheet">
<link rel="stylesheet" href="assets/css/main.css">
{inline_styles}
<script>window.MathJax = {{ tex: {{ tags: 'ams', inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']], processEscapes: true }} }};</script>
<script defer id="MathJax-script" src="https://cdn.jsdelivr.net/npm/mathjax@3.2.0/es5/tex-mml-chtml.js"></script>
<script src="assets/js/distillpub/template.v2.js"></script>
<script src="assets/js/distillpub/transforms.v2.js"></script>
<script src="assets/js/distillpub/overrides.js"></script>
<style>{styles}</style>
</head>
<d-front-matter><script async type="text/json">{front}</script></d-front-matter>
<body><div class="post distill">
<d-title><h1>{title}</h1><p>{desc}</p></d-title>
<d-byline></d-byline>
<d-article>
<d-contents><nav class="l-text figcaption"><h3>Contents</h3>{toc_html}</nav></d-contents>
{html_body}
</d-article>
<d-appendix><d-footnote-list></d-footnote-list><d-citation-list></d-citation-list></d-appendix>
</div>
<d-bibliography src="assets/bibliography/submission.bib"></d-bibliography>
<script>window.addEventListener('load', function () {{ setTimeout(function () {{
  // ?at=<id>: show only the element and what follows it (after MathJax has numbered everything)
  var id = new URLSearchParams(location.search).get('at');
  if (id === 'wide') {{  // list display equations wider than the article column
    var out = [];
    document.querySelectorAll('mjx-container[display="true"]').forEach(function (m) {{
      var w = Math.max(m.scrollWidth, m.firstElementChild ? m.firstElementChild.scrollWidth : 0);
      var cw = m.clientWidth;
      var scroll = m.style.overflowX === 'auto';
      if (w > cw + 1 && !scroll) {{ var lab = m.querySelector('[id^="mjx-eqn"]'); out.push((lab ? lab.id : '?') + ' ' + Math.round(w) + '/' + Math.round(cw)); }}
    }});
    var pre = document.createElement('pre'); pre.id = 'wide-report'; pre.textContent = 'WIDE:' + out.join(';'); document.body.appendChild(pre);
    return;
  }}
  if (id === 'refs') {{ document.querySelectorAll('d-title, d-byline, d-article').forEach(function (x) {{ x.style.display = 'none'; }}); return; }}
  var e = id && document.getElementById(id);
  if (!e) return;
  document.querySelectorAll('d-title, d-byline, d-contents').forEach(function (x) {{ x.style.display = 'none'; }});
  var art = document.querySelector('d-article'); var kids = Array.prototype.slice.call(art.children);
  var top = e; while (top.parentElement !== art) top = top.parentElement;
  var i = kids.indexOf(top);
  kids.forEach(function (k, j) {{ if (j < i || j > i + 8) k.style.display = 'none'; }}); }}, 7000); }});</script>
</body></html>'''
    # Embed the bibliography (Distill accepts inline BibTeX), so citations and the reference list also work when the
    # preview is opened from file://, where browsers block the fetch of the .bib file, and in the printed preview.pdf.
    bib = (OUT / 'assets/bibliography/submission.bib').read_text(encoding='utf-8')
    assert '</script' not in bib.lower()
    tag = '<d-bibliography src="assets/bibliography/submission.bib"></d-bibliography>'
    assert tag in page
    page = page.replace(tag, '<d-bibliography><script type="text/bibtex">\n' + bib + '\n</script></d-bibliography>')
    (OUT / 'index.html').write_text(page, encoding='utf-8')
    print('preview:', OUT / 'index.html')
    pdf = OUT / 'preview.pdf'
    subprocess.run([EDGE, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', '--virtual-time-budget=25000',
                    '--run-all-compositor-stages-before-draw', f'--print-to-pdf={pdf}',
                    (OUT / 'index.html').resolve().as_uri()], capture_output=True, timeout=180)
    print('pdf:', pdf, pdf.exists() and f'{pdf.stat().st_size / 1e3:.0f} kB')


def serve(port=8765):
    """Browsers block the bibliography fetch on file:// pages, so view the preview over HTTP."""
    import functools
    import http.server
    import webbrowser
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT))
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', port), handler)
    url = f'http://127.0.0.1:{port}/index.html'
    print('serving', url, '(Ctrl+C to stop)')
    webbrowser.open(url)
    httpd.serve_forever()


if __name__ == '__main__':
    import sys
    main()
    if '--serve' in sys.argv:
        serve()
