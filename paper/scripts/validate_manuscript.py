#!/usr/bin/env python3
"""Check manuscript references, frozen evidence, and optional original hashes."""
import argparse
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]


def strip_comments(text):
    """Remove TeX comments while preserving escaped percent signs."""
    lines = []
    for line in text.splitlines(keepends=True):
        for match in re.finditer('%', line):
            position = match.start()
            preceding = position - 1
            while preceding >= 0 and line[preceding] == '\\':
                preceding -= 1
            if (position - preceding - 1) % 2 == 0:
                line = line[:position] + '\n'
                break
        lines.append(line)
    return ''.join(lines)


def read_tex_tree(path, ancestors=()):
    """Expand active input/include files using the manuscript build directory."""
    path = path.resolve()
    assert path not in ancestors, 'Cyclic TeX input: ' + ' -> '.join(
        str(item) for item in (*ancestors, path))
    assert path.is_file(), f'Missing TeX input: {path}'
    source = strip_comments(path.read_text())

    def expand(match):
        name = match.group(1).strip()
        child = PAPER / (name if name.endswith('.tex') else name + '.tex')
        return '\n' + read_tex_tree(child, (*ancestors, path)) + '\n'

    return re.sub(r'\\(?:input|include)\s*\{([^}]+)\}', expand, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sources', action='store_true')
    args = parser.parse_args()
    text = read_tex_tree(PAPER / 'main.tex')
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    assert not [x for x, n in collections.Counter(labels).items() if n > 1], 'Duplicate labels'
    refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', text)
    assert not set(refs) - set(labels), f'Unresolved labels: {set(refs)-set(labels)}'
    bib = set(re.findall(r'@\w+\{([^,]+),', (PAPER / 'references.bib').read_text()))
    cites = set()
    for group in re.findall(r'\\cite[tp]?\{([^}]+)\}', text):
        cites.update(group.split(','))
    assert not cites-bib, f'Unresolved citations: {cites-bib}'
    for path in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
        assert (PAPER / path).exists(), path
    selected = {'particle_filter_gmm', 'normalizing_flow_sde_realnvp',
                'conditional_stochastic_interpolant'}
    with (PAPER / 'evidence/state_estimation_lorenz96_comparison.csv').open() as source:
        # The archived CSV ends with an unlabelled timing annotation, not a run.
        records = [row for row in csv.DictReader(source)
                   if row['method'] in selected and row['noise_type'] == 'gmm3']
    assert len(records) == 3 and {row['method'] for row in records} == selected, \
        'Expected one Lorenz--96 GMM3 record for each selected method'
    for record in records:
        for metric in ['rmse_all', 'rmse_obs', 'rmse_unobs', 'mae_all',
                       'coverage_90', 'mciw_all', 'mciw_obs', 'mciw_unobs']:
            value = float(record[metric])
            assert math.isfinite(value) and value >= 0, (record['method'], metric)
        assert 0 <= float(record['coverage_90']) <= 1
        assert all(float(record[metric]) > 0 for metric in ['mciw_all', 'mciw_obs', 'mciw_unobs'])
        assert int(record['eval_trajectories']) > 0 and int(record['metric_particles']) > 0
    hashes = json.loads((PAPER / 'evidence/source_manifest.json').read_text())
    expected_sources = {'evaluation/state_estimation_lorenz96_comparison.csv',
                        'evaluation/reassemble_three_model_state_estimation.csv',
                        'evaluation/aloha_sim_insertion_all_methods.csv'}
    assert len(hashes) == 3 and {record['path'] for record in hashes} == expected_sources, \
        'Active source manifest must contain exactly the three manuscript CSV sources'
    for record in hashes:
        frozen = PAPER / 'evidence' / Path(record['path']).name
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == record['sha256'], str(frozen)
    if args.check_sources:
        for record in hashes:
            raw = (ROOT / record['path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == record['sha256'], record['path']
        missing = []
        for row in csv.DictReader((PAPER / 'evidence/aloha_sim_insertion_all_methods.csv').open()):
            path = ROOT / row['source_file']
            if not path.exists():
                missing.append(row['source_file'])
        print('ALOHA underlying per-run files absent (disclosed in manuscript):')
        for path in missing:
            print('  '+path)
        assert len(missing) == 2, 'Underlying evidence availability has changed; review manuscript.'
    print(f'PASS: {len(labels)} labels, {len(cites)} citations, 3 Lorenz--96 GMM3 records, '
          f'{len(hashes)} numerical-source records; figure/input paths resolve.')


if __name__ == '__main__':
    main()
