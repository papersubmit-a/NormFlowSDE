"""Small bundle of NormFlowSDE posteriors on the robot datasets for interactive/robot-view (runs on the server, SLURM).

Reads the frozen run directories without modifying them:
  results/shared_trunk_20261114/normflowsde/<ds>/seed0/s2/posterior.npz   samples_normalized [win, M, T, D]
  results/datasets/<ds>/test.npz, cache.json                              states, observations, normalization
Window rule (deterministic): rank the test windows by the RMSE of the posterior mean (normalized units, as reported in
the paper for the robot datasets) and take the windows at quantiles 0, 0.1, ..., 0.9 plus the worst window (11).
Per window: known initial state, reference, measurements, mean, median and 5/95 % quantiles from ALL draws, and the
first K draws (no reordering) as float16. Joint values are converted to physical units (rad; ALOHA gripper dims stay in
the dataset's normalized gripper units).

Usage: python scripts/export_robot_view.py --out results/robot_view [--draws 30]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def export(ds, out, k_draws):
    run = ROOT / f'results/shared_trunk_20261114/normflowsde/{ds}/seed0/s2'
    z = np.load(run / 'posterior.npz')
    S = z['samples_normalized']                                          # [win, M, T, D] float32
    te = np.load(ROOT / f'results/datasets/{ds}/test.npz')
    meta = json.loads((ROOT / f'results/datasets/{ds}/cache.json').read_text())['metadata']
    states, obs = te['states'], te['observations']                       # [win, T+1, D], [win, T, D]
    nwin, M, T, D = S.shape
    assert states.shape == (nwin, T + 1, D) and np.abs(states[:, 1:] - z['truth_normalized']).max() == 0
    mean = S.mean(axis=1)
    rmse = np.sqrt(((mean - states[:, 1:]) ** 2).mean(axis=(1, 2)))
    run_rmse = json.loads((run / 'metrics.json').read_text())['metrics']['rmse']
    assert abs(rmse.mean() - run_rmse) < 2e-3, (rmse.mean(), run_rmse)
    order = np.argsort(rmse, kind='stable')
    picks = [int(order[int(round(q * (nwin - 1)))]) for q in QUANTILES] + [int(order[-1])]
    labels = [f'q{int(q * 100):02d}' for q in QUANTILES] + ['worst']
    mu, sd = np.array(meta['state_mean']), np.array(meta['state_std'])
    mo, so = np.array(meta['observation_mean']), np.array(meta['observation_std'])
    phys = lambda a: a * sd + mu
    sel = S[picks]                                                       # [11, M, T, D]
    q = np.quantile(sel, [0.05, 0.5, 0.95], axis=1)                     # all draws
    arrays = dict(
        window=np.array(picks), rmse=rmse[picks], x0=phys(states[picks, 0]), truth=phys(states[picks, 1:]),
        obs=obs[picks] * so + mo, mean=phys(mean[picks]), q05=phys(q[0]), q50=phys(q[1]), q95=phys(q[2]),
        draws=phys(sel[:, :k_draws]).astype(np.float16),
    )
    for k in ('episode_index', 'start_frame'):
        if k in te.files:
            arrays[k] = te[k][picks]
    np.savez_compressed(out / f'{ds}.npz', **arrays)
    info = dict(dataset=ds, seed=0, window_rule='rank by RMSE of the posterior mean (normalized units); quantiles '
                '0..0.9 and the worst window', labels=labels, windows=picks, rmse=[float(r) for r in rmse[picks]],
                mean_rmse=float(rmse.mean()), run_rmse=float(run_rmse), n_windows=int(nwin), draws_total=int(M),
                draws_exported=int(k_draws), T=int(T), D=int(D), dt=float(json.loads(
                    (ROOT / f'results/datasets/{ds}/cache.json').read_text()).get('dt', 0)),
                state_units=meta.get('state_units'))
    (out / f'{ds}.json').write_text(json.dumps(info, indent=1))
    print(ds, 'windows', picks, 'mean RMSE', round(float(rmse.mean()), 4), 'run', round(run_rmse, 4))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/robot_view')
    ap.add_argument('--draws', type=int, default=30)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for ds in ('aloha', 'reassemble'):
        export(ds, out, a.draws)
