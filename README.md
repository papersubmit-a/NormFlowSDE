# NormFlowSDE: Normalizing Flows for Amortized Bayesian State Estimation

Standalone archive of the manuscript, the code, and the frozen results behind every
number in the paper. It was extracted from the `bayesinf_jax` research repository on
2026-10-25 and contains only what the paper uses.

## Layout

```
paper/        manuscript (main.tex, sections/, figures/, tables/, references.bib, TMLR style)
experiments/  experiment scripts (experiments/normflowsde_comparison/ holds the entry points)
bayesinf/     library code the scripts import (models, flows, filters, state-estimation modules)
results/      frozen run directories, data caches, reference filters, analyses, provenance
requirements.txt
```

`results/` is organized method first:

```
results/<method>/<dataset>/seed<k>/<stage>   method in {normflowsde, nf_dpf, nsf, flowsde, normflowsde_pf}
                                             dataset in {pendulum, lorenz96, aloha, reassemble}, k in 0..2
                                             normflowsde_pf: NormFlowSDE's transition inside a bootstrap particle
                                             filter (stage learned_scale; fixed_scale = ablation), added 2026-10-31
results/filters/<dataset>/{ekf,ukf,pf,pf_gauss}   model-based reference filters (pendulum, lorenz96)
results/datasets/<dataset>/                  the data caches (cache.json, train/validation/test.npz)
results/tables/                              summary.json (every run path), table.md, build scripts
results/analyses/{calibration,equal_n,latency_filters,noise_recovery,pf_diagnostics,weak_convergence}
results/analyses/normflowsde_pf/            NormFlowSDE-PF experiment: build.py -> rows.json, table.md, paper/tables/normflowsde_pf.tex;
                                             latency_cpu/ (run_latency.sh -> rows.jsonl: same-device CPU latency of the seed-0 filters)
                                             calibration_normflowsde_pf.* (plot_calibration_pf.py), noise_recovery/ (five-panel measurement-density
                                             figures), noise_tables.py -> paper/tables/noise_normflowsde_pf.tex; originals of the paper figures unchanged
results/normflowsde_pf_gaussian/<dataset>/seed<k>/learned_scale   NormFlowSDE-PF under Gaussian observation noise (provenance/normflowsde_pf_gaussian_20261101)
results/provenance/                          launch scripts, plans and logs of the selected runs
results/SELECTED.md                          index of the selected runs and their origin
```

Each run directory stores its configuration (`run_metadata.json`), the selected
checkpoint (`model.eqx`), the test posterior (`posterior.npz`), the metrics and the
training history.

## Environment

Python 3.12 with the pinned packages in `requirements.txt` (JAX 0.10, Equinox 0.13,
Optax 0.2). All scripts are run as modules from the archive root:

```bash
cd NormFlowSDE
export PYTHONPATH=$PWD
python -m experiments.normflowsde_comparison.<script> --help
```

## Rebuilding the paper numbers and figures

The main table and the latency table:

```bash
python results/tables/build_paper_tables.py        # writes paper/tables/main_results.tex, latency.tex
```

The calibration figure, the noise-recovery figure, and the reference-filter latency:

```bash
python results/analyses/calibration/plot_calibration.py
python -m experiments.normflowsde_comparison.plot_measurement_density_recovery --help
python results/analyses/latency_filters/measure_reference_filters.py   # needs a GPU
```

The NormFlowSDE-PF table (Section "NormFlowSDE as the transition model of a particle filter"):

```bash
python results/analyses/normflowsde_pf/build.py    # writes paper/tables/normflowsde_pf.tex, updates results/tables/summary.json
python results/analyses/normflowsde_pf/plot_calibration_pf.py      # figures/calibration_normflowsde_pf.pdf
python -m experiments.normflowsde_comparison.plot_measurement_density_recovery --dataset reassemble \
  --sources results/analyses/normflowsde_pf/latency_cpu/sources.json --out results/analyses/normflowsde_pf/noise_recovery/reassemble \
  --methods normflowsde normflowsde_pf nf_dpf nsf flowsde              # figures/measurement_density_reassemble_normflowsde_pf.pdf
python results/analyses/normflowsde_pf/noise_tables.py                # paper/tables/noise_normflowsde_pf.tex (Gaussian vs GMM)
```

The manuscript compiles with `pdflatex` + `bibtex` or with `tectonic main.tex` from `paper/`.

The ALOHA and REASSEMBLE cache manifests record their raw recordings under the anonymized path
`/home/anon/bayesinf_jax`; to load them on another machine, point `NORMFLOWSDE_RAW_ROOT` at a directory that
holds `data/aloha_sim_insertion_human` and `data/reassemble` (the manifest hash is re-verified with the prefix
mapped back, and every array hash is always verified).

## Retraining

The entry points named in the reproducibility appendix of the paper, all under
`experiments/normflowsde_comparison/`:

| Purpose | Script |
|---|---|
| Build and verify the data caches, freeze and run the benchmark task graph | `filter_campaign.py` |
| NormFlowSDE stage 1 / stage 2 | `train_normflowsde_filter.py` / `improve_pendulum_normflowsde.py` |
| NF-DPF stage 1 / stage 2 | `run.py` / `improve_lorenz_baselines.py` |
| NSF-PF and FlowSDE-PF training / jitter selection | `train_state_filter.py` / `tune_state_filter_jitter.py` |
| EKF, UKF and particle-filter references | `model_based_baselines.py` |
| Latency of the learned filters | `inference_cost.py` |
| NormFlowSDE-PF (NormFlowSDE transition in a bootstrap particle filter) | `train_state_filter.py --method normflowsde --learned-scale`; launcher `results/provenance/normflowsde_pf_20261031/run_task.sh` |

The exact command lines, budgets and selection rules of the selected runs are in
`results/provenance/` and in each run's `run_metadata.json`.

## What is not included

- The `.npz` arrays: test and validation posteriors, the data caches in `results/datasets/`,
  and the reference-filter outputs in `results/filters/` (28 GB in total, 45 files above
  GitHub's 100 MB limit). They are excluded by `.gitignore` and distributed separately.
- The raw ALOHA and REASSEMBLE recordings. The windowed caches built from them are in
  `results/datasets/`, which is all that training and evaluation read.
- Superseded runs, non-selected sweep configurations and earlier campaigns. Two helper
  scripts still point at them for optional extras that the paper does not use:
  `results/tables/build_table.py` (superseded comparison rows) and
  `results/analyses/noise_recovery/build.py` (Gaussian-noise comparison rows).
- Provenance scripts under `results/provenance/` keep the absolute paths of the machines
  they ran on; they are records, not entry points.

## License

The code (`bayesinf/`, `experiments/`, and the scripts under `results/` and `paper/`)
is released under the MIT License; see `LICENSE`. The data caches and generated
artifacts (`results/datasets/`, run directories, posteriors, and checkpoints) are
**not** covered by that license and have not been approved for redistribution.
