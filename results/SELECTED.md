# results_normflowsde_paper — layout (2026-10-23, pass 3: method-first)

Only runs that enter the final benchmark table (tables/table.md) plus the analyses built on them. Everything superseded or not selected lives in results/results_arxiv/ (old campaign trees keep their relative paths there).

```
<method>/<dataset>/seed<k>/<stage>   method in {normflowsde, nf_dpf, nsf, flowsde}; dataset in {pendulum, lorenz96, aloha, reassemble}; k in 0..2
  normflowsde/.../s2   selected run (metrics.json, summary.json, model.eqx, posterior.npz, history.jsonl, run_metadata.json); s1 = its stage-1 initialization
  nf_dpf/.../s2        selected run; s1 = its stage-1 initialization
  nsf/.../jitter       selected run (jitter-tuned); train = the trained NSF it was tuned from
  flowsde/.../D<diff>  selected run (the validation-selected diffusion level, e.g. D0.01)
filters/<dataset>/{ekf,ukf,pf,pf_gauss} + summary.json   model-based reference filters (pendulum: GMM likelihood, 3 seeds; lorenz96: 2026-10-19 run on the byte-identical Lorenz cache)
datasets/<dataset>/{cache.json,train.npz,validation.npz,test.npz}   the data caches all runs above were trained and evaluated on (+ verification_*.json)
tables/            table.md, summary.json (every run path), build_table.py (rebuilds byte-identically; superseded round-4 rows and seed-0 alternatives are read from results_arxiv)
analyses/equal_n           equal sample / particle count sweep (rows.jsonl, 336 rows) on the runs above
analyses/pf_diagnostics    particle-filter diagnostics on the baseline runs
analyses/noise_recovery    measurement-density recovery figures for all four datasets (<dataset>_measurement_density.{png,pdf,json}; plot_measurement_density_recovery.py, seed 0; pendulum / Lorenz-96 residuals rescaled to physical noise units) and the Gaussian-vs-GMM noise comparison tables (build.py, table.md, noise_<dataset>.tex)
analyses/weak_convergence  weak-convergence check (seed0, ou_seed0)
analyses/latency_filters   latency of EKF/UKF/PF reference filters on an L40S with the inference_cost.py protocol (measure_reference_filters.py, job.sh -> latency.json), feeds the main table
analyses/calibration       reliability curves (coverage vs nominal level) from the saved posteriors, 3 seeds + reference filters: plot_calibration.py -> calibration.{png,pdf,json}
analyses/equal_n           also plot_sample_efficiency.py -> sample_efficiency.{png,pdf} (RMSE / coverage / latency vs M)
tables/build_paper_tables.py  writes paper/normflowsde/tables/main_results.tex and latency.tex from tables/summary.json, filters/*/summary.json, analyses/equal_n/rows.jsonl
provenance/        only what launched or selected the runs above: filter_campaign_20261020_gmm plan/state/sources (sources.json pendulum NormFlowSDE entry marked superseded), lane job scripts and per-task logs; Lorenz baseline plans, sources and HPC receipt; the nfsde_sweep2 scripts that launched the winners (sweep.sh, sweep_g/l/o/p/q.sh and their job wrappers) with the winners' logs; nfsde_stage1_optim job_reassemble.sh + pilot.sh + the b4lr1e-3 stage-1 log; reference-filter job scripts (model_based_baselines_20261019/job.sh also ran the superseded Gaussian-cache pendulum filters). Everything else moved to results_arxiv on 2026-10-23.
```

Provenance of the selected runs (old paths, now rewritten everywhere):
- normflowsde/pendulum  <- nfsde_sweep2_20261021/pendulum/rw30 (s1 + s2)
- normflowsde/lorenz96  <- nfsde_sweep2_20261021/lorenz96/lr2e-3_p0_s2lr3e-5_crps (s2) and lr2e-3_p0/s1
- normflowsde/aloha     <- filter_campaign_20261020_gmm runs/aloha/seed*/normflowsde/{s1_reference,s2_lr3e-4}
- normflowsde/reassemble <- nfsde_sweep2_20261021/reassemble/b4_a2_lr1e-4_sw0.52 (s2); s1 from nfsde_stage1_optim_20261021/reassemble/b4lr1e-3 (seed 0) and nfsde_sweep2_20261021/reassemble/b4lr1e-3_a2_lr1e-4 (seeds 1, 2)
- nf_dpf, nsf, flowsde  <- filter_campaign_20261020_gmm (pendulum, aloha, reassemble) and filter_campaign_20261020_lorenz_baselines_hpc (lorenz96; its parents were identified from the remote init_run/source_run paths in run_metadata.json: nf_dpf s1 attempt2, nsf train attempt0); the campaign's non-selected grid attempts are in results_arxiv/<campaign>/runs
- Full move list: results/REORG_20261023_pass3.txt. No latency measurement exists for these runs (results_arxiv/latency_20261020 used earlier models).
- Lorenz-96 baseline posterior.npz files (9, ~86 MB each) were copied from HPC on 2026-10-23 into nf_dpf/nsf/flowsde/lorenz96/seed*/ (they had not been synced with the campaign).
