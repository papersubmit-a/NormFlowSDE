# results — layout (2026-11-18: paper results only)

Only what the manuscript is built from: the runs behind the two benchmark tables (paper/tables/benchmark_sim.tex,
benchmark_robot.tex), the analyses behind its figures and quoted numbers, and the scripts that launched them.
Everything else was removed from this local copy on 2026-11-18 (it remains on the server, /home/anon/NormFlowSDE/results).
The direct-baseline hyperparameter sweeps (latent_nsf_sweep, latent_sde_sweep, sigker_sde_sweep) were moved to
C:\Users\anon\Documents\NormFlowSDE_results_arxiv\results\ (outside the repo).

```
shared_trunk_20261114/<method>/<dataset>/seed<k>/<stage>   benchmark campaign, 8 methods x 4 datasets x 3 seeds
    BENCHMARK_SUMMARY.sources.json   the run directory of every table cell (read by every builder below)
    BENCHMARK_SUMMARY.md             per-seed overview (its latency column predates job 36123)
    normflowsde/.../{s1,s2}          s2 = selected run, s1 = its stage-1 initialization
    nf_dpf/.../{s1,s2}, nsf/.../{train,jitter}, flowsde/.../D<diff>, normflowsde_pf/.../learned_scale
    normflowsde_pf/pendulum/seed*/fixed_scale   ablation curve of the calibration figure
    latent_nsf, latent_sde, sigker_sde           pendulum only (retrained in this campaign)
    explore_lorenz96_s2, explore_reassemble_s2   referenced by the selected NormFlowSDE runs' metadata
latent_nsf/, latent_sde/, sigker_sde/<dataset>/seed<k>/run   direct baselines on lorenz96, aloha, reassemble
                                     (not retrained in the shared-trunk campaign; launched by provenance/direct_baselines_*)
filters/<dataset>/                   model-based reference filters (EKF, UKF, bootstrap PF) on pendulum and Lorenz-96
datasets/<dataset>/                  data caches of all runs (+ verification_*.json)
tables/build_shared_trunk_tables.py  -> paper/tables/benchmark_{sim,robot}.tex and tables/shared_trunk_summary.json
analyses/
    equal_n_shared_trunk_20261117/latency_l40s   latency of all 8 methods at n in {32..2000}, one L40S (job 36123);
                                                 the table latencies are its rows at each method's protocol count
    latency_gpu_20261116_shared_trunk            protocol counts (rows.jsonl, job 35895) and sources.json of the figure jobs
    latency_filters                              reference-filter latency (latency.json)
    calibration                                  plot_calibration_shared_trunk.py -> calibration_shared_trunk.* = paper/figures/calibration.pdf;
                                                 calibration.json supplies the reference-filter curves (plot_calibration.py)
    noise_recovery_shared_trunk                  REASSEMBLE measurement-density slices (input of the grid figure)
    noise_recovery_grid                          reassemble_shared_trunk_* = paper/figures/measurement_density_reassemble_grid.pdf
    weak_convergence                             ou_seed0 -> paper/figures/weak_convergence_ou_row_print.pdf
    paper_figures_20261116                       the job that produced the calibration and noise-recovery figures
provenance/
    shared_trunk_20261114                        launch scripts, logs, build_benchmark_summary.py of the campaign
    direct_baselines_20261109/10/12              launch scripts and logs of the direct baselines
    model_based_baselines_20261019, _20261022_gmm   jobs of the reference filters (Lorenz-96; pendulum GMM likelihood)
```

Open: the CPU accuracy sweep at equal n (job 36117, server only) ran out of memory at n = 2000 on ALOHA and REASSEMBLE.
