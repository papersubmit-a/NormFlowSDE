# Benchmark summary: shared-trunk campaign (8 methods x 4 datasets x 3 seeds)

Built 2026-11-16T20:32 by `results/provenance/shared_trunk_20261114/build_benchmark_summary.py` from the run summaries on disk.

- Test split; RMSE and CRPS lower is better; 90 % coverage target 0.90.
- Mean over seeds 0-2; the three per-seed RMSE values are listed.
- Latency: milliseconds per window, seed-0 checkpoint, one NVIDIA L40S, batch size one, median of 30 calls (jobs 35895 and, for NormFlowSDE after the step-loop change of 2026-11-16, 35904).
- Latent NSF, Latent SDE and SigKer SDE on Lorenz-96, ALOHA and REASSEMBLE are the previous models (`results/<method>/...`), not retrained in this campaign.
- NormFlowSDE latency on Lorenz-96 and REASSEMBLE was measured on the checkpoint that the re-tuned run replaced (same architecture); on one A30 the old and new checkpoints time the same within noise (job 35977).

## Pendulum

| Method | RMSE seeds 0 / 1 / 2 | RMSE mean | CRPS mean | Coverage mean | Latency (ms) | Remarks |
|---|---|---|---|---|---|---|
| NormFlowSDE | 0.0707 / 0.0738 / 0.0765 | 0.0736 | 0.0387 | 0.920 | 1.8 |  |
| Latent NSF | 0.1897 / 0.1437 / 0.1269 | 0.1534 | 0.0666 | 0.889 | 1.7 |  |
| Latent SDE | 0.1045 / 0.1055 / 0.1091 | 0.1063 | 0.0531 | 0.933 | 2.0 |  |
| SigKer SDE | 0.1645 / 0.1991 / 0.1809 | 0.1815 | 0.0993 | 0.174 | 2.1 |  |
| NF-DPF | 0.0834 / 0.0809 / 0.0733 | 0.0792 | 0.0392 | 0.913 | 8.5 |  |
| NSF-PF | 0.0648 / 0.0655 / 0.0648 | **0.0650** | **0.0333** | 0.858 | 8.5 |  |
| FlowSDE-PF | 0.1200 / 0.1170 / 0.1198 | 0.1190 | 0.0625 | **0.906** | 10.7 | D0.01 |
| NormFlowSDE-PF | 0.0966 / 0.0897 / 0.1013 | 0.0959 | 0.0451 | 0.875 | 8.2 |  |

## Lorenz-96

| Method | RMSE seeds 0 / 1 / 2 | RMSE mean | CRPS mean | Coverage mean | Latency (ms) | Remarks |
|---|---|---|---|---|---|---|
| NormFlowSDE | 0.2744 / 0.2705 / 0.2726 | 0.2725 | 0.1464 | 0.910 | 4.9 | stage-2 learning rate 3e-4 (was 3e-5), chosen on validation 2026-11-16 |
| Latent NSF | 0.8400 / 0.8150 / 0.8141 | 0.8230 | 0.4021 | 0.875 | 2.9 | previous model |
| Latent SDE | 0.3364 / 0.3243 / 0.3305 | 0.3304 | 0.1836 | 0.800 | 4.2 | previous model |
| SigKer SDE | 0.4544 / 0.4603 / 0.4462 | 0.4536 | 0.2645 | 0.542 | 4.6 | previous model |
| NF-DPF | 0.2706 / 0.2661 / 0.2804 | 0.2724 | 0.1476 | 0.883 | 29.4 |  |
| NSF-PF | 0.1513 / 0.1550 / 0.1603 | **0.1556** | **0.0783** | 0.921 | 20.7 |  |
| FlowSDE-PF | 0.2974 / 0.3029 / 0.3048 | 0.3017 | 0.1634 | **0.899** | 30.3 | D0.01 |
| NormFlowSDE-PF | 0.1915 / 0.1954 / 0.2053 | 0.1974 | 0.1011 | 0.876 | 20.3 | validated every 2,500 updates instead of every 500 |

## ALOHA

| Method | RMSE seeds 0 / 1 / 2 | RMSE mean | CRPS mean | Coverage mean | Latency (ms) | Remarks |
|---|---|---|---|---|---|---|
| NormFlowSDE | 0.0523 / 0.0523 / 0.0524 | **0.0523** | **0.0291** | **0.911** | 9.8 |  |
| Latent NSF | 0.0549 / 0.0580 / 0.0547 | 0.0559 | 0.0308 | 0.878 | 5.8 | previous model |
| Latent SDE | 0.0589 / 0.0601 / 0.0600 | 0.0597 | 0.0332 | 0.884 | 8.6 | previous model |
| SigKer SDE | 0.0663 / 0.0632 / 0.0647 | 0.0648 | 0.0426 | 0.250 | 9.2 | previous model |
| NF-DPF | 0.0951 / 0.1020 / 0.0937 | 0.0969 | 0.0565 | 0.751 | 50.0 |  |
| NSF-PF | 0.0834 / 0.0829 / 0.0822 | 0.0828 | 0.0484 | 0.830 | 38.3 |  |
| FlowSDE-PF | 0.0717 / 0.0700 / 0.0707 | 0.0708 | 0.0408 | 0.868 | 48.7 | D0.05; seed 0 D0.03 stopped at update 4335 of 9000; seed 1 D0.03 stopped at update 4455 of 9000; seed 2 D0.03 stopped at update 4362 of 9000 |
| NormFlowSDE-PF | 0.0780 / 0.0776 / 0.0778 | 0.0778 | 0.0453 | 0.854 | 37.6 |  |

## REASSEMBLE

| Method | RMSE seeds 0 / 1 / 2 | RMSE mean | CRPS mean | Coverage mean | Latency (ms) | Remarks |
|---|---|---|---|---|---|---|
| NormFlowSDE | 0.0314 / 0.0329 / 0.0316 | **0.0319** | **0.0176** | 0.912 | 10.1 | 8,000 stage-2 updates (was 4,000), chosen on validation 2026-11-16; gradient guards on (seed 1 discarded 163 updates) |
| Latent NSF | 0.0330 / 0.0340 / 0.0336 | 0.0335 | 0.0180 | **0.890** | 6.4 | previous model |
| Latent SDE | 0.0363 / 0.0362 / 0.0369 | 0.0365 | 0.0199 | 0.918 | 9.6 | previous model |
| SigKer SDE | 0.0453 / 0.0451 / 0.0579 | 0.0494 | 0.0307 | 0.382 | 10.0 | previous model |
| NF-DPF | 0.0592 / 0.0606 / 0.0597 | 0.0598 | 0.0331 | 0.873 | 48.9 |  |
| NSF-PF | 0.0614 / 0.0610 / 0.0611 | 0.0612 | 0.0343 | 0.917 | 43.1 |  |
| FlowSDE-PF | 0.0514 / 0.0541 / 0.0522 | 0.0526 | 0.0297 | 0.952 | 54.5 | D0.03 |
| NormFlowSDE-PF | 0.0623 / 0.0612 / 0.0620 | 0.0618 | 0.0348 | 0.919 | 41.5 | seed 0 stopped at update 3499 of 11000; seed 2 stopped at update 3499 of 11000 |
