# Observation-noise comparison per dataset (test split, mean ± SD over 3 seeds)

Gaussian rows: all-filtering campaign of 2026-10-19 (results_arxiv/results_paper_filter_campaign_20261020; Gaussian observation noise, std 0.1 normalized), campaign recipe with per-seed validation selection. GMM rows: the final benchmark table (tables/table.md; mixture noise, weights 0.7/0.3, see datasets/*/cache.json). Lorenz--96 is observed through the mixture in both campaigns, so it has no Gaussian rows. Reference filters use the true dynamics and measurement model (EKF/UKF use the moment-matched Gaussian likelihood on GMM data; the bootstrap PF uses the exact likelihood; noise level validation-selected, 3 seeds); they exist only for the simulated systems. NormFlowSDE's GMM rows for pendulum (stage-1 rollout weight 30), Lorenz--96 (stage-1 lr 2e-3, stage-2 lr 3e-5, CRPS-selected) and REASSEMBLE (energy spread factor 0.52) use the coverage-round settings, whereas the Gaussian rows use the campaign recipe; the three baselines use the same recipe and grids in both campaigns. Bold: best RMSE / CRPS / width and coverage closest to 0.90 among the learned methods within a noise block.

## Pendulum

| Noise | Method | RMSE ↓ | CRPS ↓ | Cov. 90% (target 0.90) | Width |
|---|---|---|---|---|---|
| Gaussian | NormFlowSDE | 0.0717 ± 0.0034 | 0.0360 ± 0.0019 | 0.920 ± 0.007 | 0.2096 ± 0.0169 |
| Gaussian | NF-DPF | **0.0492 ± 0.0009** | 0.0249 ± 0.0002 | 0.849 ± 0.002 | 0.1286 ± 0.0009 |
| Gaussian | NSF | 0.0517 ± 0.0022 | **0.0249 ± 0.0010** | 0.874 ± 0.016 | **0.1261 ± 0.0008** |
| Gaussian | FlowSDE | 0.0929 ± 0.0085 | 0.0464 ± 0.0030 | **0.908 ± 0.003** | 0.2233 ± 0.0008 |
| Gaussian | EKF (reference) | 0.0421 ± 0.0000 | 0.0209 ± 0.0000 | 0.903 ± 0.001 | 0.1223 ± 0.0001 |
| Gaussian | UKF (reference) | 0.0421 ± 0.0000 | 0.0209 ± 0.0000 | 0.902 ± 0.001 | 0.1223 ± 0.0001 |
| Gaussian | Bootstrap PF (reference) | 0.0424 ± 0.0002 | 0.0213 ± 0.0002 | 0.897 ± 0.001 | 0.1224 ± 0.0001 |
| GMM | NormFlowSDE | 0.0691 ± 0.0008 | 0.0362 ± 0.0004 | 0.834 ± 0.017 | 0.1862 ± 0.0093 |
| GMM | NF-DPF | **0.0629 ± 0.0007** | **0.0320 ± 0.0003** | **0.918 ± 0.006** | 0.2017 ± 0.0017 |
| GMM | NSF | 0.0687 ± 0.0029 | 0.0346 ± 0.0014 | 0.866 ± 0.035 | **0.1805 ± 0.0110** |
| GMM | FlowSDE | 0.1144 ± 0.0069 | 0.0578 ± 0.0026 | 0.918 ± 0.003 | 0.3107 ± 0.0012 |
| GMM | EKF (reference) | 0.0578 ± 0.0000 | 0.0296 ± 0.0000 | 0.870 ± 0.001 | 0.1650 ± 0.0002 |
| GMM | UKF (reference) | 0.0578 ± 0.0000 | 0.0296 ± 0.0000 | 0.870 ± 0.001 | 0.1650 ± 0.0002 |
| GMM | Bootstrap PF (reference) | 0.0406 ± 0.0003 | 0.0192 ± 0.0001 | 0.895 ± 0.002 | 0.1129 ± 0.0000 |

## Lorenz--96

| Noise | Method | RMSE ↓ | CRPS ↓ | Cov. 90% (target 0.90) | Width |
|---|---|---|---|---|---|
| GMM | NormFlowSDE | 0.2942 ± 0.0082 | 0.1586 ± 0.0040 | 0.880 ± 0.016 | 0.8715 ± 0.0394 |
| GMM | NF-DPF | 0.2899 ± 0.0086 | 0.1585 ± 0.0051 | 0.854 ± 0.022 | 0.7811 ± 0.0645 |
| GMM | NSF | **0.2290 ± 0.0173** | **0.1199 ± 0.0103** | 0.822 ± 0.019 | **0.4692 ± 0.0053** |
| GMM | FlowSDE | 0.3227 ± 0.0067 | 0.1749 ± 0.0043 | **0.890 ± 0.009** | 0.9554 ± 0.0412 |
| GMM | EKF (reference) | 0.0229 ± 0.0000 | 0.0093 ± 0.0000 | 0.997 ± 0.000 | 0.0906 ± 0.0001 |
| GMM | UKF (reference) | 0.0230 ± 0.0000 | 0.0093 ± 0.0000 | 0.997 ± 0.000 | 0.0906 ± 0.0001 |
| GMM | Bootstrap PF (reference) | 0.0188 ± 0.0001 | 0.0077 ± 0.0000 | 0.990 ± 0.000 | 0.0674 ± 0.0000 |

## ALOHA

| Noise | Method | RMSE ↓ | CRPS ↓ | Cov. 90% (target 0.90) | Width |
|---|---|---|---|---|---|
| Gaussian | NormFlowSDE | **0.0544 ± 0.0038** | 0.0341 ± 0.0019 | 0.986 ± 0.002 | 0.3163 ± 0.0131 |
| Gaussian | NF-DPF | 0.2357 ± 0.0421 | 0.1374 ± 0.0248 | 0.679 ± 0.043 | 0.3993 ± 0.0125 |
| Gaussian | NSF | 0.1310 ± 0.0546 | 0.0819 ± 0.0380 | 0.585 ± 0.157 | 0.1829 ± 0.0112 |
| Gaussian | FlowSDE | 0.0554 ± 0.0003 | **0.0326 ± 0.0002** | **0.836 ± 0.003** | **0.1576 ± 0.0005** |
| GMM | NormFlowSDE | **0.0538 ± 0.0011** | **0.0301 ± 0.0007** | **0.871 ± 0.013** | **0.1645 ± 0.0017** |
| GMM | NF-DPF | 0.1307 ± 0.0288 | 0.0780 ± 0.0186 | 0.658 ± 0.065 | 0.2278 ± 0.0031 |
| GMM | NSF | 0.0858 ± 0.0006 | 0.0501 ± 0.0004 | 0.821 ± 0.005 | 0.2339 ± 0.0012 |
| GMM | FlowSDE | 0.0717 ± 0.0004 | 0.0413 ± 0.0002 | 0.858 ± 0.006 | 0.2147 ± 0.0047 |

## REASSEMBLE

| Noise | Method | RMSE ↓ | CRPS ↓ | Cov. 90% (target 0.90) | Width |
|---|---|---|---|---|---|
| Gaussian | NormFlowSDE | **0.0367 ± 0.0005** | 0.0248 ± 0.0004 | 0.993 ± 0.000 | 0.2532 ± 0.0050 |
| Gaussian | NF-DPF | 0.0437 ± 0.0024 | 0.0264 ± 0.0011 | 0.971 ± 0.011 | 0.2279 ± 0.0009 |
| Gaussian | NSF | 0.0445 ± 0.0001 | 0.0252 ± 0.0000 | **0.932 ± 0.000** | 0.1632 ± 0.0001 |
| Gaussian | FlowSDE | 0.0403 ± 0.0007 | **0.0228 ± 0.0004** | 0.944 ± 0.021 | **0.1543 ± 0.0149** |
| GMM | NormFlowSDE | **0.0326 ± 0.0016** | **0.0179 ± 0.0008** | **0.903 ± 0.005** | **0.1026 ± 0.0022** |
| GMM | NF-DPF | 0.0613 ± 0.0006 | 0.0340 ± 0.0003 | 0.858 ± 0.004 | 0.1760 ± 0.0021 |
| GMM | NSF | 0.0615 ± 0.0002 | 0.0345 ± 0.0001 | 0.916 ± 0.000 | 0.2140 ± 0.0003 |
| GMM | FlowSDE | 0.0530 ± 0.0012 | 0.0300 ± 0.0007 | 0.951 ± 0.001 | 0.2136 ± 0.0045 |

