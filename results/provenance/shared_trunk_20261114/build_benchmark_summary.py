"""Build results/shared_trunk_20261114/BENCHMARK_SUMMARY.md from the run summaries on disk.

  python results/provenance/shared_trunk_20261114/build_benchmark_summary.py     (from the repository root)

Every number is read from a summary.json / rows.jsonl; nothing is typed in. Sources per cell are listed in
BENCHMARK_SUMMARY.sources.json next to the table.
"""
import datetime
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
B = "results/shared_trunk_20261114"
DATASETS = dict(pendulum="Pendulum", lorenz96="Lorenz-96", aloha="ALOHA", reassemble="REASSEMBLE")
METHODS = dict(normflowsde="NormFlowSDE", latent_nsf="Latent NSF", latent_sde="Latent SDE", sigker_sde="SigKer SDE",
               nf_dpf="NF-DPF", nsf="NSF-PF", flowsde="FlowSDE-PF", normflowsde_pf="NormFlowSDE-PF")
DIRECT = ("latent_nsf", "latent_sde", "sigker_sde")
LATENCY = "results/analyses/latency_gpu_20261116_shared_trunk/rows.jsonl"
LATENCY_NORMFLOWSDE = "results/analyses/latency_gpu_20261116_shared_trunk_hoisted/rows.jsonl"
NOTES = {  # (method, dataset) -> remark on the recipe of the cell
    ("normflowsde", "lorenz96"): "stage-2 learning rate 3e-4 (was 3e-5), chosen on validation 2026-11-16",
    ("normflowsde", "reassemble"): "8,000 stage-2 updates (was 4,000), chosen on validation 2026-11-16; "
                                   "gradient guards on (seed 1 discarded 163 updates)",
    ("normflowsde_pf", "lorenz96"): "validated every 2,500 updates instead of every 500",
}


def run_dir(method, dataset, seed):
    if method in DIRECT and dataset != "pendulum":
        return f"results/{method}/{dataset}/seed{seed}/run", "previous model"
    base = f"{B}/{method}/{dataset}/seed{seed}"
    if method in ("normflowsde", "nf_dpf"):
        return f"{base}/s2", ""
    if method == "nsf":
        return f"{base}/jitter", ""
    if method == "flowsde":
        chosen = json.loads((ROOT / base / "selected.json").read_text())["selected"]
        return f"{base}/{chosen}", chosen
    if method == "normflowsde_pf":
        return f"{base}/learned_scale", ""
    return f"{base}/run", ""


def cell(method, dataset):
    rows, remarks, sources = [], [], []
    for seed in range(3):
        path, remark = run_dir(method, dataset, seed)
        summary = json.loads((ROOT / path / "summary.json").read_text())
        test = summary["test"]
        rows.append((test["rmse"], test["crps"], test["coverage_90"]))
        sources.append(path)
        if remark:
            remarks.append(remark)
        stopped = summary.get("stopped_early")
        if stopped:
            remarks.append(f"seed {seed} stopped at update {stopped['completed_updates']} of {stopped['planned_updates']}")
        # FlowSDE-PF: a diffusion level that was stopped early took part in the selection
        if method == "flowsde":
            for level in ("D0.01", "D0.03", "D0.05"):
                other = json.loads((ROOT / path).parent.joinpath(level, "summary.json").read_text()).get("stopped_early")
                if other:
                    remarks.append(f"seed {seed} {level} stopped at update {other['completed_updates']} of {other['planned_updates']}")
    return np.array(rows), sorted(set(remarks)), sources


def latency():
    table = {}
    for source in (LATENCY, LATENCY_NORMFLOWSDE):
        for line in (ROOT / source).read_text().splitlines():
            row = json.loads(line)
            if source == LATENCY_NORMFLOWSDE and row["method"] != "normflowsde":
                continue
            table[row["method"], row["dataset"]] = row["latency"]["ms_per_window_median"]
    return table


def main():
    ms = latency()
    lines = [
        "# Benchmark summary: shared-trunk campaign (8 methods x 4 datasets x 3 seeds)",
        "",
        f"Built {datetime.datetime.now().isoformat(timespec='minutes')} by "
        "`results/provenance/shared_trunk_20261114/build_benchmark_summary.py` from the run summaries on disk.",
        "",
        "- Test split; RMSE and CRPS lower is better; 90 % coverage target 0.90.",
        "- Mean over seeds 0-2; the three per-seed RMSE values are listed.",
        "- Latency: milliseconds per window, seed-0 checkpoint, one NVIDIA L40S, batch size one, median of 30 calls "
        "(jobs 35895 and, for NormFlowSDE after the step-loop change of 2026-11-16, 35904).",
        "- Latent NSF, Latent SDE and SigKer SDE on Lorenz-96, ALOHA and REASSEMBLE are the previous models "
        "(`results/<method>/...`), not retrained in this campaign.",
        "- NormFlowSDE latency on Lorenz-96 and REASSEMBLE was measured on the checkpoint that the re-tuned run replaced "
        "(same architecture); on one A30 the old and new checkpoints time the same within noise (job 35977).",
        "",
    ]
    sources, best = {}, {}
    for dataset, title in DATASETS.items():
        cells = {method: cell(method, dataset) for method in METHODS}
        means = {method: values.mean(0) for method, (values, _, _) in cells.items()}
        low_rmse = min(means, key=lambda m: means[m][0])
        low_crps = min(means, key=lambda m: means[m][1])
        near = min(means, key=lambda m: abs(means[m][2] - 0.9))
        best[dataset] = dict(rmse=low_rmse, crps=low_crps, coverage=near)
        bold = lambda text, on: f"**{text}**" if on else text
        lines += [f"## {title}", "", "| Method | RMSE seeds 0 / 1 / 2 | RMSE mean | CRPS mean | Coverage mean | Latency (ms) | Remarks |",
                  "|---|---|---|---|---|---|---|"]
        for method, name in METHODS.items():
            values, remarks, paths = cells[method]
            mean = means[method]
            remarks = remarks + ([NOTES[method, dataset]] if (method, dataset) in NOTES else [])
            lines.append(
                f"| {name} | {' / '.join(f'{v:.4f}' for v in values[:, 0])} | {bold(f'{mean[0]:.4f}', method == low_rmse)} | "
                f"{bold(f'{mean[1]:.4f}', method == low_crps)} | {bold(f'{mean[2]:.3f}', method == near)} | "
                f"{ms[method, dataset]:.1f} | {'; '.join(remarks)} |")
            sources[f"{method}/{dataset}"] = dict(
                runs=paths, test_rmse=[float(v) for v in values[:, 0]], test_crps=[float(v) for v in values[:, 1]],
                test_coverage_90=[float(v) for v in values[:, 2]], latency_ms=ms[method, dataset])
        lines.append("")
    out = ROOT / B
    (out / "BENCHMARK_SUMMARY.md").write_text("\n".join(lines))
    (out / "BENCHMARK_SUMMARY.sources.json").write_text(json.dumps(dict(cells=sources, best=best), indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
