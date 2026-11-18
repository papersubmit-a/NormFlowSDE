"""All-filtering four-method benchmark campaign, run from scratch as one frozen task graph.

Every dataset/method/seed (4 x 4 x 3 = 48 entries) is a pipeline of from-scratch training
stages with current-measurement filtering access, a matched total optimizer-update budget per
dataset (every method gets the largest previous recipe total), and validation-only
hyperparameter selection inside the campaign, per training seed:

  NormFlowSDE  native amortized filter: filtering training (dynamics in {reference,
               learned-scale + time-invariant}) -> full-horizon energy fine-tune (lr grid)
  NF-DPF       comparison-runner training -> long-window variant-B fine-tune
  NSF          learned-dynamics bootstrap PF -> process-noise jitter selection
  FlowSDE      learned-dynamics bootstrap PF, fixed diffusion D grid

Data caches are rebuilt on the cluster and must match the reference caches array by array.
Tasks share one state file, so a few GPU workers ("lanes") pack several tasks per GPU and a
chained follow-up job resumes interrupted tasks in fresh attempt directories; nothing is
deleted or overwritten.

  prepare-data  build the four caches (CPU) and verify them against expected_caches.json
  plan          freeze the task graph for a campaign root
  lane          GPU worker: claim ready tasks, run them concurrently, stop before the deadline
  status        print progress
  collect       write a build_results_paper.py source map of the selected final runs
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
REFERENCES = Path(__file__).resolve().parent / "filter_campaign_references"
DATASETS = ("pendulum", "lorenz96", "aloha", "reassemble")
METHODS = ("normflowsde", "nf_dpf", "nsf", "flowsde")
SEEDS = (0, 1, 2)
MODULE = "experiments.normflowsde_comparison."

# Total optimizer updates per method and dataset: the largest previous recipe total
# (pendulum NormFlowSDE 4400+4000, Lorenz NF-DPF 12500+6000, ALOHA NF-DPF 3000+6000,
# REASSEMBLE NF-DPF 5000+6000), so no method trains less than its tuned recipe.
BUDGET = dict(pendulum=8400, lorenz96=18500, aloha=9000, reassemble=11000)
NFSDE_FINETUNE_STEPS = 4000
NFDPF_LONGWINDOW_STEPS = dict(pendulum=4000, lorenz96=6000, aloha=6000, reassemble=6000)
# All validation windows of each split (checkpoint and hyperparameter selection).
VALIDATION_WINDOWS = dict(pendulum=48, lorenz96=200, aloha=50, reassemble=300)
NFSDE_DYNAMICS = {"reference": [], "lsnt": ["--learned-scale", "--no-time"]}
NFSDE_LEARNING_RATES = ("3e-5", "1e-4")
# GMM observation-noise campaign (opt-in, 2026-10-20): stage 2 with the learned noise scale, lr grid below.
NFSDE_LEARNING_RATES_GMM = ("1e-4", "3e-4")
FLOWSDE_DIFFUSIONS = ("0.01", "0.03", "0.05")
JITTER_GRID = "0,0.005,0.01,0.02,0.03,0.05,0.075,0.1"
# GMM campaign: joint (checkpoint, jitter) selection for the learned-dynamics PFs, grid extended above 0.05
# (results/results_arxiv/aloha_baselines_improve_20261020: selecting the checkpoint at jitter 0 picked early, poor checkpoints).
JITTER_GRID_GMM = "0,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2"
# GMM campaign: NF-DPF fine-tune keeps a supervised measurement-NLL anchor so the measurement flow stays a likelihood.
NFDPF_GMM_FLAGS = ("--measurement-nll-weight", "1", "--measurement-batch-windows", "16")
# Learned-dynamics PF capacity (comparison widths), as in results/results_normflowsde_paper/state_filter_*_20261019.
PF_CAPACITY = dict(
    pendulum=dict(nsf=70, flowsde=201, depth=2, layers=6, eval_every=400),
    lorenz96=dict(nsf=126, flowsde=242, depth=3, layers=2, eval_every=500),
    aloha=dict(nsf=68, flowsde=139, depth=3, layers=2, eval_every=250),
    reassemble=dict(nsf=71, flowsde=136, depth=3, layers=2, eval_every=500),
)
# NF-DPF comparison-runner architecture (legacy-aligned-v1 capacity profile), as in the comparison matrix.
NFDPF_CAPACITY = dict(pendulum=(61, 2, 6, 16), lorenz96=(129, 3, 2, 128), aloha=(65, 3, 2, 64),
                      reassemble=(69, 3, 2, 64))
# Conservative single-task wall-time estimates on a shared A100 (seconds): used only to avoid
# starting a task that cannot finish before the job deadline, and to order the critical path.
ESTIMATE = dict(
    nfsde_s1=dict(pendulum=4000, lorenz96=2700, aloha=1500, reassemble=2500),
    nfsde_s2=dict(pendulum=2500, lorenz96=3500, aloha=3000, reassemble=4500),
    nfdpf_s1=dict(pendulum=700, lorenz96=1000, aloha=1000, reassemble=700),
    nfdpf_s2=dict(pendulum=4500, lorenz96=7500, aloha=9000, reassemble=9000),
    nsf=dict(pendulum=1000, lorenz96=1000, aloha=2000, reassemble=2700),
    jitter=dict(pendulum=600, lorenz96=900, aloha=900, reassemble=1200),
    flowsde=dict(pendulum=1300, lorenz96=1600, aloha=4500, reassemble=4200),
)
# Relative GPU-memory weight per task kind; a lane's running weight is capped (--capacity).
WEIGHT = dict(nfdpf_s2=3, flowsde=2)
MAX_ATTEMPTS = 2
ACCESS_LABEL = "current-measurement filtering"


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

def build_cache(dataset, destination, observation_noise="gaussian"):
    from . import cache as C
    from . import run
    if dataset == "aloha":
        # As results/results_arxiv/normflowsde_filter_robot_20261018/build_ppu_cache.py (pseudo-process-uncertainty variant).
        from experiments.nf_dpf_state_estimation.datasets import _json_value, load_aloha
        import numpy as np
        data = load_aloha("pseudo_process_uncertainty" + ("_gmm" if observation_noise == "gmm" else ""))
        destination.mkdir(parents=True)
        manifest = dict(schema_version=C.SCHEMA_VERSION, dt=float(data.dt), protocol=_json_value(data.protocol),
                        metadata=_json_value(data.metadata), raw_inputs=C._raw_inputs(data), splits={})
        for name in C.SPLITS:
            arrays = {k: np.asarray(v) for k, v in getattr(data, name).items()}
            np.savez(destination / f"{name}.npz", **arrays)
            manifest["splits"][name] = dict(file=f"{name}.npz", sha256=C.file_sha256(destination / f"{name}.npz"),
                                            arrays={k: C.array_description(v) for k, v in arrays.items()})
        manifest["payload_sha256"] = C._payload_sha256(manifest)
        (destination / "cache.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    else:
        # As prepare.py: the comparison runner's default protocol for the dataset.
        extra = ["--observation-noise", "gmm"] if observation_noise == "gmm" and dataset != "lorenz96" else []
        template = run.parse_args(["--dataset", dataset, "--method", "normflowsde", "--output", str(destination), *extra])
        C.save_cache(run.load_data(template), destination)
    C.load_cache(destination)


def verify_cache(dataset, path):
    """Same data as the previous comparison: protocol, normalization and every array hash."""
    expected = json.loads((REFERENCES / "expected_caches.json").read_text())[dataset]
    manifest = json.loads((Path(path) / "cache.json").read_text())
    differences = [name for name in ("protocol", "metadata", "dt") if manifest[name] != expected[name]]
    differences += [f"splits/{split}" for split in expected["splits"]
                    if manifest["splits"][split]["arrays"] != expected["splits"][split]]
    if differences:
        raise ValueError(f"{dataset}: rebuilt cache differs from the reference cache: {differences}")


def prepare_data(args):
    caches = args.campaign / "data_cache"
    caches.mkdir(parents=True, exist_ok=True)
    report = {}
    for dataset in args.datasets:
        destination = caches / dataset
        if not (destination / "cache.json").is_file():
            if destination.exists():
                raise FileExistsError(f"Incomplete cache directory; inspect it: {destination}")
            build_cache(dataset, destination, args.observation_noise)
        if args.observation_noise == "gmm" and dataset != "lorenz96":
            noise = json.loads((destination / "cache.json").read_text())["protocol"].get("obs_noise_type")
            if noise != "gmm":
                raise ValueError(f"{dataset}: cache observation noise is {noise}, expected gmm")
            verified = "not applicable: new GMM observation-noise data (no reference cache)"
        else:
            verify_cache(dataset, destination)
            verified = "filter_campaign_references/expected_caches.json"
        report[dataset] = dict(path=str(destination), cache_json_sha256=sha256(destination / "cache.json"),
                               observation_noise=args.observation_noise, verified_against=verified)
        print(json.dumps({dataset: "verified"}), flush=True)
    atomic_json(caches / "verification.json", dict(verified_at=utc(), caches=report))


# ---------------------------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------------------------

def task(tasks, identifier, kind, dataset, argv=None, deps=(), select=None, final=None, note=""):
    if identifier in tasks:
        raise ValueError(f"Duplicate task {identifier}")
    tasks[identifier] = dict(id=identifier, kind=kind, dataset=dataset, argv=argv, deps=list(deps),
                             select=select, final=final, note=note,
                             estimate=ESTIMATE.get(kind, {}).get(dataset, 60) if argv else 5,
                             weight=WEIGHT.get(kind, 1) if argv else 0)


def plan_tasks(python, caches, smoke=False, observation_noise="gaussian", datasets=DATASETS):
    """The frozen graph. Arguments in braces are resolved at launch: {out} is the task's fresh
    attempt directory and {dir:<id>} the completed (or, for a selection, selected) run directory."""
    tasks = {}
    py = [python, "-m"]
    gmm = observation_noise == "gmm"
    for dataset in datasets:
        cache, val, budget = str(caches / dataset), str(VALIDATION_WINDOWS[dataset]), BUDGET[dataset]
        capacity = PF_CAPACITY[dataset]
        for seed in SEEDS:
            s, tag = str(seed), f"{dataset}/seed{seed}"
            # NormFlowSDE: filtering training -> energy fine-tune; dynamics then lr selected on validation.
            s1 = []
            # GMM mode: the learned noise scale is introduced only in the energy-score stage. With the entropy
            # term alone it collapses the increments (2026-10-20 GMM pendulum: s1_lsnt width 1e-4, coverage 0).
            for name, flags in ({"reference": []} if gmm else NFSDE_DYNAMICS).items():
                argv = py + [MODULE + "train_normflowsde_filter", "--dataset", dataset, "--seed", s, "--train",
                             "--data-cache", cache, "--reference-metadata", str(REFERENCES / dataset / f"seed{seed}.json"),
                             "--train-steps", str(budget - NFSDE_FINETUNE_STEPS), "--validation-windows", val,
                             "--allow-source-drift", *(["--allow-data-change"] if gmm else []), *flags,
                             "--output", "{out}"]
                if smoke:
                    argv[argv.index("--train-steps") + 1] = "3"
                    argv += ["--allow-cpu", "--smoke-updates", "3"]
                task(tasks, f"{tag}/normflowsde/s1_{name}", "nfsde_s1", dataset, argv,
                     note=f"filtering training, dynamics={name}")
                s1.append(f"{tag}/normflowsde/s1_{name}")
            task(tasks, f"{tag}/normflowsde/select_dynamics", "select", dataset, deps=s1, select="nfsde_s1")
            s2 = []
            for lr in (NFSDE_LEARNING_RATES_GMM if gmm else NFSDE_LEARNING_RATES):
                argv = py + [MODULE + "improve_pendulum_normflowsde", "--dataset", dataset, "--seed", s,
                             "--data-cache", cache, "--init-run", f"{{dir:{tag}/normflowsde/select_dynamics}}",
                             "--objective", "energy", "--learning-rate", lr, *(["--learned-scale"] if gmm else []),
                             "--steps", str(NFSDE_FINETUNE_STEPS), "--output", "{out}"]
                if smoke:
                    argv[argv.index("--steps") + 1] = "2"
                    argv += ["--validation-every", "1", "--allow-cpu", "--smoke"]
                task(tasks, f"{tag}/normflowsde/s2_lr{lr}", "nfsde_s2", dataset, argv,
                     deps=[f"{tag}/normflowsde/select_dynamics"], note=f"energy fine-tune, lr={lr}")
                s2.append(f"{tag}/normflowsde/s2_lr{lr}")
            task(tasks, f"{tag}/normflowsde/select_lr", "select", dataset, deps=s2, select="nfsde_s2",
                 final="normflowsde")

            # NF-DPF: comparison-runner training -> long-window variant B (validation-selected checkpoints).
            n2 = NFDPF_LONGWINDOW_STEPS[dataset]
            argv = py + [MODULE + "run", "--dataset", dataset, "--method", "nf_dpf",
                         "--capacity-profile", "legacy-aligned-v1", "--settings-profile", "legacy-compatible-v1",
                         *[str(v) for pair in zip(("--hidden-size", "--hidden-depth", "--flow-layers", "--context-dim"),
                                                  NFDPF_CAPACITY[dataset]) for v in pair],
                         "--seed", s, "--data-cache", cache, "--train-steps", str(budget - n2),
                         "--validation-windows", val, "--output", "{out}", "--train"]
            if dataset == "aloha":
                argv[argv.index("--seed"):argv.index("--seed")] = [
                    "--variant", "pseudo_process_uncertainty" + ("_gmm" if gmm else "")]
            elif gmm and dataset != "lorenz96":
                argv[argv.index("--seed"):argv.index("--seed")] = ["--observation-noise", "gmm"]
            if smoke:
                argv[argv.index("--train-steps") + 1] = "3"
                argv += ["--validation-every", "1", "--validation-windows", "2", "--eval-windows", "2",
                         "--posterior-samples", "4", "--num-particles", "8"]
            task(tasks, f"{tag}/nf_dpf/s1", "nfdpf_s1", dataset, argv, note="comparison-runner training")
            argv = py + [MODULE + "improve_lorenz_baselines", "--dataset", dataset, "--method", "nf_dpf",
                         "--seed", s, "--anchor-weight", "0", "--crps-weight", "1", "--mean-mse-weight", "0",
                         "--ess-weight", "0.01", "--remat", "--skip-nonfinite", "--validation-windows", val,
                         "--steps", str(n2), "--data-cache", cache, "--init-run", f"{{dir:{tag}/nf_dpf/s1}}",
                         *(NFDPF_GMM_FLAGS if gmm else ()), "--output", "{out}"]
            if dataset == "pendulum":
                # Three-stage curriculum (default horizons 8,25,60); the default batch list has four stages.
                argv[argv.index("--output"):argv.index("--output")] = ["--batches", "32,16,8"]
            if smoke:
                argv[argv.index("--steps") + 1] = "2"
                argv += ["--smoke"]
            task(tasks, f"{tag}/nf_dpf/s2", "nfdpf_s2", dataset, argv, deps=[f"{tag}/nf_dpf/s1"],
                 final="nf_dpf", note="long-window variant B fine-tune")

            # NSF-PF and FlowSDE-PF on the learned-dynamics bootstrap particle filter.
            common = ["--dataset", dataset, "--data-cache", cache, "--hidden-depth", str(capacity["depth"]),
                      "--flow-layers", str(capacity["layers"]), "--eval-every", str(capacity["eval_every"]),
                      "--steps", str(budget), "--seed", s,
                      *(["--selection-jitters", JITTER_GRID_GMM if not smoke else "0,0.05"] if gmm else [])]
            if smoke:
                common[common.index("--steps") + 1] = "2"
                common[common.index("--eval-every") + 1] = "1"
                common += ["--smoke"]
            task(tasks, f"{tag}/nsf/train", "nsf", dataset,
                 py + [MODULE + "train_state_filter", "--method", "nsf", "--hidden-size", str(capacity["nsf"]),
                       *common, "--output", "{out}"], note="NSF-PF training")
            task(tasks, f"{tag}/nsf/jitter", "jitter", dataset,
                 py + [MODULE + "tune_state_filter_jitter", "--run", f"{{dir:{tag}/nsf/train}}",
                       "--grid", (JITTER_GRID_GMM if gmm else JITTER_GRID) if not smoke else "0,0.01", "--output", "{out}"],
                 deps=[f"{tag}/nsf/train"], final="nsf", note="process-noise jitter selection")
            flows = []
            for diffusion in FLOWSDE_DIFFUSIONS:
                task(tasks, f"{tag}/flowsde/D{diffusion}", "flowsde", dataset,
                     py + [MODULE + "train_state_filter", "--method", "flowsde",
                           "--hidden-size", str(capacity["flowsde"]), *common, "--diffusion", diffusion,
                           "--output", "{out}"], note=f"FlowSDE-PF, D={diffusion}")
                flows.append(f"{tag}/flowsde/D{diffusion}")
            task(tasks, f"{tag}/flowsde/select_D", "select", dataset, deps=flows, select="flowsde", final="flowsde")
    return tasks


def critical_path(tasks):
    children = {name: [] for name in tasks}
    for name, item in tasks.items():
        for dep in item["deps"]:
            children[dep].append(name)
    memo = {}

    def length(name):
        if name not in memo:
            memo[name] = tasks[name]["estimate"] + max((length(c) for c in children[name]), default=0)
        return memo[name]
    return {name: length(name) for name in tasks}


def plan(args):
    directory = args.campaign / "plan"
    if (directory / "plan.json").exists():
        raise FileExistsError(f"Plan already frozen: {directory / 'plan.json'}")
    verification = args.campaign / "data_cache" / "verification.json"
    if not args.smoke and not verification.is_file():
        raise FileNotFoundError("Run prepare-data first; caches must be verified before planning")
    caches = args.campaign / "data_cache"
    tasks = plan_tasks(str(args.python), caches, smoke=args.smoke, observation_noise=args.observation_noise,
                       datasets=tuple(args.datasets))
    # Opt-in method subset (2026-10-20): selections depend only on their own method's runs.
    tasks = {name: item for name, item in tasks.items() if name.split("/")[2] in args.methods}
    priority = critical_path(tasks)
    for name, item in tasks.items():
        item["priority"] = priority[name]
    directory.mkdir(parents=True, exist_ok=True)
    sources = sorted({str(p.relative_to(ROOT)) for p in (ROOT / "experiments/normflowsde_comparison").glob("*.py")}
                     | {"bayesinf/applications/state_estimation/flow_sde.py"})
    frozen = dict(schema="filter-campaign-v1", created_at=utc(), campaign=str(args.campaign), root=str(ROOT),
                  python=str(args.python), smoke=args.smoke, budget=BUDGET,
                  observation_noise=args.observation_noise, datasets=list(args.datasets), methods=list(args.methods),
                  pf_selection=("joint (checkpoint, jitter) validation selection over " + JITTER_GRID_GMM
                                + "; NF-DPF fine-tune with measurement-NLL anchor " + " ".join(NFDPF_GMM_FLAGS)
                                if args.observation_noise == "gmm" else "checkpoint at jitter 0, then jitter grid"),
                  nfsde_stage2=("learned noise scale, lr grid " + ",".join(NFSDE_LEARNING_RATES_GMM)
                                if args.observation_noise == "gmm" else "bounded flow, lr grid " + ",".join(NFSDE_LEARNING_RATES)),
                  nfsde_finetune_steps=NFSDE_FINETUNE_STEPS, nfdpf_longwindow_steps=NFDPF_LONGWINDOW_STEPS,
                  validation_windows=VALIDATION_WINDOWS, grids=dict(nfsde_dynamics=NFSDE_DYNAMICS,
                  nfsde_learning_rates=NFSDE_LEARNING_RATES, flowsde_diffusions=FLOWSDE_DIFFUSIONS,
                  nsf_jitter=JITTER_GRID), pf_capacity=PF_CAPACITY, selection="per training seed, lowest validation RMSE",
                  source_sha256={name: sha256(ROOT / name) for name in sources},
                  reference_sha256={str(p.relative_to(REFERENCES)): sha256(p) for p in sorted(REFERENCES.rglob("*.json"))},
                  tasks=tasks)
    atomic_json(directory / "plan.json", frozen)
    atomic_json(directory / "state.json", dict(tasks={name: dict(status="pending", attempts=[]) for name in tasks}))
    (directory / "state.lock").touch()
    print(json.dumps(dict(status="planned", tasks=len(tasks),
                          training_tasks=sum(1 for t in tasks.values() if t["argv"]),
                          longest_chain_hours=round(max(priority.values()) / 3600, 2))), flush=True)


# ---------------------------------------------------------------------------------------------
# Lane (GPU worker)
# ---------------------------------------------------------------------------------------------

@contextmanager
def locked_state(directory):
    with (directory / "state.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = json.loads((directory / "state.json").read_text())
            yield state
            atomic_json(directory / "state.json", state)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def job_alive(job, own):
    if not job or job == own:
        return bool(job)
    if job.startswith("local-"):
        return Path(f"/proc/{job[6:]}").exists()
    try:
        output = subprocess.run(["squeue", "-h", "-j", str(job), "-o", "%T"], capture_output=True,
                                text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return True  # Unknown: never steal a task that may still be running.
    return output in {"RUNNING", "COMPLETING", "PENDING", "CONFIGURING"}


def job_deadline(fallback_seconds):
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        try:
            end = subprocess.run(["squeue", "-h", "-j", job, "-o", "%e"], capture_output=True, text=True,
                                 timeout=30).stdout.strip()
            return datetime.fromisoformat(end).timestamp()
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return time.time() + fallback_seconds


def complete(item, directory):
    """A run directory is complete when its stage wrote its final record."""
    directory = Path(directory)
    if item["kind"] in {"nfsde_s1", "nfdpf_s1"}:
        path = directory / "run_metadata.json"
        return path.is_file() and json.loads(path.read_text()).get("status") == "complete"
    return (directory / "summary.json").is_file() and (directory / "posterior.npz").is_file()


def validation_rmse(kind, directory):
    directory = Path(directory)
    if kind == "nfsde_s1":
        return json.loads((directory / "run_metadata.json").read_text())["best_validation_rmse"]
    summary = json.loads((directory / "summary.json").read_text())
    if kind == "nfsde_s2":
        return summary["best_validation"]["rmse"]
    return summary["best_validation_rmse"]


def resolve(argv, out, done_dirs):
    resolved = []
    for value in argv:
        if value == "{out}":
            value = str(out)
        elif value.startswith("{dir:") and value.endswith("}"):
            value = done_dirs[value[5:-1]]
        resolved.append(value)
    return resolved


def run_selection(name, item, plan_tasks_, state):
    candidates = {dep: state["tasks"][dep]["dir"] for dep in item["deps"]}
    scores = {dep: validation_rmse(plan_tasks_[dep]["kind"], path) for dep, path in candidates.items()}
    chosen = min(item["deps"], key=lambda dep: (scores[dep], item["deps"].index(dep)))
    return dict(status="done", dir=candidates[chosen], selected=chosen, validation_rmse=scores, finished_at=utc())


def lane(args):
    directory = args.campaign / "plan"
    frozen = json.loads((directory / "plan.json").read_text())
    tasks = frozen["tasks"]
    job = os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")
    deadline = job_deadline(args.fallback_hours * 3600)
    logs = args.campaign / "logs"
    logs.mkdir(exist_ok=True)
    environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false", MPLBACKEND="Agg",
                       OMP_NUM_THREADS="1", JAX_PLATFORMS=args.platform)
    environment.pop("MLFLOW_TRACKING_URI", None)
    running = {}  # name -> (Popen, attempt record)
    print(json.dumps(dict(lane=job, deadline=datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                          capacity=args.capacity, slots=args.slots)), flush=True)

    def finish(name, process, stopped):
        with locked_state(directory) as state:
            record = state["tasks"][name]
            attempt = record["attempts"][-1]
            attempt.update(returncode=process.returncode, ended_at=utc(), stopped_for_deadline=stopped)
            if complete(tasks[name], attempt["dir"]):
                record.update(status="done", dir=attempt["dir"], finished_at=utc())
            elif stopped:
                record["status"] = "pending"  # Resumed from scratch by a later lane.
            else:
                # Attempts flagged infrastructure_failure (e.g. files missing from a deployment) do not count.
                failures = sum(1 for a in record["attempts"]
                               if not a.get("stopped_for_deadline") and not a.get("infrastructure_failure"))
                record["status"] = "failed" if failures >= MAX_ATTEMPTS else "pending"
        print(json.dumps(dict(task=name, returncode=process.returncode, stopped=stopped,
                              status=state["tasks"][name]["status"])), flush=True)

    while True:
        now = time.time()
        for name, (process, _) in list(running.items()):
            if process.poll() is not None:
                del running[name]
                finish(name, process, stopped=False)
        if deadline - now < args.stop_margin:
            for name, (process, _) in running.items():
                process.send_signal(signal.SIGTERM)
            for name, (process, _) in list(running.items()):
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                finish(name, process, stopped=True)
            running.clear()
            print(json.dumps(dict(lane=job, status="deadline")), flush=True)
            return
        launched = []
        with locked_state(directory) as state:
            records = state["tasks"]
            for name, record in records.items():  # Reclaim tasks of lanes that ended without finishing them.
                if record["status"] == "running" and not job_alive(record["attempts"][-1]["job"], job):
                    record["attempts"][-1].update(ended_at=utc(), stopped_for_deadline=True, reclaimed_by=job)
                    record["status"] = "pending"
            done_dirs = {name: record["dir"] for name, record in records.items() if record["status"] == "done"}
            for name in [n for n, r in records.items() if r["status"] == "pending"
                         and all(records[d]["status"] == "done" for d in tasks[n]["deps"])
                         and tasks[n]["argv"] is None]:
                records[name] = run_selection(name, tasks[name], tasks, state)
                done_dirs[name] = records[name]["dir"]
            weight = sum(tasks[name]["weight"] for name in running)
            ready = sorted((n for n, r in records.items() if r["status"] == "pending" and tasks[n]["argv"]
                            and all(records[d]["status"] == "done" for d in tasks[n]["deps"])),
                           key=lambda n: -tasks[n]["priority"])
            for name in ready:
                item = tasks[name]
                if len(running) + len(launched) >= args.slots or weight + item["weight"] > args.capacity:
                    continue
                if now + args.estimate_factor * item["estimate"] + args.stop_margin > deadline:
                    continue
                attempt_number = len(records[name]["attempts"])
                out = args.campaign / "runs" / name / f"attempt{attempt_number}"
                out.parent.mkdir(parents=True, exist_ok=True)
                attempt = dict(job=job, attempt=attempt_number, dir=str(out), started_at=utc(),
                               argv=resolve(item["argv"], out, done_dirs))
                records[name]["attempts"].append(attempt)
                records[name]["status"] = "running"
                weight += item["weight"]
                launched.append((name, attempt))
            terminal = all(r["status"] in {"done", "failed"} or any(
                records[d]["status"] == "failed" for d in tasks[n]["deps"]) for n, r in records.items())
        for name, attempt in launched:
            log = (logs / (name.replace("/", "__") + f"__attempt{attempt['attempt']}.log")).open("x")
            process = subprocess.Popen(attempt["argv"], cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            running[name] = (process, attempt)
            print(json.dumps(dict(task=name, launched=attempt["dir"], pid=process.pid)), flush=True)
        if not running and terminal:
            print(json.dumps(dict(lane=job, status="campaign_terminal")), flush=True)
            return
        time.sleep(args.poll)


def status(args):
    directory = args.campaign / "plan"
    tasks = json.loads((directory / "plan.json").read_text())["tasks"]
    with locked_state(directory) as state:
        records = state["tasks"]
    counts = {}
    for name, record in records.items():
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    print(json.dumps(dict(counts=counts), indent=None))
    for name, record in records.items():
        if record["status"] in {"running", "failed"} or (args.verbose and record["status"] != "pending"):
            last = record["attempts"][-1] if record["attempts"] else {}
            print(f"{record['status']:8s} {name:45s} attempts={len(record['attempts'])} job={last.get('job')} "
                  f"started={last.get('started_at', '')[:19]}")


def collect(args):
    directory = args.campaign / "plan"
    frozen = json.loads((directory / "plan.json").read_text())
    tasks = frozen["tasks"]
    with locked_state(directory) as state:
        records = state["tasks"]
    sources = {"_note": (f"All-filtering four-method campaign {args.campaign.name} (experiments/normflowsde_comparison/"
                         "filter_campaign.py): from-scratch pipelines, equal total optimizer updates per dataset "
                         f"{BUDGET}, per-seed validation-RMSE selection inside the campaign, one rebuilt data cache "
                         "per dataset verified against the previous comparison caches.")}
    describe = dict(
        normflowsde=lambda d: (f"Native amortized filter. Filtering training ({BUDGET[d] - NFSDE_FINETUNE_STEPS} updates; "
                               "dynamics reference vs learned noise scale + time-invariant, validation-selected) then "
                               f"full-horizon energy-score fine-tune ({NFSDE_FINETUNE_STEPS} updates, lr in "
                               f"{list(NFSDE_LEARNING_RATES)}, validation-selected)."),
        nf_dpf=lambda d: (f"Comparison-runner NF-DPF ({BUDGET[d] - NFDPF_LONGWINDOW_STEPS[d]} updates) then variant-B "
                          f"long-window fine-tune ({NFDPF_LONGWINDOW_STEPS[d]} updates: horizon curriculum, weighted CRPS, "
                          "ESS penalty 0.01)."),
        nsf=lambda d: (f"NSF-PF: learned-dynamics bootstrap PF ({BUDGET[d]} updates), process-noise jitter "
                       f"validation-selected from {{{JITTER_GRID}}}."),
        flowsde=lambda d: (f"FlowSDE-PF: learned-dynamics bootstrap PF ({BUDGET[d]} updates), fixed diffusion D "
                           f"validation-selected from {list(FLOWSDE_DIFFUSIONS)}."),
    )
    incomplete = []
    gmm = frozen.get("observation_noise") == "gmm"
    if gmm:  # describe the recipes actually planned (2026-10-20 GMM campaign)
        describe["normflowsde"] = lambda d: (
            f"Native amortized filter. Filtering training ({BUDGET[d] - NFSDE_FINETUNE_STEPS} updates; dynamics reference "
            "vs learned noise scale + time-invariant, validation-selected) then full-horizon energy-score fine-tune with "
            f"the learned noise scale ({NFSDE_FINETUNE_STEPS} updates, lr in {list(NFSDE_LEARNING_RATES_GMM)}, "
            "validation-selected).")
        describe["nf_dpf"] = lambda d: (
            f"Comparison-runner NF-DPF ({BUDGET[d] - NFDPF_LONGWINDOW_STEPS[d]} updates) then variant-B long-window "
            f"fine-tune ({NFDPF_LONGWINDOW_STEPS[d]} updates: horizon curriculum, weighted CRPS, ESS penalty 0.01, "
            "supervised measurement-NLL anchor on 16 windows).")
        describe["nsf"] = lambda d: (f"NSF-PF: learned-dynamics bootstrap PF ({BUDGET[d]} updates), checkpoint and "
                                     f"process-noise jitter selected jointly on validation from {{{JITTER_GRID_GMM}}}.")
        describe["flowsde"] = lambda d: (f"FlowSDE-PF: learned-dynamics bootstrap PF ({BUDGET[d]} updates), diffusion D "
                                         f"from {list(FLOWSDE_DIFFUSIONS)} and checkpoint/jitter from "
                                         f"{{{JITTER_GRID_GMM}}} selected jointly on validation.")
    for dataset in frozen.get("datasets", DATASETS):
        methods = {}
        for method in frozen.get("methods", METHODS):
            seeds, selections = {}, {}
            for seed in SEEDS:
                name = next(n for n, t in tasks.items() if t["final"] == method
                            and n.startswith(f"{dataset}/seed{seed}/"))
                record = records[name]
                if record["status"] != "done":
                    incomplete.append(name)
                    continue
                seeds[str(seed)] = record["dir"]
                chain = {n: records[n].get("selected") for n in tasks if n.startswith(f"{dataset}/seed{seed}/{method}/")
                         and tasks[n]["argv"] is None}
                if method == "nsf":
                    chain["jitter"] = json.loads((Path(record["dir"]) / "summary.json").read_text()).get("jitter")
                selections[str(seed)] = chain
            methods[method] = dict(status="final" if len(seeds) == len(SEEDS) else "pending", access="filtering",
                                   access_label=ACCESS_LABEL, setting=describe[method](dataset), seeds=seeds,
                                   selections=selections,
                                   latency=dict(kind="metrics", note="packed concurrent GPU tasks; not an isolated timing"),
                                   note="" if len(seeds) == len(SEEDS) else "campaign tasks incomplete")
        sources[dataset] = dict(folder=dataset, data_cache=str(args.campaign / "data_cache" / dataset), methods=methods)
    output = args.output or args.campaign / "sources.json"
    atomic_json(output, sources)
    print(json.dumps(dict(sources=str(output), incomplete=incomplete)), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("prepare-data", "plan", "lane", "status", "collect"):
        q = sub.add_parser(name)
        q.add_argument("--campaign", type=Path, required=True)
        if name in ("prepare-data", "plan"):
            q.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
            q.add_argument("--observation-noise", choices=("gaussian", "gmm"), default="gaussian",
                           help="gmm (opt-in, 2026-10-20): three-component mixture observation noise on pendulum, "
                                "ALOHA and REASSEMBLE (Lorenz-96 already uses it).")
        if name == "plan":
            q.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
            q.add_argument("--python", type=Path, default=Path(sys.executable))
            q.add_argument("--smoke", action="store_true", help="Tiny CPU budgets for an end-to-end plumbing check.")
        if name == "lane":
            q.add_argument("--slots", type=int, default=8, help="Maximum concurrent tasks on this lane's GPU.")
            q.add_argument("--capacity", type=int, default=10, help="Maximum summed memory weight on the GPU.")
            q.add_argument("--estimate-factor", type=float, default=1.3)
            q.add_argument("--stop-margin", type=float, default=300.)
            q.add_argument("--fallback-hours", type=float, default=6.)
            q.add_argument("--poll", type=float, default=20.)
            q.add_argument("--platform", default="cuda")
        if name == "status":
            q.add_argument("--verbose", action="store_true")
        if name == "collect":
            q.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    args.campaign = args.campaign.resolve()
    dict(**{"prepare-data": prepare_data}, plan=plan, lane=lane, status=status, collect=collect)[args.action](args)


if __name__ == "__main__":
    main()
