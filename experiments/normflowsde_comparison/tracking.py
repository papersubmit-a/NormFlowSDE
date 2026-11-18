"""Durable MLflow REST tracking without a training-environment dependency.

Every event is written locally before upload. HTTP outages defer upload and do
not abort training; ``replay`` also works while an offline job is still running.
Only the replay process needs access to the tracking server. The server should
serve artifacts and own its database; workers never open its SQLite file.

    python -m experiments.normflowsde_comparison.tracking replay RUN_DIRECTORY \
        --tracking-uri http://127.0.0.1:5000
    python -m experiments.normflowsde_comparison.tracking replay \
        --results-root RESULTS_DIRECTORY --tracking-uri http://127.0.0.1:5000

SIGTERM/SIGINT are handled by run.py. SIGKILL, OOM kills, and machine failures
cannot execute Python cleanup: after confirming a job has stopped, use
``finalize RUN_DIRECTORY --status KILLED`` and replay to reconcile its status.
Never run two training processes in the same output directory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import getpass
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen
import uuid


def validate_uri(uri):
    """Keep credentials out of run metadata and durable tracking state."""
    if not uri:
        return None
    parsed = urlsplit(uri)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("tracking-uri must be an HTTP(S) URL without credentials, query, or fragment")
    return uri.rstrip("/")


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _flatten(values, prefix=""):
    for name, value in values.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            yield from _flatten(value, key)
        else:
            encoded = json.dumps(value, separators=(",", ":")) if not isinstance(value, str) else value
            # Stay within the limits of all supported tracking backends. Full
            # values remain in the run_metadata.json provenance artifact.
            if len(key) <= 250 and len(encoded) <= 250:
                yield {"key": key, "value": encoded}


class JournalTracker:
    def __init__(self, output, report=None, *, uri=None, experiment="normflowsde-comparison",
                 flush_every=25, timeout=2.):
        self.output = Path(output)
        self.uri = validate_uri(uri)
        self.flush_every, self.timeout = flush_every, timeout
        self.state_path = self.output / "tracking-state.json"
        self.journal_path = self.output / "tracking-events.jsonl"
        self._pending_steps, self._retry_at, self._warned = 0, 0., False
        self.output.mkdir(parents=True, exist_ok=True)
        if report is not None:
            with self._lock():
                if self.state_path.exists():
                    raise FileExistsError(f"Tracking already initialized at {self.state_path}")
                uid = uuid.uuid4().hex
                name = f"{report['dataset']}/{report['method']}/seed-{report['training_seed']}/{uid[:8]}"
                _atomic_json(self.state_path, dict(schema_version=1, uid=uid, run_name=name,
                             experiment=experiment, cursor=0, run_id=None, start_time=self._now()))
            tags = dict(dataset=report["dataset"], method=report["method"],
                        training_seed=str(report["training_seed"]),
                        inference_access=report["inference_access"],
                        output_directory=str(self.output.resolve()),
                        **{"mlflow.user": getpass.getuser(), "mlflow.runName": name})
            for name in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST"):
                if os.environ.get(name):
                    tags[name.lower()] = os.environ[name]
            provenance = dict(model=report["model_config"], protocol=report["data"]["protocol"],
                              arguments=report["arguments"],
                              source_sha256=report["source_sha256"],
                              split_sha256=report["data"]["split_sha256"],
                              packages=report["environment"]["packages"],
                              git_revision=report["git_revision"],
                              parameter_count=report.get("parameter_count"))
            self._append(dict(kind="metadata", timestamp=self._now(),
                              params=list(_flatten(provenance)),
                              tags=[dict(key=key, value=value) for key, value in tags.items()]))
            self.flush()

    @staticmethod
    def _now():
        return int(time.time() * 1000)

    @contextmanager
    def _lock(self):
        with (self.output / ".tracking.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _append(self, event):
        with self.journal_path.open("a") as journal:
            journal.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
            journal.flush()

    def metrics(self, values, step):
        timestamp = self._now()
        metrics = []
        for name, value in values.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                metrics.append(dict(key=name, value=float(value), timestamp=timestamp, step=int(step)))
        if metrics:
            self._append(dict(kind="metrics", metrics=metrics))
        self._pending_steps += 1
        if self._pending_steps >= self.flush_every:
            self.flush()
            self._pending_steps = 0

    def finish(self, status, report):
        if status not in {"FINISHED", "FAILED", "KILLED"}:
            raise ValueError("Invalid terminal tracking status")
        # Snapshots remain valid even if local metadata changes after this event.
        self._append(dict(kind="finish", status=status, timestamp=self._now(), report=report))
        self.flush(force=True)

    def _request(self, endpoint, payload=None, *, method=None, raw=None):
        url = self.uri + "/api/2.0/" + endpoint
        if method == "GET" and payload:
            url += "?" + urlencode(payload)
        data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None and method != "GET" else None)
        request = Request(url, data=data, method=method or "POST",
                          headers={"Content-Type": "application/octet-stream" if raw is not None else "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            body = response.read()
        return json.loads(body) if body else {}

    def _ensure_run(self, state):
        if state["run_id"]:
            return
        try:
            experiment = self._request("mlflow/experiments/get-by-name",
                                       {"experiment_name": state["experiment"]}, method="GET")["experiment"]
        except HTTPError as error:
            if error.code != 404:
                raise
            try:
                self._request("mlflow/experiments/create", {"name": state["experiment"]})
            except HTTPError as creation_error:
                if creation_error.code != 400:  # Concurrent workers may create it first.
                    raise
            experiment = self._request("mlflow/experiments/get-by-name",
                                       {"experiment_name": state["experiment"]}, method="GET")["experiment"]
        state["experiment_id"] = experiment["experiment_id"]
        # Recover a creation whose HTTP response was lost, rather than making
        # another run. The UID is persisted before making any network call.
        matches = self._request("mlflow/runs/search", dict(experiment_ids=[state["experiment_id"]],
                    filter=f"tags.comparison_uid = '{state['uid']}'", max_results=1)).get("runs", [])
        if matches:
            info = matches[0]["info"]
        else:
            info = self._request("mlflow/runs/create", dict(experiment_id=state["experiment_id"],
                    run_name=state["run_name"], start_time=state["start_time"],
                    tags=[dict(key="comparison_uid", value=state["uid"])]))["run"]["info"]
        state["run_id"], state["artifact_uri"] = info["run_id"], info["artifact_uri"]
        _atomic_json(self.state_path, state)

    def _batch(self, run_id, *, metrics=(), params=(), tags=()):
        # Chunks respect the 100 params/tags and 1000 metrics per request limits.
        for key, values, size in (("metrics", metrics, 900), ("params", params, 100), ("tags", tags, 100)):
            for index in range(0, len(values), size):
                self._request("mlflow/runs/log-batch", {"run_id": run_id, key: values[index:index + size]})

    def _upload_provenance(self, state, report):
        artifact_uri = state["artifact_uri"]
        if not artifact_uri.startswith("mlflow-artifacts:/"):
            raise ValueError("Tracking server must serve artifacts (mlflow-artifacts:/ URI required)")
        relative = artifact_uri[len("mlflow-artifacts:/"):].lstrip("/")
        path = "mlflow-artifacts/artifacts/" + quote(relative + "/run_metadata.json", safe="/")
        self._request(path, method="PUT", raw=(json.dumps(report, indent=2) + "\n").encode())

    def flush(self, *, force=False):
        """Best effort HTTP upload; failures retain the local cursor for replay."""
        if not self.uri or (not force and time.monotonic() < self._retry_at):
            return False
        try:
            with self._lock():
                state = json.loads(self.state_path.read_text())
                self._ensure_run(state)
                with self.journal_path.open("rb") as journal:
                    journal.seek(state["cursor"])
                    pending, cursor = [], state["cursor"]
                    while True:
                        line = journal.readline()
                        if not line or not line.endswith(b"\n"):
                            break  # A running worker may be appending this line.
                        event, end = json.loads(line), journal.tell()
                        if event["kind"] == "metrics":
                            pending.extend(event["metrics"])
                            cursor = end
                            if len(pending) < 900:
                                continue
                        else:
                            if pending:
                                self._batch(state["run_id"], metrics=pending)
                                state["cursor"] = cursor
                                _atomic_json(self.state_path, state)
                                pending = []
                            if event["kind"] == "metadata":
                                self._batch(state["run_id"], params=event["params"], tags=event["tags"])
                            elif event["kind"] == "finish":
                                self._upload_provenance(state, event["report"])
                                self._request("mlflow/runs/update", dict(run_id=state["run_id"],
                                              status=event["status"], end_time=event["timestamp"]))
                            else:
                                raise ValueError(f"Unknown tracking event {event['kind']}")
                            cursor = end
                        if pending:
                            self._batch(state["run_id"], metrics=pending)
                            pending = []
                        state["cursor"] = cursor
                        _atomic_json(self.state_path, state)
                    if pending:
                        self._batch(state["run_id"], metrics=pending)
                        state["cursor"] = cursor
                        _atomic_json(self.state_path, state)
            self._retry_at, self._warned = 0., False
            return True
        except (OSError, ValueError, KeyError) as error:
            self._retry_at = time.monotonic() + 60.
            if not self._warned:
                # Do not log URLs or HTTP response bodies, which can contain secrets.
                print(f"MLflow upload deferred ({type(error).__name__}); local tracking journal preserved.", flush=True)
                self._warned = True
            return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("replay", help="Replay one run or all journals below a results root")
    replay.add_argument("output", type=Path, nargs="?")
    replay.add_argument("--results-root", type=Path)
    replay.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    finalize = commands.add_parser("finalize", help="Reconcile an externally confirmed stopped job")
    finalize.add_argument("output", type=Path)
    finalize.add_argument("--status", required=True, choices=("FAILED", "KILLED"))
    finalize.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    args = parser.parse_args(argv)
    if args.command == "finalize":
        report_path = args.output / "run_metadata.json"
        report = json.loads(report_path.read_text())
        if report["status"] == "complete":
            parser.error("Refusing to finalize a complete run as failed/killed")
        report["status"] = "killed" if args.status == "KILLED" else "failed"
        report["termination"] = "externally confirmed stopped job"
        _atomic_json(report_path, report)
        JournalTracker(args.output, uri=args.tracking_uri).finish(args.status, report)
        return
    if bool(args.output) == bool(args.results_root):
        parser.error("Pass exactly one output directory or --results-root")
    if not args.tracking_uri:
        parser.error("replay requires --tracking-uri or MLFLOW_TRACKING_URI")
    outputs = [args.output] if args.output else sorted(path.parent for path in args.results_root.rglob("tracking-events.jsonl"))
    success = all([JournalTracker(output, uri=args.tracking_uri).flush(force=True) for output in outputs])
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
