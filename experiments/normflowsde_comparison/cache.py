"""Immutable, pickle-free benchmark arrays shared by CPU preparation and GPU runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from experiments.nf_dpf_state_estimation.datasets import BenchmarkData, _json_value


SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_description(value):
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise ValueError("Benchmark caches cannot contain Python objects")
    return dict(shape=list(array.shape), dtype=array.dtype.str,
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def _payload_sha256(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _raw_inputs(data):
    """Record files actually consumed by the two robot adapters, excluding videos."""
    data_path = data.protocol.get("data_path")
    if data_path is None:
        return []
    root = Path(data_path).resolve()
    if root.is_file():
        paths = [root]
    elif root.is_dir():
        paths = sorted(root.glob("data/**/*.parquet")) or sorted(root.glob("**/*.parquet"))
        if data.protocol.get("dataset") == "reassemble":
            paths += [root / "meta/splits.json"]
    else:
        raise FileNotFoundError(f"Raw dataset does not exist: {root}")
    return [dict(path=str(path), size_bytes=path.stat().st_size,
                 sha256=file_sha256(path)) for path in paths]


def save_cache(data, path):
    """Save exact arrays and source provenance to a new directory, atomically."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Preserve the existing cache; choose a new path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        manifest = dict(schema_version=SCHEMA_VERSION, dt=float(data.dt),
                        protocol=_json_value(data.protocol), metadata=_json_value(data.metadata),
                        raw_inputs=_raw_inputs(data), splits={})
        for name in SPLITS:
            arrays = {key: np.asarray(value) for key, value in getattr(data, name).items()}
            descriptions = {key: array_description(value) for key, value in arrays.items()}
            destination = temporary / f"{name}.npz"
            np.savez(destination, **arrays)
            manifest["splits"][name] = dict(file=destination.name,
                                           sha256=file_sha256(destination), arrays=descriptions)
        manifest["payload_sha256"] = _payload_sha256(manifest)
        (temporary / "cache.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        if path.exists():
            raise FileExistsError(f"Cache was created concurrently: {path}")
        temporary.rename(path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return path


def load_cache(path):
    """Load and verify schema, metadata, raw sources, archives and every array."""
    path = Path(path)
    manifest = json.loads((path / "cache.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark cache schema")
    payload_hash = manifest.pop("payload_sha256", None)
    if _payload_sha256(manifest) != payload_hash:
        raise ValueError("Benchmark cache metadata hash changed")
    for source in manifest["raw_inputs"]:
        raw_path = Path(source["path"])
        if not raw_path.is_file() or raw_path.stat().st_size != source["size_bytes"]:
            raise ValueError(f"Raw dataset file missing or changed: {raw_path}")
        if file_sha256(raw_path) != source["sha256"]:
            raise ValueError(f"Raw dataset hash changed: {raw_path}")
    splits = {}
    if set(manifest["splits"]) != set(SPLITS):
        raise ValueError("Benchmark cache split schema changed")
    for name in SPLITS:
        description = manifest["splits"][name]
        if description["file"] != f"{name}.npz":
            raise ValueError("Invalid benchmark cache archive name")
        archive_path = path / description["file"]
        if file_sha256(archive_path) != description["sha256"]:
            raise ValueError(f"Benchmark cache archive hash changed: {name}")
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if {key: array_description(value) for key, value in arrays.items()} != description["arrays"]:
            raise ValueError(f"Benchmark cache array schema/hash changed: {name}")
        if not {"states", "observations", "controls"}.issubset(arrays):
            raise ValueError(f"Required benchmark arrays missing: {name}")
        splits[name] = arrays
    return BenchmarkData(**splits, protocol=manifest["protocol"],
                         metadata=manifest["metadata"], dt=manifest["dt"])
