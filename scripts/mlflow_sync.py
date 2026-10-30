"""Log every run under data/runs into a local MLflow store, so runs and probes can be compared in `mlflow ui`.

Each run directory becomes one MLflow run: the checkpoint's config as params, the training log as stepped
metrics, small CSVs as artifacts. Each probe directory in it (any directory below it with a results.json) becomes a
nested run with ROC-AUC and PR-AUC per readout and label, and the official mean. Runs already in the store
are skipped by their source path, so rerunning after new runs only adds those; --rebuild logs everything again.

Example: `uv run --group mlflow scripts/mlflow_sync.py`, then `uv run --group mlflow mlflow ui --backend-store-uri sqlite:///data/mlflow.db`
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import mlflow
import torch
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

RUNS = Path("data/runs")
STORE = "sqlite:///data/mlflow.db"
ARTIFACTS = Path("data/mlartifacts").resolve().as_uri()
EXPERIMENT = "eventgpt"
OFFICIAL = ("target_1", "target_2", "target_3", "target_4")
MAX_ARTIFACT_BYTES = 2_000_000
BATCH = 1000


# -------------------------------------------------------------------------------
# Reading a run
# -------------------------------------------------------------------------------
def files(run_dir: Path, pattern: str) -> list[Path]:
    """Files matching pattern anywhere in the run directory (runs pulled from GCP keep theirs under ckpt/), caches left out."""
    return sorted(f for f in run_dir.rglob(pattern) if "cache" not in f.parts)


def checkpoint_params(run_dir: Path) -> dict[str, str]:
    """Config, step and architecture of the run's checkpoint, without loading the weights into memory."""
    ckpts = files(run_dir, "*.pt")
    if not ckpts:
        return {}
    ck = torch.load(ckpts[0], map_location="cpu", weights_only=False, mmap=True)
    out = {f"cfg.{k}": str(v) for k, v in (ck.get("cfg") or {}).items()}
    for k in ("step", "arch", "vocab_sha256"):
        if k in ck:
            out[k] = str(ck[k])
    return out


def training_metrics(run_dir: Path) -> list[Metric]:
    """Every numeric field of the training log, at its step."""
    out: list[Metric] = []
    for log in files(run_dir, "*_log.jsonl"):
        for line in log.read_text().splitlines():
            row = json.loads(line)
            step = int(row.get("step", 0))
            for k, v in row.items():
                if k != "step" and isinstance(v, (int, float)) and v == v:
                    out.append(Metric(k, float(v), 0, step))
    return out


def probe_metrics(results: dict[str, Any]) -> list[Metric]:
    """ROC-AUC and PR-AUC per readout and label, and the official mean when the file predates it."""
    out: list[Metric] = []
    for readout, labels in results["metrics"].items():
        for label, scores in labels.items():
            for name, v in scores.items():
                if isinstance(v, (int, float)) and v == v:
                    out.append(Metric(f"{readout}.{label}.{name}", float(v), 0, 0))
        if "official_mean" not in labels and all(t in labels for t in OFFICIAL):
            mean = sum(labels[t]["roc_auc"] for t in OFFICIAL) / len(OFFICIAL)
            out.append(Metric(f"{readout}.official_mean.roc_auc", mean, 0, 0))
    return out


# -------------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------------
def log_batches(client: MlflowClient, run_id: str, metrics: list[Metric], params: dict[str, str]) -> None:
    """Log params once and metrics in batches under MLflow's per-request limit."""
    items = [Param(k, v[:6000]) for k, v in params.items()]
    for i in range(0, len(items), 100):
        client.log_batch(run_id, params=items[i : i + 100])
    for i in range(0, len(metrics), BATCH):
        client.log_batch(run_id, metrics=metrics[i : i + BATCH])


def log_small_files(client: MlflowClient, run_id: str, folder: Path) -> None:
    """Attach the folder's CSV and JSON files under MAX_ARTIFACT_BYTES."""
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix in (".csv", ".json") and f.stat().st_size <= MAX_ARTIFACT_BYTES:
            client.log_artifact(run_id, str(f))


def synced(client: MlflowClient, exp_id: str) -> set[str]:
    """Source paths already in the store."""
    runs = client.search_runs([exp_id], max_results=50_000)
    return {r.data.tags["source"] for r in runs if "source" in r.data.tags}


def sync_run(client: MlflowClient, exp_id: str, run_dir: Path, done: set[str]) -> int:
    """Log one run directory and its probes; returns the number of MLflow runs created."""
    created = 0
    seed = re.search(r"-s(\d+)$", run_dir.name)
    tags = {"source": str(run_dir), "seed": seed.group(1) if seed else ""}
    if str(run_dir) in done:
        parent_id = next(r.info.run_id for r in client.search_runs([exp_id], f"tags.source = '{run_dir}'"))
    else:
        parent = client.create_run(exp_id, run_name=run_dir.name, tags=tags)
        parent_id = parent.info.run_id
        log_batches(client, parent_id, training_metrics(run_dir), checkpoint_params(run_dir))
        log_small_files(client, parent_id, run_dir)
        client.set_terminated(parent_id)
        created += 1
    for res in files(run_dir, "results.json"):
        probe_dir = res.parent
        if str(probe_dir) in done:
            continue
        results = json.loads(res.read_text())
        params = {k: str(v) for k, v in results.items() if k in ("ckpt", "step", "block", "now", "elapsed_s")}
        probe = str(probe_dir.relative_to(run_dir))
        child_tags = {**tags, "source": str(probe_dir), "probe": probe, "mlflow.parentRunId": parent_id}
        child = client.create_run(exp_id, run_name=f"{run_dir.name}/{probe}", tags=child_tags)
        log_batches(client, child.info.run_id, probe_metrics(results), params)
        log_small_files(client, child.info.run_id, probe_dir)
        client.set_terminated(child.info.run_id)
        created += 1
    return created


def main() -> None:
    """Sync data/runs into the MLflow store."""
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true", help="delete the experiment's runs and log everything again")
    a = p.parse_args()

    mlflow.set_tracking_uri(STORE)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    exp_id = exp.experiment_id if exp else client.create_experiment(EXPERIMENT, artifact_location=ARTIFACTS)
    if a.rebuild:
        for r in client.search_runs([exp_id], max_results=50_000):
            client.delete_run(r.info.run_id)
    done = set() if a.rebuild else synced(client, exp_id)
    run_dirs = sorted(d for d in RUNS.iterdir() if d.is_dir() and any(files(d, pat) for pat in ("*.pt", "*_log.jsonl", "results.json")))
    created = sum(sync_run(client, exp_id, d, done) for d in run_dirs)
    print(f"{created} MLflow runs created from {len(run_dirs)} run directories into {STORE}")


if __name__ == "__main__":
    main()
