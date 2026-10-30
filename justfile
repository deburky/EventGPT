# Command runner for EventGPT on MBD. `just` lists the recipes; `just <recipe>` runs one.
# Data lives outside the repo, on the external drive by default; override MBD_ROOT and MBD_OUT to move it.

set shell := ["zsh", "-cu"]

export MBD_ROOT := env("MBD_ROOT", "/Volumes/WD_BLACK/datasets/MBD-mini/extracted")
export MBD_OUT := env("MBD_OUT", "/Volumes/WD_BLACK/datasets/MBD-mini/tokens")

default:
    @just --list --unsorted

# -------------------------------------------------------------------------------
# Environment and checks
# -------------------------------------------------------------------------------

# Sync the uv environment including the dev group
[group('env')]
sync:
    uv sync

# Install the git pre-commit hook (ruff, pydocstyle, mypy, just --fmt) via prek
[group('env')]
hooks:
    uv run prek install

# Run every hook over the whole tree, not just the staged files
[group('env')]
hooks-all:
    uv run prek run --all-files

# Lint, format-check and type-check the package
[group('env')]
check:
    uv run ruff format --check src
    uv run ruff check src
    uv run pydocstyle src
    uv run mypy src

# Apply ruff formatting and safe fixes
[group('env')]
fix:
    uv run ruff format src
    uv run ruff check --fix src

# -------------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------------

# Build token streams from MBD-mini: vocabulary on the pretraining folds, every fold tokenized, pretraining streams packed
[confirm("rebuild the token streams, overwriting MBD_OUT?")]
[group('data')]
prepare test_fold="0":
    uv run python -m eventgpt.mbd --test-fold {{ test_fold }}

# Stream invariants per fold, one decoded client, token share per key
[group('data')]
check-tokens:
    uv run python -m eventgpt.check_tokens

# -------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------

# Short local run on MPS (about 2,000 tokens/s, so for debugging rather than results)
[group('train')]
train run="local" steps="100" *args:
    uv run python -m eventgpt.train --run {{ run }} --steps {{ steps }} --arch eventgpt --rope {{ args }}

# -------------------------------------------------------------------------------
# Evaluation (same rows, labels and test fold for both)
# -------------------------------------------------------------------------------

# Frozen-embedding probe of a checkpoint: monthly cuts, logistic head per label, ROC-AUC on the test fold (GPU; slow on MPS)
[group('eval')]
probe ckpt *args:
    uv run python -m eventgpt.probe {{ ckpt }} {{ args }}

# Aggregation baseline: hand-built features of the same rows, CatBoost per label (CPU)
[group('eval')]
baseline *args:
    uv run python -m eventgpt.baseline {{ args }}

# -------------------------------------------------------------------------------
# MLflow (local store in data/mlflow.db; see scripts/mlflow_sync.py)
# -------------------------------------------------------------------------------

# Log new runs and probes under data/runs into the MLflow store; rebuild="1" logs everything again
[group('mlflow')]
mlflow-sync rebuild="0":
    MLFLOW_DISABLE_AGENT_HINT=1 uv run --group mlflow scripts/mlflow_sync.py {{ if rebuild == "1" { "--rebuild" } else { "" } }}

# Browse and compare the runs at http://127.0.0.1:5000
[group('mlflow')]
mlflow-ui:
    MLFLOW_DISABLE_AGENT_HINT=1 uv run --group mlflow mlflow ui --backend-store-uri sqlite:///data/mlflow.db

# -------------------------------------------------------------------------------
# Google Cloud (one L4, deleted when its run ends; see scripts/gcp.py)
# -------------------------------------------------------------------------------

# Service account, bucket and data upload, once
[group('gcp')]
gcp-init:
    uv run --group gcp scripts/gcp.py init

# Train on an L4 that deletes itself afterwards; spot="1" for spot capacity, hours caps the VM's lifetime
[confirm("start a GPU VM on Google Cloud?")]
[group('gcp')]
gcp-up run spot="0" hours="6" *args:
    SPOT={{ spot }} MAX_HOURS={{ hours }} uv run --group gcp scripts/gcp.py up {{ run }} {{ args }}

# Training log of a run, mirrored to the bucket every minute
[group('gcp')]
gcp-logs run:
    uv run --group gcp scripts/gcp.py logs {{ run }}

# Startup progress from the VM's serial console, before the training log exists
[group('gcp')]
gcp-boot run:
    uv run --group gcp scripts/gcp.py boot {{ run }}

# VMs carrying the event-gpt label
[group('gcp')]
gcp-status:
    uv run --group gcp scripts/gcp.py status

# Pull a run's checkpoint and log from the bucket into data/runs/<run>
[group('gcp')]
gcp-pull run:
    uv run --group gcp scripts/gcp.py pull {{ run }}

# Delete every event-gpt VM and the NAT gateways
[group('gcp')]
gcp-down:
    uv run --group gcp scripts/gcp.py down

# Delete everything in the cloud, bucket and checkpoints included
[group('gcp')]
gcp-nuke:
    uv run --group gcp scripts/gcp.py nuke

# -------------------------------------------------------------------------------
# Paper
# -------------------------------------------------------------------------------

# Build paper/eventgpt.pdf with XeLaTeX
[group('paper')]
paper:
    make -C paper

# Stage the arXiv upload in paper/tarball
[group('paper')]
paper-zip:
    make -C paper zip
