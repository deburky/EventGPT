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
# Google Cloud (one L4, deleted when its run ends; see scripts/gcp.sh)
# -------------------------------------------------------------------------------

# Service account, bucket and data upload, once
[group('gcp')]
gcp-init:
    scripts/gcp.sh init

# Train on an L4 that deletes itself afterwards; spot="1" for spot capacity, hours caps the VM's lifetime
[confirm("start a GPU VM on Google Cloud?")]
[group('gcp')]
gcp-up run spot="0" hours="6" *args:
    SPOT={{ spot }} MAX_HOURS={{ hours }} scripts/gcp.sh up {{ run }} {{ args }}

# Training log of a run, mirrored to the bucket every minute
[group('gcp')]
gcp-logs run:
    scripts/gcp.sh logs {{ run }}

# Startup progress from the VM's serial console, before the training log exists
[group('gcp')]
gcp-boot run:
    gcloud compute instances get-serial-port-output egpt-{{ run }} --zone="$(gcloud compute instances list --filter='name=egpt-{{ run }}' --format='value(zone.basename())')" | grep startup-script | tail -30

# VMs carrying the event-gpt label
[group('gcp')]
gcp-status:
    scripts/gcp.sh status

# Pull a run's checkpoint and log from the bucket into data/runs/<run>
[group('gcp')]
gcp-pull run:
    mkdir -p data/runs/{{ run }}
    gcloud storage rsync --recursive "gs://event-gpt-$(gcloud config get-value project 2>/dev/null)/runs/{{ run }}" data/runs/{{ run }}

# Delete every event-gpt VM and the NAT gateways
[group('gcp')]
gcp-down:
    scripts/gcp.sh down

# Delete everything in the cloud, bucket and checkpoints included
[group('gcp')]
gcp-nuke:
    scripts/gcp.sh nuke

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
