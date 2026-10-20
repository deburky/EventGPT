#!/usr/bin/env bash
# Startup script of a training VM, run as root on first boot by scripts/gcp.sh up.
# Pulls the code and data from the bucket, trains, uploads the checkpoint and log, then deletes the VM.
# The VM is deleted on failure too: the EXIT trap runs whatever happens after the guard below.

set -uxo pipefail

md() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
NAME="$(md name)"
ZONE="$(md zone | awk -F/ '{print $NF}')"
BUCKET="$(md attributes/bucket)"
RUN="$(md attributes/run)"
ARGS="$(md attributes/train-args)"
PROBE="$(md attributes/probe || echo 1)"  # 1 trains, probes and scores zero-shot; 0 only trains; only probes; zero-shot only scores
PROBE_ARGS="$(md attributes/probe-args || echo --batch 32)"
DATA_PREFIX="$(md attributes/data-prefix || echo data)"  # bucket folder of the token streams, e.g. data-trx for a variant
WORK=/opt/egpt
OUT="gs://$BUCKET/runs/$RUN"

# startup scripts run again on every boot; a VM that rebooted mid-run should not start over
if [[ -f "$WORK/started" ]]; then
    exit 0
fi
mkdir -p "$WORK/data" "$WORK/ckpt"
touch "$WORK/started"
export HOME=/root UV_CACHE_DIR="$WORK/uv-cache" UV_PYTHON_INSTALL_DIR="$WORK/python"

finish() {
    # the log is written through tee in a process substitution, which can lag the process that just exited:
    # without the pause the last lines, a traceback among them, miss the final upload
    sleep 5
    sync
    gcloud storage cp "$WORK/train.log" "$OUT/train.log"
    gcloud storage rsync --recursive "$WORK/ckpt" "$OUT/ckpt"
    gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet
}
trap finish EXIT
exec > >(tee -a "$WORK/train.log") 2>&1

# -------------------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------------------
for _ in $(seq 60); do
    nvidia-smi >/dev/null 2>&1 && break
    sleep 10
done
nvidia-smi || exit 1
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
gcloud storage cp "$OUT/code.tar.gz" "$WORK/code.tar.gz"
mkdir -p "$WORK/code" && tar -xzf "$WORK/code.tar.gz" -C "$WORK/code"
gcloud storage rsync --recursive "gs://$BUCKET/$DATA_PREFIX" "$WORK/data"
gcloud storage rsync --recursive "gs://$BUCKET/raw" "$WORK/raw"
# a rerun of the same run name resumes from whatever the last VM mirrored
gcloud storage rsync --recursive "$OUT/ckpt" "$WORK/ckpt" || true
cd "$WORK/code" && uv sync --no-dev --python 3.12 || exit 1

# -------------------------------------------------------------------------------
# Training, then the probe, with the log mirrored every minute and the checkpoint folder (probe cache included) every five
# -------------------------------------------------------------------------------
(
    i=0
    while sleep 60; do
        gcloud storage cp "$WORK/train.log" "$OUT/train.log" --quiet >/dev/null 2>&1
        i=$((i + 1))
        if ((i % 5 == 0)); then
            gcloud storage rsync --recursive "$WORK/ckpt" "$OUT/ckpt" --quiet >/dev/null 2>&1
        fi
    done
) &
export MBD_OUT="$WORK/data" MBD_ROOT="$WORK/raw" EGPT_CKPT="$WORK/ckpt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # length-sorted batches grow steadily; avoid fragmentation
if [[ "$PROBE" == "1" || "$PROBE" == "0" ]]; then
    # shellcheck disable=SC2086  # ARGS is a list of flags
    uv run --no-dev python -m eventgpt.train --run "$RUN" $ARGS || exit 1
fi
if [[ "$PROBE" == "1" || "$PROBE" == "only" ]]; then
    # shellcheck disable=SC2086  # PROBE_ARGS is a list of flags
    uv run --no-dev python -m eventgpt.probe "$WORK/ckpt/$RUN.pt" $PROBE_ARGS
fi
if [[ "$PROBE" == "1" || "$PROBE" == "only" || "$PROBE" == "zero-shot" ]]; then
    # zero-shot event targets on the probe's rows, which the probe wrote (or a previous VM mirrored) to ckpt/probe
    uv run --no-dev python -m eventgpt.zero_shot "$WORK/ckpt/$RUN.pt" --probe-dir "$WORK/ckpt/probe" --out "$WORK/ckpt/zero_shot.csv"
fi
