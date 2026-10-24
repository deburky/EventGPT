#!/usr/bin/env bash
# Train EventGPT on a single L4 in Google Cloud, with nothing left running afterwards.
#
#   scripts/gcp.sh init                    service account, bucket, data upload (once)
#   scripts/gcp.sh up <run> [train args]   push the code and start a VM that trains, probes, uploads and deletes
#                                          itself; rerunning a run name resumes from its last mirrored checkpoint
#   scripts/gcp.sh logs <run>              the run's training log, synced to the bucket every minute
#   scripts/gcp.sh status                  VMs carrying the event-gpt label
#   scripts/gcp.sh down                    delete every VM carrying the label, and the NAT gateways
#   scripts/gcp.sh nuke                    also delete the bucket, service account and role
#
# Three things stop a VM from outliving its run: the startup script deletes the VM when training ends
# or fails, --max-run-duration makes Google delete it after MAX_HOURS whatever happens, and `down`
# deletes anything labeled project=event-gpt. Checkpoints and logs live in the bucket under runs/<run>/.
#
# The project's organization policy forbids external IPs on VMs, so the VM gets none and reaches PyPI and the
# uv installer through a Cloud NAT on the default network, created by init.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-europe-west4}"
# every European zone with L4s, tried in order until one has capacity
ZONES="${GCP_ZONES:-europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c europe-west3-a europe-west3-b europe-west2-a europe-west2-b europe-west6-b europe-west6-c}"
BUCKET="${GCP_BUCKET:-event-gpt-${PROJECT}}"
DATA="${MBD_OUT:-/Volumes/WD_BLACK/datasets/MBD-mini/tokens}"
RAW="${MBD_ROOT:-/Volumes/WD_BLACK/datasets/MBD-mini/extracted}"
SA_NAME="event-gpt-runner"
SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
ROLE="eventGptSelfDelete"
ROUTER="event-gpt-router"
NAT="event-gpt-nat"
MACHINE="${GCP_MACHINE:-g2-standard-8}"
IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"
MAX_HOURS="${MAX_HOURS:-6}"
PROBE="${PROBE:-1}"  # 1 probes the final checkpoint before the VM deletes itself; only skips training and probes the run's checkpoint
PROBE_ARGS="${PROBE_ARGS:---batch 32}"  # flags for eventgpt.evaluation.probe, space-separated (no semicolons: the metadata delimiter)
SPOT="${SPOT:-0}"
DATA_PREFIX="${DATA_PREFIX:-data}"  # bucket folder of the token streams the VM trains and probes on
LABEL="project=event-gpt"
DEFAULT_ARGS="--arch eventgpt --rope --steps 20000 --batch 8 --accum 2"

# -------------------------------------------------------------------------------
# One-time setup
# -------------------------------------------------------------------------------
ensure_nat() {
    # Cloud NAT is regional, so every region a VM lands in needs its own router and gateway
    local region="$1"
    # another project's gateway (graph-credit, say) may already cover every subnet in the region. GCP allows one
    # such NAT per network and region, and the other project deletes its own when its run ends, which would cut
    # this VM off mid-run, so the region is skipped (return 1) rather than shared
    local other
    other="$(gcloud compute routers list --filter="region~$region\$ AND name!=$ROUTER" --format='value(name)')"
    for r in $other; do
        if gcloud compute routers nats list --router="$r" --region="$region" --format='value(sourceSubnetworkIpRangesToNat)' \
            | grep -q ALL_SUBNETWORKS_ALL_IP_RANGES; then
            echo "skipping $region: router $r there has its own NAT"
            return 1
        fi
    done
    if ! gcloud compute routers describe "$ROUTER" --region="$region" >/dev/null 2>&1; then
        gcloud compute routers create "$ROUTER" --network=default --region="$region" >/dev/null
    fi
    if ! gcloud compute routers nats describe "$NAT" --router="$ROUTER" --region="$region" >/dev/null 2>&1; then
        gcloud compute routers nats create "$NAT" --router="$ROUTER" --region="$region" \
            --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges >/dev/null
    fi
}

init() {
    gcloud services enable compute.googleapis.com storage.googleapis.com iam.googleapis.com
    if ! gcloud iam service-accounts describe "$SA" >/dev/null 2>&1; then
        gcloud iam service-accounts create "$SA_NAME" --display-name="EventGPT training VM"
    fi
    if ! gcloud iam roles describe "$ROLE" --project="$PROJECT" >/dev/null 2>&1; then
        gcloud iam roles create "$ROLE" --project="$PROJECT" --title="Delete own VM" \
            --permissions=compute.instances.delete,compute.instances.get,compute.zoneOperations.get
    fi
    gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" \
        --role="projects/$PROJECT/roles/$ROLE" --condition=None >/dev/null
    if ! gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
        gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access
    fi
    gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$SA" \
        --role=roles/storage.objectAdmin >/dev/null
    gcloud storage rsync --recursive "$DATA/vocab" "gs://$BUCKET/data/vocab"
    gcloud storage rsync --recursive "$DATA/pretrain" "gs://$BUCKET/data/pretrain"
    gcloud storage cp "$DATA/test_fold.txt" "gs://$BUCKET/data/test_fold.txt"
    for k in 0 1 2 3 4; do
        gcloud storage rsync --recursive "$DATA/fold=$k" "gs://$BUCKET/data/fold=$k"
    done
    gcloud storage rsync --recursive "$RAW/targets" "gs://$BUCKET/raw/targets"
    echo "ready: gs://$BUCKET, service account $SA"
}

# -------------------------------------------------------------------------------
# Runs
# -------------------------------------------------------------------------------
push_code() {
    local tarball
    tarball="$(mktemp -d)/event-gpt.tar.gz"
    tar -czf "$tarball" -C "$ROOT" pyproject.toml uv.lock README.md src
    gcloud storage cp "$tarball" "gs://$BUCKET/runs/$1/code.tar.gz"
}

up() {
    local run="$1"
    shift
    local args="${*:-$DEFAULT_ARGS}"
    if [[ ! "$run" =~ ^[a-z0-9-]+$ ]]; then
        echo "run name must be lowercase letters, digits and dashes (it becomes a VM name and a label)" >&2
        exit 1
    fi
    # the project's GPU quota is one L4 across regions, shared with other projects' VMs
    local busy
    busy="$(gcloud compute instances list --filter='guestAccelerators:* AND status!=TERMINATED' --format='value(name)')"
    if [[ -n "$busy" ]]; then
        echo "a GPU VM is already running ($busy); the quota is one L4, so wait for it to finish" >&2
        exit 1
    fi
    push_code "$run"
    local provisioning=(--provisioning-model=STANDARD)
    if [[ "$SPOT" == "1" ]]; then
        provisioning=(--provisioning-model=SPOT)
    fi
    local err
    err="$(mktemp)"
    for zone in $ZONES; do
        echo "trying $zone"
        ensure_nat "${zone%-*}" || continue
        if gcloud compute instances create "egpt-$run" \
            --zone="$zone" \
            --machine-type="$MACHINE" \
            --accelerator=type=nvidia-l4,count=1 \
            --maintenance-policy=TERMINATE \
            "${provisioning[@]}" \
            --max-run-duration="${MAX_HOURS}h" \
            --instance-termination-action=DELETE \
            --image-project=deeplearning-platform-release \
            --image-family="$IMAGE_FAMILY" \
            --boot-disk-size=200GB \
            --no-address \
            --service-account="$SA" \
            --scopes=cloud-platform \
            --labels="$LABEL,run=$run" \
            --metadata="^;^install-nvidia-driver=True;bucket=$BUCKET;run=$run;probe=$PROBE;probe-args=$PROBE_ARGS;data-prefix=$DATA_PREFIX;train-args=$args" \
            --metadata-from-file=startup-script="$ROOT/scripts/gcp_startup.sh" 2>"$err"; then
            echo "started egpt-$run in $zone (deleted after training, or after ${MAX_HOURS}h at most)"
            echo "log: scripts/gcp.sh logs $run"
            return 0
        fi
        cat "$err" >&2
        # only a stockout is worth another zone; anything else fails the same way everywhere
        if ! grep -qE "ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources" "$err"; then
            drop_nats
            exit 1
        fi
    done
    echo "no zone in [$ZONES] had an L4 free" >&2
    drop_nats
    exit 1
}

logs() {
    gcloud storage cat "gs://$BUCKET/runs/$1/train.log"
}

status() {
    gcloud compute instances list --filter="labels.project=event-gpt" \
        --format="table(name,zone.basename(),status,labels.run,creationTimestamp.date('%Y-%m-%d %H:%M'))"
}

drop_nats() {
    # a NAT holds an external IP that is billed even with no VM behind it; up recreates them on demand
    gcloud compute routers list --filter="name=$ROUTER" --format="value(name,region.basename())" |
        while read -r name region; do
            gcloud compute routers delete "$name" --region="$region" --quiet
        done
}

down() {
    local rows
    rows="$(gcloud compute instances list --filter="labels.project=event-gpt" --format="value(name,zone.basename())")"
    while read -r name zone; do
        [[ -n "$name" ]] && gcloud compute instances delete "$name" --zone="$zone" --quiet
    done <<<"$rows"
    drop_nats
}

nuke() {
    down
    read -r -p "delete gs://$BUCKET with every checkpoint in it, the NAT, the service account and the role? [y/N] " ok
    [[ "$ok" == "y" ]] || exit 1
    gcloud storage rm --recursive "gs://$BUCKET"
    gcloud projects remove-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" \
        --role="projects/$PROJECT/roles/$ROLE" --condition=None >/dev/null
    gcloud iam service-accounts delete "$SA" --quiet
    gcloud iam roles delete "$ROLE" --project="$PROJECT" --quiet
}

cmd="${1:-}"
shift || true
case "$cmd" in
    init) init ;;
    up) up "$@" ;;
    logs) logs "$@" ;;
    status) status ;;
    down) down ;;
    nuke) nuke ;;
    *) sed -n '2,10p' "$0" ;;
esac
