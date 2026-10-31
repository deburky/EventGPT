"""Train EventGPT on a single L4 in Google Cloud, with nothing left running afterwards.

    uv run --group gcp scripts/gcp.py init                    service account, bucket, data upload (once)
    uv run --group gcp scripts/gcp.py up <run> [train args]   push the code and start a VM that trains, probes,
                                                              uploads and deletes itself; rerunning a run name
                                                              resumes from its last mirrored checkpoint
    uv run --group gcp scripts/gcp.py logs <run>              the run's training log, synced to the bucket every minute
    uv run --group gcp scripts/gcp.py boot <run>              startup progress from the VM's serial console
    uv run --group gcp scripts/gcp.py pull <run>              the run's checkpoint and log, into data/runs/<run>
    uv run --group gcp scripts/gcp.py wait <run>              block until the run's VM has deleted itself, then pull
    uv run --group gcp scripts/gcp.py queue <runs.toml>       start each run of the file in turn, retrying while no
                                                              zone has an L4 free, and pull each when its VM is gone
    uv run --group gcp scripts/gcp.py status                  VMs carrying the event-gpt label
    uv run --group gcp scripts/gcp.py down                    delete every VM carrying the label, and the NAT gateways
    uv run --group gcp scripts/gcp.py nuke                    also delete the bucket, service account and role

Three things stop a VM from outliving its run: the startup script (scripts/gcp_startup.sh) deletes the VM when
training ends or fails, max_run_duration makes Google delete it after MAX_HOURS whatever happens, and `down`
deletes anything labeled project=event-gpt. Checkpoints and logs live in the bucket under runs/<run>/.

The project's organization policy forbids external IPs on VMs, so the VM gets none and reaches PyPI and the
uv installer through a Cloud NAT on the default network, created on demand by `up`.

Credentials come from Application Default Credentials (`gcloud auth application-default login`), or, without
them, from the gcloud CLI's own login, whose access token is fetched again before it expires. Settings are
read from the environment: GCP_PROJECT, GCP_REGION, GCP_ZONES, GCP_BUCKET, GCP_MACHINE, MAX_HOURS, PROBE,
PROBE_ARGS, SPOT, DATA_PREFIX, MBD_OUT and MBD_ROOT.

A queue file lists runs as TOML tables; each takes the settings of `up` and falls back to the environment's:

    [[run]]
    name = "hier-rope-attn-s42"
    args = ""                  # training flags; empty means DEFAULT_ARGS
    probe = "only"
    probe_args = "--batch 32 --now --attn --out /opt/egpt/ckpt/probe-attn"
    max_hours = 4
    files = { "probe" = "bucket:runs/hier-rope-s42/ckpt/probe" }   # copied to runs/<name>/ckpt/<key> before launch

A files source is a local file or folder, or bucket:<prefix> for an object or folder already in the bucket, copied
there without a download.

With probe = "module", the VM runs `python -m <probe_args>` (an evaluation module and its flags) in place of training,
on the files copied in; /opt/egpt/ckpt is the run's checkpoint folder there.
"""

import io
import os
import re
import subprocess
import sys
import tarfile
import time
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import google.auth
import google.auth.credentials
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import compute_v1, iam_admin_v1, resourcemanager_v3, service_usage_v1
from google.cloud.storage import Bucket, Client
from google.iam.v1 import policy_pb2

ROOT = Path(__file__).resolve().parents[1]


# -------------------------------------------------------------------------------
# Credentials
# -------------------------------------------------------------------------------
def gcloud(*args: str) -> str:
    """Output of a gcloud CLI command, "" when gcloud is missing or fails."""
    try:
        out = subprocess.run(["gcloud", *args], capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


class GcloudCredentials(google.auth.credentials.Credentials):
    """The gcloud CLI's login, for machines without Application Default Credentials; refreshed every 45 minutes."""

    def refresh(self, request: Any) -> None:
        """Fetch a new access token from gcloud."""
        token = gcloud("auth", "print-access-token")
        if not token:
            sys.exit("no credentials: run `gcloud auth application-default login` (or `gcloud auth login`)")
        self.token = token
        # google.auth compares expiry with a naive UTC time
        self.expiry = (datetime.now(UTC) + timedelta(minutes=45)).replace(tzinfo=None)


def credentials() -> tuple[google.auth.credentials.Credentials, str]:
    """Application Default Credentials and their project, or the gcloud CLI's login and configured project."""
    try:
        creds, project = google.auth.default()
        return creds, project or gcloud("config", "get-value", "project")
    except DefaultCredentialsError:
        return GcloudCredentials(), gcloud("config", "get-value", "project")


CREDS, DEFAULT_PROJECT = credentials()
PROJECT = os.environ.get("GCP_PROJECT") or DEFAULT_PROJECT
REGION = os.environ.get("GCP_REGION", "europe-west4")
# every European zone with L4s, then the US ones, tried in order until one has capacity
ZONES = os.environ.get(
    "GCP_ZONES",
    "europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c europe-west3-a europe-west3-b "
    "europe-west2-a europe-west2-b europe-west6-b europe-west6-c "
    "us-central1-a us-central1-b us-central1-c us-east1-c us-east1-d us-east4-a us-east4-c us-west1-a us-west1-b",
).split()
BUCKET = os.environ.get("GCP_BUCKET", f"event-gpt-{PROJECT}")
DATA = Path(os.environ.get("MBD_OUT", ROOT / "data/MBD-mini/tokens"))
RAW = Path(os.environ.get("MBD_ROOT", ROOT / "data/MBD-mini/extracted"))
SA_NAME = "event-gpt-runner"
SA = f"{SA_NAME}@{PROJECT}.iam.gserviceaccount.com"
ROLE = "eventGptSelfDelete"
ROLE_NAME = f"projects/{PROJECT}/roles/{ROLE}"
ROUTER = "event-gpt-router"
NAT = "event-gpt-nat"
MACHINE = os.environ.get("GCP_MACHINE", "g2-standard-8")
IMAGE = "projects/deeplearning-platform-release/global/images/family/common-cu129-ubuntu-2204-nvidia-580"
MAX_HOURS = int(os.environ.get("MAX_HOURS", "6"))
PROBE = os.environ.get("PROBE", "1")  # 1 probes the final checkpoint before the VM deletes itself; only skips training
PROBE_ARGS = os.environ.get("PROBE_ARGS", "--batch 32")  # flags for eventgpt.evaluation.probe
SPOT = os.environ.get("SPOT", "0") == "1"
DATA_PREFIX = os.environ.get("DATA_PREFIX", "data")  # bucket folder of the token streams the VM trains and probes on
LABELS = {"project": "event-gpt"}
DEFAULT_ARGS = "--arch eventgpt --rope --steps 20000 --batch 8 --accum 2"
# no L4 free in the zone, or the project's one-GPU quota taken by a VM started a moment earlier
STOCKOUT = re.compile(r"ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|QUOTA_EXCEEDED|GPUS_ALL_REGIONS")
RETRY_SECONDS = 300
POLL_SECONDS = 120


@dataclass
class Run:
    """One run's name, training flags and VM settings; the defaults come from the environment."""

    name: str
    args: str = DEFAULT_ARGS
    probe: str = PROBE
    probe_args: str = PROBE_ARGS
    max_hours: int = MAX_HOURS
    spot: bool = SPOT
    data_prefix: str = DATA_PREFIX
    files: dict[str, str] = field(default_factory=dict)


# -------------------------------------------------------------------------------
# Storage
# -------------------------------------------------------------------------------
def bucket() -> Bucket:
    """The run bucket."""
    return Client(project=PROJECT, credentials=CREDS).bucket(BUCKET)


def upload_tree(local: Path, prefix: str) -> None:
    """Upload every file under local to prefix/, skipping objects that already exist with the same size."""
    b = bucket()
    have = {blob.name: blob.size for blob in b.list_blobs(prefix=f"{prefix}/")}
    files = [p for p in sorted(local.rglob("*")) if p.is_file()]
    sent = 0
    for p in files:
        name = f"{prefix}/{p.relative_to(local).as_posix()}"
        if have.get(name) != p.stat().st_size:
            b.blob(name).upload_from_filename(str(p))
            sent += 1
    print(f"{prefix}: {sent} of {len(files)} files uploaded")


def copy_in_bucket(src: str, dest: str) -> None:
    """Copy the object src, or every object under the folder src, to dest inside the bucket."""
    b = bucket()
    one = b.get_blob(src)
    if one is not None:
        b.copy_blob(one, b, dest)
        return
    blobs = list(b.list_blobs(prefix=f"{src.rstrip('/')}/"))
    if not blobs:
        sys.exit(f"nothing at gs://{BUCKET}/{src}")
    for blob in blobs:
        b.copy_blob(blob, b, dest + blob.name.removeprefix(src.rstrip("/")))
    print(f"{src}: {len(blobs)} objects copied to {dest}")


# -------------------------------------------------------------------------------
# One-time setup
# -------------------------------------------------------------------------------
def ensure_nat(region: str) -> bool:
    """Create the router and NAT gateway in region; False when another project's NAT already covers it.

    Cloud NAT is regional, so every region a VM lands in needs its own router and gateway. Another project's
    gateway may already cover every subnet in the region. GCP allows one such NAT per network and region, and
    the other project deletes its own when its run ends, which would cut this VM off mid-run, so the region is
    skipped rather than shared.
    """
    routers = compute_v1.RoutersClient(credentials=CREDS)
    ours = None
    for r in routers.list(project=PROJECT, region=region):
        if r.name == ROUTER:
            ours = r
        elif any(n.source_subnetwork_ip_ranges_to_nat == "ALL_SUBNETWORKS_ALL_IP_RANGES" for n in r.nats):
            print(f"skipping {region}: router {r.name} there has its own NAT")
            return False
    nat = compute_v1.RouterNat(
        name=NAT,
        nat_ip_allocate_option="AUTO_ONLY",
        source_subnetwork_ip_ranges_to_nat="ALL_SUBNETWORKS_ALL_IP_RANGES",
    )
    if ours is None:
        router = compute_v1.Router(name=ROUTER, network="global/networks/default", nats=[nat])
        routers.insert(project=PROJECT, region=region, router_resource=router).result()
    elif not any(n.name == NAT for n in ours.nats):
        ours.nats.append(nat)
        routers.patch(project=PROJECT, region=region, router=ROUTER, router_resource=ours).result()
    return True


def bind(policy: policy_pb2.Policy, role: str, member: str) -> bool:
    """Add member to role in policy; False when the binding was already there."""
    for b in policy.bindings:
        if b.role == role and not b.HasField("condition"):
            if member in b.members:
                return False
            b.members.append(member)
            return True
    policy.bindings.add(role=role, members=[member])
    return True


def init() -> None:
    """Enable the APIs and create the service account, its role, the bucket and the data upload."""
    usage = service_usage_v1.ServiceUsageClient(credentials=CREDS)
    apis = ["compute.googleapis.com", "storage.googleapis.com", "iam.googleapis.com"]
    usage.batch_enable_services(request={"parent": f"projects/{PROJECT}", "service_ids": apis}).result()
    iam = iam_admin_v1.IAMClient(credentials=CREDS)
    try:
        iam.get_service_account(name=f"projects/{PROJECT}/serviceAccounts/{SA}")
    except NotFound:
        account = iam_admin_v1.ServiceAccount(display_name="EventGPT training VM")
        iam.create_service_account(name=f"projects/{PROJECT}", account_id=SA_NAME, service_account=account)
    try:
        iam.get_role(request=iam_admin_v1.GetRoleRequest(name=ROLE_NAME))
    except NotFound:
        role = iam_admin_v1.Role(
            title="Delete own VM",
            included_permissions=["compute.instances.delete", "compute.instances.get", "compute.zoneOperations.get"],
        )
        iam.create_role(request=iam_admin_v1.CreateRoleRequest(parent=f"projects/{PROJECT}", role_id=ROLE, role=role))
    projects = resourcemanager_v3.ProjectsClient(credentials=CREDS)
    policy = projects.get_iam_policy(resource=f"projects/{PROJECT}")
    if bind(policy, ROLE_NAME, f"serviceAccount:{SA}"):
        projects.set_iam_policy(request={"resource": f"projects/{PROJECT}", "policy": policy})
    client = Client(project=PROJECT, credentials=CREDS)
    b = client.bucket(BUCKET)
    if not b.exists():
        b.iam_configuration.uniform_bucket_level_access_enabled = True
        b = client.create_bucket(b, location=REGION)
    bucket_policy = b.get_iam_policy(requested_policy_version=3)
    member = f"serviceAccount:{SA}"
    if not any(x["role"] == "roles/storage.objectAdmin" and member in x["members"] for x in bucket_policy.bindings):
        bucket_policy.bindings.append({"role": "roles/storage.objectAdmin", "members": {member}})
        b.set_iam_policy(bucket_policy)
    upload_tree(DATA / "vocab", "data/vocab")
    upload_tree(DATA / "pretrain", "data/pretrain")
    b.blob("data/test_fold.txt").upload_from_filename(str(DATA / "test_fold.txt"))
    for k in range(5):
        upload_tree(DATA / f"fold={k}", f"data/fold={k}")
    upload_tree(RAW / "targets", "raw/targets")
    print(f"ready: gs://{BUCKET}, service account {SA}")


# -------------------------------------------------------------------------------
# Runs
# -------------------------------------------------------------------------------
def push_code(run: str) -> None:
    """Upload the package sources, which the VM installs with uv."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("pyproject.toml", "uv.lock", "README.md", "src"):
            tar.add(ROOT / name, arcname=name)
    bucket().blob(f"runs/{run}/code.tar.gz").upload_from_string(buf.getvalue())


def gpu_vms() -> list[str]:
    """Names of the project's running VMs with an accelerator; the quota is one L4, shared with other projects."""
    out = []
    for _, scoped in compute_v1.InstancesClient(credentials=CREDS).aggregated_list(project=PROJECT):
        for vm in scoped.instances:
            if vm.guest_accelerators and vm.status != "TERMINATED":
                out.append(vm.name)
    return out


def instance(run: Run, zone: str) -> compute_v1.Instance:
    """The VM of a run in zone: one L4, no external IP, deleted after max_hours at most."""
    scheduling = compute_v1.Scheduling(
        on_host_maintenance="TERMINATE",
        provisioning_model="SPOT" if run.spot else "STANDARD",
        max_run_duration=compute_v1.Duration(seconds=run.max_hours * 3600),
        instance_termination_action="DELETE",
    )
    if run.spot:
        scheduling.automatic_restart = False
    metadata = {
        "install-nvidia-driver": "True",
        "bucket": BUCKET,
        "run": run.name,
        "probe": run.probe,
        "probe-args": run.probe_args,
        "data-prefix": run.data_prefix,
        "train-args": run.args,
        "startup-script": (ROOT / "scripts/gcp_startup.sh").read_text(),
    }
    return compute_v1.Instance(
        name=f"egpt-{run.name}",
        machine_type=f"zones/{zone}/machineTypes/{MACHINE}",
        guest_accelerators=[compute_v1.AcceleratorConfig(accelerator_type=f"zones/{zone}/acceleratorTypes/nvidia-l4", accelerator_count=1)],
        scheduling=scheduling,
        disks=[
            compute_v1.AttachedDisk(
                boot=True,
                auto_delete=True,
                initialize_params=compute_v1.AttachedDiskInitializeParams(source_image=IMAGE, disk_size_gb=200),
            )
        ],
        network_interfaces=[compute_v1.NetworkInterface(network="global/networks/default")],
        service_accounts=[compute_v1.ServiceAccount(email=SA, scopes=["https://www.googleapis.com/auth/cloud-platform"])],
        labels={**LABELS, "run": run.name},
        metadata=compute_v1.Metadata(items=[compute_v1.Items(key=k, value=v) for k, v in metadata.items()]),
    )


def start(run: Run) -> bool:
    """Push the code and start the run's VM in the first zone with an L4 free; False when the L4 is taken or every zone is out."""
    if not re.fullmatch(r"[a-z0-9-]+", run.name):
        sys.exit("run name must be lowercase letters, digits and dashes (it becomes a VM name and a label)")
    busy = gpu_vms()
    if busy:
        print(f"a GPU VM is already running ({' '.join(busy)}); the quota is one L4")
        return False
    push_code(run.name)
    for rel, src in run.files.items():
        dest = f"runs/{run.name}/ckpt/{rel}"
        if src.startswith("bucket:"):
            copy_in_bucket(src.removeprefix("bucket:"), dest)
        elif (ROOT / src).is_dir():
            upload_tree(ROOT / src, dest)
        else:
            bucket().blob(dest).upload_from_filename(str(ROOT / src))
    client = compute_v1.InstancesClient(credentials=CREDS)
    for zone in ZONES:
        print(f"trying {zone}")
        if not ensure_nat(zone.rsplit("-", 1)[0]):
            continue
        try:
            client.insert(project=PROJECT, zone=zone, instance_resource=instance(run, zone)).result()
        except GoogleAPICallError as e:
            # only a stockout is worth another zone; anything else fails the same way everywhere
            if not STOCKOUT.search(str(e)):
                print(e, file=sys.stderr)
                drop_nats(force=False)
                sys.exit(1)
            continue
        print(f"started egpt-{run.name} in {zone} (deleted after training, or after {run.max_hours}h at most)")
        print(f"log: uv run --group gcp scripts/gcp.py logs {run.name}")
        return True
    drop_nats(force=False)
    print(f"no zone in [{' '.join(ZONES)}] had an L4 free")
    return False


def up(run: Run, retry: bool = False) -> None:
    """Start the run; with retry, try every zone again each RETRY_SECONDS until one has an L4 free."""
    while not start(run):
        if not retry:
            sys.exit(1)
        print(f"{time.strftime('%H:%M')} {run.name}: retrying in {RETRY_SECONDS // 60} min", flush=True)
        time.sleep(RETRY_SECONDS)


def wait(name: str) -> None:
    """Block until the run's VM has deleted itself, then pull its files."""
    while any(vm.name == f"egpt-{name}" for vm in labeled_vms()):
        time.sleep(POLL_SECONDS)
    print(f"{time.strftime('%H:%M')} {name}: VM gone", flush=True)
    pull(name)


def queue(path: Path) -> None:
    """Run every [[run]] of a TOML file in turn on the one L4."""
    runs = [Run(**spec) for spec in tomllib.loads(path.read_text())["run"]]
    for run in runs:
        run.args = run.args or DEFAULT_ARGS
        up(run, retry=True)
        wait(run.name)
    print(f"{time.strftime('%H:%M')} queue done: {' '.join(r.name for r in runs)}", flush=True)


def logs(run: str) -> None:
    """Print the run's training log."""
    print(bucket().blob(f"runs/{run}/train.log").download_as_text(), end="")


def boot(run: str) -> None:
    """Print the startup script's lines from the VM's serial console, before the training log exists."""
    client = compute_v1.InstancesClient(credentials=CREDS)
    vm = next((v for v in labeled_vms() if v.name == f"egpt-{run}"), None)
    if vm is None:
        sys.exit(f"no VM egpt-{run}")
    zone = vm.zone.rsplit("/", 1)[-1]
    out = client.get_serial_port_output(project=PROJECT, zone=zone, instance=vm.name).contents
    print("\n".join([line for line in out.splitlines() if "startup-script" in line][-30:]))


def pull(run: str) -> None:
    """Download the run's checkpoint folder and log into data/runs/<run>, skipping files already there."""
    dest = ROOT / "data/runs" / run
    got = 0
    for blob in bucket().list_blobs(prefix=f"runs/{run}/"):
        path = dest / blob.name.removeprefix(f"runs/{run}/")
        if path.exists() and path.stat().st_size == blob.size:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(path))
        got += 1
    print(f"{got} files to {dest}")


def labeled_vms() -> list[compute_v1.Instance]:
    """The project's VMs carrying the event-gpt label."""
    out: list[compute_v1.Instance] = []
    request = compute_v1.AggregatedListInstancesRequest(project=PROJECT, filter="labels.project=event-gpt")
    for _, scoped in compute_v1.InstancesClient(credentials=CREDS).aggregated_list(request=request):
        out.extend(scoped.instances)
    return out


def status() -> None:
    """List the labeled VMs."""
    print(f"{'NAME':32} {'ZONE':16} {'STATUS':12} {'RUN':24} CREATED")
    for vm in labeled_vms():
        zone = vm.zone.rsplit("/", 1)[-1]
        print(f"{vm.name:32} {zone:16} {vm.status:12} {vm.labels.get('run', ''):24} {vm.creation_timestamp[:16]}")


def drop_nats(force: bool = True) -> None:
    """Delete the routers; a NAT holds an external IP that is billed even with no VM behind it.

    Without force the routers stay while any GPU VM exists, since a run started by another queue a moment ago
    may still be installing its packages through them.
    """
    if not force and gpu_vms():
        return
    routers = compute_v1.RoutersClient(credentials=CREDS)
    request = compute_v1.AggregatedListRoutersRequest(project=PROJECT, filter=f"name={ROUTER}")
    for scope, scoped in routers.aggregated_list(request=request):
        for r in scoped.routers:
            region = scope.rsplit("/", 1)[-1]
            routers.delete(project=PROJECT, region=region, router=r.name).result()
            print(f"deleted router {r.name} in {region}")


def down() -> None:
    """Delete every labeled VM and the NAT gateways."""
    client = compute_v1.InstancesClient(credentials=CREDS)
    for vm in labeled_vms():
        zone = vm.zone.rsplit("/", 1)[-1]
        client.delete(project=PROJECT, zone=zone, instance=vm.name).result()
        print(f"deleted {vm.name} in {zone}")
    drop_nats()


def nuke() -> None:
    """Also delete the bucket with every checkpoint, the service account and the role."""
    down()
    ok = input(f"delete gs://{BUCKET} with every checkpoint in it, the NAT, the service account and the role? [y/N] ")
    if ok != "y":
        sys.exit(1)
    b = bucket()
    for blob in b.list_blobs():
        blob.delete()
    b.delete()
    projects = resourcemanager_v3.ProjectsClient(credentials=CREDS)
    policy = projects.get_iam_policy(resource=f"projects/{PROJECT}")
    for binding in policy.bindings:
        if binding.role == ROLE_NAME and f"serviceAccount:{SA}" in binding.members:
            binding.members.remove(f"serviceAccount:{SA}")
    projects.set_iam_policy(request={"resource": f"projects/{PROJECT}", "policy": policy})
    iam = iam_admin_v1.IAMClient(credentials=CREDS)
    iam.delete_service_account(name=f"projects/{PROJECT}/serviceAccounts/{SA}")
    iam.delete_role(request=iam_admin_v1.DeleteRoleRequest(name=ROLE_NAME))


# -------------------------------------------------------------------------------
# Command line
# -------------------------------------------------------------------------------
def main() -> None:
    """Dispatch the subcommand."""
    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if not PROJECT:
        sys.exit("no project: set GCP_PROJECT or `gcloud config set project <id>`")
    if cmd == "init":
        init()
    elif cmd == "up" and rest:
        up(Run(rest[0], " ".join(rest[1:]) or DEFAULT_ARGS))
    elif cmd == "logs" and rest:
        logs(rest[0])
    elif cmd == "boot" and rest:
        boot(rest[0])
    elif cmd == "pull" and rest:
        pull(rest[0])
    elif cmd == "wait" and rest:
        wait(rest[0])
    elif cmd == "queue" and rest:
        queue(Path(rest[0]))
    elif cmd == "status":
        status()
    elif cmd == "down":
        down()
    elif cmd == "nuke":
        nuke()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
