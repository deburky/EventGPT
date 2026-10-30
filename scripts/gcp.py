"""Train EventGPT on a single L4 in Google Cloud, with nothing left running afterwards.

    uv run --group gcp scripts/gcp.py init                    service account, bucket, data upload (once)
    uv run --group gcp scripts/gcp.py up <run> [train args]   push the code and start a VM that trains, probes,
                                                              uploads and deletes itself; rerunning a run name
                                                              resumes from its last mirrored checkpoint
    uv run --group gcp scripts/gcp.py logs <run>              the run's training log, synced to the bucket every minute
    uv run --group gcp scripts/gcp.py boot <run>              startup progress from the VM's serial console
    uv run --group gcp scripts/gcp.py pull <run>              the run's checkpoint and log, into data/runs/<run>
    uv run --group gcp scripts/gcp.py status                  VMs carrying the event-gpt label
    uv run --group gcp scripts/gcp.py down                    delete every VM carrying the label, and the NAT gateways
    uv run --group gcp scripts/gcp.py nuke                    also delete the bucket, service account and role

Three things stop a VM from outliving its run: the startup script (scripts/gcp_startup.sh) deletes the VM when
training ends or fails, max_run_duration makes Google delete it after MAX_HOURS whatever happens, and `down`
deletes anything labeled project=event-gpt. Checkpoints and logs live in the bucket under runs/<run>/.

The project's organization policy forbids external IPs on VMs, so the VM gets none and reaches PyPI and the
uv installer through a Cloud NAT on the default network, created on demand by `up`.

Credentials come from Application Default Credentials (`gcloud auth application-default login`). Settings are
read from the environment: GCP_PROJECT, GCP_REGION, GCP_ZONES, GCP_BUCKET, GCP_MACHINE, MAX_HOURS, PROBE,
PROBE_ARGS, SPOT, DATA_PREFIX, MBD_OUT and MBD_ROOT.
"""

import io
import os
import re
import sys
import tarfile
from pathlib import Path

import google.auth
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import compute_v1, iam_admin_v1, resourcemanager_v3, service_usage_v1
from google.cloud.storage import Bucket, Client
from google.iam.v1 import policy_pb2

ROOT = Path(__file__).resolve().parents[1]


def default_project() -> str:
    """The project of Application Default Credentials, or "" without them."""
    try:
        return google.auth.default()[1] or ""
    except DefaultCredentialsError:
        return ""


PROJECT = os.environ.get("GCP_PROJECT") or default_project()
REGION = os.environ.get("GCP_REGION", "europe-west4")
# every European zone with L4s, tried in order until one has capacity
ZONES = os.environ.get(
    "GCP_ZONES",
    "europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c europe-west3-a europe-west3-b "
    "europe-west2-a europe-west2-b europe-west6-b europe-west6-c",
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
STOCKOUT = re.compile(r"ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources")


# -------------------------------------------------------------------------------
# Storage
# -------------------------------------------------------------------------------
def bucket() -> Bucket:
    """The run bucket."""
    return Client(project=PROJECT).bucket(BUCKET)


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
    routers = compute_v1.RoutersClient()
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
    usage = service_usage_v1.ServiceUsageClient()
    apis = ["compute.googleapis.com", "storage.googleapis.com", "iam.googleapis.com"]
    usage.batch_enable_services(request={"parent": f"projects/{PROJECT}", "service_ids": apis}).result()
    iam = iam_admin_v1.IAMClient()
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
    projects = resourcemanager_v3.ProjectsClient()
    policy = projects.get_iam_policy(resource=f"projects/{PROJECT}")
    if bind(policy, ROLE_NAME, f"serviceAccount:{SA}"):
        projects.set_iam_policy(request={"resource": f"projects/{PROJECT}", "policy": policy})
    client = Client(project=PROJECT)
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
    for _, scoped in compute_v1.InstancesClient().aggregated_list(project=PROJECT):
        for vm in scoped.instances:
            if vm.guest_accelerators and vm.status != "TERMINATED":
                out.append(vm.name)
    return out


def instance(run: str, zone: str, args: str) -> compute_v1.Instance:
    """The VM of a run in zone: one L4, no external IP, deleted after MAX_HOURS at most."""
    scheduling = compute_v1.Scheduling(
        on_host_maintenance="TERMINATE",
        provisioning_model="SPOT" if SPOT else "STANDARD",
        max_run_duration=compute_v1.Duration(seconds=MAX_HOURS * 3600),
        instance_termination_action="DELETE",
    )
    if SPOT:
        scheduling.automatic_restart = False
    metadata = {
        "install-nvidia-driver": "True",
        "bucket": BUCKET,
        "run": run,
        "probe": PROBE,
        "probe-args": PROBE_ARGS,
        "data-prefix": DATA_PREFIX,
        "train-args": args,
        "startup-script": (ROOT / "scripts/gcp_startup.sh").read_text(),
    }
    return compute_v1.Instance(
        name=f"egpt-{run}",
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
        labels={**LABELS, "run": run},
        metadata=compute_v1.Metadata(items=[compute_v1.Items(key=k, value=v) for k, v in metadata.items()]),
    )


def up(run: str, args: str) -> None:
    """Push the code and start the run's VM in the first zone with an L4 free."""
    if not re.fullmatch(r"[a-z0-9-]+", run):
        sys.exit("run name must be lowercase letters, digits and dashes (it becomes a VM name and a label)")
    busy = gpu_vms()
    if busy:
        sys.exit(f"a GPU VM is already running ({' '.join(busy)}); the quota is one L4, so wait for it to finish")
    push_code(run)
    client = compute_v1.InstancesClient()
    for zone in ZONES:
        print(f"trying {zone}")
        if not ensure_nat(zone.rsplit("-", 1)[0]):
            continue
        try:
            client.insert(project=PROJECT, zone=zone, instance_resource=instance(run, zone, args)).result()
        except GoogleAPICallError as e:
            print(e, file=sys.stderr)
            # only a stockout is worth another zone; anything else fails the same way everywhere
            if not STOCKOUT.search(str(e)):
                drop_nats()
                sys.exit(1)
            continue
        print(f"started egpt-{run} in {zone} (deleted after training, or after {MAX_HOURS}h at most)")
        print(f"log: uv run --group gcp scripts/gcp.py logs {run}")
        return
    drop_nats()
    sys.exit(f"no zone in [{' '.join(ZONES)}] had an L4 free")


def logs(run: str) -> None:
    """Print the run's training log."""
    print(bucket().blob(f"runs/{run}/train.log").download_as_text(), end="")


def boot(run: str) -> None:
    """Print the startup script's lines from the VM's serial console, before the training log exists."""
    client = compute_v1.InstancesClient()
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
    for _, scoped in compute_v1.InstancesClient().aggregated_list(request=request):
        out.extend(scoped.instances)
    return out


def status() -> None:
    """List the labeled VMs."""
    print(f"{'NAME':32} {'ZONE':16} {'STATUS':12} {'RUN':24} CREATED")
    for vm in labeled_vms():
        zone = vm.zone.rsplit("/", 1)[-1]
        print(f"{vm.name:32} {zone:16} {vm.status:12} {vm.labels.get('run', ''):24} {vm.creation_timestamp[:16]}")


def drop_nats() -> None:
    """Delete the routers; a NAT holds an external IP that is billed even with no VM behind it."""
    routers = compute_v1.RoutersClient()
    request = compute_v1.AggregatedListRoutersRequest(project=PROJECT, filter=f"name={ROUTER}")
    for scope, scoped in routers.aggregated_list(request=request):
        for r in scoped.routers:
            region = scope.rsplit("/", 1)[-1]
            routers.delete(project=PROJECT, region=region, router=r.name).result()
            print(f"deleted router {r.name} in {region}")


def down() -> None:
    """Delete every labeled VM and the NAT gateways."""
    client = compute_v1.InstancesClient()
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
    projects = resourcemanager_v3.ProjectsClient()
    policy = projects.get_iam_policy(resource=f"projects/{PROJECT}")
    for binding in policy.bindings:
        if binding.role == ROLE_NAME and f"serviceAccount:{SA}" in binding.members:
            binding.members.remove(f"serviceAccount:{SA}")
    projects.set_iam_policy(request={"resource": f"projects/{PROJECT}", "policy": policy})
    iam = iam_admin_v1.IAMClient()
    iam.delete_service_account(name=f"projects/{PROJECT}/serviceAccounts/{SA}")
    iam.delete_role(request=iam_admin_v1.DeleteRoleRequest(name=ROLE_NAME))


# -------------------------------------------------------------------------------
# Command line
# -------------------------------------------------------------------------------
def main() -> None:
    """Dispatch the subcommand."""
    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if not PROJECT:
        sys.exit("no project: set GCP_PROJECT or run `gcloud auth application-default login`")
    if cmd == "init":
        init()
    elif cmd == "up" and rest:
        up(rest[0], " ".join(rest[1:]) or DEFAULT_ARGS)
    elif cmd == "logs" and rest:
        logs(rest[0])
    elif cmd == "boot" and rest:
        boot(rest[0])
    elif cmd == "pull" and rest:
        pull(rest[0])
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
