"""CPU checks for review-only manifests and immutable-backup archive serving."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
import uuid
import zipfile
from pathlib import Path

import pytest

from macfit_training.service.backup import export_archive
from macfit_training.service.settings import Settings
from macfit_training.service.store import JobStore

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy/macfit-training"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, DEPLOY / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pull = load_script("backup_pull")
server = load_script("archive_server")
bootstrap = load_script("bootstrap")
renderer = load_script("render_cluster")


def test_archive_readiness_normalizes_configured_secret(monkeypatch):
    secret = "synthetic-gateway-secret-for-tests"
    monkeypatch.setenv("MACFIT_TRAINING_GATEWAY_SECRET", " \r\n" + secret + "\n")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps({"available": False, "auth": {"required": True}}).encode()

    def request_ready(request, *, timeout):
        assert request.get_header("X-training-gateway") == secret
        assert timeout == 2
        return Response()

    monkeypatch.setattr(server.urllib.request, "urlopen", request_ready)
    assert server.ready()


def test_render_is_digest_pinned_private_and_read_only_root_filesystems():
    document = renderer.render(ROOT)
    assert document["kind"] == "List"
    resources = {row["kind"]: row for row in document["items"] if row["kind"] != "ConfigMap"}
    assert "Secret" not in resources and "Route" not in resources
    pv = resources["PersistentVolume"]["spec"]
    assert pv["capacity"]["storage"] == "200Gi" and pv["persistentVolumeReclaimPolicy"] == "Retain"
    deployment = resources["Deployment"]["spec"]
    assert deployment["strategy"]["type"] == "Recreate"
    pod = deployment["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"]["runAsUser"] == 1000740000
    assert [container["name"] for container in pod["containers"]] == [
        "bridge",
        "ssh-tunnel",
        "archive",
    ]
    assert "readinessProbe" not in pod["containers"][-1]
    for container in pod["containers"] + pod["initContainers"]:
        assert "@sha256:" in container["image"]
        assert container["securityContext"]["readOnlyRootFilesystem"] is True
        assert container["securityContext"]["allowPrivilegeEscalation"] is False
    cron = resources["CronJob"]["spec"]
    assert cron["schedule"] == "*/10 * * * *" and cron["concurrencyPolicy"] == "Forbid"
    verify = cron["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]
    assert verify["command"][-2:] == ["--keep", "3"]
    for argument, value in (("--max-archive-gib", "25"), ("--max-unpacked-gib", "32")):
        assert verify["command"][verify["command"].index(argument) + 1] == value
    with pytest.raises(ValueError, match="pinned"):
        renderer.render(ROOT, python_image="python:latest")


def test_bridge_ingress_allows_only_same_namespace_frontend_on_http():
    document = renderer.render(ROOT, namespace="isolated-training")
    policies = [row for row in document["items"] if row["kind"] == "NetworkPolicy"]
    assert len(policies) == 1
    policy = policies[0]
    assert policy["apiVersion"] == "networking.k8s.io/v1"
    assert policy["metadata"] == {
        "name": "macfit-training-bridge",
        "namespace": "isolated-training",
    }
    # No namespaceSelector means the peer must be in this policy's namespace.
    # Exact comparison prevents extra sources, ports or egress isolation slipping in.
    assert policy["spec"] == {
        "podSelector": {"matchLabels": {"app": "macfit-training-bridge"}},
        "policyTypes": ["Ingress"],
        "ingress": [
            {
                "from": [{"podSelector": {"matchLabels": {"app": "mac-fit"}}}],
                "ports": [{"protocol": "TCP", "port": 8080}],
            }
        ],
    }
    deployment = next(row for row in document["items"] if row["kind"] == "Deployment")
    assert deployment["spec"]["template"]["metadata"]["labels"] == policy["spec"][
        "podSelector"
    ]["matchLabels"]


def test_source_bundle_roundtrip_and_hash_boundary(tmp_path):
    document = renderer.render(ROOT)
    cm = next(row for row in document["items"] if "binaryData" in row)
    encoded = cm["binaryData"]["source.zip"]
    assert len(encoded) < 900 * 1024
    raw = base64.b64decode(encoded)
    bundle = tmp_path / "source.zip"
    bundle.write_bytes(raw)
    destination = tmp_path / "unpacked"
    bootstrap.source(bundle, destination, hashlib.sha256(raw).hexdigest())
    relative = "src/macfit_training/service/api.py"
    assert (destination / relative).read_bytes() == (ROOT / relative).read_bytes()
    with pytest.raises(ValueError, match="digest"):
        bootstrap.source(bundle, tmp_path / "wrong", "a" * 64)
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("src/../../escape.py", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        bootstrap.source(
            bundle, tmp_path / "unsafe", hashlib.sha256(bundle.read_bytes()).hexdigest()
        )
    assert not (tmp_path / "escape.py").exists()


@pytest.mark.parametrize(
    "host", ["-oProxyCommand=bad", "host;command", "user@host", "host:22", "999.1.1.1"]
)
def test_gpu_host_rejects_non_host_values(host):
    with pytest.raises(ValueError):
        renderer.render(ROOT, gpu_host=host)


def test_deployment_coordinates_are_supplied_as_environment_not_script_literals():
    document = renderer.render(
        ROOT,
        gpu_host="gpu.example.net",
        node="example-node",
        uid=1000700001,
        firebase_project_id="example-project",
    )
    deployment = next(row for row in document["items"] if row["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    assert pod["securityContext"]["runAsUser"] == 1000700001
    assert pod["nodeSelector"] == {"kubernetes.io/hostname": "example-node"}
    ssh = next(row for row in pod["containers"] if row["name"] == "ssh-tunnel")
    assert {row["name"]: row.get("value") for row in ssh["env"]}[
        "MACFIT_GPU_HOST"
    ] == "gpu.example.net"
    archive = next(row for row in pod["containers"] if row["name"] == "archive")
    assert {row["name"]: row.get("value") for row in archive["env"]}[
        "MACFIT_FIREBASE_PROJECT_ID"
    ] == "example-project"


def small_bundle(tmp_path):
    bundle = tmp_path / "source.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("src/example.py", "value = 42\n")
    return bundle, hashlib.sha256(bundle.read_bytes()).hexdigest()


def test_source_retry_reuses_only_verified_publication(tmp_path):
    bundle, digest = small_bundle(tmp_path)
    destination = tmp_path / "unpacked"
    bootstrap.source(bundle, destination, digest)
    published = destination / "src/example.py"
    identity = (published.stat().st_ino, published.stat().st_mtime_ns)
    bootstrap.source(bundle, destination, digest)
    assert (published.stat().st_ino, published.stat().st_mtime_ns) == identity
    published.write_text("changed = True\n")
    with pytest.raises(ValueError, match="bytes changed"):
        bootstrap.source(bundle, destination, digest)
    assert published.read_text() == "changed = True\n"


def test_interrupted_source_unpack_is_not_published_and_retry_recovers(tmp_path, monkeypatch):
    bundle, digest = small_bundle(tmp_path)
    destination = tmp_path / "unpacked"
    original_open = Path.open

    def interrupted_open(path, *args, **kwargs):
        if path.name == ".source-manifest.json":
            raise OSError("Simulated interruption after source files were written")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", interrupted_open)
        with pytest.raises(OSError, match="interruption"):
            bootstrap.source(bundle, destination, digest)
    assert not destination.exists()
    # SIGKILL can bypass finally: the next init removes only its own exact staging names.
    abandoned = tmp_path / (".unpacked.pending-" + "a" * 32)
    abandoned.mkdir()
    (abandoned / "partial.py").write_text("incomplete")
    unrelated = tmp_path / ".unpacked.pending-do-not-delete"
    unrelated.mkdir()
    bootstrap.source(bundle, destination, digest)
    assert not abandoned.exists() and unrelated.exists()
    assert (destination / "src/example.py").read_text() == "value = 42\n"
    assert (destination / ".source-manifest.json").is_file()


@pytest.fixture
def archive(tmp_path):
    source = tmp_path / "source"
    store = JobStore(Settings(source, "synthetic-gateway-secret-for-tests"))
    store.create(
        "owner",
        str(uuid.uuid4()),
        str(uuid.uuid4()),
        "generation",
        {
            "base_model": {"id": "test"},
            "generation_config": {"walltime_seconds": 3600},
        },
    )
    payload = tmp_path / "snapshot.tgz"
    with payload.open("wb") as stream:
        export_archive(source, stream)
    root = tmp_path / "archive"
    receipt = pull.publish_archive(payload, root, keep=3)
    return root, receipt


def test_serving_copy_preserves_verified_snapshot_and_survives_pointer_loss(archive):
    root, receipt = archive
    original = (root / "latest/data/jobs.sqlite3").read_bytes()
    current, copied_receipt = server.prepare_copy(root)
    assert copied_receipt == receipt
    assert current.parent == root / "serving"
    JobStore(Settings(current, "synthetic-gateway-secret-for-tests", archive_only=True))
    assert (root / "latest/data/jobs.sqlite3").read_bytes() == original
    assert (current / "jobs.sqlite3").stat().st_ino != (
        root / "latest/data/jobs.sqlite3"
    ).stat().st_ino
    (root / "latest").unlink()
    assert server.previous_copy(root) == (current, receipt)


def test_archive_api_denies_mutations_and_requires_owner_authentication(archive, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    root, _ = archive
    data, _ = server.prepare_copy(root)
    token = "synthetic-gateway-secret-for-tests"
    monkeypatch.setenv("MACFIT_TRAINING_GATEWAY_SECRET", token)
    headers = {"X-Training-Gateway": token}
    before = (root / "latest/data/jobs.sqlite3").read_bytes()
    with TestClient(server.archive_app(data)) as client:
        cap = client.get("/api/training/capabilities", headers=headers)
        assert cap.status_code == 200 and cap.json()["available"] is False
        assert cap.headers["X-MacFit-Archive"] == "true"
        assert client.get("/api/training/jobs", headers=headers).status_code == 401
        assert client.get("/api/training/jobs").status_code == 403
        rejected = client.post("/api/training/jobs", headers=headers, json={})
        assert (
            rejected.status_code == 503 and rejected.json()["error"]["code"] == "archive_read_only"
        )
    assert (root / "latest/data/jobs.sqlite3").read_bytes() == before
    assert "torch" not in sys.modules
