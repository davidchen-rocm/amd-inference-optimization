#!/usr/bin/env python3
"""Render a reviewable Kubernetes List; this program never calls oc or the cluster API."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import ipaddress
import json
import re
import zipfile
from pathlib import Path

PYTHON_IMAGE = (
    "docker.io/library/python@sha256:"
    "782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254"
)
SSH_IMAGE = (
    "docker.io/alpine/git@sha256:c0280cf9572316299b08544065d3bf35db65043d5e3963982ec50647d2746e26"
)
UID = 1000740000
NODE = "example-openshift-node"
NAME = "macfit-training"
PVC = NAME + "-backups"
SCRIPTS = (
    "bridge.py",
    "tunnel.sh",
    "backup_pull.py",
    "backup_fetch.sh",
    "bootstrap.py",
    "archive_server.py",
    "archive-requirements.txt",
)


def bundle_source(repo: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted((repo / "src").rglob("*.py")):
            if source.is_symlink():
                raise ValueError("Source files must not be symlinks")
            info = zipfile.ZipInfo(
                source.relative_to(repo).as_posix(), date_time=(2026, 1, 1, 0, 0, 0)
            )
            info.external_attr = 0o100600 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source.read_bytes())
    return buffer.getvalue()


def render(
    repo: Path,
    namespace: str = "mac-fit",
    *,
    python_image=PYTHON_IMAGE,
    ssh_image=SSH_IMAGE,
    gpu_host: str = "203.0.113.10",
    node: str = NODE,
    uid: int = UID,
    firebase_project_id: str = "macfit-example",
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", namespace):
        raise ValueError("Invalid namespace")
    if re.fullmatch(r"[0-9.]+", gpu_host):
        ipaddress.IPv4Address(gpu_host)
    elif len(gpu_host) > 253 or any(
        not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
        for label in gpu_host.split(".")
    ):
        raise ValueError("GPU host must be an IPv4 address or DNS hostname")
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,61}[a-z0-9])?", node):
        raise ValueError("Invalid node hostname label")
    if type(uid) is not int or not 1 <= uid < 2**31:
        raise ValueError("A non-root namespace UID is required")
    if not re.fullmatch(r"[a-z][a-z0-9-]{3,100}", firebase_project_id):
        raise ValueError("Invalid Firebase project ID")
    if any(
        not re.fullmatch(r"[A-Za-z0-9./_-]+@sha256:[a-f0-9]{64}", image)
        for image in (python_image, ssh_image)
    ):
        raise ValueError("Both images must be pinned by immutable SHA-256 digest")
    deployment = repo / "deploy/macfit-training"
    scripts = {name: (deployment / name).read_text() for name in SCRIPTS}
    scripts["passwd"] = (
        "root:x:0:0:root:/root:/bin/sh\n"
        f"macfit:x:{uid}:{uid}:MacFit service:/tmp:/bin/sh\n"
        "nobody:x:65534:65534:nobody:/nonexistent:/bin/false\n"
    )
    source = bundle_source(repo)
    digest = hashlib.sha256(source).hexdigest()
    source_cm = NAME + "-source-" + digest[:12]
    scripts_digest = hashlib.sha256(json.dumps(scripts, sort_keys=True).encode()).hexdigest()
    scripts_cm = NAME + "-scripts-" + scripts_digest[:12]
    encoded = base64.b64encode(source).decode()
    if len(encoded) > 900 * 1024 or len(json.dumps(scripts).encode()) > 900 * 1024:
        raise ValueError("Source or scripts exceed the safe ConfigMap budget")

    def metadata(name, *, cluster=False, labels=None):
        return {
            "name": name,
            **({} if cluster else {"namespace": namespace}),
            **({"labels": labels} if labels else {}),
        }

    def mount(name, path, **kwargs):
        return {"name": name, "mountPath": path, **kwargs}

    def cm_volume(name, config):
        return {"name": name, "configMap": {"name": config, "defaultMode": 0o444}}

    def secret_volume(name, keys):
        return {
            "name": name,
            "secret": {
                "secretName": NAME + "-connection",
                "defaultMode": 0o444,
                "items": [{"key": key, "path": path} for key, path in keys],
            },
        }

    container_security = {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
        "runAsNonRoot": True,
    }
    pod_security = {
        "runAsNonRoot": True,
        "runAsUser": uid,
        "runAsGroup": uid,
        "fsGroup": uid,
        "fsGroupChangePolicy": "OnRootMismatch",
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    common_env = [
        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "HOME", "value": "/tmp"},
    ]
    base_mounts = [
        mount("scripts", "/opt/training", readOnly=True),
        mount("source", "/work/source", readOnly=True),
    ]
    volumes = [
        cm_volume("scripts", scripts_cm),
        cm_volume("bundle", source_cm),
        {"name": "source", "emptyDir": {"sizeLimit": "32Mi"}},
        {"name": "archive", "persistentVolumeClaim": {"claimName": PVC}},
    ]

    def container(name, image, command, mounts, *, env=None, memory="256Mi", cpu="250m"):
        return {
            "name": name,
            "image": image,
            "imagePullPolicy": "IfNotPresent",
            "command": command,
            "securityContext": container_security,
            "volumeMounts": mounts,
            "env": [*common_env, *(env or [])],
            "resources": {
                "requests": {"cpu": "25m", "memory": "32Mi"},
                "limits": {"cpu": cpu, "memory": memory},
            },
        }

    source_init = container(
        "unpack-source",
        python_image,
        [
            "python",
            "/opt/training/bootstrap.py",
            "source",
            "--bundle",
            "/opt/bundle/source.zip",
            "--destination",
            "/work/source/unpacked",
            "--sha256",
            digest,
        ],
        [
            mount("scripts", "/opt/training", readOnly=True),
            mount("bundle", "/opt/bundle", readOnly=True),
            mount("source", "/work/source"),
            mount("python-tmp", "/tmp"),
        ],
    )
    python_path = "/work/source/unpacked/src"
    python_env = [{"name": "PYTHONPATH", "value": python_path}]
    vendor_init = container(
        "prepare-cpu-runtime",
        python_image,
        [
            "python",
            "/opt/training/bootstrap.py",
            "vendor",
            "--requirements",
            "/opt/training/archive-requirements.txt",
            "--root",
            "/archive/vendor",
        ],
        [*base_mounts, mount("archive", "/archive"), mount("python-tmp", "/tmp")],
        env=python_env,
        memory="1Gi",
        cpu="1",
    )
    bridge = container(
        "bridge",
        python_image,
        ["python", "/opt/training/bridge.py"],
        [
            mount("scripts", "/opt/training", readOnly=True),
            mount("gateway", "/run/training-secrets", readOnly=True),
            mount("bridge-tmp", "/tmp"),
        ],
        env=[
            {"name": "TRAINING_ARCHIVE_HOST", "value": "127.0.0.1"},
            {"name": "TRAINING_ARCHIVE_PORT", "value": "8793"},
        ],
        memory="128Mi",
        cpu="250m",
    )
    bridge.update(
        ports=[{"name": "http", "containerPort": 8080}],
        readinessProbe={
            "httpGet": {"path": "/healthz", "port": "http"},
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "failureThreshold": 3,
        },
        livenessProbe={
            "tcpSocket": {"port": "http"},
            "initialDelaySeconds": 10,
            "periodSeconds": 20,
            "timeoutSeconds": 2,
            "failureThreshold": 3,
        },
    )
    ssh = container(
        "ssh-tunnel",
        ssh_image,
        ["/bin/sh", "/opt/training/tunnel.sh"],
        [
            mount("scripts", "/opt/training", readOnly=True),
            mount("scripts", "/etc/passwd", subPath="passwd", readOnly=True),
            mount("ssh-secrets", "/run/training-secrets", readOnly=True),
            mount("ssh-tmp", "/tmp"),
        ],
        env=[{"name": "MACFIT_GPU_HOST", "value": gpu_host}],
        memory="64Mi",
        cpu="100m",
    )
    archive = container(
        "archive",
        python_image,
        ["python", "/opt/training/archive_server.py", "watch"],
        [*base_mounts, mount("archive", "/archive"), mount("python-tmp", "/tmp")],
        env=[
            {"name": "PYTHONPATH", "value": python_path + ":/archive/vendor/current"},
            {"name": "MACFIT_ARCHIVE_ONLY", "value": "1"},
            {"name": "MACFIT_FIREBASE_PROJECT_ID", "value": firebase_project_id},
            {
                "name": "MACFIT_TRAINING_GATEWAY_SECRET",
                "valueFrom": {
                    "secretKeyRef": {"name": NAME + "-connection", "key": "gateway-token"}
                },
            },
        ],
        memory="1Gi",
        cpu="1",
    )
    archive["ports"] = [{"name": "archive", "containerPort": 8793}]
    # Waiting for the initial backup must not remove the healthy GPU bridge.
    bridge_volumes = [
        *volumes,
        secret_volume("gateway", [("gateway-token", "gateway-token")]),
        secret_volume("ssh-secrets", [("bridge-key", "ssh-key"), ("known-hosts", "known-hosts")]),
        *[
            {"name": name, "emptyDir": {"sizeLimit": "256Mi"}}
            for name in ("python-tmp", "bridge-tmp", "ssh-tmp")
        ],
    ]
    labels = {"app": NAME + "-bridge"}
    pod = {
        "serviceAccountName": NAME,
        "automountServiceAccountToken": False,
        "securityContext": pod_security,
        "nodeSelector": {"kubernetes.io/hostname": node},
        "terminationGracePeriodSeconds": 40,
        "initContainers": [source_init, vendor_init],
        "containers": [bridge, ssh, archive],
        "volumes": bridge_volumes,
    }
    fetch = container(
        "fetch-private-snapshot",
        ssh_image,
        ["/bin/sh", "/opt/training/backup_fetch.sh"],
        [
            mount("scripts", "/opt/training", readOnly=True),
            mount("scripts", "/etc/passwd", subPath="passwd", readOnly=True),
            mount("backup-secrets", "/run/backup-secrets", readOnly=True),
            mount("ssh-tmp", "/tmp"),
            mount("archive", "/archive"),
        ],
        env=[{"name": "MACFIT_GPU_HOST", "value": gpu_host}],
        memory="128Mi",
        cpu="250m",
    )
    verify = container(
        "verify-and-publish",
        python_image,
        [
            "python",
            "/opt/training/backup_pull.py",
            "verify",
            "--archive",
            "/archive/incoming/snapshot.tgz",
            "--archive-root",
            "/archive",
            "--max-archive-gib",
            "25",
            "--max-unpacked-gib",
            "32",
            "--keep",
            "3",
        ],
        [*base_mounts, mount("archive", "/archive"), mount("python-tmp", "/tmp")],
        env=python_env,
        memory="512Mi",
        cpu="1",
    )
    backup_pod = {
        "serviceAccountName": NAME,
        "automountServiceAccountToken": False,
        "restartPolicy": "Never",
        "securityContext": pod_security,
        "nodeSelector": {"kubernetes.io/hostname": node},
        "terminationGracePeriodSeconds": 30,
        "initContainers": [source_init, fetch],
        "containers": [verify],
        "volumes": [
            *volumes,
            secret_volume(
                "backup-secrets", [("backup-key", "backup-key"), ("known-hosts", "known-hosts")]
            ),
            {"name": "python-tmp", "emptyDir": {"sizeLimit": "256Mi"}},
            {"name": "ssh-tmp", "emptyDir": {"sizeLimit": "32Mi"}},
        ],
    }
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "ServiceAccount",
                "metadata": metadata(NAME),
                "automountServiceAccountToken": False,
            },
            {
                "apiVersion": "v1",
                "kind": "PersistentVolume",
                "metadata": metadata(PVC, cluster=True),
                "spec": {
                    "capacity": {"storage": "200Gi"},
                    "volumeMode": "Filesystem",
                    "accessModes": ["ReadWriteOnce"],
                    "persistentVolumeReclaimPolicy": "Retain",
                    "storageClassName": "",
                    "claimRef": {"namespace": namespace, "name": PVC},
                    "local": {"path": "/var/mnt/macfit-training-backups"},
                    "nodeAffinity": {
                        "required": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": "kubernetes.io/hostname",
                                            "operator": "In",
                                            "values": [node],
                                        }
                                    ]
                                }
                            ]
                        }
                    },
                },
            },
            {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": metadata(PVC),
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "volumeMode": "Filesystem",
                    "storageClassName": "",
                    "volumeName": PVC,
                    "resources": {"requests": {"storage": "200Gi"}},
                },
            },
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": metadata(source_cm),
                "immutable": True,
                "binaryData": {"source.zip": encoded},
            },
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": metadata(scripts_cm),
                "immutable": True,
                "data": scripts,
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": metadata(NAME + "-bridge", labels=labels),
                "spec": {
                    "replicas": 1,
                    "strategy": {"type": "Recreate"},
                    "selector": {"matchLabels": labels},
                    "template": {"metadata": {"labels": labels}, "spec": pod},
                },
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": metadata(NAME + "-bridge"),
                "spec": {
                    "type": "ClusterIP",
                    "selector": labels,
                    "ports": [{"name": "http", "port": 8080, "targetPort": "http"}],
                },
            },
            {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": metadata(NAME + "-bridge"),
                "spec": {
                    "podSelector": {"matchLabels": labels},
                    "policyTypes": ["Ingress"],
                    "ingress": [
                        {
                            "from": [{"podSelector": {"matchLabels": {"app": "mac-fit"}}}],
                            "ports": [{"protocol": "TCP", "port": 8080}],
                        }
                    ],
                },
            },
            {
                "apiVersion": "batch/v1",
                "kind": "CronJob",
                "metadata": metadata(NAME + "-backup"),
                "spec": {
                    "schedule": "*/10 * * * *",
                    "concurrencyPolicy": "Forbid",
                    "startingDeadlineSeconds": 180,
                    "successfulJobsHistoryLimit": 2,
                    "failedJobsHistoryLimit": 3,
                    "jobTemplate": {
                        "spec": {
                            "backoffLimit": 1,
                            "activeDeadlineSeconds": 1200,
                            "template": {
                                "metadata": {"labels": {"app": NAME + "-backup"}},
                                "spec": backup_pod,
                            },
                        }
                    },
                },
            },
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--namespace", default="mac-fit")
    parser.add_argument("--gpu-host", required=True)
    parser.add_argument("--node", default=NODE)
    parser.add_argument("--uid", type=int, default=UID)
    parser.add_argument("--firebase-project-id", default="macfit-example")
    parser.add_argument("--python-image", default=PYTHON_IMAGE)
    parser.add_argument("--ssh-image", default=SSH_IMAGE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    document = render(
        args.repo,
        args.namespace,
        python_image=args.python_image,
        ssh_image=args.ssh_image,
        gpu_host=args.gpu_host,
        node=args.node,
        uid=args.uid,
        firebase_project_id=args.firebase_project_id,
    )
    encoded = json.dumps(document, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded)
    else:
        print(encoded, end="")
