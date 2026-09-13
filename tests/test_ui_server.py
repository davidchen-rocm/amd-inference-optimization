from __future__ import annotations

import hashlib
import http.client
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
from typer.testing import CliRunner

from amd_inference_opt.cli import app
from amd_inference_opt.experiment_bundle import refresh_experiment_bundle
from amd_inference_opt.frontend_api import (
    ControlPlaneReader,
    ReadOnlyStoreSource,
    project_quality_summary,
)
from amd_inference_opt.models import OptimizationTask, WorkflowRecord
from amd_inference_opt.store import ExperimentStore
from amd_inference_opt.ui_server import (
    DEFAULT_UI_HOST,
    DEFAULT_UI_PORT,
    create_control_plane_server,
    is_loopback_host,
)


def _make_store(root: Path, task_id: str = "ui-run") -> ExperimentStore:
    store = ExperimentStore(root)
    task = OptimizationTask.model_validate(
        {
            "id": task_id,
            "model": {
                "path": "/models/qwen.gguf",
                "architecture": "qwen",
                "quantization": "Q5_K_M",
            },
            "runtime": {"repo_path": "/src/llama.cpp", "base_commit": "abc123"},
            "gpu": {"gfx_target": "gfx1201", "name": "RX 9070 XT"},
            "mcp": {"command": ["/opt/rocm-agent-mcp"]},
        }
    )
    store.create_task(task)
    store.save_workflow(WorkflowRecord(task_id=task_id))
    store.save_evidence_json(
        task_id,
        "ui/example",
        {"tg128": 87.25, "quality": "pass"},
        producer="test",
    )
    store.append_event(task_id, "baseline_recorded", {"summary": "tg128 baseline saved"})
    return store


@contextmanager
def _running_server(reader: ControlPlaneReader) -> Iterator[tuple[str, int]]:
    server = create_control_plane_server(reader, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield str(host), int(port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(
    address: tuple[str, int], method: str, path: str
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(*address, timeout=3)
    connection.request(method, path)
    response = connection.getresponse()
    body = response.read()
    headers = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, headers, body


def _reader(root: Path) -> ControlPlaneReader:
    return ControlPlaneReader([ReadOnlyStoreSource("main", root, label="Main store")])


def _snapshot(root: Path) -> dict[str, tuple[int, str]]:
    values: dict[str, tuple[int, str]] = {}
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            data = path.read_bytes()
            values[str(path.relative_to(root))] = (
                path.stat().st_mtime_ns,
                hashlib.sha256(data).hexdigest(),
            )
    return values


def test_api_and_all_five_ui_screens_are_served_read_only(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _make_store(root)
    before = _snapshot(root)

    with _running_server(_reader(root)) as address:
        paths = (
            "/api/v1/meta",
            "/api/v1/schema",
            "/api/v1/runs",
            "/api/v1/runs/main/ui-run",
            "/api/v1/runs/main/ui-run/optimization-map",
            "/api/v1/runs/main/ui-run/events",
            "/api/v1/runs/main/ui-run/artifacts",
        )
        for path in paths:
            status, headers, body = _request(address, "GET", path)
            assert status == 200
            assert headers["cache-control"] == "no-store"
            assert headers["x-content-type-options"] == "nosniff"
            assert headers["x-frame-options"] == "DENY"
            assert "default-src 'self'" in headers["content-security-policy"]
            assert "access-control-allow-origin" not in headers
            assert json.loads(body)

        artifacts = json.loads(_request(address, "GET", paths[-1])[2])["items"]
        previewable = next(item for item in artifacts if item["preview_available"])
        preview_path = quote(previewable["path"], safe="")
        status, _, body = _request(
            address,
            "GET",
            f"/api/v1/runs/main/ui-run/artifacts/preview?path={preview_path}",
        )
        assert status == 200
        assert json.loads(body)["artifact"]["integrity"] == "verified"

        screens = (
            "/",
            "/runs/main/ui-run",
            "/runs/main/ui-run/map",
            "/runs/main/ui-run/experiments/example",
            f"/runs/main/ui-run/artifacts?path={preview_path}",
        )
        for path in screens:
            status, headers, body = _request(address, "GET", path)
            assert status == 200
            assert headers["content-type"].startswith("text/html")
            assert b"READ ONLY" in body

        status, headers, body = _request(address, "GET", "/assets/app.js")
        assert status == 200
        assert headers["content-type"].startswith(("application/javascript", "text/javascript"))
        assert body.count(b"function ") >= 10

    assert _snapshot(root) == before


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def test_every_mutating_or_preflight_method_is_rejected(tmp_path: Path, method: str) -> None:
    root = tmp_path / "store"
    _make_store(root)
    with _running_server(_reader(root)) as address:
        status, headers, body = _request(address, method, "/api/v1/runs")
    assert status == 405
    assert headers["cache-control"] == "no-store"
    assert "access-control-allow-origin" not in headers
    assert json.loads(body)["code"] == "read_only"


def test_api_rejects_path_traversal_and_unregistered_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _make_store(root)
    with _running_server(_reader(root)) as address:
        unsafe = quote("../task.json", safe="")
        status, _, body = _request(
            address,
            "GET",
            f"/api/v1/runs/main/ui-run/artifacts/preview?path={unsafe}",
        )
        assert status == 404
        assert json.loads(body)["code"] == "read_error"

        status, _, _ = _request(
            address,
            "GET",
            "/api/v1/runs/main/ui-run/artifacts/preview?path=not-registered.json",
        )
        assert status == 404

        status, _, _ = _request(address, "GET", "/api/v1/runs/main/%2e%2e")
        assert status == 404


def test_api_rejects_registered_artifact_replaced_by_symlink(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = _make_store(root)
    manifest = store.load_json("ui-run", "artifacts/manifest.json")
    relative = next(
        path
        for path, artifact in manifest["artifacts"].items()
        if artifact["media_type"] == "application/json" and "evidence" in path
    )
    artifact_path = root / "ui-run" / relative
    target = artifact_path.with_name("symlink-target.json")
    target.write_bytes(artifact_path.read_bytes())
    artifact_path.unlink()
    artifact_path.symlink_to(target.name)

    with _running_server(_reader(root)) as address:
        status, _, body = _request(
            address,
            "GET",
            "/api/v1/runs/main/ui-run/artifacts/preview?path=" + quote(relative, safe=""),
        )
    assert status == 404
    assert json.loads(body)["code"] == "read_error"


def test_read_only_ui_projects_vllm_workflow_stage(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = ExperimentStore(root)
    task = OptimizationTask.model_validate(
        {
            "id": "mi300x-vllm-run",
            "campaign_kind": "vllm_mi300x",
            "model": {
                "path": "/models/Qwen3",
                "architecture": "qwen3",
                "quantization": "bf16",
            },
            "runtime": {
                "name": "vllm",
                "deployment": "native",
                "version": "0.10.1",
                "executable": "/opt/venv/bin/python",
                "executable_sha256": "a" * 64,
                "environment_manifest_sha256": "b" * 64,
            },
            "gpu": {
                "gfx_target": "gfx942",
                "name": "AMD Instinct MI300X",
            },
            "workload": {
                "kind": "online_serving",
                "input_tokens": 1024,
                "output_tokens": 256,
                "num_prompts": 100,
                "concurrency": 8,
            },
            "mcp": {"command": ["/opt/rocm-agent-mcp"]},
        }
    )
    store.create_task(task)
    store.save_json(
        task.id,
        "state/vllm-workflow.json",
        {
            "schema_version": 1,
            "task_id": task.id,
            "config_sha256": "c" * 64,
            "current_stage": "BASELINE",
            "status": "ACTIVE",
            "completions": [],
            "profile_attempt_count": 0,
            "experiment_count": 0,
            "rerun_count": 0,
            "revision": 2,
            "updated_at": "2026-08-23T00:00:00Z",
        },
        producer="test",
    )

    source = ReadOnlyStoreSource("main", root)
    summary = source.summary(task.id)
    detail = source.detail(task.id)

    assert summary.runtime == "vllm"
    assert summary.gfx == "gfx942"
    assert summary.stage == "BASELINE"
    assert summary.status == "ACTIVE"
    assert [(item.name, item.status) for item in detail.workflow_stages] == [
        ("BASELINE", "RUNNING")
    ]


def test_gfx1201_capability_matrix_projects_real_campaign_outcomes(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = _make_store(root)
    evidence = store.save_evidence_json(
        "ui-run",
        "gfx1201/mixed-precision/result",
        {"winner": "q5"},
        producer="test",
    )
    store.save_json(
        "ui-run",
        "state/gfx1201-campaign.json",
        {
            "current_stage": "RUN_HIP_GRAPH_AB",
            "status": "ACTIVE",
            "revision": 4,
            "updated_at": "2026-08-18T00:00:00Z",
            "completions": [
                {
                    "stage": "RUN_MIXED_BIT",
                    "result": {
                        "capability": "mixed_precision",
                        "outcome": "ACCEPT",
                        "summary": "Q5 selected on the Pareto frontier",
                        "experiment_ids": ["q5-live"],
                        "evidence": [evidence.model_dump(mode="json")],
                    },
                }
            ],
        },
        producer="test",
    )

    detail = ReadOnlyStoreSource("main", root).detail("ui-run")
    capabilities = {item.id: item for item in detail.capabilities}
    assert capabilities["mixed_precision"].status == "ACCEPT"
    assert capabilities["mixed_precision"].experiment_ids == ["q5-live"]
    assert capabilities["hip_graph_ab"].status == "RUNNING"
    assert capabilities["memory_reuse_audit"].status == "NOT_STARTED"


def test_experiment_metrics_quality_and_gate_are_projected(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = _make_store(root)
    store.save_json(
        "ui-run",
        "artifacts/baseline.json",
        {
            "benchmark": {
                "metrics": {
                    "tokens_per_second_tg128": {
                        "unit": "tokens/s",
                        "samples": [100.0, 100.5, 99.5],
                    }
                }
            }
        },
        producer="test",
    )
    store.save_json(
        "ui-run",
        "experiments/candidate/spec.json",
        {
            "hypothesis_id": "hypothesis-1",
            "change": {"kind": "source_patch", "description": "change one kernel"},
        },
        producer="test",
    )
    store.save_json(
        "ui-run",
        "experiments/candidate/e2e-result.json",
        {
            "status": "SUCCEEDED",
            "metrics": {
                "tokens_per_second_tg128": {
                    "unit": "tokens/s",
                    "samples": [110.0, 111.0, 109.0],
                }
            },
        },
        producer="test",
    )
    store.save_json(
        "ui-run",
        "experiments/candidate/gate-decision.json",
        {
            "outcome": "ACCEPT",
            "metric_improvements_percent": {"tokens_per_second_tg128": 10.0},
            "reasons": ["passed"],
        },
        producer="test",
    )
    store.save_json(
        "ui-run",
        "experiments/candidate/quality-result.json",
        {"status": "SUCCEEDED", "perplexity": 3.4},
        producer="test",
    )

    detail = ReadOnlyStoreSource("main", root).detail("ui-run")
    assert [(item.id, item.status) for item in detail.experiments] == [
        ("candidate", "ACCEPT")
    ]
    assert detail.metrics[0].baseline == 100.0
    assert detail.metrics[0].candidate == 110.0
    assert detail.metrics[0].delta_percent == 10.0
    assert detail.quality == {"status": "SUCCEEDED", "perplexity": 3.4}
    assert detail.quality_summary is not None
    assert detail.quality_summary.policy == "provisional-math-100.v1"
    assert detail.quality_summary.perplexity.candidate == 3.4


def test_balanced_quality_summary_projects_math_general_and_ppl() -> None:
    summary = project_quality_summary(
        {
            "quality_policy": "balanced-200.v1",
            "status": "SUCCEEDED",
            "baseline": {
                "accuracies": {"math_accuracy": 0.70, "general_accuracy": 0.72},
                "perplexity": 2.0,
            },
            "candidate": {
                "accuracies": {"math_accuracy": 0.68, "general_accuracy": 0.71},
                "perplexity": 2.01,
                "math_correct": 68,
                "math_total": 100,
                "general_correct": 71,
                "general_total": 100,
            },
        }
    )

    assert summary is not None
    assert summary.suite_ids == ["math-100.v1", "general-100.v1"]
    assert summary.math.candidate_percent == pytest.approx(68)
    assert summary.math.delta_points == pytest.approx(-2)
    assert summary.general.candidate_percent == pytest.approx(71)
    assert summary.general.delta_points == pytest.approx(-1)
    assert summary.perplexity.candidate == pytest.approx(2.01)
    assert summary.perplexity.delta_percent == pytest.approx(0.5)


def _save_bundle_candidate(store: ExperimentStore) -> None:
    patch = store.task_dir("ui-run") / "bundle-source.patch"
    patch.write_text(
        "diff --git a/kernel.cu b/kernel.cu\n"
        "--- a/kernel.cu\n"
        "+++ b/kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old_mapping\n"
        "+four_wave_mapping\n",
        encoding="utf-8",
    )
    store.save_json(
        "ui-run",
        "artifacts/baseline.json",
        {
            "benchmark": {
                "metrics": {
                    "tokens_per_second_tg128": {
                        "unit": "tokens/s",
                        "samples": [100.0, 100.0, 100.0],
                    },
                    "tokens_per_second_tg512": {
                        "unit": "tokens/s",
                        "samples": [102.0, 102.0, 102.0],
                    },
                }
            }
        },
        producer="test",
    )
    store.save_json(
        "ui-run",
        "experiments/bundled/spec.json",
        {
            "hypothesis_id": "shape-wave-map",
            "change": {
                "kind": "source_patch",
                "description": "map the real matrix shape to four waves",
                "patch_path": str(patch),
                "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            },
        },
        producer="test",
    )
    store.save_json(
        "ui-run",
        "experiments/bundled/e2e-result.json",
        {
            "status": "SUCCEEDED",
            "metrics": {
                "tokens_per_second_tg128": {
                    "unit": "tokens/s",
                    "samples": [110.0, 110.0, 110.0],
                },
                "tokens_per_second_tg512": {
                    "unit": "tokens/s",
                    "samples": [111.0, 111.0, 111.0],
                },
            },
        },
        producer="test",
    )
    store.save_json(
        "ui-run",
        "experiments/bundled/gate-decision.json",
        {"outcome": "ACCEPT", "reasons": ["performance and quality passed"]},
        producer="test",
    )
    store.save_json(
        "ui-run",
        "experiments/bundled/quality-result.json",
        {"status": "SUCCEEDED", "perplexity": 3.4, "correctness_passed": True},
        producer="test",
    )
    refresh_experiment_bundle(store, "ui-run", "bundled")


def test_verified_bundle_is_preferred_and_has_a_single_experiment_endpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    store = _make_store(root)
    _save_bundle_candidate(store)

    source = ReadOnlyStoreSource("main", root)
    experiment = source.experiment("ui-run", "bundled")
    assert experiment.projection == "bundle"
    assert experiment.change_summary == "map the real matrix shape to four waves"
    assert experiment.decision == "ACCEPT"
    assert {metric.name for metric in experiment.candidate_metrics} == {
        "tokens_per_second_tg128",
        "tokens_per_second_tg512",
    }
    assert experiment.artifact_count >= 5
    assert experiment.source_digest is not None

    with _running_server(_reader(root)) as address:
        status, _, body = _request(
            address,
            "GET",
            "/api/v1/runs/main/ui-run/experiments/bundled",
        )
    payload = json.loads(body)
    assert status == 200
    assert payload["projection"] == "bundle"
    assert payload["decision"] == "ACCEPT"


def test_invalid_bundle_falls_back_to_legacy_evidence(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = _make_store(root)
    _save_bundle_candidate(store)
    summary_path = root / "ui-run/experiments/bundled/summary.json"
    summary_path.write_text(summary_path.read_text() + "\n", encoding="utf-8")

    detail = ReadOnlyStoreSource("main", root).detail("ui-run")
    experiment = next(item for item in detail.experiments if item.id == "bundled")
    assert experiment.projection == "legacy"
    assert experiment.decision == "ACCEPT"
    assert any("invalid experiment bundle" in warning for warning in detail.warnings)


def test_bind_defaults_and_loopback_enforcement(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _make_store(root)
    reader = _reader(root)

    assert DEFAULT_UI_HOST == "127.0.0.1"
    assert DEFAULT_UI_PORT == 4561
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("example.com")
    with pytest.raises(ValueError, match="loopback"):
        create_control_plane_server(reader, host="0.0.0.0")


def test_cli_parses_repeated_store_specs_without_constructing_writable_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _make_store(first, "one-run")
    _make_store(second, "two-run")
    observed: dict[str, Any] = {}

    def fake_serve(
        reader: ControlPlaneReader,
        *,
        host: str,
        port: int,
        draft_store: Any,
        announce: Any,
    ) -> None:
        observed["source_ids"] = list(reader.sources)
        observed["host"] = host
        observed["port"] = port
        observed["draft_store"] = draft_store
        observed["announce"] = announce

    monkeypatch.setattr("amd_inference_opt.ui_server.serve_control_plane", fake_serve)
    result = CliRunner().invoke(
        app,
        [
            "ui",
            "--store",
            f"first={first}",
            "--store",
            f"second={second}",
            "--draft-db",
            str(tmp_path / "control" / "control-plane.sqlite3"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed["source_ids"] == ["first", "second"]
    assert observed["host"] == "127.0.0.1"
    assert observed["port"] == 4561
    assert observed["draft_store"].path.name == "control-plane.sqlite3"


def test_cli_rejects_unsafe_store_and_host_specs(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _make_store(root)
    runner = CliRunner()

    malformed = runner.invoke(app, ["ui", "--store", str(root)])
    assert malformed.exit_code == 2
    assert "ID=PATH" in malformed.stderr

    non_loopback = runner.invoke(
        app,
        ["ui", "--store", f"main={root}", "--host", "0.0.0.0"],
    )
    assert non_loopback.exit_code == 2
    assert "loopback" in non_loopback.stderr


def test_static_assets_do_not_use_inline_scripts_or_remote_dependencies(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _make_store(root)
    with _running_server(_reader(root)) as address:
        _, _, html = _request(address, "GET", "/")
        _, _, script = _request(address, "GET", "/assets/app.js")

    assert b'<script defer src="/assets/app.js?v=6"></script>' in html
    assert b"http://" not in html and b"https://" not in html
    assert b"innerHTML" not in script
    assert b"await loadCurrentScreen()" not in script
    assert script.count(b"table.append(TableHead(") == 6
    assert b"function OptimizationMapBoard(" in script
    assert b'["Experiment", "Change", "tg128", "tg512", "Quality", "Decision"]' in script
    assert b"Advanced artifacts (" in script
    assert b"document.createElementNS" in script
    assert os.linesep.encode() in script
