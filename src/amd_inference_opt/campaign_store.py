"""Revision-checked campaign persistence inside an ``ExperimentStore`` task."""

from __future__ import annotations

from pathlib import Path

from .campaign import CampaignEngine
from .campaign_models import CampaignConfig, CampaignRecord
from .evidence_catalog import probe_evidence_capabilities
from .models import ArtifactRef
from .platform_probe import LocalPlatformEvidence, probe_local_platform
from .runtime_health import capture_runtime_health
from .session_breakdown import persist_session_breakdown
from .store import ExperimentStore, StoreError

CAMPAIGN_STATE_PATH = "state/campaign.json"


class CampaignStoreError(RuntimeError):
    """Campaign state is missing, duplicated, or being updated from stale state."""


class CampaignStore:
    """Persist one outer campaign in the task directory owning all of its state."""

    def __init__(self, store: ExperimentStore | str | Path) -> None:
        self.store = store if isinstance(store, ExperimentStore) else ExperimentStore(store)

    def create(self, config: CampaignConfig) -> CampaignRecord:
        """Create and persist revision zero without replacing an existing campaign."""

        task_id = config.task.id
        task_dir = self.store.task_dir(task_id)
        if not task_dir.exists():
            try:
                self.store.create_task(config.task)
            except StoreError as error:
                raise CampaignStoreError(str(error)) from error
        else:
            try:
                persisted_task = self.store.load_task(task_id)
            except StoreError as error:
                raise CampaignStoreError(str(error)) from error
            if persisted_task != config.task:
                raise CampaignStoreError(
                    "campaign task does not match the task already stored in its directory"
                )

        try:
            with self.store.task_lock(task_id):
                state_path = self.store.task_dir(task_id) / CAMPAIGN_STATE_PATH
                if state_path.exists() or state_path.is_symlink():
                    raise CampaignStoreError(f"campaign already exists: {config.id}")
                platform = probe_local_platform(config.task.gpu)
                platform_ref = self.store.save_evidence_json(
                    task_id,
                    "campaign/platform",
                    platform,
                    producer="local-platform-probe",
                )
                self.store.save_evidence_json(
                    task_id,
                    "campaign/evidence-capabilities",
                    probe_evidence_capabilities(),
                    producer="evidence-capability-probe",
                )
                self.store.save_evidence_json(
                    task_id,
                    "campaign/runtime-health",
                    capture_runtime_health(
                        workspace=self.store.root,
                        extra_paths=(
                            config.task.runtime.repo_path,
                            config.task.model.path,
                        ),
                    ),
                    producer="runtime-health-probe",
                )
                record = CampaignEngine.new(
                    config,
                    platform_evidence=platform_ref,
                )
                self.store.save_json(
                    task_id,
                    CAMPAIGN_STATE_PATH,
                    record,
                    producer="campaign",
                )
                persist_session_breakdown(
                    self.store,
                    record,
                    platform=platform,
                )
                self.store.append_event(
                    task_id,
                    "campaign_created",
                    {"campaign_id": config.id, "revision": record.revision},
                )
                return record
        except StoreError as error:
            raise CampaignStoreError(str(error)) from error

    def _load_for_task(self, task_id: str) -> CampaignRecord:
        try:
            return self.store.load_json(task_id, CAMPAIGN_STATE_PATH, CampaignRecord)
        except StoreError as error:
            raise CampaignStoreError(str(error)) from error

    def load(self, identifier: str) -> CampaignRecord:
        """Load by owning task id, or by campaign id when those ids differ."""

        try:
            direct_dir = self.store.task_dir(identifier)
        except StoreError as error:
            raise CampaignStoreError(str(error)) from error
        if direct_dir.is_dir():
            return self._load_for_task(identifier)

        matches: list[CampaignRecord] = []
        for directory in self.store.root.iterdir():
            if not directory.is_dir():
                continue
            state_path = directory / CAMPAIGN_STATE_PATH
            if not state_path.is_file() or state_path.is_symlink():
                continue
            try:
                record = self._load_for_task(directory.name)
            except CampaignStoreError:
                continue
            if record.campaign_id == identifier:
                matches.append(record)
        if not matches:
            raise CampaignStoreError(f"campaign not found: {identifier}")
        if len(matches) > 1:
            raise CampaignStoreError(f"campaign id is ambiguous: {identifier}")
        return matches[0]

    def save(self, record: CampaignRecord) -> ArtifactRef:
        """Atomically save exactly the next revision and reject stale writers."""

        task_id = record.task_id
        try:
            with self.store.task_lock(task_id):
                current = self._load_for_task(task_id)
                if current.config != record.config:
                    raise CampaignStoreError("campaign configuration is immutable")
                expected_revision = current.revision + 1
                if record.revision != expected_revision:
                    raise CampaignStoreError(
                        "campaign revision conflict: "
                        f"expected {expected_revision}, got {record.revision}"
                    )
                platform = None
                if record.platform_evidence is not None:
                    if not self.store.verify_artifact(task_id, record.platform_evidence):
                        raise CampaignStoreError("platform evidence failed integrity verification")
                    platform = self.store.load_json(
                        task_id,
                        record.platform_evidence.path,
                        LocalPlatformEvidence,
                    )
                reference = self.store.save_json(
                    task_id,
                    CAMPAIGN_STATE_PATH,
                    record,
                    producer="campaign",
                )
                persist_session_breakdown(
                    self.store,
                    record,
                    platform=platform,
                )
                self.store.append_event(
                    task_id,
                    "campaign_saved",
                    {
                        "campaign_id": record.campaign_id,
                        "revision": record.revision,
                        "stage": record.current_stage.value,
                        "status": record.status.value,
                    },
                )
                return reference
        except StoreError as error:
            raise CampaignStoreError(str(error)) from error


__all__ = [
    "CAMPAIGN_STATE_PATH",
    "CampaignStore",
    "CampaignStoreError",
]
