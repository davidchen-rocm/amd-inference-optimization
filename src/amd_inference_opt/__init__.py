"""AMD inference optimization workflow primitives."""

from .architecture import ArchitectureProfile, architecture_profile
from .campaign import CampaignEngine
from .campaign_models import (
    CampaignConfig,
    CampaignRecord,
    CampaignStage,
    CampaignStatus,
    CandidateDisposition,
)
from .control_policy import (
    ActionSpec,
    CampaignControlPolicy,
    OptimizationLedger,
    ProfileRefreshPolicy,
    action_catalogue,
    content_fingerprint,
)
from .evidence_catalog import EvidenceCapabilitySnapshot, probe_evidence_capabilities
from .experiment_bundle import (
    ExperimentBundleManifest,
    ExperimentBundleSummary,
    ExperimentCatalog,
    load_experiment_bundle,
    refresh_experiment_bundle,
    refresh_task_bundles,
)
from .kernel_manifest import KernelShapeManifest, build_kernel_shape_manifest
from .model_library import (
    ModelLibraryCatalog,
    ModelRole,
    load_model_catalog,
    scan_model_library,
)
from .models import (
    AccuracyRequirement,
    AgentDecision,
    DecisionOutcome,
    OptimizationTask,
    WorkflowRecord,
    WorkflowStage,
)
from .project_config import ProjectConfig, load_project_config
from .session_breakdown import SessionBreakdownV1, build_session_breakdown

__all__ = [
    "AgentDecision",
    "AccuracyRequirement",
    "ActionSpec",
    "ArchitectureProfile",
    "CampaignConfig",
    "CampaignEngine",
    "CampaignRecord",
    "CampaignStage",
    "CampaignStatus",
    "CampaignControlPolicy",
    "CandidateDisposition",
    "DecisionOutcome",
    "EvidenceCapabilitySnapshot",
    "ExperimentBundleManifest",
    "ExperimentBundleSummary",
    "ExperimentCatalog",
    "KernelShapeManifest",
    "ModelLibraryCatalog",
    "ModelRole",
    "OptimizationTask",
    "OptimizationLedger",
    "ProfileRefreshPolicy",
    "ProjectConfig",
    "SessionBreakdownV1",
    "WorkflowRecord",
    "WorkflowStage",
    "action_catalogue",
    "architecture_profile",
    "build_session_breakdown",
    "build_kernel_shape_manifest",
    "content_fingerprint",
    "load_experiment_bundle",
    "load_model_catalog",
    "load_project_config",
    "probe_evidence_capabilities",
    "refresh_experiment_bundle",
    "refresh_task_bundles",
    "scan_model_library",
]

__version__ = "0.1.0"
