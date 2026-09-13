import json
from pathlib import Path

from amd_inference_opt.models import AgentDecision

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "skill_forward_test"


def test_fresh_agent_decision_obeys_old_mapping_forward_test_contract() -> None:
    context = json.loads((FIXTURE_DIR / "agent-context-old-mapping.json").read_text())
    raw_decision = json.loads((FIXTURE_DIR / "fresh-agent-decision.json").read_text())
    decision = AgentDecision.model_validate(raw_decision)

    available_ids = {item["id"] for item in context["evidence"]}
    assert set(decision.evidence_used) <= available_ids
    assert all("@sha256:" in item for item in decision.evidence_used)
    assert decision.requested_profile_level is None
    assert decision.proposed_next_stage.value == "CREATE_EXPERIMENT"
    assert decision.proposed_experiment is None

    missing = " ".join(decision.missing_evidence).lower()
    assert "unsupported" in missing
    assert "not integrated" in missing
    assert "unknown" in missing

    hypothesis = decision.hypothesis
    assert hypothesis is not None
    assert "split-k" in (hypothesis.id + hypothesis.proposed_change).lower()
    assert "unset llama_q4_rdna_mapping" in hypothesis.proposed_change.lower()
    required_validation = " ".join(hypothesis.required_validation)
    assert "binary_sha256=" + "9" * 64 in required_validation
    assert any("performance-eligible" in item for item in hypothesis.required_validation)
    assert "before any expensive quality evaluation" in required_validation

    serialized = json.dumps(raw_decision)
    assert '"outcome"' not in serialized
    assert '"gate_decision"' not in serialized
