"""Stable schemas and config loader for the read-only optimization capability map."""

from __future__ import annotations

import json
import re
from datetime import datetime
from enum import StrEnum
from importlib import resources
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .models import StrictModel, utc_now

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class OptimizationLayer(StrEnum):
    MODEL = "MODEL"
    KERNEL = "KERNEL"
    RUNTIME = "RUNTIME"
    EVIDENCE = "EVIDENCE"
    VALIDATION = "VALIDATION"


class CapabilityAvailability(StrEnum):
    IMPLEMENTED = "IMPLEMENTED"
    PARTIAL = "PARTIAL"
    PLANNED = "PLANNED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class CapabilityRunStatus(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"
    OPPORTUNITY_FOUND = "OPPORTUNITY_FOUND"
    NO_ACTION = "NO_ACTION"
    UNAVAILABLE = "UNAVAILABLE"


class OptimizationRelation(StrEnum):
    DEPENDS_ON = "DEPENDS_ON"
    PRODUCES = "PRODUCES"
    SELECTS = "SELECTS"
    OBSERVES = "OBSERVES"
    VALIDATES = "VALIDATES"
    FEEDS = "FEEDS"


class OptimizationEdgeState(StrEnum):
    INACTIVE = "INACTIVE"
    READY = "READY"
    ACTIVE = "ACTIVE"
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"


class OptimizationNodeTemplate(StrictModel):
    id: str
    title: str
    layer: OptimizationLayer
    availability: CapabilityAvailability
    summary: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("optimization node id must be lowercase kebab-case")
        return value

    @field_validator("title", "summary")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("optimization node text cannot be empty")
        return value


class OptimizationEdgeTemplate(StrictModel):
    source: str
    target: str
    relation: OptimizationRelation
    label: str

    @field_validator("source", "target")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("optimization edge ids must be lowercase kebab-case")
        return value


class OptimizationTopologyConfig(StrictModel):
    schema_name: Literal["gpuopt.optimization-topology-config.v1"] = Field(
        default="gpuopt.optimization-topology-config.v1", alias="schema"
    )
    nodes: list[OptimizationNodeTemplate] = Field(min_length=1)
    edges: list[OptimizationEdgeTemplate] = Field(min_length=1)

    @model_validator(mode="after")
    def graph_is_valid(self) -> OptimizationTopologyConfig:
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("optimization topology node ids must be unique")
        known = set(ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError("optimization topology edge references an unknown node")
            if edge.source == edge.target:
                raise ValueError("optimization topology cannot contain self edges")
        coordinates = [(edge.source, edge.target, edge.relation) for edge in self.edges]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("optimization topology edges must be unique")
        return self


class OptimizationEvidenceLink(StrictModel):
    path: str
    sha256: str | None = None


class OptimizationNodeView(StrictModel):
    id: str
    title: str
    layer: OptimizationLayer
    availability: CapabilityAvailability
    run_status: CapabilityRunStatus
    summary: str
    status_summary: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)
    evidence: list[OptimizationEvidenceLink] = Field(default_factory=list)


class OptimizationEdgeView(StrictModel):
    source: str
    target: str
    relation: OptimizationRelation
    label: str
    state: OptimizationEdgeState


class OptimizationMapV1(StrictModel):
    schema_name: Literal["gpuopt.optimization-map.v1"] = Field(
        default="gpuopt.optimization-map.v1", alias="schema"
    )
    generated_at: datetime = Field(default_factory=utc_now)
    source_id: str
    run_id: str
    read_only: Literal[True] = True
    layers: list[OptimizationLayer] = Field(
        default_factory=lambda: list(OptimizationLayer)
    )
    nodes: list[OptimizationNodeView]
    edges: list[OptimizationEdgeView]

    @model_validator(mode="after")
    def view_graph_is_valid(self) -> OptimizationMapV1:
        ids = {node.id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("optimization map node ids must be unique")
        if any(edge.source not in ids or edge.target not in ids for edge in self.edges):
            raise ValueError("optimization map edge references an unknown node")
        return self


def load_default_topology() -> OptimizationTopologyConfig:
    payload = (
        resources.files("amd_inference_opt.frontend")
        .joinpath("optimization-topology.json")
        .read_text(encoding="utf-8")
    )
    return OptimizationTopologyConfig.model_validate(json.loads(payload))


__all__ = [
    "CapabilityAvailability",
    "CapabilityRunStatus",
    "OptimizationEdgeState",
    "OptimizationEdgeTemplate",
    "OptimizationEdgeView",
    "OptimizationEvidenceLink",
    "OptimizationLayer",
    "OptimizationMapV1",
    "OptimizationNodeTemplate",
    "OptimizationNodeView",
    "OptimizationRelation",
    "OptimizationTopologyConfig",
    "load_default_topology",
]
