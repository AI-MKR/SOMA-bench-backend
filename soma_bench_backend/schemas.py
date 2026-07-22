from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


BenchmarkType = Literal["swebench_verified", "swe_explorer_explore", "swe_explorer_edit"]
EvaluationBackend = Literal["soma_benchmark", "command"]
EvaluationStatus = Literal["queued", "running", "completed", "failed"]
CaseRunStatus = Literal["completed", "failed", "timed_out"]
LeaderboardStatus = Literal["qualified", "not_qualified", "running", "failed"]


class CompetitionCreate(BaseModel):
    name: str
    description: str = ""
    benchmark_types: list[BenchmarkType] = Field(
        default_factory=lambda: [
            "swebench_verified",
            "swe_explorer_explore",
            "swe_explorer_edit",
        ]
    )
    screening_threshold: float = 0.0


class CompetitionRead(BaseModel):
    id: int
    name: str
    description: str
    benchmark_types: list[str]
    screening_threshold: float
    created_at: datetime


class BenchmarkCaseInput(BaseModel):
    instance_id: str
    benchmark_type: BenchmarkType
    dataset_name: str = ""
    split: str = "test"
    title: str = ""
    repo: str = ""
    prompt: str
    baseline_resolved_count: int = Field(ge=0)
    baseline_input_tokens: int | None = Field(default=None, ge=0)
    baseline_cached_input_tokens: int | None = Field(default=None, ge=0)
    baseline_output_tokens: int | None = Field(default=None, ge=0)
    baseline_duration_seconds: float | None = Field(default=None, ge=0)
    baseline_hit_file_rate: float | None = Field(default=None, ge=0, le=1)
    baseline_noise_rate: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkCaseImportRequest(BaseModel):
    cases: list[BenchmarkCaseInput]


class HuggingFaceCaseImportRequest(BaseModel):
    dataset_name: str
    benchmark_type: BenchmarkType
    split: str = "test"
    limit: int | None = Field(default=None, ge=1)
    instance_ids: list[str] = Field(default_factory=list)


class BenchmarkCaseRead(BenchmarkCaseInput):
    id: int
    competition_id: int
    created_at: datetime


class SubmissionCreate(BaseModel):
    miner_hotkey: str
    display_name: str = ""
    submission_root: str = "."
    entry_command: str = ""
    compressor_path: str
    environment: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubmissionRead(BaseModel):
    id: int
    competition_id: int
    miner_hotkey: str
    display_name: str
    submission_root: str
    entry_command: str
    compressor_path: str
    environment: dict[str, str]
    metadata: dict[str, Any]
    created_at: datetime


class EvaluationCreate(BaseModel):
    submission_id: int
    backend: EvaluationBackend = "soma_benchmark"
    attempts_per_case: int = Field(default=5, ge=5, le=5)
    timeout_seconds: float | None = Field(default=None, gt=0)
    agent_name: str = "copilot"
    execute: bool = True
    swerebench_eval: bool = True
    extra_args: list[str] = Field(default_factory=list)


class CaseMetrics(BaseModel):
    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)
    files_hit_rate: float | None = Field(default=None, ge=0, le=1)
    noise_rate: float | None = Field(default=None, ge=0, le=1)


class CaseResultPayload(BaseModel):
    resolved: bool
    patch: str = ""
    metrics: CaseMetrics = Field(default_factory=CaseMetrics)
    notes: str = ""
    artifacts: dict[str, Any] = Field(default_factory=dict)


class EvaluationRead(BaseModel):
    id: int
    competition_id: int
    submission_id: int
    status: EvaluationStatus
    attempts_per_case: int
    timeout_seconds: float
    overall_score: float | None
    qualified: bool | None
    error_text: str
    summary: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    updated_at: datetime


class LeaderboardEntry(BaseModel):
    competition_id: int
    submission_id: int
    evaluation_id: int
    miner_hotkey: str
    display_name: str
    overall_score: float
    quality_score: float
    efficiency_score: float
    status: LeaderboardStatus
    screener_passed: bool
    category_scores: dict[str, float]
    summary: dict[str, Any]
    updated_at: datetime


class DashboardMetrics(BaseModel):
    top_score: float | None
    total_uploads: int
    total_evaluations: int
    status_counts: dict[str, int]


class DashboardValidator(BaseModel):
    name: str
    status: str
    is_archive: bool = False


class DashboardPayload(BaseModel):
    competition: CompetitionRead
    metrics: DashboardMetrics
    validators: list[DashboardValidator]
    leaderboard: list[LeaderboardEntry]
