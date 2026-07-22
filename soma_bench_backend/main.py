from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status

from .config import get_settings
from .db import (
    connect,
    create_competition,
    create_evaluation,
    create_submission,
    count_evaluations,
    get_competition,
    get_evaluation,
    get_submission,
    import_benchmark_cases,
    init_db,
    list_benchmark_cases,
    list_competitions,
    list_evaluation_logs,
    list_latest_evaluations,
    list_leaderboard,
    list_submissions,
)
from .executor import LocalBenchmarkExecutor
from .schemas import (
    BenchmarkCaseImportRequest,
    BenchmarkCaseRead,
    CompetitionCreate,
    CompetitionRead,
    EvaluationCreate,
    EvaluationLogRead,
    EvaluationRead,
    DashboardMetrics,
    DashboardPayload,
    DashboardValidator,
    HuggingFaceCaseImportRequest,
    LeaderboardEntry,
    SubmissionCreate,
    SubmissionRead,
)

settings = get_settings()
executor = LocalBenchmarkExecutor(settings)
app = FastAPI(title=settings.app_name, version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    init_db(settings)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "database_path": str(settings.database_path),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post(
    "/competitions",
    response_model=CompetitionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_competition_endpoint(payload: CompetitionCreate) -> CompetitionRead:
    with connect(settings) as db:
        row = create_competition(
            db,
            name=payload.name,
            description=payload.description,
            benchmark_types=payload.benchmark_types,
            screening_threshold=payload.screening_threshold,
        )
    return CompetitionRead.model_validate(row)


@app.get("/competitions", response_model=list[CompetitionRead])
def list_competitions_endpoint() -> list[CompetitionRead]:
    with connect(settings) as db:
        rows = list_competitions(db)
    return [CompetitionRead.model_validate(row) for row in rows]


@app.post(
    "/competitions/{competition_id}/cases/import",
    response_model=list[BenchmarkCaseRead],
)
def import_cases_endpoint(
    competition_id: int,
    payload: BenchmarkCaseImportRequest,
) -> list[BenchmarkCaseRead]:
    with connect(settings) as db:
        competition = get_competition(db, competition_id)
        if competition is None:
            raise HTTPException(status_code=404, detail="Competition not found.")
        rows = import_benchmark_cases(
            db,
            competition_id=competition_id,
            cases=[case.model_dump() for case in payload.cases],
        )
    return [BenchmarkCaseRead.model_validate(row) for row in rows]


@app.post(
    "/competitions/{competition_id}/cases/import-huggingface",
    response_model=list[BenchmarkCaseRead],
)
def import_huggingface_cases_endpoint(
    competition_id: int,
    payload: HuggingFaceCaseImportRequest,
) -> list[BenchmarkCaseRead]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Install project dependencies so the datasets package is available.",
        ) from exc

    dataset = load_dataset(payload.dataset_name, split=payload.split)
    wanted_ids = set(payload.instance_ids)
    cases = []
    for row in dataset:
        instance_id = str(row.get("instance_id") or row.get("id") or "").strip()
        if not instance_id:
            continue
        if wanted_ids and instance_id not in wanted_ids:
            continue
        prompt = str(
            row.get("problem_statement")
            or row.get("prompt")
            or row.get("issue")
            or row.get("text")
            or ""
        )
        repo = str(row.get("repo") or row.get("repository") or "")
        metadata = _jsonable(dict(row))
        cases.append(
            {
                "instance_id": instance_id,
                "benchmark_type": payload.benchmark_type,
                "dataset_name": payload.dataset_name,
                "split": payload.split,
                "title": str(row.get("title") or instance_id),
                "repo": repo,
                "prompt": prompt,
                "baseline_resolved_count": int(row.get("baseline_resolved_count") or 5),
                "baseline_input_tokens": _optional_int(row.get("baseline_input_tokens")),
                "baseline_cached_input_tokens": _optional_int(row.get("baseline_cached_input_tokens")),
                "baseline_output_tokens": _optional_int(row.get("baseline_output_tokens")),
                "baseline_duration_seconds": _optional_float(row.get("baseline_duration_seconds")),
                "baseline_hit_file_rate": _optional_float(row.get("baseline_hit_file_rate")),
                "baseline_noise_rate": _optional_float(row.get("baseline_noise_rate")),
                "metadata": metadata,
            }
        )
        if payload.limit is not None and len(cases) >= payload.limit:
            break

    if not cases:
        raise HTTPException(status_code=400, detail="No benchmark cases matched import request.")

    with connect(settings) as db:
        competition = get_competition(db, competition_id)
        if competition is None:
            raise HTTPException(status_code=404, detail="Competition not found.")
        rows = import_benchmark_cases(
            db,
            competition_id=competition_id,
            cases=cases,
        )
    return [BenchmarkCaseRead.model_validate(row) for row in rows]


@app.get(
    "/competitions/{competition_id}/cases",
    response_model=list[BenchmarkCaseRead],
)
def list_cases_endpoint(competition_id: int) -> list[BenchmarkCaseRead]:
    with connect(settings) as db:
        competition = get_competition(db, competition_id)
        if competition is None:
            raise HTTPException(status_code=404, detail="Competition not found.")
        rows = list_benchmark_cases(db, competition_id)
    return [BenchmarkCaseRead.model_validate(row) for row in rows]


@app.post(
    "/competitions/{competition_id}/submissions",
    response_model=SubmissionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_submission_endpoint(
    competition_id: int,
    payload: SubmissionCreate,
) -> SubmissionRead:
    submission_root = Path(payload.submission_root).expanduser().resolve()
    if not submission_root.is_dir():
        raise HTTPException(
            status_code=400,
            detail="submission_root must be an existing directory.",
        )
    compressor_path = Path(payload.compressor_path).expanduser().resolve()
    if not compressor_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="compressor_path must be an existing Python file.",
        )
    with connect(settings) as db:
        competition = get_competition(db, competition_id)
        if competition is None:
            raise HTTPException(status_code=404, detail="Competition not found.")
        row = create_submission(
            db,
            competition_id=competition_id,
            miner_hotkey=payload.miner_hotkey,
            display_name=payload.display_name,
            submission_root=submission_root,
            entry_command=payload.entry_command,
            compressor_path=compressor_path,
            environment=payload.environment,
            metadata=payload.metadata,
        )
    return SubmissionRead.model_validate(row)


@app.get(
    "/competitions/{competition_id}/submissions",
    response_model=list[SubmissionRead],
)
def list_submissions_endpoint(competition_id: int) -> list[SubmissionRead]:
    with connect(settings) as db:
        competition = get_competition(db, competition_id)
        if competition is None:
            raise HTTPException(status_code=404, detail="Competition not found.")
        rows = list_submissions(db, competition_id)
    return [SubmissionRead.model_validate(row) for row in rows]


@app.post(
    "/competitions/{competition_id}/evaluations",
    response_model=EvaluationRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_evaluation_endpoint(
    competition_id: int,
    payload: EvaluationCreate,
) -> EvaluationRead:
    timeout_seconds = payload.timeout_seconds or settings.default_timeout_seconds
    attempts_per_case = 5
    with connect(settings) as db:
        competition = get_competition(db, competition_id)
        if competition is None:
            raise HTTPException(status_code=404, detail="Competition not found.")
        submission = get_submission(db, payload.submission_id)
        if submission is None or int(submission["competition_id"]) != competition_id:
            raise HTTPException(
                status_code=404,
                detail="Submission not found in competition.",
            )
        if not list_benchmark_cases(db, competition_id):
            raise HTTPException(
                status_code=400,
                detail="Competition has no benchmark cases.",
            )
        row = create_evaluation(
            db,
            competition_id=competition_id,
            submission_id=payload.submission_id,
            attempts_per_case=attempts_per_case,
            timeout_seconds=timeout_seconds,
        )
    executor.schedule(row["id"])
    return EvaluationRead.model_validate(_with_evaluation_state(row))


@app.get("/evaluations/{evaluation_id}", response_model=EvaluationRead)
def get_evaluation_endpoint(evaluation_id: int) -> EvaluationRead:
    with connect(settings) as db:
        row = get_evaluation(db, evaluation_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Evaluation not found.")
    return EvaluationRead.model_validate(_with_evaluation_state(row))


@app.get("/evaluations/{evaluation_id}/logs", response_model=list[EvaluationLogRead])
def list_evaluation_logs_endpoint(evaluation_id: int) -> list[EvaluationLogRead]:
    with connect(settings) as db:
        row = get_evaluation(db, evaluation_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Evaluation not found.")
        rows = list_evaluation_logs(db, evaluation_id)
    return [EvaluationLogRead.model_validate(row) for row in rows]


@app.get(
    "/competitions/{competition_id}/leaderboard",
    response_model=list[LeaderboardEntry],
)
def get_leaderboard_endpoint(competition_id: int) -> list[LeaderboardEntry]:
    with connect(settings) as db:
        competition = get_competition(db, competition_id)
        if competition is None:
            raise HTTPException(status_code=404, detail="Competition not found.")
        rows = list_leaderboard(db, competition_id)
    return [LeaderboardEntry.model_validate(row) for row in rows]


@app.get(
    "/competitions/{competition_id}/dashboard",
    response_model=DashboardPayload,
)
def get_dashboard_endpoint(competition_id: int) -> DashboardPayload:
    with connect(settings) as db:
        competition = get_competition(db, competition_id)
        if competition is None:
            raise HTTPException(status_code=404, detail="Competition not found.")
        submissions = list_submissions(db, competition_id)
        leaderboard_rows = list_leaderboard(db, competition_id)
        latest_evaluations = list_latest_evaluations(db, competition_id)
        total_evaluations = count_evaluations(db, competition_id)

    leaderboard = _dashboard_leaderboard_entries(
        submissions=submissions,
        leaderboard_rows=leaderboard_rows,
        latest_evaluations=latest_evaluations,
    )
    status_counts = {
        state: sum(1 for row in leaderboard if row.evaluation_state == state)
        for state in ["screening", "qualified", "not_qualified", "evaluating"]
    }
    top_score = max((float(row["overall_score"]) for row in leaderboard_rows), default=None)
    payload = {
        "competition": CompetitionRead.model_validate(competition),
        "metrics": DashboardMetrics(
            top_score=top_score,
            total_uploads=len(submissions),
            total_evaluations=total_evaluations,
            status_counts=status_counts,
        ),
        "validators": [
            DashboardValidator(name="local-validator", status="working", is_archive=False)
        ],
        "leaderboard": leaderboard,
    }
    return DashboardPayload.model_validate(payload)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _evaluation_state_from_row(
    row: dict[str, Any] | None,
    *,
    has_leaderboard_score: bool = False,
) -> tuple[str, str]:
    if row is None:
        return "screening", "screening"
    if row["status"] == "queued":
        return "screening", "screening"
    if row["status"] == "running":
        return "evaluating", "evaluating"
    if row["status"] == "failed":
        return "not_qualified", "not qualified"
    if row["status"] == "completed":
        if bool(row.get("qualified")):
            return "qualified", "qualified"
        return "not_qualified", "not qualified"
    return "screening", "screening"


def _with_evaluation_state(
    row: dict[str, Any],
    *,
    has_leaderboard_score: bool = False,
) -> dict[str, Any]:
    mutated = dict(row)
    state, label = _evaluation_state_from_row(
        row,
        has_leaderboard_score=has_leaderboard_score,
    )
    mutated["evaluation_state"] = state
    mutated["evaluation_state_label"] = label
    return mutated


def _dashboard_leaderboard_entries(
    *,
    submissions: list[dict[str, Any]],
    leaderboard_rows: list[dict[str, Any]],
    latest_evaluations: list[dict[str, Any]],
) -> list[LeaderboardEntry]:
    leaderboard_by_submission_id = {
        int(row["submission_id"]): row
        for row in leaderboard_rows
    }
    latest_evaluation_by_submission_id = {
        int(row["submission_id"]): row
        for row in latest_evaluations
    }
    entries: list[LeaderboardEntry] = []

    for row in leaderboard_rows:
        state = str(row.get("evaluation_state") or "")
        if not state:
            state, label = _evaluation_state_from_row(
                latest_evaluation_by_submission_id.get(int(row["submission_id"])),
                has_leaderboard_score=True,
            )
            row = {
                **row,
                "evaluation_state": state,
                "evaluation_state_label": label,
            }
        entries.append(LeaderboardEntry.model_validate(row))

    for submission in submissions:
        submission_id = int(submission["id"])
        if submission_id in leaderboard_by_submission_id:
            continue
        evaluation = latest_evaluation_by_submission_id.get(submission_id)
        state, label = _evaluation_state_from_row(evaluation)
        summary = {
            "competition_name": "",
            "evaluation_state": state,
            "evaluation_state_label": label,
            "screener": {
                "passed": False,
                "status": "pending" if state in {"screening", "evaluating"} else "failed",
                "threshold": None,
                "score": None,
            },
            "tasks": 0,
            "passed_without_compression": 0,
            "passed_with_compression": 0,
            "tokens_without_compression": {
                "input": None,
                "cache": None,
                "output": None,
                "weighted": None,
            },
            "tokens_with_compression": {
                "input": None,
                "cache": None,
                "output": None,
                "weighted": None,
            },
            "total_weighted": None,
            "baseline_weighted": None,
            "evaluation_details": {
                "evaluation_state": state,
                "evaluation_state_label": label,
                "task_details": [],
            },
        }
        entries.append(
            LeaderboardEntry.model_validate(
                {
                    "competition_id": int(submission["competition_id"]),
                    "submission_id": submission_id,
                    "evaluation_id": int(evaluation["id"]) if evaluation else 0,
                    "miner_hotkey": submission["miner_hotkey"],
                    "display_name": submission["display_name"] or submission["miner_hotkey"],
                    "overall_score": -1.0,
                    "quality_score": -1.0,
                    "efficiency_score": -1.0,
                    "status": state,
                    "evaluation_state": state,
                    "evaluation_state_label": label,
                    "screener_passed": False,
                    "category_scores": {},
                    "summary": summary,
                    "updated_at": evaluation["updated_at"] if evaluation else submission["created_at"],
                }
            )
        )

    return sorted(
        entries,
        key=lambda row: (
            row.overall_score,
            row.updated_at,
        ),
        reverse=True,
    )
