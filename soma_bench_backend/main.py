from __future__ import annotations

from datetime import datetime
from pathlib import Path

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
    EvaluationRead,
    DashboardMetrics,
    DashboardPayload,
    DashboardValidator,
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
    attempts_per_case = min(payload.attempts_per_case, settings.max_attempts_per_case)
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
    return EvaluationRead.model_validate(row)


@app.get("/evaluations/{evaluation_id}", response_model=EvaluationRead)
def get_evaluation_endpoint(evaluation_id: int) -> EvaluationRead:
    with connect(settings) as db:
        row = get_evaluation(db, evaluation_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Evaluation not found.")
    return EvaluationRead.model_validate(row)


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
        total_evaluations = count_evaluations(db, competition_id)

    leaderboard = [LeaderboardEntry.model_validate(row) for row in leaderboard_rows]
    status_counts = {
        "qualified": sum(1 for row in leaderboard if row.status == "qualified"),
        "not_qualified": sum(1 for row in leaderboard if row.status == "not_qualified"),
        "screening": max(0, len(submissions) - len(leaderboard)),
    }
    top_score = max((row.overall_score for row in leaderboard), default=None)
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
