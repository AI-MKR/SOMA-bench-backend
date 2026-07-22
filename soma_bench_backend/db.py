from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import Settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_factory(cursor: sqlite3.Cursor, row: tuple[Any, ...]) -> dict[str, Any]:
    return {column[0]: row[index] for index, column in enumerate(cursor.description)}


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _competition_from_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["benchmark_types"] = _loads(row["benchmark_types_json"], [])
    del row["benchmark_types_json"]
    return row


def _case_from_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["metadata"] = _loads(row["metadata_json"], {})
    del row["metadata_json"]
    return row


def _submission_from_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["environment"] = _loads(row["environment_json"], {})
    row["metadata"] = _loads(row["metadata_json"], {})
    del row["environment_json"]
    del row["metadata_json"]
    return row


def _evaluation_from_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["summary"] = _loads(row["summary_json"], {})
    del row["summary_json"]
    row["qualified"] = None if row["qualified"] is None else bool(row["qualified"])
    return row


def _leaderboard_from_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["screener_passed"] = bool(row["screener_passed"])
    row["category_scores"] = _loads(row["category_scores_json"], {})
    row["summary"] = _loads(row["summary_json"], {})
    row["evaluation_state"] = row["summary"].get(
        "evaluation_state",
        "scored" if row["status"] == "qualified" else row["status"],
    )
    row["evaluation_state_label"] = row["summary"].get(
        "evaluation_state_label",
        "not qualified" if row["evaluation_state"] == "not_qualified" else row["evaluation_state"],
    )
    del row["category_scores_json"]
    del row["summary_json"]
    return row


@contextmanager
def connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(settings.database_path, check_same_thread=False)
    connection.row_factory = _row_factory
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db(settings: Settings) -> None:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(settings) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS competitions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              description TEXT NOT NULL DEFAULT '',
              benchmark_types_json TEXT NOT NULL,
              screening_threshold REAL NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS benchmark_cases (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              competition_id INTEGER NOT NULL REFERENCES competitions(id) ON DELETE CASCADE,
              instance_id TEXT NOT NULL,
              benchmark_type TEXT NOT NULL,
              dataset_name TEXT NOT NULL DEFAULT '',
              split TEXT NOT NULL DEFAULT 'test',
              title TEXT NOT NULL DEFAULT '',
              repo TEXT NOT NULL DEFAULT '',
              prompt TEXT NOT NULL,
              baseline_resolved_count INTEGER NOT NULL DEFAULT 0,
              baseline_input_tokens INTEGER,
              baseline_cached_input_tokens INTEGER,
              baseline_output_tokens INTEGER,
              baseline_duration_seconds REAL,
              baseline_hit_file_rate REAL,
              baseline_noise_rate REAL,
              metadata_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              UNIQUE(competition_id, instance_id, benchmark_type)
            );

            CREATE TABLE IF NOT EXISTS submissions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              competition_id INTEGER NOT NULL REFERENCES competitions(id) ON DELETE CASCADE,
              miner_hotkey TEXT NOT NULL,
              display_name TEXT NOT NULL DEFAULT '',
              submission_root TEXT NOT NULL,
              entry_command TEXT NOT NULL,
              compressor_path TEXT NOT NULL DEFAULT '',
              environment_json TEXT NOT NULL DEFAULT '{}',
              metadata_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              competition_id INTEGER NOT NULL REFERENCES competitions(id) ON DELETE CASCADE,
              submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
              status TEXT NOT NULL,
              attempts_per_case INTEGER NOT NULL,
              timeout_seconds REAL NOT NULL,
              overall_score REAL,
              qualified INTEGER,
              error_text TEXT NOT NULL DEFAULT '',
              summary_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              started_at TEXT,
              finished_at TEXT,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_case_results (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              evaluation_id INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
              benchmark_case_id INTEGER NOT NULL REFERENCES benchmark_cases(id) ON DELETE CASCADE,
              attempt_index INTEGER NOT NULL,
              status TEXT NOT NULL,
              resolved INTEGER NOT NULL DEFAULT 0,
              input_tokens INTEGER,
              cached_input_tokens INTEGER,
              output_tokens INTEGER,
              duration_seconds REAL,
              files_hit_rate REAL,
              noise_rate REAL,
              patch_path TEXT NOT NULL DEFAULT '',
              stdout_path TEXT NOT NULL DEFAULT '',
              stderr_path TEXT NOT NULL DEFAULT '',
              error_text TEXT NOT NULL DEFAULT '',
              raw_result_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              UNIQUE(evaluation_id, benchmark_case_id, attempt_index)
            );

            CREATE TABLE IF NOT EXISTS leaderboard_scores (
              competition_id INTEGER NOT NULL REFERENCES competitions(id) ON DELETE CASCADE,
              submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
              evaluation_id INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
              overall_score REAL NOT NULL,
              quality_score REAL NOT NULL DEFAULT -1,
              efficiency_score REAL NOT NULL DEFAULT -1,
              status TEXT NOT NULL,
              screener_passed INTEGER NOT NULL,
              category_scores_json TEXT NOT NULL,
              summary_json TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (competition_id, submission_id)
            );
            """
        )
        _ensure_column(db, "leaderboard_scores", "quality_score", "REAL NOT NULL DEFAULT -1")
        _ensure_column(db, "leaderboard_scores", "efficiency_score", "REAL NOT NULL DEFAULT -1")
        _ensure_column(db, "benchmark_cases", "dataset_name", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(db, "benchmark_cases", "split", "TEXT NOT NULL DEFAULT 'test'")
        _ensure_column(db, "submissions", "compressor_path", "TEXT NOT NULL DEFAULT ''")


def _ensure_column(
    db: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    columns = {
        row["name"]
        for row in db.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        db.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


def create_competition(
    db: sqlite3.Connection,
    *,
    name: str,
    description: str,
    benchmark_types: list[str],
    screening_threshold: float,
) -> dict[str, Any]:
    created_at = utc_now()
    cursor = db.execute(
        """
        INSERT INTO competitions (
          name,
          description,
          benchmark_types_json,
          screening_threshold,
          created_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            name,
            description,
            json.dumps(benchmark_types),
            screening_threshold,
            created_at,
        ),
    )
    return get_competition(db, int(cursor.lastrowid))


def list_competitions(db: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = db.execute(
        "SELECT * FROM competitions ORDER BY created_at DESC, id DESC"
    ).fetchall()
    return [_competition_from_row(row) for row in rows]


def get_competition(db: sqlite3.Connection, competition_id: int) -> dict[str, Any] | None:
    row = db.execute(
        "SELECT * FROM competitions WHERE id = ?",
        (competition_id,),
    ).fetchone()
    return None if row is None else _competition_from_row(row)


def import_benchmark_cases(
    db: sqlite3.Connection,
    *,
    competition_id: int,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    created_at = utc_now()
    for case in cases:
        db.execute(
            """
            INSERT INTO benchmark_cases (
              competition_id,
              instance_id,
              benchmark_type,
              dataset_name,
              split,
              title,
              repo,
              prompt,
              baseline_resolved_count,
              baseline_input_tokens,
              baseline_cached_input_tokens,
              baseline_output_tokens,
              baseline_duration_seconds,
              baseline_hit_file_rate,
              baseline_noise_rate,
              metadata_json,
              created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(competition_id, instance_id, benchmark_type)
            DO UPDATE SET
              dataset_name = excluded.dataset_name,
              split = excluded.split,
              title = excluded.title,
              repo = excluded.repo,
              prompt = excluded.prompt,
              baseline_resolved_count = excluded.baseline_resolved_count,
              baseline_input_tokens = excluded.baseline_input_tokens,
              baseline_cached_input_tokens = excluded.baseline_cached_input_tokens,
              baseline_output_tokens = excluded.baseline_output_tokens,
              baseline_duration_seconds = excluded.baseline_duration_seconds,
              baseline_hit_file_rate = excluded.baseline_hit_file_rate,
              baseline_noise_rate = excluded.baseline_noise_rate,
              metadata_json = excluded.metadata_json
            """,
            (
                competition_id,
                case["instance_id"],
                case["benchmark_type"],
                case.get("dataset_name", ""),
                case.get("split", "test"),
                case.get("title", ""),
                case.get("repo", ""),
                case["prompt"],
                case["baseline_resolved_count"],
                case.get("baseline_input_tokens"),
                case.get("baseline_cached_input_tokens"),
                case.get("baseline_output_tokens"),
                case.get("baseline_duration_seconds"),
                case.get("baseline_hit_file_rate"),
                case.get("baseline_noise_rate"),
                json.dumps(case.get("metadata", {})),
                created_at,
            ),
        )
    return list_benchmark_cases(db, competition_id)


def list_benchmark_cases(
    db: sqlite3.Connection,
    competition_id: int,
) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT * FROM benchmark_cases
        WHERE competition_id = ?
        ORDER BY benchmark_type, instance_id, id
        """,
        (competition_id,),
    ).fetchall()
    return [_case_from_row(row) for row in rows]


def update_benchmark_case_baseline(
    db: sqlite3.Connection,
    *,
    benchmark_case_id: int,
    baseline_resolved_count: int,
    baseline_input_tokens: int | None,
    baseline_cached_input_tokens: int | None,
    baseline_output_tokens: int | None,
    baseline_duration_seconds: float | None,
) -> dict[str, Any]:
    db.execute(
        """
        UPDATE benchmark_cases
        SET baseline_resolved_count = ?,
            baseline_input_tokens = ?,
            baseline_cached_input_tokens = ?,
            baseline_output_tokens = ?,
            baseline_duration_seconds = ?
        WHERE id = ?
        """,
        (
            baseline_resolved_count,
            baseline_input_tokens,
            baseline_cached_input_tokens,
            baseline_output_tokens,
            baseline_duration_seconds,
            benchmark_case_id,
        ),
    )
    row = db.execute(
        "SELECT * FROM benchmark_cases WHERE id = ?",
        (benchmark_case_id,),
    ).fetchone()
    return _case_from_row(row)


def create_submission(
    db: sqlite3.Connection,
    *,
    competition_id: int,
    miner_hotkey: str,
    display_name: str,
    submission_root: Path,
    entry_command: str,
    compressor_path: Path,
    environment: dict[str, str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    created_at = utc_now()
    cursor = db.execute(
        """
        INSERT INTO submissions (
          competition_id,
          miner_hotkey,
          display_name,
          submission_root,
          entry_command,
          compressor_path,
          environment_json,
          metadata_json,
          created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            competition_id,
            miner_hotkey,
            display_name,
            str(submission_root),
            entry_command,
            str(compressor_path),
            json.dumps(environment),
            json.dumps(metadata),
            created_at,
        ),
    )
    return get_submission(db, int(cursor.lastrowid))


def get_submission(db: sqlite3.Connection, submission_id: int) -> dict[str, Any] | None:
    row = db.execute(
        "SELECT * FROM submissions WHERE id = ?",
        (submission_id,),
    ).fetchone()
    return None if row is None else _submission_from_row(row)


def list_submissions(
    db: sqlite3.Connection,
    competition_id: int,
) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT * FROM submissions
        WHERE competition_id = ?
        ORDER BY created_at DESC, id DESC
        """,
        (competition_id,),
    ).fetchall()
    return [_submission_from_row(row) for row in rows]


def create_evaluation(
    db: sqlite3.Connection,
    *,
    competition_id: int,
    submission_id: int,
    attempts_per_case: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    created_at = utc_now()
    cursor = db.execute(
        """
        INSERT INTO evaluations (
          competition_id,
          submission_id,
          status,
          attempts_per_case,
          timeout_seconds,
          created_at,
          updated_at
        ) VALUES (?, ?, 'queued', ?, ?, ?, ?)
        """,
        (
            competition_id,
            submission_id,
            attempts_per_case,
            timeout_seconds,
            created_at,
            created_at,
        ),
    )
    return get_evaluation(db, int(cursor.lastrowid))


def get_evaluation(db: sqlite3.Connection, evaluation_id: int) -> dict[str, Any] | None:
    row = db.execute(
        "SELECT * FROM evaluations WHERE id = ?",
        (evaluation_id,),
    ).fetchone()
    return None if row is None else _evaluation_from_row(row)


def list_latest_evaluations(
    db: sqlite3.Connection,
    competition_id: int,
) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT e.*
        FROM evaluations e
        JOIN (
          SELECT submission_id, MAX(id) AS id
          FROM evaluations
          WHERE competition_id = ?
          GROUP BY submission_id
        ) latest ON latest.id = e.id
        ORDER BY e.updated_at DESC, e.id DESC
        """,
        (competition_id,),
    ).fetchall()
    return [_evaluation_from_row(row) for row in rows]


def set_evaluation_running(db: sqlite3.Connection, evaluation_id: int) -> None:
    now = utc_now()
    db.execute(
        """
        UPDATE evaluations
        SET status = 'running',
            started_at = COALESCE(started_at, ?),
            updated_at = ?
        WHERE id = ?
        """,
        (now, now, evaluation_id),
    )


def complete_evaluation(
    db: sqlite3.Connection,
    *,
    evaluation_id: int,
    overall_score: float,
    qualified: bool,
    summary: dict[str, Any],
) -> None:
    now = utc_now()
    db.execute(
        """
        UPDATE evaluations
        SET status = 'completed',
            overall_score = ?,
            qualified = ?,
            summary_json = ?,
            finished_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            overall_score,
            int(qualified),
            json.dumps(summary),
            now,
            now,
            evaluation_id,
        ),
    )


def fail_evaluation(
    db: sqlite3.Connection,
    *,
    evaluation_id: int,
    error_text: str,
    summary: dict[str, Any] | None = None,
) -> None:
    now = utc_now()
    db.execute(
        """
        UPDATE evaluations
        SET status = 'failed',
            error_text = ?,
            summary_json = ?,
            finished_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            error_text,
            json.dumps(summary or {}),
            now,
            now,
            evaluation_id,
        ),
    )


def insert_case_result(
    db: sqlite3.Connection,
    *,
    evaluation_id: int,
    benchmark_case_id: int,
    attempt_index: int,
    status: str,
    resolved: bool,
    input_tokens: int | None,
    cached_input_tokens: int | None,
    output_tokens: int | None,
    duration_seconds: float | None,
    files_hit_rate: float | None,
    noise_rate: float | None,
    patch_path: str,
    stdout_path: str,
    stderr_path: str,
    error_text: str,
    raw_result: dict[str, Any],
) -> None:
    db.execute(
        """
        INSERT OR REPLACE INTO evaluation_case_results (
          evaluation_id,
          benchmark_case_id,
          attempt_index,
          status,
          resolved,
          input_tokens,
          cached_input_tokens,
          output_tokens,
          duration_seconds,
          files_hit_rate,
          noise_rate,
          patch_path,
          stdout_path,
          stderr_path,
          error_text,
          raw_result_json,
          created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evaluation_id,
            benchmark_case_id,
            attempt_index,
            status,
            int(resolved),
            input_tokens,
            cached_input_tokens,
            output_tokens,
            duration_seconds,
            files_hit_rate,
            noise_rate,
            patch_path,
            stdout_path,
            stderr_path,
            error_text,
            json.dumps(raw_result),
            utc_now(),
        ),
    )


def list_case_results(
    db: sqlite3.Connection,
    evaluation_id: int,
) -> list[dict[str, Any]]:
    return db.execute(
        """
        SELECT * FROM evaluation_case_results
        WHERE evaluation_id = ?
        ORDER BY benchmark_case_id, attempt_index
        """,
        (evaluation_id,),
    ).fetchall()


def upsert_leaderboard_entry(db: sqlite3.Connection, entry: dict[str, Any]) -> None:
    db.execute(
        """
        INSERT INTO leaderboard_scores (
          competition_id,
          submission_id,
          evaluation_id,
          overall_score,
          quality_score,
          efficiency_score,
          status,
          screener_passed,
          category_scores_json,
          summary_json,
          updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(competition_id, submission_id)
        DO UPDATE SET
          evaluation_id = excluded.evaluation_id,
          overall_score = excluded.overall_score,
          quality_score = excluded.quality_score,
          efficiency_score = excluded.efficiency_score,
          status = excluded.status,
          screener_passed = excluded.screener_passed,
          category_scores_json = excluded.category_scores_json,
          summary_json = excluded.summary_json,
          updated_at = excluded.updated_at
        """,
        (
            entry["competition_id"],
            entry["submission_id"],
            entry["evaluation_id"],
            entry["overall_score"],
            entry["quality_score"],
            entry["efficiency_score"],
            entry["status"],
            int(entry["screener_passed"]),
            json.dumps(entry["category_scores"]),
            json.dumps(entry["summary"]),
            utc_now(),
        ),
    )


def list_leaderboard(
    db: sqlite3.Connection,
    competition_id: int,
) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT
          lb.*,
          s.miner_hotkey,
          s.display_name
        FROM leaderboard_scores lb
        JOIN submissions s ON s.id = lb.submission_id
        WHERE lb.competition_id = ?
        ORDER BY lb.overall_score DESC, lb.updated_at DESC
        """,
        (competition_id,),
    ).fetchall()
    return [_leaderboard_from_row(row) for row in rows]


def count_evaluations(
    db: sqlite3.Connection,
    competition_id: int,
) -> int:
    row = db.execute(
        "SELECT COUNT(*) AS count FROM evaluations WHERE competition_id = ?",
        (competition_id,),
    ).fetchone()
    return int(row["count"])
