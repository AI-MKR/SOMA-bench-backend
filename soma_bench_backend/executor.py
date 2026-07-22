from __future__ import annotations

import json
import os
import subprocess
import threading
import traceback
from pathlib import Path
from typing import Any

from . import db
from .config import Settings
from .scoring import build_leaderboard_entry


class LocalBenchmarkExecutor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._active_runs: set[int] = set()

    def schedule(self, evaluation_id: int) -> None:
        with self._lock:
            if evaluation_id in self._active_runs:
                return
            self._active_runs.add(evaluation_id)
        thread = threading.Thread(
            target=self._run_evaluation,
            args=(evaluation_id,),
            daemon=True,
            name=f"soma-bench-eval-{evaluation_id}",
        )
        thread.start()

    def _run_evaluation(self, evaluation_id: int) -> None:
        try:
            with db.connect(self.settings) as connection:
                evaluation_row = db.get_evaluation(connection, evaluation_id)
                if evaluation_row is None:
                    return
                submission_row = db.get_submission(connection, evaluation_row["submission_id"])
                competition_row = db.get_competition(
                    connection, evaluation_row["competition_id"]
                )
                if submission_row is None or competition_row is None:
                    db.fail_evaluation(
                        connection,
                        evaluation_id=evaluation_id,
                        error_text="Missing submission or competition.",
                    )
                    return
                case_rows = db.list_benchmark_cases(
                    connection, competition_row["id"]
                )
                if not case_rows:
                    db.fail_evaluation(
                        connection,
                        evaluation_id=evaluation_id,
                        error_text="Competition has no benchmark cases.",
                    )
                    return
                db.set_evaluation_running(connection, evaluation_id)

            run_dir = self.settings.runs_dir / f"evaluation-{evaluation_id}"
            run_dir.mkdir(parents=True, exist_ok=True)

            for case_row in case_rows:
                for attempt_index in range(1, evaluation_row["attempts_per_case"] + 1):
                    self._run_case_attempt(
                        evaluation_id=evaluation_id,
                        submission_row=submission_row,
                        case_row=case_row,
                        timeout_seconds=evaluation_row["timeout_seconds"],
                        attempt_index=attempt_index,
                        run_dir=run_dir,
                    )

            with db.connect(self.settings) as connection:
                result_rows = db.list_case_results(connection, evaluation_id)
                leaderboard_entry = build_leaderboard_entry(
                    competition_row=competition_row,
                    submission_row=submission_row,
                    case_rows=case_rows,
                    result_rows=result_rows,
                    evaluation_id=evaluation_id,
                    settings=self.settings,
                )
                db.upsert_leaderboard_entry(connection, leaderboard_entry)
                db.complete_evaluation(
                    connection,
                    evaluation_id=evaluation_id,
                    overall_score=leaderboard_entry["overall_score"],
                    qualified=leaderboard_entry["screener_passed"],
                    summary=leaderboard_entry["summary"],
                )
        except Exception as exc:
            with db.connect(self.settings) as connection:
                db.fail_evaluation(
                    connection,
                    evaluation_id=evaluation_id,
                    error_text=f"{type(exc).__name__}: {exc}",
                    summary={"traceback": traceback.format_exc()},
                )
        finally:
            with self._lock:
                self._active_runs.discard(evaluation_id)

    def _run_case_attempt(
        self,
        *,
        evaluation_id: int,
        submission_row: dict[str, Any],
        case_row: dict[str, Any],
        timeout_seconds: float,
        attempt_index: int,
        run_dir: Path,
    ) -> None:
        case_dir = run_dir / case_row["benchmark_type"] / case_row["instance_id"] / str(
            attempt_index
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        case_payload_path = case_dir / "case.json"
        result_payload_path = case_dir / "result.json"
        patch_path = case_dir / "patch.diff"
        stdout_path = case_dir / "stdout.log"
        stderr_path = case_dir / "stderr.log"

        case_payload = {
            "evaluation_id": evaluation_id,
            "attempt_index": attempt_index,
            "instance_id": case_row["instance_id"],
            "benchmark_type": case_row["benchmark_type"],
            "title": case_row["title"],
            "repo": case_row["repo"],
            "prompt": case_row["prompt"],
            "baseline": {
                "resolved_count": case_row["baseline_resolved_count"],
                "input_tokens": case_row["baseline_input_tokens"],
                "cached_input_tokens": case_row["baseline_cached_input_tokens"],
                "output_tokens": case_row["baseline_output_tokens"],
                "duration_seconds": case_row["baseline_duration_seconds"],
                "hit_file_rate": case_row["baseline_hit_file_rate"],
                "noise_rate": case_row["baseline_noise_rate"],
            },
            "metadata": case_row["metadata"],
        }
        case_payload_path.write_text(
            json.dumps(case_payload, indent=2, ensure_ascii=True), encoding="utf-8"
        )

        env = os.environ.copy()
        env.update(submission_row["environment"])
        env["SOMA_BENCH_CASE_PATH"] = str(case_payload_path)
        env["SOMA_BENCH_RESULT_PATH"] = str(result_payload_path)
        env["SOMA_BENCH_EVALUATION_ID"] = str(evaluation_id)
        env["SOMA_BENCH_ATTEMPT_INDEX"] = str(attempt_index)
        env["SOMA_BENCH_INSTANCE_ID"] = case_row["instance_id"]
        env["SOMA_BENCH_BENCHMARK_TYPE"] = case_row["benchmark_type"]
        env["SOMA_BENCH_TIMEOUT_SECONDS"] = str(timeout_seconds)

        status = "failed"
        error_text = ""
        raw_result: dict[str, Any] = {}
        resolved = False
        input_tokens = None
        cached_input_tokens = None
        output_tokens = None
        duration_seconds = None
        files_hit_rate = None
        noise_rate = None

        try:
            completed = subprocess.run(
                submission_row["entry_command"],
                cwd=submission_row["submission_root"],
                env=env,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_path.write_text(completed.stderr or "", encoding="utf-8")
            raw_result = self._load_result_payload(
                result_payload_path=result_payload_path,
                stdout_text=completed.stdout or "",
            )
            resolved = bool(raw_result.get("resolved"))
            metrics = raw_result.get("metrics") or {}
            input_tokens = self._to_int(metrics.get("input_tokens"))
            cached_input_tokens = self._to_int(metrics.get("cached_input_tokens"))
            output_tokens = self._to_int(metrics.get("output_tokens"))
            duration_seconds = self._to_float(metrics.get("duration_seconds"))
            files_hit_rate = self._to_float(metrics.get("files_hit_rate"))
            noise_rate = self._to_float(metrics.get("noise_rate"))
            patch_text = str(raw_result.get("patch") or "")
            patch_path.write_text(patch_text, encoding="utf-8")
            if completed.returncode == 0:
                status = "completed"
            else:
                status = "failed"
                error_text = f"Command exited with code {completed.returncode}."
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text((exc.stdout or "") if isinstance(exc.stdout, str) else "", encoding="utf-8")
            stderr_path.write_text((exc.stderr or "") if isinstance(exc.stderr, str) else "", encoding="utf-8")
            status = "timed_out"
            error_text = f"Timed out after {timeout_seconds} seconds."
        except Exception as exc:
            status = "failed"
            error_text = f"{type(exc).__name__}: {exc}"

        with db.connect(self.settings) as connection:
            db.insert_case_result(
                connection,
                evaluation_id=evaluation_id,
                benchmark_case_id=case_row["id"],
                attempt_index=attempt_index,
                status=status,
                resolved=resolved,
                input_tokens=input_tokens,
                cached_input_tokens=cached_input_tokens,
                output_tokens=output_tokens,
                duration_seconds=duration_seconds,
                files_hit_rate=files_hit_rate,
                noise_rate=noise_rate,
                patch_path=str(patch_path),
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                error_text=error_text,
                raw_result=raw_result,
            )

    @staticmethod
    def _load_result_payload(
        *,
        result_payload_path: Path,
        stdout_text: str,
    ) -> dict[str, Any]:
        if result_payload_path.is_file():
            return json.loads(result_payload_path.read_text(encoding="utf-8"))

        stdout_text = stdout_text.strip()
        if not stdout_text:
            return {}
        return json.loads(stdout_text)

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)
