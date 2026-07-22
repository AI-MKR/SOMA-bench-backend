from __future__ import annotations

import json
import os
import shlex
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
                    db.insert_evaluation_log(
                        connection,
                        evaluation_id=evaluation_id,
                        level="error",
                        event="evaluation_failed",
                        message="Missing submission or competition.",
                        details={
                            "submission_id": evaluation_row["submission_id"],
                            "competition_id": evaluation_row["competition_id"],
                        },
                    )
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
                    db.insert_evaluation_log(
                        connection,
                        evaluation_id=evaluation_id,
                        level="error",
                        event="evaluation_failed",
                        message="Competition has no benchmark cases.",
                        details={"competition_id": competition_row["id"]},
                    )
                    db.fail_evaluation(
                        connection,
                        evaluation_id=evaluation_id,
                        error_text="Competition has no benchmark cases.",
                    )
                    return
                db.set_evaluation_running(connection, evaluation_id)
                db.insert_evaluation_log(
                    connection,
                    evaluation_id=evaluation_id,
                    level="info",
                    event="evaluation_started",
                    message="Evaluation started.",
                    details={
                        "competition_id": competition_row["id"],
                        "submission_id": submission_row["id"],
                        "case_count": len(case_rows),
                        "attempts_per_case": evaluation_row["attempts_per_case"],
                        "timeout_seconds": evaluation_row["timeout_seconds"],
                    },
                )

            run_dir = self.settings.runs_dir / f"evaluation-{evaluation_id}"
            run_dir.mkdir(parents=True, exist_ok=True)

            for case_row in case_rows:
                with db.connect(self.settings) as connection:
                    db.insert_evaluation_log(
                        connection,
                        evaluation_id=evaluation_id,
                        level="info",
                        event="case_started",
                        message=f"Started benchmark case {case_row['instance_id']}.",
                        details={
                            "benchmark_case_id": case_row["id"],
                            "instance_id": case_row["instance_id"],
                            "benchmark_type": case_row["benchmark_type"],
                        },
                    )
                baseline_missing = (
                    case_row["baseline_input_tokens"] is None
                    or case_row["baseline_output_tokens"] is None
                )
                if baseline_missing:
                    with db.connect(self.settings) as connection:
                        db.insert_evaluation_log(
                            connection,
                            evaluation_id=evaluation_id,
                            level="info",
                            event="baseline_started",
                            message=f"Started baseline run for {case_row['instance_id']}.",
                            details={
                                "benchmark_case_id": case_row["id"],
                                "instance_id": case_row["instance_id"],
                            },
                        )
                case_row = self._ensure_case_baseline(
                    case_row=case_row,
                    timeout_seconds=evaluation_row["timeout_seconds"],
                    run_dir=run_dir,
                )
                if baseline_missing:
                    with db.connect(self.settings) as connection:
                        db.insert_evaluation_log(
                            connection,
                            evaluation_id=evaluation_id,
                            level="info",
                            event="baseline_finished",
                            message=f"Finished baseline run for {case_row['instance_id']}.",
                            details={
                                "benchmark_case_id": case_row["id"],
                                "instance_id": case_row["instance_id"],
                                "baseline_resolved_count": case_row["baseline_resolved_count"],
                                "baseline_input_tokens": case_row["baseline_input_tokens"],
                                "baseline_cached_input_tokens": case_row["baseline_cached_input_tokens"],
                                "baseline_output_tokens": case_row["baseline_output_tokens"],
                                "baseline_duration_seconds": case_row["baseline_duration_seconds"],
                            },
                        )
                for attempt_index in range(1, evaluation_row["attempts_per_case"] + 1):
                    self._run_case_attempt(
                        evaluation_id=evaluation_id,
                        submission_row=submission_row,
                        case_row=case_row,
                        timeout_seconds=evaluation_row["timeout_seconds"],
                        attempt_index=attempt_index,
                        run_dir=run_dir,
                        run_label="miner",
                        baseline=False,
                        store_result=True,
                    )

            with db.connect(self.settings) as connection:
                case_rows = db.list_benchmark_cases(
                    connection, competition_row["id"]
                )
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
                db.insert_evaluation_log(
                    connection,
                    evaluation_id=evaluation_id,
                    level="info",
                    event="evaluation_scored",
                    message="Evaluation scored and leaderboard entry updated.",
                    details={
                        "overall_score": leaderboard_entry["overall_score"],
                        "quality_score": leaderboard_entry["quality_score"],
                        "efficiency_score": leaderboard_entry["efficiency_score"],
                        "screener_passed": leaderboard_entry["screener_passed"],
                        "evaluation_state": leaderboard_entry["evaluation_state"],
                        "attempt_count": leaderboard_entry["summary"]["attempt_count"],
                        "completed_attempts": leaderboard_entry["summary"]["completed_attempts"],
                        "failed_attempts": leaderboard_entry["summary"]["failed_attempts"],
                        "timed_out_attempts": leaderboard_entry["summary"]["timed_out_attempts"],
                    },
                )
                db.complete_evaluation(
                    connection,
                    evaluation_id=evaluation_id,
                    overall_score=leaderboard_entry["overall_score"],
                    qualified=leaderboard_entry["screener_passed"],
                    summary=leaderboard_entry["summary"],
                )
        except Exception as exc:
            with db.connect(self.settings) as connection:
                db.insert_evaluation_log(
                    connection,
                    evaluation_id=evaluation_id,
                    level="error",
                    event="evaluation_failed",
                    message=f"{type(exc).__name__}: {exc}",
                    details={"traceback": traceback.format_exc()},
                )
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
        submission_row: dict[str, Any] | None,
        case_row: dict[str, Any],
        timeout_seconds: float,
        attempt_index: int,
        run_dir: Path,
        run_label: str,
        baseline: bool,
        store_result: bool,
    ) -> dict[str, Any]:
        case_dir = run_dir / run_label / case_row["benchmark_type"] / case_row["instance_id"] / str(attempt_index)
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
        if submission_row is not None:
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
            command, command_cwd = self._build_command(
                submission_row=submission_row,
                case_row=case_row,
                case_dir=case_dir,
                baseline=baseline,
            )
            if store_result:
                with db.connect(self.settings) as connection:
                    db.insert_evaluation_log(
                        connection,
                        evaluation_id=evaluation_id,
                        level="info",
                        event="attempt_started",
                        message=f"Started attempt {attempt_index} for {case_row['instance_id']}.",
                        details={
                            "benchmark_case_id": case_row["id"],
                            "instance_id": case_row["instance_id"],
                            "benchmark_type": case_row["benchmark_type"],
                            "attempt_index": attempt_index,
                            "command_cwd": command_cwd,
                            "case_dir": str(case_dir),
                            "case_payload_path": str(case_payload_path),
                        },
                    )
            completed = subprocess.run(
                command,
                cwd=command_cwd,
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
                case_dir=case_dir,
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

        result = {
            "status": status,
            "resolved": resolved,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "duration_seconds": duration_seconds,
            "files_hit_rate": files_hit_rate,
            "noise_rate": noise_rate,
            "patch_path": str(patch_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "error_text": error_text,
            "raw_result": raw_result,
        }

        if store_result:
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
                log_level = "info" if status == "completed" else "error"
                db.insert_evaluation_log(
                    connection,
                    evaluation_id=evaluation_id,
                    level=log_level,
                    event="attempt_finished",
                    message=f"Finished attempt {attempt_index} for {case_row['instance_id']} with status {status}.",
                    details={
                        "benchmark_case_id": case_row["id"],
                        "instance_id": case_row["instance_id"],
                        "benchmark_type": case_row["benchmark_type"],
                        "attempt_index": attempt_index,
                        "status": status,
                        "resolved": resolved,
                        "input_tokens": input_tokens,
                        "cached_input_tokens": cached_input_tokens,
                        "output_tokens": output_tokens,
                        "duration_seconds": duration_seconds,
                        "files_hit_rate": files_hit_rate,
                        "noise_rate": noise_rate,
                        "patch_path": str(patch_path),
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                        "error_text": error_text,
                        "raw_status": raw_result.get("status"),
                        "raw_error": raw_result.get("error"),
                    },
                )
        return result

    def _ensure_case_baseline(
        self,
        *,
        case_row: dict[str, Any],
        timeout_seconds: float,
        run_dir: Path,
    ) -> dict[str, Any]:
        if (
            case_row["baseline_input_tokens"] is not None
            and case_row["baseline_output_tokens"] is not None
        ):
            return case_row

        baseline_results = [
            self._run_case_attempt(
                evaluation_id=0,
                submission_row=None,
                case_row=case_row,
                timeout_seconds=timeout_seconds,
                attempt_index=attempt_index,
                run_dir=run_dir,
                run_label="baseline",
                baseline=True,
                store_result=False,
            )
            for attempt_index in range(1, self.settings.default_attempts_per_case + 1)
        ]
        resolved_count = sum(1 for result in baseline_results if result["resolved"])
        with db.connect(self.settings) as connection:
            return db.update_benchmark_case_baseline(
                connection,
                benchmark_case_id=case_row["id"],
                baseline_resolved_count=resolved_count,
                baseline_input_tokens=_average_int(result["input_tokens"] for result in baseline_results),
                baseline_cached_input_tokens=_average_int(
                    result["cached_input_tokens"] for result in baseline_results
                ),
                baseline_output_tokens=_average_int(result["output_tokens"] for result in baseline_results),
                baseline_duration_seconds=_average_float(
                    result["duration_seconds"] for result in baseline_results
                ),
            )

    @staticmethod
    def _load_result_payload(
        *,
        result_payload_path: Path,
        stdout_text: str,
        case_dir: Path,
    ) -> dict[str, Any]:
        if result_payload_path.is_file():
            return json.loads(result_payload_path.read_text(encoding="utf-8"))

        parsed = _parse_soma_benchmark_result(case_dir)
        if parsed:
            return parsed

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

    def _build_command(
        self,
        *,
        submission_row: dict[str, Any] | None,
        case_row: dict[str, Any],
        case_dir: Path,
        baseline: bool,
    ) -> tuple[str, str]:
        if submission_row is not None and submission_row["entry_command"].strip():
            return submission_row["entry_command"], submission_row["submission_root"]

        if self.settings.soma_benchmark_repo is None:
            raise RuntimeError(
                "SOMA_BENCHMARK_REPO must point to a local DendriteHQ/SOMA-benchmark checkout."
            )

        dataset_name = case_row["dataset_name"] or _default_dataset_name(case_row["benchmark_type"])
        command = shlex.split(self.settings.soma_benchmark_runner)
        command.extend(
            [
                "benchmark-solve",
                "--agent-name",
                self.settings.default_agent_name,
                "--benchmark",
                dataset_name,
                "--instance-id",
                case_row["instance_id"],
                "--benchmark-type",
                case_row["benchmark_type"],
                "--output-dir",
                str(case_dir / "soma-benchmark"),
            ]
        )
        if not baseline:
            if submission_row is None:
                raise RuntimeError("Miner submission is required for compressed evaluation.")
            command.extend(["--copilot-compression-script-path", submission_row["compressor_path"]])
        command.extend(["--execute", "--swerebench-eval"])
        return (
            " ".join(shlex.quote(part) for part in command),
            str(self.settings.soma_benchmark_repo),
        )


def _default_dataset_name(benchmark_type: str) -> str:
    if benchmark_type == "swebench_verified":
        return "SWE-bench/SWE-bench_Verified"
    return "SWE-Explore-Bench/SWE-Explore-Bench"


def _parse_soma_benchmark_result(case_dir: Path) -> dict[str, Any]:
    bench_dir = case_dir / "soma-benchmark"
    output_jsonl = bench_dir / "output.jsonl"
    rows: list[dict[str, Any]] = []
    if output_jsonl.is_file():
        for line in output_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    row = rows[-1] if rows else {}
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    token_usage = metadata.get("token_usage") if isinstance(metadata.get("token_usage"), dict) else {}
    patch_eval = _load_json(bench_dir / "evaluation-summary.json")
    patch_capture = metadata.get("patch_capture") if isinstance(metadata.get("patch_capture"), dict) else {}

    resolved = bool(patch_eval.get("resolved")) if patch_eval else bool(row.get("resolved", False))
    input_tokens = _first_int(
        token_usage,
        "input_tokens",
        "prompt_tokens",
        "total_input_tokens",
    )
    cached_input_tokens = _cached_input_tokens(token_usage)
    output_tokens = _first_int(
        token_usage,
        "output_tokens",
        "completion_tokens",
        "total_output_tokens",
    )

    patch_path = patch_capture.get("patch_path") if isinstance(patch_capture.get("patch_path"), str) else ""
    patch_text = ""
    if patch_path and Path(patch_path).is_file():
        patch_text = Path(patch_path).read_text(encoding="utf-8")

    return {
        "resolved": resolved,
        "patch": patch_text,
        "metrics": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "duration_seconds": _first_float(metadata, "duration_seconds", "elapsed_seconds"),
            "files_hit_rate": _first_float(metadata, "files_hit_rate", "hit_file_rate"),
            "noise_rate": _first_float(metadata, "noise_rate"),
        },
        "artifacts": {
            "soma_benchmark_output": row,
            "evaluation_summary": patch_eval,
            "output_jsonl": str(output_jsonl),
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _first_int(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return int(value)
    return None


def _first_float(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return float(value)
    return None


def _cached_input_tokens(data: dict[str, Any]) -> int | None:
    direct = _first_int(data, "cached_input_tokens", "total_cached_input_tokens")
    if direct is not None:
        return direct
    cache_read = data.get("cache_read_tokens")
    cache_creation = data.get("cache_creation_tokens")
    if cache_read is None and cache_creation is None:
        return None
    return int(cache_read or 0) + int(cache_creation or 0)


def _average_int(values: Any) -> int | None:
    filtered = [int(value) for value in values if value is not None]
    if not filtered:
        return None
    return int(round(sum(filtered) / len(filtered)))


def _average_float(values: Any) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)
