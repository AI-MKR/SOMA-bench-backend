from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log, log2
from typing import Any

from .config import Settings


@dataclass
class TaskScore:
    benchmark_case_id: int
    instance_id: str
    benchmark_type: str
    x: int
    y: int
    attempts: int
    tokens_without_compression: float | None
    tokens_with_compression: float | None
    baseline_tokens: dict[str, float | int | None]
    compressed_tokens: dict[str, float | int | None]
    score: float | None
    pool: str
    hard_boost: float
    quality_ratio: float
    efficiency_ratio: float | None
    efficiency_score: float


def compute_weighted_tokens(
    *,
    input_tokens: int | None,
    cached_input_tokens: int | None,
    output_tokens: int | None,
    settings: Settings,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    cached_input_tokens = 0 if cached_input_tokens is None else cached_input_tokens
    if input_tokens < 0 or cached_input_tokens < 0 or output_tokens < 0:
        return None
    return (
        settings.input_token_weight * float(input_tokens)
        + settings.cached_input_token_weight * float(cached_input_tokens)
        + settings.output_token_weight * float(output_tokens)
    )


def _compression_ratio(
    tokens_without_compression: float | None,
    tokens_with_compression: float | None,
) -> float:
    if (
        tokens_without_compression is None
        or tokens_with_compression is None
        or tokens_without_compression <= 0
        or tokens_with_compression <= 0
    ):
        return 0.0
    ratio = float(tokens_without_compression) / float(tokens_with_compression)
    return max(-2.0, min(log(ratio), 2.0))


def compute_swe_task_score(
    *,
    x: int,
    y: int,
    tokens_without_compression: float | None,
    tokens_with_compression: float | None,
) -> tuple[float | None, str, float]:
    if x <= 1:
        if y == 0:
            return None, "excluded", 0.0
        r = _compression_ratio(tokens_without_compression, tokens_with_compression)
        if x == 1 and y == 1:
            score = r
        else:
            denom = max(1, 5 - x)
            score = max(-2.0, min(r + ((y - x) / denom), 3.0))
        return score, "hard_boost", max(0.0, score)

    penalty_threshold = int(0.8 * x)
    penalty_threshold = max(1, penalty_threshold)
    r = _compression_ratio(tokens_without_compression, tokens_with_compression)
    if y < penalty_threshold:
        score = max(-4.0, min(-2.0 - 2.0 * (1.0 - (y / penalty_threshold)), -2.0))
    elif y <= x:
        score = r
    else:
        denom = max(1, 5 - x)
        score = max(-2.0, min(r + ((y - x) / denom), 3.0))
    return score, "main", 0.0


def compute_explore_task_score(
    *,
    miner_quality: float,
    baseline_quality: float,
    tokens_without_compression: float | None,
    tokens_with_compression: float | None,
) -> float:
    quality_margin = miner_quality - baseline_quality
    if quality_margin <= -0.2:
        return -2.0
    if (
        tokens_without_compression is None
        or tokens_with_compression is None
        or tokens_without_compression <= 0
        or tokens_with_compression <= 0
    ):
        token_score = 0.0
    else:
        token_score = max(-2.0, min(2.0 * log2(tokens_without_compression / tokens_with_compression), 2.0))
    quality_gate = 1.0 if quality_margin >= 0 else max(0.0, (quality_margin + 0.2) / 0.2)
    return quality_gate * token_score


def _normalized_score(raw_score: float) -> float:
    clamped = max(-4.0, min(raw_score, 3.0))
    return (2.0 * clamped + 1.0) / 7.0


def _quality_ratio(*, baseline_resolved_count: int, miner_resolved_count: int) -> float:
    if baseline_resolved_count <= 0:
        return 1.0 if miner_resolved_count > 0 else 0.0
    return max(0.0, min(float(miner_resolved_count) / float(baseline_resolved_count), 1.0))


def _efficiency_components(
    *,
    tokens_without_compression: float | None,
    tokens_with_compression: float | None,
) -> tuple[float | None, float]:
    if (
        tokens_without_compression is None
        or tokens_with_compression is None
        or tokens_without_compression <= 0
        or tokens_with_compression <= 0
    ):
        return None, 0.0
    ratio = float(tokens_without_compression) / float(tokens_with_compression)
    return ratio, max(-1.0, min(_compression_ratio(tokens_without_compression, tokens_with_compression) / 2.0, 1.0))


def _average_optional_number(values: list[int | float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _token_breakdown(
    *,
    input_tokens: int | float | None,
    cached_input_tokens: int | float | None,
    output_tokens: int | float | None,
    settings: Settings,
) -> dict[str, float | int | None]:
    weighted = compute_weighted_tokens(
        input_tokens=int(input_tokens) if input_tokens is not None else None,
        cached_input_tokens=int(cached_input_tokens) if cached_input_tokens is not None else None,
        output_tokens=int(output_tokens) if output_tokens is not None else None,
        settings=settings,
    )
    return {
        "input": input_tokens,
        "cache": cached_input_tokens,
        "output": output_tokens,
        "weighted": weighted,
    }


def _compressed_token_breakdown(
    runs: list[dict[str, Any]],
    settings: Settings,
) -> dict[str, float | int | None]:
    completed_runs = [row for row in runs if row["status"] == "completed"]
    return _token_breakdown(
        input_tokens=_average_optional_number(
            [row.get("input_tokens") for row in completed_runs]
        ),
        cached_input_tokens=_average_optional_number(
            [row.get("cached_input_tokens") for row in completed_runs]
        ),
        output_tokens=_average_optional_number(
            [row.get("output_tokens") for row in completed_runs]
        ),
        settings=settings,
    )


def _token_savings(
    *,
    baseline_weighted: float | None,
    compressed_weighted: float | None,
) -> dict[str, float | None]:
    if (
        baseline_weighted is None
        or compressed_weighted is None
        or baseline_weighted <= 0
    ):
        return {
            "weighted_saved": None,
            "weighted_ratio": None,
            "weighted_saved_percent": None,
        }
    saved = baseline_weighted - compressed_weighted
    return {
        "weighted_saved": saved,
        "weighted_ratio": compressed_weighted / baseline_weighted,
        "weighted_saved_percent": (saved / baseline_weighted) * 100.0,
    }


def build_category_score(
    *,
    benchmark_type: str,
    case_rows: list[dict[str, Any]],
    case_result_rows: dict[int, list[dict[str, Any]]],
    settings: Settings,
) -> tuple[float, dict[str, Any]]:
    task_scores: list[TaskScore] = []
    if benchmark_type == "swe_explorer_explore":
        return _build_explore_category_score(
            benchmark_type=benchmark_type,
            case_rows=case_rows,
            case_result_rows=case_result_rows,
            settings=settings,
        )

    main_scores: list[tuple[float, float]] = []
    hard_boost_total = 0.0

    for case_row in case_rows:
        runs = case_result_rows.get(case_row["id"], [])
        completed_runs = [row for row in runs if row["status"] == "completed"]
        resolved_runs = [row for row in runs if bool(row["resolved"])]
        miner_tokens = [
            value
            for value in (
                compute_weighted_tokens(
                    input_tokens=row["input_tokens"],
                    cached_input_tokens=row["cached_input_tokens"],
                    output_tokens=row["output_tokens"],
                    settings=settings,
                )
                for row in resolved_runs
            )
            if value is not None and value > 0
        ]
        tok_a = sum(miner_tokens) / len(miner_tokens) if miner_tokens else None
        tok_b = compute_weighted_tokens(
            input_tokens=case_row["baseline_input_tokens"],
            cached_input_tokens=case_row["baseline_cached_input_tokens"],
            output_tokens=case_row["baseline_output_tokens"],
            settings=settings,
        )
        baseline_tokens = _token_breakdown(
            input_tokens=case_row["baseline_input_tokens"],
            cached_input_tokens=case_row["baseline_cached_input_tokens"],
            output_tokens=case_row["baseline_output_tokens"],
            settings=settings,
        )
        compressed_tokens = _compressed_token_breakdown(runs, settings)
        x = int(case_row["baseline_resolved_count"])
        y = len(resolved_runs)
        score, pool, hard_boost = compute_swe_task_score(
            x=x,
            y=y,
            tokens_without_compression=tok_b,
            tokens_with_compression=tok_a,
        )
        efficiency_ratio, efficiency_score = _efficiency_components(
            tokens_without_compression=tok_b,
            tokens_with_compression=tok_a,
        )
        task_scores.append(
            TaskScore(
                benchmark_case_id=case_row["id"],
                instance_id=str(case_row.get("instance_id") or case_row["id"]),
                benchmark_type=benchmark_type,
                x=x,
                y=y,
                attempts=len(completed_runs),
                tokens_without_compression=tok_b,
                tokens_with_compression=tok_a,
                baseline_tokens=baseline_tokens,
                compressed_tokens=compressed_tokens,
                score=score,
                pool=pool,
                hard_boost=hard_boost,
                quality_ratio=_quality_ratio(
                    baseline_resolved_count=x,
                    miner_resolved_count=y,
                ),
                efficiency_ratio=efficiency_ratio,
                efficiency_score=efficiency_score,
            )
        )
        if score is not None and pool == "main":
            weight = float(x) ** (1.0 / 3.0)
            main_scores.append((score, weight))
        if pool == "hard_boost":
            hard_boost_total += hard_boost

    main_score = (
        sum(score * weight for score, weight in main_scores)
        / sum(weight for _, weight in main_scores)
        if main_scores
        else 0.0
    )
    scored_tasks = [item for item in task_scores if item.score is not None]
    hard_boost = hard_boost_total / len(scored_tasks) if scored_tasks else 0.0
    raw_total = main_score + hard_boost
    normalized = _normalized_score(raw_total)
    quality_score = (
        (2.0 * (sum(item.quality_ratio for item in task_scores) / len(task_scores))) - 1.0
        if task_scores
        else -1.0
    )
    efficiency_score = (
        sum(item.efficiency_score for item in task_scores) / len(task_scores)
        if task_scores
        else -1.0
    )

    summary = {
        "benchmark_type": benchmark_type,
        "task_count": len(case_rows),
        "scored_task_count": len(scored_tasks),
        "main_score": main_score,
        "hard_boost": hard_boost,
        "raw_total": raw_total,
        "normalized_score": normalized,
        "quality_score": quality_score,
        "efficiency_score": efficiency_score,
        "tasks": [
            {
                "benchmark_case_id": item.benchmark_case_id,
                "instance_id": item.instance_id,
                "benchmark_type": item.benchmark_type,
                "attempts": item.attempts,
                "passed_without_compression": item.x,
                "passed_with_compression": item.y,
                "x": item.x,
                "y": item.y,
                "score": item.score,
                "pool": item.pool,
                "hard_boost": item.hard_boost,
                "tokens_without_compression": item.tokens_without_compression,
                "tokens_with_compression": item.tokens_with_compression,
                "baseline_tokens": item.baseline_tokens,
                "compressed_tokens": item.compressed_tokens,
                "token_savings": _token_savings(
                    baseline_weighted=item.baseline_tokens["weighted"],
                    compressed_weighted=item.compressed_tokens["weighted"],
                ),
                "quality_ratio": item.quality_ratio,
                "efficiency_ratio": item.efficiency_ratio,
                "efficiency_score": item.efficiency_score,
            }
            for item in task_scores
        ],
    }
    return normalized, summary


def _build_explore_category_score(
    *,
    benchmark_type: str,
    case_rows: list[dict[str, Any]],
    case_result_rows: dict[int, list[dict[str, Any]]],
    settings: Settings,
) -> tuple[float, dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for case_row in case_rows:
        runs = case_result_rows.get(case_row["id"], [])
        completed_runs = [row for row in runs if row["status"] == "completed"]
        tok_b = compute_weighted_tokens(
            input_tokens=case_row["baseline_input_tokens"],
            cached_input_tokens=case_row["baseline_cached_input_tokens"],
            output_tokens=case_row["baseline_output_tokens"],
            settings=settings,
        )
        baseline_tokens = _token_breakdown(
            input_tokens=case_row["baseline_input_tokens"],
            cached_input_tokens=case_row["baseline_cached_input_tokens"],
            output_tokens=case_row["baseline_output_tokens"],
            settings=settings,
        )
        compressed_tokens = _compressed_token_breakdown(runs, settings)
        miner_tokens = [
            value
            for value in (
                compute_weighted_tokens(
                    input_tokens=row["input_tokens"],
                    cached_input_tokens=row["cached_input_tokens"],
                    output_tokens=row["output_tokens"],
                    settings=settings,
                )
                for row in completed_runs
            )
            if value is not None and value > 0
        ]
        tok_a = sum(miner_tokens) / len(miner_tokens) if miner_tokens else None
        baseline_quality = _quality_from_rates(
            hit_rate=case_row["baseline_hit_file_rate"],
            noise_rate=case_row["baseline_noise_rate"],
            fallback=1.0,
        )
        miner_quality_values = [
            _quality_from_rates(
                hit_rate=row["files_hit_rate"],
                noise_rate=row["noise_rate"],
                fallback=1.0 if bool(row["resolved"]) else 0.0,
            )
            for row in completed_runs
        ]
        miner_quality = (
            sum(miner_quality_values) / len(miner_quality_values)
            if miner_quality_values
            else 0.0
        )
        score = compute_explore_task_score(
            miner_quality=miner_quality,
            baseline_quality=baseline_quality,
            tokens_without_compression=tok_b,
            tokens_with_compression=tok_a,
        )
        efficiency_ratio, efficiency_score = _efficiency_components(
            tokens_without_compression=tok_b,
            tokens_with_compression=tok_a,
        )
        tasks.append(
            {
                "benchmark_case_id": case_row["id"],
                "instance_id": str(case_row.get("instance_id") or case_row["id"]),
                "benchmark_type": benchmark_type,
                "attempts": len(completed_runs),
                "passed_without_compression": int(case_row.get("baseline_resolved_count") or 0),
                "passed_with_compression": sum(1 for row in completed_runs if bool(row["resolved"])),
                "score": score,
                "normalized_score": max(-1.0, min(score / 2.0, 1.0)),
                "baseline_quality": baseline_quality,
                "miner_quality": miner_quality,
                "quality_ratio": max(0.0, min(miner_quality / baseline_quality, 1.0))
                if baseline_quality > 0
                else miner_quality,
                "efficiency_ratio": efficiency_ratio,
                "efficiency_score": efficiency_score,
                "tokens_without_compression": tok_b,
                "tokens_with_compression": tok_a,
                "baseline_tokens": baseline_tokens,
                "compressed_tokens": compressed_tokens,
                "token_savings": _token_savings(
                    baseline_weighted=baseline_tokens["weighted"],
                    compressed_weighted=compressed_tokens["weighted"],
                ),
            }
        )

    normalized = (
        sum(task["normalized_score"] for task in tasks) / len(tasks)
        if tasks
        else -1.0
    )
    quality_score = (
        (2.0 * (sum(task["quality_ratio"] for task in tasks) / len(tasks))) - 1.0
        if tasks
        else -1.0
    )
    efficiency_score = (
        sum(task["efficiency_score"] for task in tasks) / len(tasks)
        if tasks
        else -1.0
    )
    return normalized, {
        "benchmark_type": benchmark_type,
        "task_count": len(case_rows),
        "scored_task_count": len(tasks),
        "normalized_score": normalized,
        "quality_score": quality_score,
        "efficiency_score": efficiency_score,
        "tasks": tasks,
    }


def _quality_from_rates(
    *,
    hit_rate: float | None,
    noise_rate: float | None,
    fallback: float,
) -> float:
    if hit_rate is None:
        return fallback
    return max(-1.0, min(float(hit_rate) - float(noise_rate or 0.0), 1.0))


def _sum_optional_numbers(values: list[int | float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered)


def _aggregate_detail_metrics(
    *,
    competition_row: dict[str, Any],
    case_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    category_summaries: dict[str, Any],
    overall_score: float,
    quality_score: float,
    efficiency_score: float,
    screener_passed: bool,
) -> dict[str, Any]:
    task_details = [
        task
        for summary in category_summaries.values()
        for task in summary.get("tasks", [])
    ]
    baseline_input = _sum_optional_numbers(
        [case.get("baseline_input_tokens") for case in case_rows]
    )
    baseline_cache = _sum_optional_numbers(
        [case.get("baseline_cached_input_tokens") for case in case_rows]
    )
    baseline_output = _sum_optional_numbers(
        [case.get("baseline_output_tokens") for case in case_rows]
    )
    baseline_weighted = _sum_optional_numbers(
        [
            task.get("baseline_tokens", {}).get("weighted")
            for task in task_details
        ]
    )
    compressed_input = _sum_optional_numbers(
        [
            task.get("compressed_tokens", {}).get("input")
            for task in task_details
        ]
    )
    compressed_cache = _sum_optional_numbers(
        [
            task.get("compressed_tokens", {}).get("cache")
            for task in task_details
        ]
    )
    compressed_output = _sum_optional_numbers(
        [
            task.get("compressed_tokens", {}).get("output")
            for task in task_details
        ]
    )
    compressed_weighted = _sum_optional_numbers(
        [
            task.get("compressed_tokens", {}).get("weighted")
            for task in task_details
        ]
    )
    passed_without_compression = sum(
        int(case.get("baseline_resolved_count") or 0)
        for case in case_rows
    )
    passed_with_compression = sum(1 for row in result_rows if bool(row["resolved"]))
    return {
        "screener": {
            "passed": screener_passed,
            "status": "passed" if screener_passed else "failed",
            "threshold": float(competition_row["screening_threshold"]),
            "score": overall_score,
        },
        "tasks": len(case_rows),
        "attempts": len(result_rows),
        "passed_without_compression": passed_without_compression,
        "passed_with_compression": passed_with_compression,
        "tokens_without_compression": {
            "input": baseline_input,
            "cache": baseline_cache,
            "output": baseline_output,
            "weighted": baseline_weighted,
        },
        "tokens_with_compression": {
            "input": compressed_input,
            "cache": compressed_cache,
            "output": compressed_output,
            "weighted": compressed_weighted,
        },
        "baseline_weighted": baseline_weighted,
        "total_weighted": compressed_weighted,
        "token_savings": _token_savings(
            baseline_weighted=baseline_weighted,
            compressed_weighted=compressed_weighted,
        ),
        "scores": {
            "overall": overall_score,
            "quality": quality_score,
            "efficiency": efficiency_score,
        },
        "task_details": task_details,
    }


def build_leaderboard_entry(
    *,
    competition_row: dict[str, Any],
    submission_row: dict[str, Any],
    case_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    evaluation_id: int,
    settings: Settings,
) -> dict[str, Any]:
    grouped_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_results: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for case_row in case_rows:
        grouped_cases[case_row["benchmark_type"]].append(case_row)
    for result_row in result_rows:
        grouped_results[result_row["benchmark_case_id"]].append(result_row)

    category_scores: dict[str, float] = {}
    category_summaries: dict[str, Any] = {}
    category_quality_scores: dict[str, float] = {}
    category_efficiency_scores: dict[str, float] = {}
    completed_attempts = sum(1 for row in result_rows if row["status"] == "completed")
    failed_attempts = sum(1 for row in result_rows if row["status"] == "failed")
    timed_out_attempts = sum(1 for row in result_rows if row["status"] == "timed_out")
    resolved_attempts = sum(1 for row in result_rows if bool(row["resolved"]))

    for benchmark_type, typed_cases in grouped_cases.items():
        score, summary = build_category_score(
            benchmark_type=benchmark_type,
            case_rows=typed_cases,
            case_result_rows=grouped_results,
            settings=settings,
        )
        category_scores[benchmark_type] = score
        category_summaries[benchmark_type] = summary
        category_quality_scores[benchmark_type] = float(summary["quality_score"])
        category_efficiency_scores[benchmark_type] = float(summary["efficiency_score"])

    overall_score = (
        sum(category_scores.values()) / len(category_scores) if category_scores else -1.0
    )
    quality_score = (
        sum(category_quality_scores.values()) / len(category_quality_scores)
        if category_quality_scores
        else -1.0
    )
    efficiency_score = (
        sum(category_efficiency_scores.values()) / len(category_efficiency_scores)
        if category_efficiency_scores
        else -1.0
    )
    screener_passed = overall_score >= float(competition_row["screening_threshold"])
    status = "qualified" if screener_passed else "not_qualified"
    evaluation_details = _aggregate_detail_metrics(
        competition_row=competition_row,
        case_rows=case_rows,
        result_rows=result_rows,
        category_summaries=category_summaries,
        overall_score=overall_score,
        quality_score=quality_score,
        efficiency_score=efficiency_score,
        screener_passed=screener_passed,
    )

    summary = {
        "competition_name": competition_row["name"],
        "screener": evaluation_details["screener"],
        "tasks": evaluation_details["tasks"],
        "passed_without_compression": evaluation_details["passed_without_compression"],
        "passed_with_compression": evaluation_details["passed_with_compression"],
        "tokens_without_compression": evaluation_details["tokens_without_compression"],
        "tokens_with_compression": evaluation_details["tokens_with_compression"],
        "total_weighted": evaluation_details["total_weighted"],
        "baseline_weighted": evaluation_details["baseline_weighted"],
        "token_savings": evaluation_details["token_savings"],
        "attempt_count": len(result_rows),
        "completed_attempts": completed_attempts,
        "failed_attempts": failed_attempts,
        "timed_out_attempts": timed_out_attempts,
        "resolved_attempts": resolved_attempts,
        "score_components": {
            "overall_score": overall_score,
            "quality_score": quality_score,
            "efficiency_score": efficiency_score,
            "category_quality_scores": category_quality_scores,
            "category_efficiency_scores": category_efficiency_scores,
        },
        "evaluation_details": evaluation_details,
        "category_summaries": category_summaries,
    }

    return {
        "competition_id": int(competition_row["id"]),
        "submission_id": int(submission_row["id"]),
        "evaluation_id": int(evaluation_id),
        "miner_hotkey": submission_row["miner_hotkey"],
        "display_name": submission_row["display_name"] or submission_row["miner_hotkey"],
        "overall_score": overall_score,
        "quality_score": quality_score,
        "efficiency_score": efficiency_score,
        "status": status,
        "screener_passed": screener_passed,
        "category_scores": category_scores,
        "summary": summary,
    }
