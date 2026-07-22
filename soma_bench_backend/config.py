from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str
    data_dir: Path
    database_path: Path
    runs_dir: Path
    default_timeout_seconds: float
    max_attempts_per_case: int
    input_token_weight: float
    cached_input_token_weight: float
    output_token_weight: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.getenv("SOMA_BENCH_DATA_DIR", "data")).expanduser().resolve()
    db_path = Path(
        os.getenv("SOMA_BENCH_DB_PATH", str(data_dir / "soma-bench.db"))
    ).expanduser().resolve()
    runs_dir = data_dir / "runs"

    data_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        app_name="SOMA Local Benchmark Backend",
        data_dir=data_dir,
        database_path=db_path,
        runs_dir=runs_dir,
        default_timeout_seconds=float(
            os.getenv("SOMA_BENCH_DEFAULT_TIMEOUT_SECONDS", "1800")
        ),
        max_attempts_per_case=max(
            1, int(os.getenv("SOMA_BENCH_MAX_ATTEMPTS_PER_CASE", "5"))
        ),
        input_token_weight=float(os.getenv("SOMA_BENCH_INPUT_TOKEN_WEIGHT", "1.0")),
        cached_input_token_weight=float(
            os.getenv("SOMA_BENCH_CACHED_INPUT_TOKEN_WEIGHT", "0.1")
        ),
        output_token_weight=float(os.getenv("SOMA_BENCH_OUTPUT_TOKEN_WEIGHT", "3.0")),
    )
