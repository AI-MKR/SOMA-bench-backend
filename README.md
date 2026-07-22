# SOMA Local Benchmark Backend

Backend-only local benchmark system for SOMA-style miner evaluation.

This project is intentionally shaped around what is visible in:

- `DendriteHQ/SOMA`
- `DendriteHQ/SOMA-benchmark`
- `https://thesoma.ai/dashboard`

The implemented local backend mirrors the core SOMA flow:

1. create a competition
2. register benchmark cases with baseline metrics
3. register miner submissions
4. run local evaluations against those cases
5. compute leaderboard scores from quality + efficiency

## What this backend implements

- `FastAPI` service
- `SQLite` persistence
- competition registry
- benchmark case registry
- miner submission registry
- background local evaluator
- per-case artifact capture
- SOMA-style normalized SWE scoring
- explicit quality and efficiency scores
- leaderboard API
- dashboard-style API payload

## What it does not implement

- frontend
- bittensor wallet/signature flow
- validator heartbeats
- OpenClaw/Copilot orchestration
- distributed validators

Those pieces are intentionally omitted because the requirement here is local backend benchmarking.

## Observed SOMA behavior this project follows

From the current public code and dashboard:

- miners submit one solution for an active competition
- validators pull benchmark work and submit per-task results
- leaderboard entries expose category scores and an overall normalized score
- current live dashboard data is code-evaluation oriented, with categories such as:
  - `swebench_verified`
  - `swe_explorer_edit`
- scoring mixes quality and efficiency via weighted token usage

## Miner compressor contract

The miner code being evaluated is a context compressor. It is called between the
agent and the model during a SOMA-benchmark run.

Expected miner file:

```python
def compress_messages(messages=None, path=None, metadata=None) -> list:
    return messages if isinstance(messages, list) else []
```

The backend registers each miner submission with:

- `submission_root`: working directory
- `compressor_path`: Python file exposing `compress_messages(...)`
- `entry_command`: optional override for custom command-mode evaluation

Normal SOMA-style evaluation does not call `entry_command`. It runs:

```bash
uv run python -m soma_bench benchmark-solve \
  --agent-name copilot \
  --benchmark DATASET \
  --instance-id INSTANCE_ID \
  --benchmark-type BENCHMARK_TYPE \
  --execute \
  --swerebench-eval \
  --copilot-compression-script-path /path/to/miner.py
```

The command is executed inside the local `DendriteHQ/SOMA-benchmark` checkout
configured by `SOMA_BENCHMARK_REPO`.

The older command-mode path is still supported for custom local experiments. In
that mode the command should write this result shape:

```json
{
  "resolved": true,
  "patch": "--- optional diff text ---",
  "metrics": {
    "input_tokens": 1200,
    "cached_input_tokens": 500,
    "output_tokens": 220,
    "duration_seconds": 18.4,
    "files_hit_rate": 0.9,
    "noise_rate": 0.1
  },
  "notes": "optional",
  "artifacts": {
    "anything": "optional"
  }
}
```

`files_hit_rate` and `noise_rate` are optional unless you want to add your own
exploration-style analysis. The current backend score path uses the SWE-style
quality/efficiency path for `swebench_verified` and `swe_explorer_edit`.

## Benchmark case import contract

Each case should include:

- `instance_id`
- `benchmark_type`
- `dataset_name`
- `split`
- `prompt`
- baseline quality count:
  - `baseline_resolved_count`
- baseline token metrics:
  - `baseline_input_tokens`
  - `baseline_cached_input_tokens`
  - `baseline_output_tokens`

Minimal example:

```json
{
  "cases": [
    {
      "instance_id": "django__django-12345",
      "benchmark_type": "swebench_verified",
      "dataset_name": "SWE-bench/SWE-bench_Verified",
      "split": "test",
      "title": "Fix failing admin filter behavior",
      "repo": "django/django",
      "prompt": "Problem statement here",
      "baseline_resolved_count": 3,
      "baseline_input_tokens": 8200,
      "baseline_cached_input_tokens": 1500,
      "baseline_output_tokens": 900
    }
  ]
}
```

You can also import directly from Hugging Face:

```http
POST /competitions/1/cases/import-huggingface
```

```json
{
  "dataset_name": "SWE-bench/SWE-bench_Verified",
  "benchmark_type": "swebench_verified",
  "split": "test",
  "limit": 10
}
```

For SWE-Explore:

```json
{
  "dataset_name": "SWE-Explore-Bench/SWE-Explore-Bench",
  "benchmark_type": "swe_explorer_explore",
  "split": "test"
}
```

## API overview

- `GET /health`
- `POST /competitions`
- `GET /competitions`
- `POST /competitions/{competition_id}/cases/import`
- `POST /competitions/{competition_id}/cases/import-huggingface`
- `GET /competitions/{competition_id}/cases`
- `POST /competitions/{competition_id}/submissions`
- `GET /competitions/{competition_id}/submissions`
- `POST /competitions/{competition_id}/evaluations`
- `GET /evaluations/{evaluation_id}`
- `GET /competitions/{competition_id}/leaderboard`
- `GET /competitions/{competition_id}/dashboard`

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn soma_bench_backend.main:app --reload
```

Default storage:

- database: `./data/soma-bench.db`
- run artifacts: `./data/runs`

Override with:

- `SOMA_BENCH_DATA_DIR`
- `SOMA_BENCH_DB_PATH`
- `SOMA_BENCHMARK_REPO`
- `SOMA_BENCHMARK_RUNNER`
- `SOMA_BENCH_DEFAULT_AGENT_NAME`

## Notes on scoring

This backend uses the public SWE scoring formulas documented in SOMA's
`docs/miner/scoring.md` and reflected by the platform scoring route:

- weighted token usage:
  - input = `1.0`
  - cached input = `0.1`
  - output = `3.0`
- per-task score combines:
  - baseline resolved count
  - miner resolved count
  - compression ratio from baseline tokens vs miner tokens
- category score is normalized to `[-1, 1]`
- overall score is the mean of category scores for the submission
- `quality_score` is a normalized resolution score based on miner resolved
  attempts versus baseline resolved counts
- `efficiency_score` is a normalized weighted-token savings score

Each benchmark problem is evaluated exactly 5 miner attempts. If baseline token
metrics are missing, the backend first runs the same problem 5 times without the
miner compressor and caches those baseline metrics on the benchmark case.

For `swe_explorer_explore`, quality uses exploration metrics when available:

```text
quality = files_hit_rate - noise_rate
```

Token savings are only rewarded when exploration quality is preserved.

## Suggested next step

Populate a competition with real SWE-bench-style cases and baseline metrics from
your local SOMA benchmark data, then point miner submissions at a wrapper
command that emits the result JSON described above.
