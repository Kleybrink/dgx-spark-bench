# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **LLM benchmarking framework** that measures performance of local models served by [Ollama](https://ollama.com). It runs one or more evaluation prompts across multiple models and think-modes, collecting GPU/latency/throughput metrics, scoring answer quality via an LLM judge, and produces CSV summaries and PNG charts.

## Development Setup

```bash
# Activate the virtual environment (Python 3.12.3)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Ollama must be running on localhost:11434 (or set OLLAMA_HOST env var)
```

## Running the Benchmark

```bash
# Run the full benchmark (reads config.yaml, writes to results/)
python benchmark.py

# The benchmark is resumable: interrupted runs are saved to results/wip/wip.db (SQLite)
# Re-running benchmark.py will skip already-completed model/think-mode/prompt combos

# Interactive exploration via JupyterLab
jupyter lab
```

## Configuration

All benchmark parameters live in `config.yaml`:
- `models` — list of Ollama model names to benchmark
- `prompts` — list of prompt entries, each with `name`, `prompt`, and optional `golden_answer` (supports 1 to n prompts; for backward compat a single top-level `prompt` key is auto-wrapped)
- `ollama_base_url` — overridden by `OLLAMA_HOST` env var
- `runs_per_mode` — number of repetitions per model × think-mode × prompt combination
- `keep_alive`, `timeout_s`, `warmup_timeout_s` — Ollama session and per-run timeouts
- `vram_settle_s`, `vram_free_target` — VRAM cleanup thresholds between runs
- `judge_model`, `judge_temperature`, `judge_think` — LLM judge settings (omit `judge_model` to disable judging)

## Architecture

```
benchmark.py                   ← Entry point: loads config, delegates to orchestrator
config.yaml                    ← Models, prompts, judge config, and benchmark parameters
requirements.txt               ← Direct Python dependencies (httpx, matplotlib, pandas, psutil, pyyaml)
ollama_benchmark/
  orchestrator.py              ← High-level loop: model → think → prompt → run; calls analysis/plots
  runner.py   run_one()        ← Single HTTP streaming request to /api/chat; collects all metrics
  judge.py    judge_response() ← LLM-as-a-Judge: scores answer against golden_answer
  gpu.py      GPUSampler       ← Background thread sampling nvidia-smi every 0.5s
  warmup.py   warmup()         ← Subprocess-based model validation with hard timeout
  cleanup.py  force_cleanup()  ← Multi-stage VRAM release between models
  wip.py      WIPTracker       ← SQLite-backed resumability; tracks completed run tuples
  analysis.py                  ← build_summary, print_prompt_comparison, category & ranking analysis
  display.py                   ← Live leaderboard during runs + final leaderboard from summary
  plots.py                     ← Accuracy chart, category chart, CSV export
  config.py   BenchmarkConfig  ← Loads and validates config.yaml
  logging_config.py            ← Logging setup and configuration
results/                       ← Output CSVs, PNG charts, and REPORT.md per run
results/viewer.html            ← Browser-based interactive viewer for benchmark results
results/wip/wip.db             ← Resumable in-progress state (SQLite)
```

### Key Design Decisions

**Multi-prompt support**: `config.yaml` holds a `prompts` list; each entry has `name`, `prompt`, and optional `golden_answer`. Prompt names follow the convention `category--difficulty--short_name` (e.g. `math--easy--arithmetic`), which is parsed by `analysis._parse_prompt_name()` for category-level aggregation. The benchmark loop is: model → think_mode → prompt → run. Warmup is prompt-independent (validates model + think_mode once).

**Per-run cleanup sequence** (`cleanup.force_cleanup`): API unload → kill orphaned `ollama runner` processes → drop Linux page caches → poll until VRAM < target %. This ensures a clean GPU state before each model.

**Warmup before benchmarking** (`warmup.warmup`): Runs a minimal inference in a subprocess with a hard timeout. Returns `"ok"`, `"no_think"`, `"not_found"`, `"timeout"`, or `"error"`. Models that return `"not_found"` are skipped entirely; `"no_think"` means only `think=False` runs are attempted.

**Streaming HTTP** (`runner.run_one`): Uses `httpx` to POST to `/api/chat` with `stream=True`. TTFT is captured on first token. A `GPUSampler` background thread polls `nvidia-smi` throughout. Returns `None` on timeout (the run is dropped from results). The full response text is stored in `answer_text` for judging.

**LLM-as-a-Judge** (`judge.judge_response`): After each successful run, the model's `answer_text` is sent to a separate judge model (non-streaming) together with the prompt's `golden_answer`. The judge returns `judge_score` (0.0/0.5/1.0), `judge_pass` (bool), and `judge_reasoning`. Disabled if `judge_model` is absent from config or the prompt has no `golden_answer`.

**WIP resumability**: Results are stored in `results/wip/wip.db` (SQLite, WAL mode). Each run is inserted as a row with 4 key columns (`model`, `think`, `prompt_name`, `run`) plus a `data` column containing the full row dict as JSON. On startup, completed tuples are loaded from the DB to skip already-finished runs.

**Analysis pipeline** (`save_results` in `orchestrator.py`): After all runs complete, the pipeline produces:
1. `build_summary` — groups by `(model, think, prompt_name)` and computes mean/std/median/min/max for all metrics plus `judge_pass_rate`
2. `print_prompt_comparison` — compact per-prompt table with one row per (model, think) showing score, decode tok/s, wall latency, TTFT
3. `build_model_ranking` / `print_model_ranking` — weighted composite score (judge pass rate 55%, decode tok/s 20%, e2e tok/s 10%, TTFT 7.5%, wall latency 7.5%) with rank-based (percentile) normalisation; falls back to speed-only weights when judge data is unavailable
4. `print_final_leaderboard` — detailed table sorted by decode tok/s with std and pass rate

Verbose reporting functions (`print_results`, `print_interpretation`, `print_cross_model_comparison`, `print_category_summary`) are retained in `analysis.py` for notebook/interactive use but are not called by the orchestrator.

**Output artefacts**: An accuracy chart (judge pass rate by model), a category chart (judge score by category × model), plus CSVs for raw data, summary, category summary, and model ranking.

## Linting

```bash
ruff check .
ruff format .
```
