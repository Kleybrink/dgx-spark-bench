# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated benchmarking framework for local LLMs served by Ollama on an NVIDIA DGX Spark. Measures latency, throughput, and answer quality (via LLM-as-a-Judge) across multiple models, prompts, and reasoning modes.

## Commands

```bash
# Install dependencies (Python 3.12+, uv preferred)
uv sync

# Run the full benchmark
uv run python benchmark.py

# Alternative: activate venv first
source .venv/bin/activate && python benchmark.py
```

There are no tests, linter config, or CI. The project is a single-purpose benchmarking tool.

## Architecture

**Entry point:** `benchmark.py` — loads config, creates WIP tracker, runs benchmark, saves results.

**Pipeline flow:** `config.yaml` → orchestrator → runner → judge → analysis → plots/CSVs

Key modules in `ollama_benchmark/`:

- `**orchestrator.py`** — Main loop: iterates model x think_mode x prompt x run. Handles warmup, cleanup between models, judging, and timeout tracking. Calls `save_results()` to produce final output.
- `**runner.py**` — Executes a single inference via streaming HTTP (`httpx`) to Ollama's `/api/chat`. Collects wall time, TTFT, decode throughput, and GPU metrics. Returns `None` on timeout.
- `**judge.py**` — Sends model response + golden answer to a separate judge LLM. Scores 0.0/0.5/1.0 using a system prompt rubric. Non-streaming request.
- `**wip.py**` — SQLite-backed resumability. Tracks completed (model, think, prompt, run) tuples. Re-running `benchmark.py` skips finished combinations.
- `**cleanup.py**` — Multi-stage VRAM release between models (API unload, process kill, page cache drop).
- `**warmup.py**` — Validates model availability and think-mode support before benchmarking.
- `**gpu.py**` — Background `nvidia-smi` sampling thread (every 0.5s during inference).
- `**analysis.py**` — Builds summary DataFrames, category breakdowns, and model rankings from raw results.
- `**plots.py**` — Generates PNG charts (with `adjustText` for label collision avoidance) and exports CSVs to `results/<timestamp>/`.
- `**display.py**` — Live leaderboard during benchmark and final formatted output. Both sorted by composite score.
- `**config.py**` — Loads and validates `config.yaml` into a `BenchmarkConfig` dataclass.

## Key Design Details

- **Resumability:** WIP state lives in `results/wip/wip.db` (SQLite with WAL). Each completed run is persisted immediately. Delete `wip.db` to start fresh.
- **Think modes:** Every model is tested in both think (reasoning) and no-think (direct) modes. Models that don't support thinking are auto-detected during warmup and skipped.
- **Prompt naming:** Prompts use `category--difficulty--short_name` convention (e.g., `math--easy--arithmetic`). The `--` delimiter drives automatic category-level aggregation in analysis.
- **Timeout:** 5-minute default (`timeout_s: 300` in config). Timed-out runs are recorded as failures with `judge_score: 0.0`.
- **Judge:** Uses a separate Ollama model (configured via `judge_model` in `config.yaml`). The judge scores against golden answers embedded in the config.
- **Results:** Written to `results/<timestamp>/` with CSVs, PNGs, and REPORT.md. A `results/viewer.html` provides interactive browser-based viewing.
- **Composite score:** Weighted combination of pass rate (65%) and wall time (35%), using rank-based percentile normalisation. Defined in `analysis.py:_RANKING_WEIGHTS`. Only two independent, user-facing metrics — no redundant speed metrics. Leaderboards in `display.py` sort by this score.
- **Label collision avoidance:** Scatter plots use `adjustText` library to automatically reposition overlapping labels with connector lines.

