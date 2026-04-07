# Ollama Benchmark: Local LLM Performance on NVIDIA DGX Spark

An automated benchmarking framework that measures the real-world usability of local LLMs served by [Ollama](https://ollama.com). It evaluates models across multiple prompts and reasoning modes, collecting latency/throughput metrics and scoring answer quality via an LLM-as-a-Judge pipeline.

Built for — and tested on — the [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) (128 GB unified memory, 273 GB/s bandwidth).

## Why This Exists

Public LLM leaderboards rank models by raw capability under ideal conditions. But on local hardware, a model that fits in memory and produces correct answers *slowly* can be worse than a smaller model that runs 7x faster. This benchmark answers a different question: **which models are genuinely practical for interactive local use?**

## Quick Start

**Python 3.12+** (see `.python-version`). Dependencies are declared in [`pyproject.toml`](pyproject.toml); lockfile: [`uv.lock`](uv.lock).

### With uv (recommended)

```bash
# 1. Install dependencies and the local package (editable)
uv sync

# 2. Ensure Ollama is running
#    (default: localhost:11434, or set OLLAMA_HOST)

# 3. Configure models and prompts — edit config.yaml

# 4. Run the benchmark
uv run python benchmark.py
# or: source .venv/bin/activate && python benchmark.py
```

### With pip

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
python benchmark.py
```

Results are written to `results/<timestamp>/`.

The benchmark is **resumable** — interrupted runs are saved to `results/wip/wip.db` (SQLite). Re-running `benchmark.py` skips already-completed combinations.

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│  config.yaml │────▶│  Orchestrator │────▶│  Runner      │────▶│  Judge      │
│  models,     │     │  model × mode │     │  streaming   │     │  LLM scores │
│  prompts     │     │  × prompt     │     │  HTTP to     │     │  answer vs  │
│              │     │  × run loop   │     │  Ollama API  │     │  golden     │
└─────────────┘     └──────────────┘     └──────────────┘     └─────────────┘
                           │                     │
                    ┌──────┴──────┐        ┌─────┴──────┐
                    │ Cleanup     │        │ GPU Sampler │
                    │ VRAM release│        │ nvidia-smi  │
                    │ between     │        │ every 0.5s  │
                    │ models      │        └────────────┘
                    └─────────────┘
                           │
                    ┌──────┴──────┐
                    │  Analysis   │──▶ CSVs, PNGs, leaderboards
                    └─────────────┘
```

**Per model × mode × prompt × run**, the framework:

1. **Warms up** the model with a minimal inference request
2. **Cleans up** GPU state (API unload, process kill, page cache drop)
3. **Streams** a prompt to Ollama's `/api/chat`, measuring TTFT and decode speed
4. **Judges** the response against a golden answer using a separate LLM (scoring 0.0 / 0.5 / 1.0)
5. **Persists** the result to SQLite for resumability

After all runs complete, the analysis pipeline produces:
- **Leaderboard** — sorted by pass rate (primary) and wall time (tiebreaker)
- **Category breakdown** — per-category pass rates across 9 domains
- **Charts** — accuracy, throughput scatter, wall time vs accuracy, category heatmap
- **CSVs** — raw data, summary, category summary, model ranking

## Project Structure

```
benchmark.py               Entry point
pyproject.toml             Project metadata and Python dependencies
uv.lock                    Locked dependency versions (uv)
config.yaml                Models, prompts, judge config, parameters
ollama_benchmark/
  orchestrator.py           Main loop: model → think → prompt → run
  runner.py                 Streaming HTTP to Ollama, metrics collection
  judge.py                  LLM-as-a-Judge scoring
  gpu.py                    Background nvidia-smi sampling
  warmup.py                 Model validation before benchmarking
  cleanup.py                Multi-stage VRAM release between models
  wip.py                    SQLite-backed resumability
  analysis.py               Summary, ranking, category analysis
  display.py                Live + final leaderboards
  plots.py                  Chart generation and CSV export
  config.py                 Config loading and validation
  logging_config.py         Logging setup
results/
  <timestamp>/              Per-run output (CSVs, PNGs, REPORT.md)
  viewer.html               Interactive browser-based results viewer
  wip/wip.db                In-progress state (SQLite)
```

## Configuration

All benchmark parameters live in `config.yaml`:

| Parameter | Purpose |
|-----------|---------|
| `models` | List of Ollama model tags to benchmark |
| `prompts` | List of `{name, prompt, golden_answer}` entries |
| `runs_per_mode` | Repetitions per model/mode/prompt combination |
| `timeout_s` | Max seconds per inference run |
| `judge_model` | Model used for scoring (omit to disable judging) |
| `ollama_base_url` | Ollama endpoint (overridden by `OLLAMA_HOST` env var) |

Prompt names follow the convention `category--difficulty--short_name` (e.g. `math--easy--arithmetic`) for automatic category-level aggregation.

## Evaluation Design

**21 prompts** across **9 categories** (logic, code, math, NLP, knowledge, instruction-following, spatial, data, multi-domain) at varying difficulty levels. Every prompt:

- Includes **distractor information** to test whether models ignore noise
- Requires **JSON-formatted output** to test instruction compliance
- Has a **golden answer** for automated scoring

Each inference run has a **5-minute timeout** (`timeout_s` in config). Runs that exceed this limit are discarded and count as failures.

Each model is tested in both **think** (reasoning) and **no-think** (direct) modes.

## Key Learnings

The latest benchmark results (21 models, 39 model-mode combinations, 21 prompts) produced several findings relevant to anyone running LLMs locally on bandwidth-constrained hardware. For full details, see [`REPORT.md`](results/20260407_135131/REPORT.md).

### MoE architecture is the decisive factor

On a single-GPU system, inference is memory-bandwidth-bound: `decode tok/s ≈ bandwidth / active_params`. MoE and cascade models keep all parameters in memory for quality but only read active experts per token. A 30B MoE with 3.6B active params runs at 69 tok/s — while a 27B dense model crawls at 11 tok/s. **Total parameter count is meaningless; active parameter count determines speed.**

### Think mode transforms quality

Enabling reasoning gives large quality gains for most models. The biggest jumps:

| Model | No-Think | Think | Delta |
|-------|:--------:|:-----:|:-----:|
| Nemotron Cascade 2 30B | 44% | 100% | +56 pp |
| Nemotron 3 Nano 30B | 44% | 98% | +54 pp |
| Nemotron 3 Nano 4B | 42% | 89% | +48 pp |
| GLM-4.7-Flash | 49% | 90% | +42 pp |

A 4B MoE with thinking beats a 122B model without it. Not all models benefit: `qwen3.5:27b` and `gemma4:26b-a4b-it-q8_0` slightly regress with thinking enabled.

### Dense models can compete on quality — if they're good enough

`gemma4:26b` reaches **100% pass rate** with thinking and **96.4% without** — proving that a well-tuned dense model can match MoE quality. The trade-off is speed: 27.6 tok/s decode vs 67–69 tok/s for the fastest MoE models.

### Cloud leaderboard rankings invert locally

The model with the highest cloud intelligence index (Qwen3.5:27b, AA Index 42) is the **worst** local performer (rank 38/39). Memory bandwidth constraints completely reshape the quality-speed trade-off.

### Top recommendations

| Model | Think | Pass Rate | Wall (s) | Composite | Best For |
|-------|:-----:|:---------:|:--------:|:---------:|----------|
| gpt-oss:20b | Yes | 100% | 14.3 | 0.883 | Perfect accuracy, fastest wall time |
| gpt-oss:20b | No | 100% | 15.3 | 0.864 | Perfect accuracy, no reasoning overhead |
| gpt-oss:120b | Yes | 100% | 16.1 | 0.855 | Perfect accuracy, viable 120B MoE |
| nemotron-cascade-2:30b | Yes | 100% | 18.5 | 0.828 | Perfect accuracy, fastest streaming (66.8 tok/s) |
| nemotron-3-nano:30b | Yes | 97.6% | 21.2 | 0.732 | Near-perfect, highest decode speed (69 tok/s) |

## Latest Results

Benchmark run from 2026-04-07: 21 models, 39 model-mode combinations, 21 prompts across 9 categories, 2 runs per combination. All values are means across runs. For the full analysis, see [`REPORT.md`](results/20260407_135131/REPORT.md).

### Composite Score

The leaderboard uses a weighted composite score combining two user-facing dimensions:

| Metric | Weight | Direction |
|--------|--------|-----------|
| Pass rate (quality) | 65% | Higher is better |
| Wall latency (speed) | 35% | Lower is better |

Rank-based (percentile) normalisation eliminates sensitivity to outliers. No redundant speed metrics — wall time captures the full user experience including TTFT, decode speed, and output length.

### Leaderboard

Ranked by composite score (65% quality, 35% wall time):

| Rank | Model | Think | Pass Rate | Decode (tok/s) | TTFT (s) | Wall (s) | Composite |
|-----:|-------|:-----:|----------:|---------------:|---------:|---------:|----------:|
| 1 | gpt-oss:20b | Yes | 100.0% | 52.7 | 0.33 | 14.3 | 0.883 |
| 2 | gpt-oss:20b | No | 100.0% | 55.2 | 0.33 | 15.3 | 0.864 |
| 3 | gpt-oss:120b | Yes | 100.0% | 40.0 | 0.58 | 16.1 | 0.855 |
| 4 | nemotron-cascade-2:30b | Yes | 100.0% | 66.8 | 0.31 | 18.5 | 0.828 |
| 5 | gemma4:26b | No | 96.4% | 28.3 | 1.26 | 15.2 | 0.780 |
| 6 | gpt-oss:120b | No | 96.4% | 39.2 | 0.60 | 18.0 | 0.743 |
| 7 | nemotron-3-nano:30b | Yes | 97.6% | 69.0 | 0.30 | 21.2 | 0.732 |
| 8 | nemotron-3-nano:4b | Yes | 89.3% | 66.5 | 0.19 | 11.2 | 0.707 |
| 9 | gemma4:26b | Yes | 100.0% | 27.6 | 1.14 | 67.1 | 0.680 |
| 10 | qwen3.5:35b | Yes | 98.8% | 56.4 | 0.37 | 54.7 | 0.647 |
| 11 | gemma4:26b-a4b-it-q8_0 | No | 91.7% | 23.1 | 1.36 | 19.1 | 0.647 |
| 12 | qwen3-coder-next | No | 85.7% | 44.1 | 0.45 | 13.3 | 0.610 |
| 13 | gemma4:e2b | Yes | 91.7% | 48.3 | 0.86 | 34.2 | 0.582 |
| 14 | gemma4:e4b | Yes | 94.0% | 29.1 | 1.00 | 48.4 | 0.580 |
| 15 | nemotron-3-super:120b | Yes | 95.2% | 19.8 | 21.96 | 61.4 | 0.570 |
| 16 | qwen3.5:35b | No | 78.6% | 57.3 | 0.37 | 12.9 | 0.542 |
| 17 | glm-4.7-flash | Yes | 90.5% | 55.0 | 0.29 | 36.2 | 0.539 |
| 18 | gemma4:e4b | No | 85.7% | 30.2 | 0.90 | 26.0 | 0.490 |
| 19 | qwen3.5:4b | No | 78.6% | 54.9 | 0.25 | 16.9 | 0.487 |
| 20 | lfm2:24b | No | 71.4% | 71.4 | 0.18 | 11.8 | 0.466 |
| 21 | gemma4:26b-a4b-it-q8_0 | Yes | 90.5% | 22.8 | 1.21 | 84.7 | 0.465 |
| 22 | glm-4.7-flash | No | 48.8% | 59.1 | 0.30 | 2.4 | 0.425 |
| 23 | gemma4:e2b | No | 76.2% | 47.6 | 0.87 | 18.9 | 0.416 |
| 24 | nemotron-3-nano:30b | No | 44.0% | 70.1 | 0.31 | 1.1 | 0.409 |
| 25 | nemotron-cascade-2:30b | No | 44.0% | 68.4 | 0.31 | 1.4 | 0.400 |
| 26 | qwen3.5:9b | Yes | 88.1% | 34.0 | 0.28 | 94.0 | 0.395 |
| 27 | nemotron-3-nano:4b | No | 41.7% | 68.4 | 0.19 | 1.0 | 0.384 |
| 28 | qwen3.5:122b | No | 79.8% | 22.2 | 30.77 | 52.9 | 0.383 |
| 29 | qwen3.5:122b | Yes | 88.1% | 21.8 | 26.86 | 141.1 | 0.377 |
| 30 | qwen3.5:4b | Yes | 81.0% | 53.8 | 0.25 | 88.6 | 0.345 |
| 31 | nemotron-3-super:120b | No | 72.6% | 20.0 | 21.56 | 31.1 | 0.336 |
| 32 | devstral-small-2:24b | No | 76.2% | 9.5 | 1.03 | 44.0 | 0.324 |
| 33 | qwen3.5:2b | No | 40.5% | 87.7 | 0.20 | 11.7 | 0.321 |
| 34 | qwen3.5:2b | Yes | 78.6% | 85.3 | 0.19 | 84.2 | 0.312 |
| 35 | qwen3.5:27b | No | 70.2% | 11.2 | 0.55 | 32.0 | 0.292 |
| 36 | qwen3.5:9b | No | 65.5% | 32.2 | 0.29 | 22.8 | 0.286 |
| 37 | qwen3.5:0.8b | No | 34.5% | 159.2 | 0.18 | 22.8 | 0.175 |
| 38 | qwen3.5:27b | Yes | 66.7% | 10.9 | 0.61 | 218.5 | 0.137 |
| 39 | qwen3.5:0.8b | Yes | 44.0% | 156.6 | 0.29 | 137.4 | 0.087 |

### Charts

![Accuracy by model](results/20260407_135131/ollama_benchmark_accuracy_20260407_135131.png)

![Wall time vs accuracy](results/20260407_135131/ollama_benchmark_walltime_vs_accuracy_20260407_135131.png)

![Throughput scatter](results/20260407_135131/ollama_benchmark_throughput_scatter_20260407_135131.png)

![Category performance](results/20260407_135131/ollama_benchmark_category_20260407_135131.png)


## Disclaimer

Benchmark results are provided "as is" without warranty of any kind. Results may vary depending on hardware configuration, software versions, model quantisation, thermal conditions, and system load. No guarantee of accuracy or reproducibility is made.

Mention of specific models, vendors, or products does not imply endorsement or affiliation. The authors are not liable for any decisions or outcomes based on these results.

See [LICENSE](LICENSE) for the full terms.

## License

MIT License — see [LICENSE](LICENSE).
