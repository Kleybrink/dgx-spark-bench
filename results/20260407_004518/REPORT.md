# Benchmark Report: Local LLM Inference on NVIDIA DGX Spark

**Date**: 2026-04-07
**Benchmark run**: `results/20260407_004518/`

---

## 1. Executive Summary

This run benchmarks **19 models** across **35 model x mode combinations** on the NVIDIA DGX Spark (128 GB unified memory, 273 GB/s bandwidth) using Ollama for local inference.

The ranking uses **no composite score**. Instead, results are sorted by the two dimensions that matter to the user:

1. **Pass Rate** (primary) — was the answer correct?
2. **Wall Time** (tiebreaker) — how long did I wait?

Decode speed and TTFT are shown as supplementary context but do not affect ranking order. This avoids hidden weighting choices that can distort the picture.

**Key findings:**

1. **Five model-mode combinations achieve 100% pass rate.** Of these, `gpt-oss:20b` (think) is fastest at 14.3s mean wall time; `nemotron-cascade-2:30b` (think) has the highest decode throughput at 66.8 tok/s.
2. **`gemma4:26b` joins the 100% club.** With thinking enabled it reaches perfect accuracy, though at a higher wall time (67.1s) than the MoE/cascade leaders. Even without thinking it scores 96.4% — one of the strongest no-think results in the benchmark.
3. **Think mode remains essential for most models.** The largest gains: nemotron-cascade-2:30b (+56 pp), nemotron-3-nano:30b (+54 pp), nemotron-3-nano:4b (+48 pp), glm-4.7-flash (+42 pp).
4. **Dense models above ~27B active parameters remain impractical** for interactive local use due to high TTFT and wall time (`qwen3.5:122b`: 141s, `nemotron-3-super:120b`: 61s).

---

## 2. Full Ranking

Sorted by pass rate (descending), then wall time (ascending). Decode speed and TTFT shown for context.

| Rank | Model | Think | Pass Rate | Decode (tok/s) | TTFT (s) | Wall (s) |
|-----:|-------|:-----:|----------:|---------------:|---------:|---------:|
| 1 | gpt-oss:20b | Yes | 100.0% | 52.7 | 0.33 | 14.3 |
| 2 | gpt-oss:20b | No | 100.0% | 55.2 | 0.33 | 15.3 |
| 3 | gpt-oss:120b | Yes | 100.0% | 40.0 | 0.58 | 16.1 |
| 4 | nemotron-cascade-2:30b | Yes | 100.0% | 66.8 | 0.31 | 18.5 |
| 5 | gemma4:26b | Yes | 100.0% | 27.6 | 1.14 | 67.1 |
| 6 | qwen3.5:35b | Yes | 98.8% | 56.4 | 0.37 | 54.7 |
| 7 | nemotron-3-nano:30b | Yes | 97.6% | 69.0 | 0.30 | 21.2 |
| 8 | gemma4:26b | No | 96.4% | 28.3 | 1.26 | 15.2 |
| 9 | gpt-oss:120b | No | 96.4% | 39.2 | 0.60 | 18.0 |
| 10 | nemotron-3-super:120b | Yes | 95.2% | 19.8 | 21.96 | 61.4 |
| 11 | gemma4:26b-a4b-it-q8_0 | No | 91.7% | 23.1 | 1.36 | 19.1 |
| 12 | glm-4.7-flash | Yes | 90.5% | 55.0 | 0.29 | 36.2 |
| 13 | gemma4:26b-a4b-it-q8_0 | Yes | 90.5% | 22.8 | 1.21 | 84.7 |
| 14 | nemotron-3-nano:4b | Yes | 89.3% | 66.5 | 0.19 | 11.2 |
| 15 | qwen3.5:9b | Yes | 88.1% | 34.0 | 0.28 | 94.0 |
| 16 | qwen3.5:122b | Yes | 88.1% | 21.8 | 26.86 | 141.1 |
| 17 | qwen3-coder-next:latest | No | 85.7% | 44.1 | 0.45 | 13.3 |
| 18 | qwen3.5:4b | Yes | 81.0% | 53.8 | 0.25 | 88.6 |
| 19 | qwen3.5:122b | No | 79.8% | 22.2 | 30.77 | 52.9 |
| 20 | qwen3.5:35b | No | 78.6% | 57.3 | 0.37 | 12.9 |
| 21 | qwen3.5:4b | No | 78.6% | 54.9 | 0.25 | 16.9 |
| 22 | qwen3.5:2b | Yes | 78.6% | 85.3 | 0.19 | 84.2 |
| 23 | devstral-small-2:24b | No | 76.2% | 9.5 | 1.03 | 44.0 |
| 24 | nemotron-3-super:120b | No | 72.6% | 20.0 | 21.56 | 31.1 |
| 25 | lfm2:24b | No | 71.4% | 71.4 | 0.18 | 11.8 |
| 26 | qwen3.5:27b | No | 70.2% | 11.2 | 0.55 | 32.0 |
| 27 | qwen3.5:27b | Yes | 66.7% | 10.9 | 0.61 | 218.5 |
| 28 | qwen3.5:9b | No | 65.5% | 32.2 | 0.29 | 22.8 |
| 29 | glm-4.7-flash | No | 48.8% | 59.1 | 0.30 | 2.4 |
| 30 | nemotron-3-nano:30b | No | 44.0% | 70.1 | 0.31 | 1.1 |
| 31 | nemotron-cascade-2:30b | No | 44.0% | 68.4 | 0.31 | 1.4 |
| 32 | qwen3.5:0.8b | Yes | 44.0% | 156.6 | 0.29 | 137.4 |
| 33 | nemotron-3-nano:4b | No | 41.7% | 68.4 | 0.19 | 1.0 |
| 34 | qwen3.5:2b | No | 40.5% | 87.7 | 0.20 | 11.7 |
| 35 | qwen3.5:0.8b | No | 34.5% | 159.2 | 0.18 | 22.8 |

---

## 3. Think Mode Impact

| Model | No-Think | Think | Delta |
|-------|-------------------:|----------------:|------:|
| nemotron-cascade-2:30b | 44.0% | 100.0% | **+56.0 pp** |
| nemotron-3-nano:30b | 44.0% | 97.6% | **+53.6 pp** |
| nemotron-3-nano:4b | 41.7% | 89.3% | **+47.6 pp** |
| glm-4.7-flash | 48.8% | 90.5% | **+41.7 pp** |
| qwen3.5:2b | 40.5% | 78.6% | +38.1 pp |
| nemotron-3-super:120b | 72.6% | 95.2% | +22.6 pp |
| qwen3.5:9b | 65.5% | 88.1% | +22.6 pp |
| qwen3.5:35b | 78.6% | 98.8% | +20.2 pp |
| qwen3.5:0.8b | 34.5% | 44.0% | +9.5 pp |
| qwen3.5:122b | 79.8% | 88.1% | +8.3 pp |
| gemma4:26b | 96.4% | 100.0% | +3.6 pp |
| gpt-oss:120b | 96.4% | 100.0% | +3.6 pp |
| qwen3.5:4b | 78.6% | 81.0% | +2.4 pp |
| gpt-oss:20b | 100.0% | 100.0% | 0.0 pp |
| gemma4:26b-a4b-it-q8_0 | 91.7% | 90.5% | -1.2 pp |
| qwen3.5:27b | 70.2% | 66.7% | -3.6 pp |

Think mode improves quality for nearly every model. The exceptions (`gemma4:26b-a4b-it-q8_0`, `qwen3.5:27b`) show that longer reasoning traces can hurt when they increase wall time without adding quality — especially on slower dense architectures where timeouts become a factor.

---

## 4. Recommendations

### If you need the best answer

Pick any of the five 100%-pass-rate combinations. Among those, `gpt-oss:20b` has the shortest wall time (14–15s), while `nemotron-cascade-2:30b` (think) has the highest decode throughput (66.8 tok/s) for the most responsive streaming experience.

### If you want the best quality/speed balance

| Model | Think | Pass Rate | Wall (s) | Why |
|-------|:-----:|----------:|---------:|-----|
| gpt-oss:20b | No | 100.0% | 15.3 | Perfect accuracy, no reasoning overhead |
| nemotron-cascade-2:30b | Yes | 100.0% | 18.5 | Perfect accuracy, fastest streaming (66.8 tok/s decode) |
| nemotron-3-nano:30b | Yes | 97.6% | 21.2 | Near-perfect, highest raw decode speed (69 tok/s) |
| gemma4:26b | No | 96.4% | 15.2 | Strong quality without thinking, short wall time |

### If you need low latency (sub-15s wall time)

| Model | Think | Pass Rate | Wall (s) |
|-------|:-----:|----------:|---------:|
| nemotron-3-nano:4b | Yes | 89.3% | 11.2 |
| lfm2:24b | No | 71.4% | 11.8 |
| qwen3-coder-next:latest | No | 85.7% | 13.3 |
| qwen3.5:35b | No | 78.6% | 12.9 |
| gpt-oss:20b | Yes | 100.0% | 14.3 |

### Models to avoid for interactive local use

| Model | Issue |
|-------|-------|
| qwen3.5:27b (think) | 66.7% pass rate at 218.5s wall — worst quality-speed ratio |
| qwen3.5:122b | 27–31s TTFT, 53–141s wall — use cloud API instead |
| qwen3.5:0.8b | 34–44% pass rate — too inaccurate for any serious task |

---

## 5. Methodology

- **19 models**, 21 prompts across 9 categories, 2 runs per model/mode/prompt combination.
- Each model tested in **think** (reasoning) and **no-think** (direct) mode.
- Pass rate = fraction of runs scored 1.0 by the LLM judge (`qwen3.5:35b`, temperature 0.1).
- Wall time = mean end-to-end time from request to complete response.
- **No composite score.** The ranking is sorted by pass rate (descending), then wall time (ascending). This avoids arbitrary weighting decisions that can distort results. Decode speed and TTFT are shown for context but do not affect rank order.

---

## Appendix: Data Sources

- `results/20260407_004518/ollama_benchmark_ranking_20260407_004518.csv`
- `results/20260407_004518/ollama_benchmark_category_summary_20260407_004518.csv`
- `results/20260407_004518/ollama_benchmark_raw_20260407_004518.csv`
