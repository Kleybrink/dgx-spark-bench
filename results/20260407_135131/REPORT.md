# Benchmark Report: Local LLM Inference on NVIDIA DGX Spark

**Date**: 2026-04-07
**Benchmark run**: `results/20260407_135131/`

---

## 1. Executive Summary

This benchmark measured 21 models (39 model-mode combinations) across 21 prompts on the NVIDIA DGX Spark (128 GB unified memory, 273 GB/s bandwidth) to determine which LLMs are genuinely practical for local interactive use.

**Key findings:**

1. **MoE architecture remains the decisive factor.** On a bandwidth-bound single-GPU system, only active parameters consume bandwidth per token. MoE models with 3-4B active parameters achieve 55-69 tok/s decode, while dense models at similar total sizes crawl at 10-28 tok/s.

2. **Think mode transforms quality.** The biggest gains come from MoE models: nemotron-cascade-2:30b jumps from 44% to 100% (+56 pp), nemotron-3-nano:30b from 44% to 98% (+54 pp). A 4B MoE with thinking beats a 122B model without it.

3. **Five models achieve 100% pass rate with thinking:** gpt-oss:20b, gpt-oss:120b, nemotron-cascade-2:30b, gemma4:26b, and (new) gpt-oss:20b also without thinking. The field has become more competitive at the top.

4. **Dense models can compete on quality.** gemma4:26b reaches 100% (think) and 96.4% (no-think) -- proving strong dense models match MoE quality. The trade-off is speed: 28 tok/s vs 67-69 tok/s for top MoE models.

**Top 3 recommendations:**

| Model | Why |
|-------|-----|
| **gpt-oss:20b** (think) | 100% accuracy, fastest wall time (14.3s), perfect in both modes |
| **nemotron-cascade-2:30b** (think) | 100% accuracy, fastest streaming (66.8 tok/s), 18.5s wall |
| **gpt-oss:120b** (think) | 100% accuracy, viable 120B MoE at 40 tok/s, 16.1s wall |

---

## 2. Hardware & Test Setup

### Hardware: NVIDIA DGX Spark

| Spec | Value |
|------|-------|
| SoC | NVIDIA GB10 (Blackwell GPU + Grace ARM CPU) |
| Memory | 128 GB LPDDR5x unified (CPU + GPU shared) |
| Memory bandwidth | 273 GB/s |
| CUDA cores | 6,144 |
| CPU | 20-core NVIDIA Grace (ARM) |

### Software Stack

| Component | Configuration |
|-----------|---------------|
| Inference server | [Ollama](https://ollama.com) |
| KV cache quantization | `OLLAMA_KV_CACHE_TYPE=q8` |
| Flash attention | `OLLAMA_FLASH_ATTENTION=true` |
| Parallel requests | `OLLAMA_NUM_PARALLEL=2` |
| Model format | GGUF (Ollama default quantization per model tag) |

### Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Models configured | 21 (19 successfully ran; `gemma4:e2b`, `gemma4:e4b` added since last run) |
| Prompts | 21 across 9 categories |
| Runs per mode | 2 |
| Timeout | 300 seconds per run |
| Judge model | `qwen3.5:35b` (temperature 0.1, no-think mode) |
| Judge scoring | 0.0 (fail), 0.5 (partial), 1.0 (pass) |

---

## 3. Composite Score

The ranking uses a weighted composite with rank-based (percentile) normalisation across two user-facing dimensions:

| Metric | Weight | Direction |
|--------|--------|-----------|
| Judge pass rate (quality) | 65% | Higher is better |
| Wall latency (speed) | 35% | Lower is better |

**Why these two?** They are the dimensions users actually experience: "Is the answer correct?" and "How long did I wait?" Unlike prior versions that included decode tok/s, e2e tok/s, and TTFT (which are highly correlated and double-count speed), this formulation uses two independent, interpretable metrics. Wall time naturally captures the full user experience including TTFT, decode speed, and output length.

Rank-based normalisation (percentile position) is used instead of min-max scaling to eliminate sensitivity to outliers.

---

## 4. Results Overview

### Full Ranking

All 39 model x mode combinations, sorted by composite score:

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
| 12 | qwen3-coder-next:latest | No | 85.7% | 44.1 | 0.45 | 13.3 | 0.610 |
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

![Accuracy by model](ollama_benchmark_accuracy_20260407_135131.png)

![Wall time vs accuracy](ollama_benchmark_walltime_vs_accuracy_20260407_135131.png)

![Throughput scatter](ollama_benchmark_throughput_scatter_20260407_135131.png)

![Category performance](ollama_benchmark_category_20260407_135131.png)

### Think Mode Impact

| Model | No-Think | Think | Delta |
|-------|:--------:|:-----:|:-----:|
| Nemotron Cascade 2 30B | 44.0% | 100.0% | **+56.0 pp** |
| Nemotron 3 Nano 30B | 44.0% | 97.6% | **+53.6 pp** |
| Nemotron 3 Nano 4B | 41.7% | 89.3% | **+47.6 pp** |
| GLM-4.7-Flash | 48.8% | 90.5% | **+41.7 pp** |
| Qwen3.5:2b | 40.5% | 78.6% | +38.1 pp |
| Nemotron 3 Super 120B | 72.6% | 95.2% | +22.6 pp |
| Qwen3.5:9b | 65.5% | 88.1% | +22.6 pp |
| Qwen3.5:35b | 78.6% | 98.8% | +20.2 pp |
| Gemma4:e2b | 76.2% | 91.7% | +15.5 pp |
| Qwen3.5:0.8b | 34.5% | 44.0% | +9.5 pp |
| Gemma4:e4b | 85.7% | 94.0% | +8.3 pp |
| Qwen3.5:122b | 79.8% | 88.1% | +8.3 pp |
| gpt-oss:120b | 96.4% | 100.0% | +3.6 pp |
| gemma4:26b | 96.4% | 100.0% | +3.6 pp |
| Qwen3.5:4b | 78.6% | 81.0% | +2.4 pp |
| gpt-oss:20b | **100.0%** | **100.0%** | 0.0 pp |
| gemma4:26b-a4b-it-q8_0 | 91.7% | 90.5% | -1.2 pp |
| Qwen3.5:27b | 70.2% | 66.7% | -3.6 pp |

**gpt-oss:20b** remains the only model achieving 100% pass rate in both think and no-think modes. **gemma4:26b** joins the 100% think-mode club this run, alongside gpt-oss:120b and nemotron-cascade-2:30b. **qwen3.5:27b** and **gemma4:26b-a4b-it-q8_0** are the only models where think mode hurts -- in qwen3.5:27b's case because reasoning generates very long outputs that frequently hit the 300-second timeout (218s average wall time).

---

## 5. Category-Level Analysis

### Per-Category Pass Rates (Think Mode, Top 6 Models)

| Category | gpt-oss:20b | gpt-oss:120b | nemotron-cascade-2:30b | gemma4:26b | nemotron-3-nano:30b | qwen3.5:35b |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logic** (5 prompts) | 100% | 100% | 100% | 100% | 90% | 100% |
| **Math** (3 prompts) | 100% | 100% | 100% | 100% | 100% | 100% |
| **Code** (4 prompts) | 100% | 100% | 100% | 100% | 100% | 100% |
| **NLP** (3 prompts) | 100% | 100% | 100% | 100% | 100% | 92% |
| **Knowledge** (2 prompts) | 100% | 100% | 100% | 100% | 100% | 100% |
| **Instruction-following** (1 prompt) | 100% | 100% | 100% | 100% | 100% | 100% |
| **Spatial** (1 prompt) | 100% | 100% | 100% | 100% | 100% | 100% |
| **Data** (1 prompt) | 100% | 100% | 100% | 100% | 100% | 100% |
| **Multi-domain** (1 prompt) | 100% | 100% | 100% | 100% | 100% | 100% |

The top 4 models (gpt-oss:20b, gpt-oss:120b, nemotron-cascade-2:30b, gemma4:26b) achieve **100% across all 9 categories** with thinking enabled. nemotron-3-nano:30b's only weakness is hard logic (90%), and qwen3.5:35b slightly falters on NLP (92%).

**Logic remains the hardest category overall.** The hard constraint-satisfaction puzzles (grid puzzle, scheduling) are the primary differentiator. In no-think mode, most MoE models score 20-40% on logic, jumping to 85-100% with thinking.

---

## 6. New Models in This Run

### Gemma4 Variants (e2b, e4b)

Two new Gemma4 "edge" variants were added:

| Model | Think | Pass Rate | Decode (tok/s) | Wall (s) | Composite |
|-------|:-----:|----------:|---------------:|---------:|----------:|
| gemma4:e2b | Yes | 91.7% | 48.3 | 34.2 | 0.582 |
| gemma4:e2b | No | 76.2% | 47.6 | 18.9 | 0.416 |
| gemma4:e4b | Yes | 94.0% | 29.1 | 48.4 | 0.580 |
| gemma4:e4b | No | 85.7% | 30.2 | 26.0 | 0.490 |

Both are solid mid-tier performers. gemma4:e2b is faster (48 tok/s) but less accurate; gemma4:e4b is more accurate (94% think) but slower (29 tok/s). Neither matches the full gemma4:26b (100% think, 28 tok/s) -- the larger model's quality advantage outweighs its similar speed.

---

## 7. Lessons Learned

### 1. MoE Architecture is King on DGX Spark

On a single-GPU system, LLM inference during token generation is memory-bandwidth-bound:

```
decode tok/s ~ memory_bandwidth / active_params_in_memory
```

With 273 GB/s bandwidth, a 27B dense model at Q8 (~27 GB) yields ~10 tok/s, matching the observed 11 tok/s for Qwen3.5:27b. A 30B MoE with 3.6B active parameters achieves ~76 tok/s theoretical, matching the observed 69 tok/s for Nemotron 3 Nano 30B.

### 2. The Composite Score Validates the Quality-First Philosophy

The new composite score (65% quality, 35% wall time) produces rankings that align well with practical usability:

- Models with 100% pass rate AND fast wall times dominate the top (gpt-oss:20b, gpt-oss:120b, nemotron-cascade-2:30b)
- gemma4:26b no-think (96.4%, 15s) ranks above gemma4:26b think (100%, 67s) -- correctly reflecting that the user waits 4x longer for a marginal quality gain
- Models that are fast but inaccurate (nemotron-3-nano:4b no-think: 42%, 1s) rank low despite excellent speed

### 3. Wall Time Penalises Think Mode Fairly

Unlike tok/s metrics, wall time captures what the user actually experiences. Think mode models produce more tokens and therefore take longer -- but the user genuinely waits longer. The composite reflects this trade-off correctly: gemma4:26b think (100%, 67s, rank 9) vs no-think (96.4%, 15s, rank 5). If 96.4% accuracy suffices, the no-think mode saves 52 seconds per prompt.

### 4. Cloud Leaderboard Rankings Invert Locally

| Model | AA Intelligence Index | Local Rank | Composite |
|-------|:---:|:---:|:---:|
| Qwen3.5:27b | 42 (highest) | **38** (near worst) | 0.137 |
| Nemotron 3 Nano 30B | 13 (lowest) | **7** | 0.732 |
| gpt-oss:20b | N/A | **1** (best) | 0.883 |

### 5. Five Models Now Achieve Perfect Accuracy

With nemotron-cascade-2:30b and gemma4:26b joining the 100% club (think mode), there are now five configurations with perfect pass rates. The differentiator between them is pure speed:

| Model | Think | Wall (s) | Why Choose |
|-------|:-----:|:--------:|------------|
| gpt-oss:20b | Yes | 14.3 | Fastest overall |
| gpt-oss:20b | No | 15.3 | No reasoning overhead |
| gpt-oss:120b | Yes | 16.1 | Largest model, still fast |
| nemotron-cascade-2:30b | Yes | 18.5 | Highest decode speed (66.8 tok/s) |
| gemma4:26b | Yes | 67.1 | Dense alternative, slower but competitive |

---

## 8. Recommendations

### Tier 1 -- Daily Drivers

| Model | Pass Rate | Decode (tok/s) | TTFT (s) | Wall (s) | Composite | Best For |
|-------|---:|---:|---:|---:|---:|---|
| **gpt-oss:20b** (think) | 100% | 52.7 | 0.33 | 14.3 | 0.883 | Best overall: perfect accuracy, fastest wall time |
| **gpt-oss:20b** (no-think) | 100% | 55.2 | 0.33 | 15.3 | 0.864 | Perfect without reasoning overhead |
| **gpt-oss:120b** (think) | 100% | 40.0 | 0.58 | 16.1 | 0.855 | Largest viable MoE, perfect accuracy |
| **nemotron-cascade-2:30b** (think) | 100% | 66.8 | 0.31 | 18.5 | 0.828 | Perfect accuracy, fastest streaming |

### Tier 2 -- Strong Alternatives

| Model | Pass Rate | Decode (tok/s) | Wall (s) | Composite | Best For |
|-------|---:|---:|---:|---:|---|
| **gemma4:26b** (no-think) | 96.4% | 28.3 | 15.2 | 0.780 | Dense model, fast wall time, near-perfect |
| **nemotron-3-nano:30b** (think) | 97.6% | 69.0 | 21.2 | 0.732 | Fastest decode, near-perfect quality |
| **nemotron-3-nano:4b** (think) | 89.3% | 66.5 | 11.2 | 0.707 | Lowest latency, good for edge/constrained use |

### Tier 3 -- Not Recommended for DGX Spark

| Model | Issue |
|-------|-------|
| **qwen3.5:122b** | 22 tok/s, 27-31s TTFT -- use cloud API (6x faster) |
| **nemotron-3-super:120b** | 20 tok/s, 22s TTFT -- same issue |
| **qwen3.5:27b** | 11 tok/s, 219s wall with thinking -- worst composite (0.137) |
| **qwen3.5:0.8b** | 157 tok/s but 34-44% accuracy -- too inaccurate |
| **devstral-small-2:24b** | 9.5 tok/s, 76% accuracy -- slowest decode, no think mode |

---

## Appendix: Data Sources

- **Benchmark data:** `results/20260407_135131/` (ranking CSV, category summary CSV, summary CSV, raw CSV)
- **Benchmark configuration:** `config.yaml`
- **Ollama configuration:** `ollama_settings/ollama.service`
