# Comprehensive Benchmark Report: Local LLM Inference on NVIDIA DGX Spark

**Date**: 2026-03-20
**Benchmark run**: `results/20260320_224129/`

---

## 1. Executive Summary

The NVIDIA DGX Spark is a desktop AI supercomputer with 128 GB of unified memory and 273 GB/s memory bandwidth — enough to hold any open-weights model up to ~120B parameters. But *fitting* a model in memory and *running it usably* are very different things. This benchmark measured 16 models across 21 prompts to find which LLMs are genuinely practical for local interactive use.

**Key findings:**

1. **Mixture-of-Experts (MoE) and cascade models dominate.** On a single-GPU, memory-bandwidth-bound system like the DGX Spark, only active parameters consume bandwidth during inference. MoE models with 3–4B active parameters achieve 56–71 tok/s decode speed — within 10–20% of cloud API throughput — while dense models at similar total parameter counts crawl. The new Nemotron Cascade 2 30B takes the top spot with 100% accuracy at 66.8 tok/s.

2. **Think mode (reasoning) is essential for quality.** Every model showed dramatic quality improvements with reasoning enabled. The think/no-think quality gap often exceeds the gap between entirely different model sizes (e.g., Nemotron Cascade 2 30B: 44% → 100%, GLM-4.7-Flash: 49% → 90%).

3. **Public leaderboard rankings don't predict local usability.** Models that top the Artificial Analysis Intelligence Index (like Qwen3.5:27b, Index 42) can be nearly unusable locally (11 tok/s), while low-ranked models like Nemotron 3 Nano 30B (Index 13) become top local performers.

4. **Large MoE models can be practical.** gpt-oss:120b achieves 100% accuracy (think mode) at 40 tok/s with just 0.58s TTFT — proving that 120B-class models can be viable locally when they use MoE with moderate active parameters, unlike dense 120B+ models which suffer 20–30s TTFT.

**Top 3 recommendations:**

| Model | Why |
|-------|-----|
| **nemotron-cascade-2:30b** (think) | 100% accuracy at 66.8 tok/s — fastest perfect-accuracy model, new #1 overall |
| **nemotron-3-nano:30b** (think) | 97.6% accuracy at 69 tok/s — fastest high-quality MoE model, excellent all-rounder |
| **gpt-oss:20b** (no-think) | 100% accuracy at 55 tok/s — perfect scores without reasoning overhead |

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
| AI performance | Up to 1 PFLOP FP4 |
| Power | 240W PSU, ~140W TDP for SoC |

**Note:** The GB10 does not report GPU memory usage through `nvidia-smi` (shows "Not Supported"), so VRAM utilization metrics are unavailable in this benchmark.

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
| Models configured | 16 (all 16 ran successfully) |
| Prompts | 21 across 9 categories |
| Runs per mode | 2 |
| Timeout | 300 seconds per run |
| Judge model | `qwen3.5:35b` (temperature 0.1, no-think mode) |
| Judge scoring | 0.0 (fail), 0.5 (partial), 1.0 (pass) |

---

## 3. Methodology

### Prompt Design

21 prompts span 9 categories at varying difficulty levels:

| Category | Prompts | Difficulty Range | Examples |
|----------|---------|-----------------|----------|
| Logic | 5 | Easy → Hard | Syllogistic deduction, grid puzzles, constraint scheduling |
| Code | 4 | Easy → Hard | String tracing, algorithm tracing, matrix computation, code generation |
| Math | 3 | Easy → Hard | Word problems, multi-step percentages, combinatorics |
| NLP | 3 | Easy → Hard | Fact extraction, sentiment analysis, reading comprehension |
| Knowledge | 2 | Easy → Medium | Basic science, common misconceptions |
| Instruction-following | 1 | Medium | Multi-step data filtering and sorting |
| Spatial | 1 | Medium | Grid-based spatial reasoning |
| Data | 1 | Medium | Tabular aggregation and analysis |
| Multi-domain | 1 | Hard | Road trip planning (arithmetic + logic + ordering) |

**Distractor design:** Every prompt includes at least one irrelevant piece of information (e.g., "The bakery has 3 employees and a cat named Muffin," "The university cafeteria serves pizza on Wednesdays") to test whether models correctly ignore noise and follow instructions precisely.

**Output format:** All prompts require JSON-formatted answers, which tests both reasoning ability and instruction compliance.

### Evaluation Pipeline

1. **Warmup:** Each model is validated with a minimal inference request before benchmarking.
2. **Cleanup:** Between models, Ollama unloads the prior model, orphaned processes are killed, Linux page caches are dropped, and the system waits until VRAM settles.
3. **Inference:** Streaming HTTP to Ollama's `/api/chat` endpoint. TTFT captured on first token.
4. **Judging:** After each run, the response is scored by `qwen3.5:35b` against a golden answer.
5. **Resumability:** All results are persisted to SQLite; interrupted runs resume automatically.

### Composite Score

The ranking uses a weighted composite with rank-based (percentile) normalisation:

| Metric | Weight |
|--------|--------|
| Judge pass rate | 55% |
| Decode tok/s | 20% |
| End-to-end tok/s | 10% |
| TTFT (lower is better) | 7.5% |
| Wall latency (lower is better) | 7.5% |

---

## 4. Results Overview

### Full Ranking

All 30 model × mode combinations, sorted by composite score:

| Rank | Model | Think | Pass Rate | Decode (tok/s) | E2E (tok/s) | TTFT (s) | Wall (s) | Composite |
|-----:|-------|:-----:|----------:|---------------:|------------:|---------:|---------:|----------:|
| 1 | nemotron-cascade-2:30b | Yes | 100.0% | 66.8 | 64.0 | 0.31 | 18.5 | 0.811 |
| 2 | nemotron-3-nano:30b | Yes | 97.6% | 69.0 | 66.0 | 0.30 | 21.2 | 0.784 |
| 3 | gpt-oss:20b | No | 100.0% | 55.2 | 53.0 | 0.33 | 15.3 | 0.768 |
| 4 | gpt-oss:20b | Yes | 100.0% | 52.7 | 50.4 | 0.33 | 14.3 | 0.739 |
| 5 | nemotron-3-nano:4b | Yes | 89.3% | 66.5 | 64.0 | 0.19 | 11.2 | 0.721 |
| 6 | qwen3.5:35b | Yes | 98.8% | 56.4 | 54.9 | 0.37 | 54.7 | 0.701 |
| 7 | gpt-oss:120b | Yes | 100.0% | 40.0 | 37.6 | 0.58 | 16.1 | 0.680 |
| 8 | glm-4.7-flash | Yes | 90.5% | 55.0 | 53.7 | 0.29 | 36.2 | 0.634 |
| 9 | qwen3.5:2b | Yes | 78.6% | 85.3 | 83.0 | 0.19 | 84.2 | 0.599 |
| 10 | lfm2:24b | No | 71.4% | 71.4 | 64.2 | 0.18 | 11.8 | 0.579 |
| 11 | gpt-oss:120b | No | 96.4% | 39.2 | 37.0 | 0.60 | 18.0 | 0.573 |
| 12 | qwen3.5:4b | Yes | 81.0% | 53.8 | 52.8 | 0.25 | 88.6 | 0.516 |
| 13 | qwen3-coder-next | No | 85.7% | 44.1 | 37.0 | 0.45 | 13.3 | 0.497 |
| 14 | qwen3.5:4b | No | 78.6% | 54.9 | 46.5 | 0.25 | 16.9 | 0.491 |
| 15 | qwen3.5:35b | No | 78.6% | 57.3 | 43.1 | 0.37 | 12.9 | 0.485 |
| 16 | qwen3.5:9b | Yes | 88.1% | 34.0 | 33.6 | 0.28 | 94.0 | 0.485 |
| 17 | nemotron-3-super:120b | Yes | 95.2% | 19.8 | 10.9 | 22.0 | 61.4 | 0.466 |
| 18 | qwen3.5:0.8b | Yes | 44.0% | 156.6 | 148.2 | 0.29 | 137.4 | 0.426 |
| 19 | qwen3.5:2b | No | 40.5% | 87.7 | 64.3 | 0.20 | 11.7 | 0.416 |
| 20 | qwen3.5:0.8b | No | 34.5% | 159.2 | 121.3 | 0.18 | 22.8 | 0.403 |
| 21 | qwen3.5:122b | Yes | 88.1% | 21.8 | 16.5 | 26.9 | 141.1 | 0.401 |
| 22 | nemotron-3-nano:30b | No | 44.0% | 70.1 | 44.9 | 0.31 | 1.1 | 0.395 |
| 23 | glm-4.7-flash | No | 48.8% | 59.1 | 42.6 | 0.30 | 2.4 | 0.384 |
| 24 | nemotron-3-nano:4b | No | 41.7% | 68.4 | 49.2 | 0.19 | 1.0 | 0.384 |
| 25 | nemotron-cascade-2:30b | No | 44.0% | 68.4 | 45.2 | 0.31 | 1.4 | 0.378 |
| 26 | qwen3.5:122b | No | 79.8% | 22.2 | 3.9 | 30.8 | 52.9 | 0.340 |
| 27 | qwen3.5:9b | No | 65.5% | 32.2 | 27.2 | 0.29 | 22.8 | 0.275 |
| 28 | nemotron-3-super:120b | No | 72.6% | 20.0 | 5.6 | 21.6 | 31.1 | 0.269 |
| 29 | qwen3.5:27b | No | 70.2% | 11.2 | 9.9 | 0.55 | 32.0 | 0.228 |
| 30 | qwen3.5:27b | Yes | 66.7% | 10.9 | 10.8 | 0.61 | 218.5 | 0.172 |

### Charts

![Accuracy by model](ollama_benchmark_accuracy_20260320_224129.png)

![Wall time vs accuracy](ollama_benchmark_walltime_vs_accuracy_20260320_224129.png)

![Throughput scatter](ollama_benchmark_throughput_scatter_20260320_224129.png)

![Category performance](ollama_benchmark_category_20260320_224129.png)

### Think Mode Impact

The quality difference between think (reasoning) and no-think mode is stark:

| Model | No-Think Pass Rate | Think Pass Rate | Delta |
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
| gpt-oss:120b | 96.4% | 100.0% | +3.6 pp |
| qwen3.5:4b | 78.6% | 81.0% | +2.4 pp |
| gpt-oss:20b | **100.0%** | **100.0%** | 0.0 pp |
| qwen3.5:27b | 70.2% | 66.7% | -3.5 pp |

Notable observations: **nemotron-cascade-2:30b** shows the largest think-mode gain in the entire benchmark (+56 pp), jumping from 44% to a perfect 100%. **gpt-oss:20b** remains the only model to achieve a perfect 100% pass rate in *both* think and no-think modes. **qwen3.5:27b** is *worse* with thinking because reasoning generates very long outputs that frequently hit the 300-second timeout (218s average wall time), degrading overall quality.

---

## 5. Category-Level Analysis

### Per-Category Pass Rates (Think Mode, Top Models)

| Category | nemotron-cascade-2:30b | gpt-oss:120b | gpt-oss:20b (no-think) | qwen3.5:35b | nemotron-3-nano:30b | glm-4.7-flash | nemotron-3-nano:4b | qwen3.5:9b |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logic** (5 prompts) | 100% | 100% | 100% | 100% | 90% | 85% | 80% | 70% |
| **Math** (3 prompts) | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| **Code** (4 prompts) | 100% | 100% | 100% | 100% | 100% | 87.5% | 93.8% | 100% |
| **NLP** (3 prompts) | 100% | 100% | 100% | 91.7% | 100% | 83.3% | 100% | 100% |
| **Knowledge** (2 prompts) | 100% | 100% | 100% | 100% | 100% | 87.5% | 62.5% | 100% |
| **Instruction-following** (1 prompt) | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 0% |
| **Spatial** (1 prompt) | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| **Data** (1 prompt) | 100% | 100% | 100% | 100% | 100% | 100% | 75% | 100% |
| **Multi-domain** (1 prompt) | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

### Key Category Observations

**Logic (hardest category overall):** The hard constraint-satisfaction puzzles (grid puzzle, scheduling) are the primary differentiator between models. nemotron-cascade-2:30b (think), gpt-oss:120b (think), gpt-oss:20b, qwen3.5:35b, and qwen3-coder-next achieve 100% in logic. Most models score 70–90%, with qwen3.5:27b and qwen3.5:0.8b dropping to 50% and 30% respectively.

**Code — tracing vs generation:** Code tracing tasks (predicting exact output) are particularly difficult because they require precise step-by-step execution without "fuzzy" reasoning. In no-think mode, most models struggle with code (50–69% pass rates). With thinking, all top models reach 87–100%.

**Math — combinatorics as the hard ceiling:** The easy and medium math problems are solved by nearly every model with thinking enabled. The hard combinatorics problem (`P(exactly one ball of each color) = 3/11`) is where smaller models falter — qwen3.5:0.8b achieves only 66.7% even with thinking.

**Instruction-following — a binary cliff:**
- **No-think mode:** 13 out of 16 models score **0%**. Only gpt-oss:20b (100%), gpt-oss:120b (100%), and the two NVIDIA Nemotron Nano models (50% each) manage any success. The instruction-following task requires multi-step data processing (filtering, sorting, computing an average) which most models skip or garble without reasoning.
- **Think mode:** Most models jump to 100%, but notably **qwen3.5:2b, 4b, and 9b still score 0%** even with reasoning enabled. The smaller dense Qwen3.5 models appear to lack the instruction-compliance fine-tuning that larger models and MoE variants have.

---

## 6. Comparison with Artificial Analysis Leaderboard

To contextualise local DGX Spark performance, we compare against [Artificial Analysis](https://artificialanalysis.ai/leaderboards/models) cloud API benchmarks for the same model families. AA provides an Intelligence Index (quality) and median output speed (throughput) measured across commercial API providers.

### Quality vs Speed Comparison

| Model | AA Intell. Index | AA Speed (tok/s) | Local Pass Rate (think) | Local Decode (tok/s) | Speed Ratio (AA/Local) |
|-------|:---:|---:|---:|---:|---:|
| Qwen3.5 122B | 42 | 133.7 | 88.1% | 21.8 | 6.1x |
| Qwen3.5 27B | 42 | 89.6 | 66.7% | 10.9 | 8.2x |
| Qwen3.5 35B (MoE) | 37 | 175.2 | 98.8% | 56.4 | 3.1x |
| Nemotron 3 Super 120B | 36 | 465.4 | 95.2% | 19.8 | 23.5x |
| Qwen3.5 9B | 32 | 65.0 | 88.1% | 34.0 | 1.9x |
| GLM-4.7-Flash (MoE) | 30 | 64.6 | 90.5% | 55.0 | 1.2x |
| Qwen3.5 4B | 27 | N/A | 81.0% | 53.8 | — |
| Nemotron 3 Nano 30B (MoE) | 13 | 77.8 | 97.6% | 69.0 | 1.1x |
| LFM2 24B (MoE) | 10 | 212.1 | 71.4% | 71.4 | 3.0x |

*AA data sourced from [artificialanalysis.ai](https://artificialanalysis.ai/leaderboards/models) as of March 2026. Speed is median output tokens/s across providers. Nemotron Cascade 2 30B is not yet listed on Artificial Analysis.*

### Speed Gap Analysis

The speed ratio between cloud APIs and local DGX Spark inference reveals a clear architectural pattern:

| Architecture | Models | Typical Speed Ratio | Assessment |
|-------------|--------|:---:|------------|
| Small MoE (3–4B active) | Nemotron Nano 30B, GLM-4.7-Flash | **1.1–1.2x** | **API-competitive** — DGX Spark matches cloud speed |
| Medium MoE (~3B active, larger total) | Qwen3.5:35b, LFM2:24b | 3.0–3.1x | Acceptable — 3x slower but still interactive (50+ tok/s) |
| Small dense (4–9B) | Qwen3.5:9b | 1.9x | Acceptable for 9B; larger dense models degrade rapidly |
| Large models (120B+) | Qwen3.5:122b, Nemotron Super 120B | **6–24x** | **Not competitive** — cloud APIs run 6–24x faster via tensor parallelism |

### Quality Ranking Divergence

The Artificial Analysis Intelligence Index and our local benchmark rankings diverge significantly because they measure different things:

- **AA Index** measures raw model capability on standardized benchmarks (MMLU, GPQA, etc.) under ideal conditions with no time pressure.
- **Our benchmark** measures *practical usability* — can the model solve real-world tasks within a 300-second window on local hardware?

This explains why **Nemotron 3 Nano 30B** (AA Index: 13, lowest in our comparison) ranks as a **top local model** (97.6% pass rate, 69 tok/s). Its MoE architecture with 3.6B active parameters is perfectly matched to DGX Spark's bandwidth constraints, and it has enough reasoning capability to handle our 21-prompt suite when thinking is enabled.

Conversely, **Qwen3.5:27b** (AA Index: 42, tied for highest) is the **worst local model** (rank #30). Its 27B dense parameters saturate the memory bus at just 11 tok/s, and reasoning generates such long outputs that many prompts time out.

---

## 7. Lessons Learned

### 1. MoE and Cascade Architectures are King on DGX Spark

On a single-GPU system, LLM inference during token generation is fundamentally memory-bandwidth-bound: the model weights must be read from memory once per output token. Throughput is approximately:

```
decode tok/s ≈ memory_bandwidth / model_size_in_memory
```

With 273 GB/s bandwidth, a 27B dense model at Q8 quantization (~27 GB) yields roughly `273 / 27 ≈ 10 tok/s` — matching the observed 11 tok/s for Qwen3.5:27b.

MoE models exploit this by keeping all parameters in memory (for quality) while only *reading* the active expert parameters each token (for speed). A 30B-total MoE model with 3.6B active parameters achieves `273 / 3.6 ≈ 76 tok/s` theoretical — matching the observed 69 tok/s for Nemotron 3 Nano 30B.

**Nemotron Cascade 2 30B** extends this principle with a cascade architecture: it routes tokens through a smaller fast model first and only escalates to larger experts when needed, achieving 66.8 tok/s decode while maintaining 100% accuracy — the best quality-speed combination in the benchmark.

This makes MoE and cascade architectures the optimal choice for DGX Spark: you get large-model quality at small-model speed.

### 2. Most Dense 120B+ Models are Impractical — but MoE Variants Can Be Viable

Dense 120B+ models like Nemotron 3 Super 120B and Qwen3.5:122b achieve reasonable decode speeds (20–22 tok/s) but suffer from 20–30 second TTFT (time to first token) due to the massive prompt evaluation phase on a single GPU. Cloud API providers achieve 6–24x higher throughput by distributing inference across multiple GPUs via tensor parallelism — something the DGX Spark cannot do.

**However, gpt-oss:120b proves this isn't universal.** With 100% accuracy (think mode), 40 tok/s decode, and just 0.58s TTFT, it delivers a fully interactive experience despite being a 120B model. Its TTFT is 38–53x faster than other 120B models (vs 22–27s), strongly suggesting an MoE architecture with moderate active parameters (~7B based on the 40 tok/s speed). This makes gpt-oss:120b the highest-accuracy model in the benchmark while remaining genuinely usable.

**Recommendation:** For *dense* models above ~35B active parameters, use a cloud API. But MoE 120B+ models with moderate active parameters can be excellent local choices — always benchmark before deciding.

### 3. Think Mode is Essential for Quality

The average quality improvement from enabling thinking is **+24 percentage points** across all models. The most dramatic gains come from MoE and cascade models that have fast decode but relatively weak "System 1" (no-think) responses:

- Nemotron Cascade 2 30B: 44% → 100% (+56 pp)
- Nemotron 3 Nano 30B: 44% → 98% (+54 pp)
- Nemotron 3 Nano 4B: 42% → 89% (+48 pp)
- GLM-4.7-Flash: 49% → 90% (+42 pp)

The quality gap between think and no-think modes often exceeds the gap between entirely different model sizes. Enabling reasoning on a 4B MoE model yields better results than running a 122B model without reasoning.

**Exception:** gpt-oss:20b achieves 100% accuracy in *both* think and no-think modes — the only model where reasoning makes zero difference. gpt-oss:120b comes close, scoring 96.4% without thinking and 100% with. Both models appear to have exceptionally strong instruction-following fine-tuning.

### 4. Public Leaderboard Rankings Don't Predict Local Usability

| Model | AA Intelligence Index | Local Rank | Local Composite |
|-------|:---:|:---:|:---:|
| Qwen3.5:27b | 42 (highest) | **30** (worst) | 0.172 |
| Nemotron 3 Nano 30B | 13 (lowest) | **2** (second best) | 0.784 |

The AA Intelligence Index measures *what a model can do*; our benchmark measures *what a model can do on this specific hardware in a reasonable time*. The DGX Spark's memory bandwidth bottleneck inverts the usual quality hierarchy: a smaller MoE model that runs 7x faster outperforms a larger dense model that produces superior tokens — but too slowly.

### 5. Optimal Sweet Spot: 4B–35B Active Parameters

Models in the 4B–35B active parameter range achieve the best balance of quality and speed on DGX Spark:

| Active Params | Decode (tok/s) | Best Pass Rate | Example |
|:---:|:---:|:---:|---|
| < 1B | 157 | 44% | qwen3.5:0.8b — fast but too inaccurate |
| 2–4B | 54–88 | 89% | nemotron-3-nano:4b, qwen3.5:2b |
| 3–4B (MoE/cascade) | 56–71 | 100% | nemotron-cascade-2:30b, qwen3.5:35b, nemotron-3-nano:30b |
| ~7B (MoE in large model) | 40 | 100% | gpt-oss:120b — viable 120B via MoE |
| 9–27B (dense) | 11–34 | 88% | qwen3.5:9b — acceptable; 27b — too slow |
| 12–22B (active in large MoE) | 20–22 | 95% | nemotron-3-super:120b — usable but slow TTFT |

Below 2B active parameters, quality drops sharply. Above ~35B active, speed becomes prohibitive for interactive use (< 25 tok/s).

### 6. The "Qwen3.5:27b Problem"

Despite being a non-MoE model at "only" 27B parameters, Qwen3.5:27b runs at just 10.9–11.2 tok/s — slower than MoE models with 4–5x more total parameters. This is a pure memory bandwidth bottleneck: at Q8 quantization, the 27B model occupies ~27 GB and requires a full read per generated token.

By contrast, Qwen3.5:35b (MoE) has 35B total parameters but only ~3B active, so it runs at 56 tok/s — **5x faster** than the smaller 27b dense model. This counter-intuitive result illustrates why always benchmarking before deploying is essential. Parameter count alone tells you nothing about local inference performance.

### 7. Instruction-Following Without Reasoning is Universally Poor

The instruction-following task (filter fruits by price threshold, sort alphabetically, compute average) seems simple, but requires precise multi-step execution with a specific output format:

- **No-think mode:** 13 of 16 models score **0%**. Only gpt-oss:20b (100%), gpt-oss:120b (100%), and the two Nemotron Nano models (50%) manage any success. Most models either skip the filtering step, fail to sort correctly, or produce malformed JSON.
- **Think mode:** Most models jump to 100%, but the smaller dense Qwen3.5 models (2b, 4b, 9b) still score **0%** even with reasoning enabled, suggesting they lack the instruction-compliance fine-tuning present in larger and MoE variants.

This has practical implications: if your use case requires structured output (JSON, filtered lists, formatted tables), you need both thinking enabled *and* a model with strong instruction-following capability.

### 8. DGX Spark Matches API Speed for Small MoE Models

For the right model choices, the DGX Spark eliminates the cloud API speed advantage entirely:

| Model | Local Decode (tok/s) | AA API Speed (tok/s) | Ratio |
|-------|---:|---:|:---:|
| Nemotron 3 Nano 30B | 69.0 | 77.8 | **1.1x** |
| GLM-4.7-Flash | 55.0 | 64.6 | **1.2x** |

At 1.1–1.2x cloud speed, the DGX Spark is genuinely competitive for these models — with zero per-token cost, complete data privacy, no rate limits, and no internet dependency. For teams running thousands of inference calls per day with sensitive data, this eliminates the need for API access entirely for the best local models.

---

## 8. Recommendations

### Tier 1 — Highly Recommended (daily driver quality + speed)

| Model | Pass Rate | Decode (tok/s) | TTFT (s) | Wall (s) | Best For |
|-------|---:|---:|---:|---:|---|
| **nemotron-cascade-2:30b** (think) | 100% | 66.8 | 0.31 | 18.5 | Perfect accuracy at top speed — new #1 overall |
| **nemotron-3-nano:30b** (think) | 97.6% | 69.0 | 0.30 | 21.2 | Fastest high-quality model, excellent all-rounder |
| **gpt-oss:20b** (no-think) | 100% | 55.2 | 0.33 | 15.3 | Highest accuracy, no reasoning overhead |
| **qwen3.5:35b** (think) | 98.8% | 56.4 | 0.37 | 54.7 | Best open model for hard tasks (100% on logic) |

### Tier 2 — Good for Specific Use Cases

| Model | Pass Rate | Decode (tok/s) | TTFT (s) | Wall (s) | Best For |
|-------|---:|---:|---:|---:|---|
| **gpt-oss:120b** (think) | 100% | 40.0 | 0.58 | 16.1 | Highest accuracy overall (100% all categories), viable 120B MoE |
| **nemotron-3-nano:4b** (think) | 89.3% | 66.5 | 0.19 | 11.2 | Latency-sensitive applications, edge deployment |
| **glm-4.7-flash** (think) | 90.5% | 55.0 | 0.29 | 36.2 | Good alternative with permissive licensing |
| **qwen3.5:2b** (think) | 78.6% | 85.3 | 0.19 | 84.2 | Fastest usable thinking model (but fails instruction-following) |
| **qwen3-coder-next** (no-think) | 85.7% | 44.1 | 0.45 | 13.3 | Code-focused tasks (100% on logic and math) |

### Tier 3 — Not Recommended for DGX Spark

| Model | Issue |
|-------|-------|
| **qwen3.5:122b** | 22 tok/s decode, 27s TTFT — use the cloud API instead (6x faster) |
| **nemotron-3-super:120b** | 20 tok/s decode, 22s TTFT — same issue (24x faster via API) |
| **qwen3.5:27b** | 11 tok/s decode, 219s wall time with thinking — worst quality/speed ratio |
| **qwen3.5:0.8b** | 157 tok/s but only 44% pass rate — too inaccurate for any serious task |
| **lfm2:24b** (no-think only) | 71 tok/s but 71.4% accuracy — fast but unreliable without think mode |

### Decision Flowchart

```
Need highest accuracy?
  └─ Yes → Need 100% on ALL categories?
           └─ Yes → nemotron-cascade-2:30b (think) or gpt-oss:120b (think): perfect scores across all 9 categories
           └─ No → gpt-oss:20b (no-think) or qwen3.5:35b (think)
  └─ No → Need fastest response?
           └─ Yes → nemotron-3-nano:4b (think): 0.19s TTFT, 11s wall
           └─ No → Need best all-rounder?
                    └─ Yes → nemotron-cascade-2:30b (think): 100% at 66.8 tok/s
                    └─ No → Need structured output / JSON?
                             └─ Yes → nemotron-cascade-2:30b, gpt-oss:20b, gpt-oss:120b, qwen3.5:35b, or nemotron-3-nano:30b (all with think)
                             └─ No → qwen3.5:2b (think) for maximum throughput at acceptable quality
```

---

## Appendix: Data Sources

- **Benchmark data:** `results/20260320_224129/` (ranking CSV, category summary CSV, summary CSV)
- **Benchmark configuration:** `config.yaml`
- **Ollama configuration:** `ollama_settings/ollama.service`
- **Artificial Analysis Leaderboard:** [artificialanalysis.ai/leaderboards/models](https://artificialanalysis.ai/leaderboards/models)
- **DGX Spark Hardware Specs:** [docs.nvidia.com/dgx/dgx-spark/hardware.html](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
