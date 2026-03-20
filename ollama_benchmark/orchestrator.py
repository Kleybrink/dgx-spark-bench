"""High-level benchmark orchestrator — runs the full model × think × prompt loop."""

import logging
import os
import time

import pandas as pd

from .cleanup import force_cleanup, kill_ollama_runners
from .config import BenchmarkConfig
from .display import fmt_elapsed, print_leaderboard, print_final_leaderboard
from .judge import judge_response
from .runner import run_one
from .warmup import warmup
from .wip import WIPTracker
from .analysis import (
    build_summary,
    build_category_summary,
    build_model_ranking,
    print_prompt_comparison,
    print_model_ranking,
)
from .plots import (
    export_csvs,
    generate_accuracy_chart,
    generate_category_chart,
    generate_throughput_scatter_chart,
    generate_walltime_vs_accuracy_chart,
)

logger = logging.getLogger(__name__)


def print_config(cfg: BenchmarkConfig, wip: WIPTracker):
    """Print a compact summary of what the benchmark will do."""
    total = len(cfg.models) * 2 * len(cfg.prompts) * cfg.runs_per_mode
    logger.info(
        "Benchmark plan: %d models × 2 think modes × %d prompts × %d runs = %d total",
        len(cfg.models),
        len(cfg.prompts),
        cfg.runs_per_mode,
        total,
    )
    print(
        f"Benchmark: {len(cfg.models)} models × 2 think modes × {len(cfg.prompts)} prompts"
        f" × {cfg.runs_per_mode} runs = {total} total runs"
    )
    print(f"Prompts:   {', '.join(p['name'] for p in cfg.prompts)}")
    print(f"Timeouts:  warmup={cfg.warmup_timeout_s}s  run={cfg.timeout_s}s")
    print(f"Ollama:    {cfg.ollama_base_url}")
    if cfg.judge_model:
        print(
            f"Judge:     {cfg.judge_model}  (temp={cfg.judge_temperature}, think={cfg.judge_think})"
        )
    else:
        print("Judge:     disabled")
    if wip.rows:
        print(f"Resuming:  {len(wip.rows)} runs loaded from WIP")
    print()


def run_benchmark(cfg: BenchmarkConfig, wip: WIPTracker):
    """Execute the full benchmark loop: model → think_mode → prompt → run.

    Handles warmup, cleanup, judging, timeouts, and prints progress.
    """
    skip_think: set[str] = set()
    not_found: set[str] = set()
    timed_out: dict[str, int] = {}

    bench_start = time.monotonic()

    for model_idx, model in enumerate(cfg.models, 1):
        if model in not_found:
            continue

        logger.info("Model %d/%d: %s", model_idx, len(cfg.models), model)

        _print_model_header(model_idx, len(cfg.models), model, bench_start)

        cleanup_status = force_cleanup(
            vram_settle_timeout_s=cfg.vram_settle_s,
            vram_free_target_pct=cfg.vram_free_target,
            quiet=True,
        )
        logger.info("Cleanup status: %s", cleanup_status)
        print(f"  Cleanup: {cleanup_status}")

        for think_mode in (False, True):
            if think_mode and model in skip_think:
                continue

            think_label = "T" if think_mode else "F"

            if wip.all_done(model, think_mode, cfg.prompts, cfg.runs_per_mode):
                print(f"  Warmup think={think_label}: skipped (all runs in WIP)")
                continue

            status = warmup(
                model,
                think_mode,
                keep_alive=cfg.keep_alive,
                timeout_s=cfg.warmup_timeout_s,
            )

            if _handle_warmup_status(
                status, model, think_mode, think_label, cfg, not_found, skip_think
            ):
                if status in ("not_found", "timeout", "error") and not think_mode:
                    break
                continue

            print(f"  Warmup think={think_label}: ok")

            _run_prompts(cfg, wip, model, think_mode, think_label, timed_out)

        print_leaderboard(wip.rows, model_idx, len(cfg.models))

    _print_warnings(not_found, skip_think, timed_out)


def save_results(cfg: BenchmarkConfig, wip: WIPTracker):
    """Analyse collected data, print summaries, and write CSVs + plots."""
    df = pd.DataFrame(wip.rows)

    if df.empty:
        raise RuntimeError(
            "No benchmark data collected. Check that at least one model is pulled "
            "and responding. Run `ollama list` to verify."
        )

    summary = build_summary(df)

    cat_summary = build_category_summary(summary)
    ranking = build_model_ranking(summary)

    print_prompt_comparison(summary)
    print_model_ranking(ranking)
    print_final_leaderboard(summary)

    ts_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", ts_str)
    os.makedirs(results_dir, exist_ok=True)

    plot_paths: list[str] = []
    acc_path = generate_accuracy_chart(summary, results_dir, ts_str, quiet=True)
    if acc_path:
        plot_paths.append(acc_path)
    cat_path = generate_category_chart(cat_summary, results_dir, ts_str, quiet=True)
    if cat_path:
        plot_paths.append(cat_path)
    wta_path = generate_walltime_vs_accuracy_chart(
        summary, results_dir, ts_str, quiet=True
    )
    if wta_path:
        plot_paths.append(wta_path)
    tp_path = generate_throughput_scatter_chart(
        summary, results_dir, ts_str, quiet=True
    )
    if tp_path:
        plot_paths.append(tp_path)

    csv_paths = export_csvs(
        df,
        summary,
        results_dir,
        ts_str,
        cat_summary=cat_summary if not cat_summary.empty else None,
        ranking=ranking if not ranking.empty else None,
        quiet=True,
    )
    logger.info(
        "Saved %d plots and %d CSVs to %s/",
        len(plot_paths),
        len(csv_paths),
        results_dir,
    )
    print(
        f"\nSaved {len(plot_paths)} plots and {len(csv_paths)} CSVs to {results_dir}/"
    )

    print("\nFinal cleanup …")
    force_cleanup(quiet=True)
    print("Done.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_model_header(
    model_idx: int, total_models: int, model: str, bench_start: float
):
    elapsed = time.monotonic() - bench_start
    models_done = model_idx - 1
    if models_done > 0:
        eta = elapsed / models_done * (total_models - models_done)
        eta_str = f"  ETA ~{fmt_elapsed(eta)}"
    else:
        eta_str = ""
    print(
        f"\n[Model {model_idx}/{total_models}] {model} | elapsed {fmt_elapsed(elapsed)}{eta_str}"
    )


def _handle_warmup_status(
    status: str,
    model: str,
    think_mode: bool,
    think_label: str,
    cfg: BenchmarkConfig,
    not_found: set[str],
    skip_think: set[str],
) -> bool:
    """Handle non-ok warmup status. Returns True if the caller should skip/break."""
    if status == "ok":
        return False

    if status == "not_found":
        not_found.add(model)
        logger.warning("Model not found: %s", model)
        print(f"  Warmup think={think_label}: NOT FOUND — skipping model")
        return True

    if status == "no_think":
        skip_think.add(model)
        logger.warning("Model %s does not support think mode", model)
        print(f"  Warmup think={think_label}: no_think — skipping think runs")
        return True

    if status == "timeout":
        if not think_mode:
            not_found.add(model)
            logger.warning(
                "Warmup timeout for %s (think=%s) after %ss",
                model,
                think_label,
                cfg.warmup_timeout_s,
            )
            print(
                f"  Warmup think={think_label}: TIMEOUT ({cfg.warmup_timeout_s}s) — skipping model"
            )
        else:
            skip_think.add(model)
            logger.warning("Warmup timeout for %s think=T — skipping think runs", model)
            print(f"  Warmup think={think_label}: TIMEOUT — skipping think runs")
        kill_ollama_runners()
        return True

    # status == "error" or anything unexpected
    if not think_mode:
        not_found.add(model)
        logger.error(
            "Warmup error for %s (think=%s) — skipping model", model, think_label
        )
        print(f"  Warmup think={think_label}: ERROR — skipping model")
    else:
        skip_think.add(model)
        logger.error("Warmup error for %s think=T — skipping think runs", model)
        print(f"  Warmup think={think_label}: ERROR — skipping think runs")
    return True


def _run_prompts(
    cfg: BenchmarkConfig,
    wip: WIPTracker,
    model: str,
    think_mode: bool,
    think_label: str,
    timed_out: dict[str, int],
):
    """Execute all prompt × run combinations for one model/think_mode pair."""
    for run_idx in range(1, cfg.runs_per_mode + 1):
        for prompt_entry in cfg.prompts:
            prompt_name = prompt_entry["name"]
            prompt_text = prompt_entry["prompt"]
            golden_answer = prompt_entry.get("golden_answer")
            if wip.is_done(model, think_mode, prompt_name, run_idx):
                continue

            tag = f"  {model} | {think_label} | {prompt_name} | {run_idx}/{cfg.runs_per_mode}"
            print(tag, end="  ", flush=True)

            logger.info(
                "Run start: %s | think=%s | %s | %d/%d",
                model,
                think_label,
                prompt_name,
                run_idx,
                cfg.runs_per_mode,
            )

            try:
                row = run_one(
                    model=model,
                    prompt=prompt_text,
                    think=think_mode,
                    run_idx=run_idx,
                    base_url=cfg.ollama_base_url,
                    keep_alive=cfg.keep_alive,
                    timeout_s=cfg.timeout_s,
                )
            except Exception as e:
                logger.error(
                    "Run exception: %s | think=%s | %s: %s",
                    model,
                    think_label,
                    prompt_name,
                    e,
                    exc_info=True,
                )
                print(f"ERROR: {e}")
                continue

            if row is None:
                timed_out[model] = timed_out.get(model, 0) + 1
                logger.warning(
                    "Run timeout: %s | think=%s | %s (%ss)",
                    model,
                    think_label,
                    prompt_name,
                    cfg.timeout_s,
                )
                # Create stub row so timeout counts in results
                row = {
                    "model": model,
                    "think": think_mode,
                    "run": run_idx,
                    "prompt_name": prompt_name,
                    "wall_latency_s": float(cfg.timeout_s),
                    "judge_score": 0.0,
                    "timed_out": True,
                }
                wip.append(row)
                print(f"TIMEOUT ({cfg.timeout_s}s)")
                kill_ollama_runners()
                continue

            row["prompt_name"] = prompt_name

            judge_str = _maybe_judge(cfg, row, golden_answer, prompt_text)

            wip.append(row)
            _print_run_metrics(row, judge_str)


def _maybe_judge(
    cfg: BenchmarkConfig, row: dict, golden_answer: str | None, prompt_text: str = ""
) -> str:
    """Run the LLM judge if configured. Returns a string for display."""
    if not (cfg.judge_model and golden_answer and row.get("answer_text")):
        return ""

    judgment = judge_response(
        answer_text=row["answer_text"],
        golden_answer=golden_answer,
        judge_model=cfg.judge_model,
        base_url=cfg.ollama_base_url,
        temperature=cfg.judge_temperature,
        think=cfg.judge_think,
        prompt=prompt_text,
    )
    row.update(judgment)
    score = judgment.get("judge_score")
    err = judgment.get("judge_error")
    if err is not None:
        logger.warning("Judge error for %s: %s", row.get("model"), err)
    else:
        logger.info(
            "Judge scored %s for %s: %s",
            score,
            row.get("model"),
            row.get("prompt_name", ""),
        )
    return f"  judge={score}" if err is None else f"  judge_err={err}"


def _print_run_metrics(row: dict, judge_str: str):
    parts = [f"wall={row['wall_latency_s']:.1f}s"]
    if row["ttft_s"] is not None:
        parts.append(f"ttft={row['ttft_s']:.2f}s")
    if row["decode_tokens_per_s"] is not None:
        parts.append(f"dec={row['decode_tokens_per_s']:.1f} tok/s")
    print("  ".join(parts) + judge_str)


def _print_warnings(
    not_found: set[str], skip_think: set[str], timed_out: dict[str, int]
):
    if not_found:
        print(f"\n✗  Not found / skipped entirely: {sorted(not_found)}")
    skip_only = skip_think - not_found
    if skip_only:
        print(f"⚠  think=False only: {sorted(skip_only)}")
    if timed_out:
        print(f"⏱  Timed-out runs: { {k: v for k, v in sorted(timed_out.items())} }")
