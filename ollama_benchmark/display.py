"""Display utilities for compact benchmark output."""

import logging
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_float(val) -> float | None:
    """Convert to float, returning None for missing/non-numeric/NaN values."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if f != f else f  # NaN → None
    except (ValueError, TypeError):
        return None


def fmt_elapsed(seconds: float) -> str:
    """Format seconds as '12m34s' or '1h23m'."""
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h{m:02d}m"


def print_leaderboard(rows: list[dict], completed_models: int, total_models: int):
    """Print a compact leaderboard with one row per (model, think).

    Sorted by % correct descending, then total wall time ascending.
    """
    if not rows:
        return

    logger.debug(
        "Leaderboard update: %d rows, %d/%d models",
        len(rows),
        completed_models,
        total_models,
    )

    # Aggregate by (model, think)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["model"], r["think"])
        groups[key].append(r)

    agg_rows = []
    for (model, think), runs in groups.items():
        dec = [
            v
            for r in runs
            if (v := _safe_float(r.get("decode_tokens_per_s"))) is not None
        ]
        ttft = [v for r in runs if (v := _safe_float(r.get("ttft_s"))) is not None]
        prompt_walls: dict[str, list[float]] = defaultdict(list)
        for r in runs:
            v = _safe_float(r.get("wall_latency_s"))
            if v is not None:
                prompt_walls[r.get("prompt_name", "")].append(v)
        wall_sum = (
            sum(sum(vals) / len(vals) for vals in prompt_walls.values())
            if prompt_walls
            else None
        )
        judge = [
            v for r in runs if (v := _safe_float(r.get("judge_score"))) is not None
        ]

        score_sum = sum(judge)
        judge_total = len(judge)
        correct_pct = (score_sum / judge_total * 100) if judge_total else None

        agg_rows.append(
            {
                "model": model,
                "think": think,
                "correct_pct": correct_pct,
                "wall_sum": wall_sum,
                "ttft_avg": sum(ttft) / len(ttft) if ttft else None,
                "dec_avg": sum(dec) / len(dec) if dec else None,
            }
        )

    # Sort by correct_pct descending, then wall_sum ascending
    agg_rows.sort(
        key=lambda r: (
            -(r["correct_pct"] if r["correct_pct"] is not None else -1),
            r["wall_sum"] if r["wall_sum"] is not None else float("inf"),
        )
    )

    # Dynamic column widths
    model_w = max(len("Model"), max(len(str(r["model"])) for r in agg_rows))

    header = f"--- Leaderboard ({completed_models}/{total_models} models done) "
    header += "-" * max(0, 79 - len(header))
    print(f"\n{header}")
    print(
        f"  {'Model':<{model_w}}  {'Think':>5}"
        f"  {'Correct':>7}  {'Wall(sum)':>9}  {'TTFT':>6}  {'dec tok/s':>9}"
    )

    for r in agg_rows:
        think_str = "T" if r["think"] else "F"
        corr_str = (
            f"{r['correct_pct']:6.0f}%" if r["correct_pct"] is not None else f"{'—':>7}"
        )
        wall_str = (
            f"{r['wall_sum']:8.1f}s" if r["wall_sum"] is not None else f"{'—':>9}"
        )
        ttft_str = (
            f"{r['ttft_avg']:5.2f}s" if r["ttft_avg"] is not None else f"{'—':>6}"
        )
        dec_str = f"{r['dec_avg']:9.1f}" if r["dec_avg"] is not None else f"{'—':>9}"

        print(
            f"  {r['model']:<{model_w}}  {think_str:>5}"
            f"  {corr_str}  {wall_str}  {ttft_str}  {dec_str}"
        )

    print("-" * 79)


def print_final_leaderboard(summary: pd.DataFrame):
    """Print a final leaderboard with one row per (model, think).

    Sorted by % correct descending, then total wall time ascending.
    """
    if summary.empty:
        return

    print("\n" + "=" * 79)
    print("  FINAL LEADERBOARD")
    print("=" * 79)

    # Aggregate from per-prompt summary to per (model, think)
    has_judge = (
        "judge_score_sum" in summary.columns and "judge_total_count" in summary.columns
    )

    agg_rows = []
    for key, sub in summary.groupby(["model", "think"], dropna=False):
        assert isinstance(key, tuple)
        model, think = key
        dec_vals = sub["decode_tokens_per_s"].dropna()
        dec_avg = dec_vals.mean() if not dec_vals.empty else None
        dec_std = dec_vals.std() if len(dec_vals) > 1 else None

        ttft_vals = sub["ttft_s"].dropna()
        ttft_avg = ttft_vals.mean() if not ttft_vals.empty else None

        wall_vals = sub["wall_latency_s"].dropna()
        wall_sum = wall_vals.sum() if not wall_vals.empty else None

        correct_pct = None
        if has_judge:
            score_sum = sub["judge_score_sum"].sum()
            total_count = sub["judge_total_count"].sum()
            if total_count > 0:
                correct_pct = score_sum / total_count * 100

        agg_rows.append(
            {
                "model": model,
                "think": think,
                "correct_pct": correct_pct,
                "wall_sum": wall_sum,
                "ttft_avg": ttft_avg,
                "dec_avg": dec_avg,
                "dec_std": dec_std,
            }
        )

    # Sort by correct_pct descending, then wall_sum ascending
    agg_rows.sort(
        key=lambda r: (
            -(r["correct_pct"] if r["correct_pct"] is not None else -1),
            r["wall_sum"] if r["wall_sum"] is not None else float("inf"),
        )
    )

    model_w = max(len("Model"), max(len(str(r["model"])) for r in agg_rows))

    has_std = any(r["dec_std"] is not None for r in agg_rows)

    header = (
        f"  {'Model':<{model_w}}  {'Think':>5}"
        f"  {'Correct':>7}  {'Wall(sum)':>9}  {'TTFT':>6}  {'dec tok/s':>9}"
    )
    if has_std:
        header += f"  {'+-std':>7}"
    print(header)

    for r in agg_rows:
        think_str = "T" if r["think"] else "F"
        corr_str = (
            f"{r['correct_pct']:6.0f}%"
            if r["correct_pct"] is not None
            else f"{'--':>7}"
        )
        wall_str = (
            f"{r['wall_sum']:8.1f}s" if r["wall_sum"] is not None else f"{'--':>9}"
        )
        ttft_str = (
            f"{r['ttft_avg']:5.2f}s" if r["ttft_avg"] is not None else f"{'--':>6}"
        )
        dec_str = f"{r['dec_avg']:9.1f}" if r["dec_avg"] is not None else f"{'--':>9}"

        line = (
            f"  {r['model']:<{model_w}}  {think_str:>5}"
            f"  {corr_str}  {wall_str}  {ttft_str}  {dec_str}"
        )
        if has_std:
            std_val = r["dec_std"]
            std_str = f"  {std_val:7.1f}" if std_val is not None else f"  {'--':>7}"
            line += std_str
        print(line)

    print("=" * 79)
