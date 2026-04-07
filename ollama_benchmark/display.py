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


def _compute_composite(
    agg_rows: list[dict],
    quality_weight: float = 0.65,
    speed_weight: float = 0.35,
):
    """Add a 'composite' key to each row using rank-based normalisation.

    Uses correct_pct (higher=better) and wall_sum (lower=better).
    """
    n = len(agg_rows)
    if n <= 1:
        for r in agg_rows:
            r["composite"] = 0.50
        return

    # Extract values, replacing None with median
    def _filled(key, rows):
        vals = [r[key] for r in rows if r[key] is not None]
        med = sorted(vals)[len(vals) // 2] if vals else 0.0
        return [r[key] if r[key] is not None else med for r in rows]

    corr_vals = _filled("correct_pct", agg_rows)
    wall_vals = _filled("wall_sum", agg_rows)

    def _rank_norm(vals, invert=False):
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                j += 1
            avg_rank = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        normed = [r / (n - 1) for r in ranks]
        return [1.0 - v for v in normed] if invert else normed

    corr_norm = _rank_norm(corr_vals)
    wall_norm = _rank_norm(wall_vals, invert=True)

    for i, r in enumerate(agg_rows):
        r["composite"] = quality_weight * corr_norm[i] + speed_weight * wall_norm[i]


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

    # Compute composite score via rank-based normalisation
    _compute_composite(agg_rows)

    # Sort by composite descending
    agg_rows.sort(key=lambda r: -r["composite"])

    # Dynamic column widths
    model_w = max(len("Model"), max(len(str(r["model"])) for r in agg_rows))

    header = f"--- Leaderboard ({completed_models}/{total_models} models done) "
    header += "-" * max(0, 79 - len(header))
    print(f"\n{header}")
    print(
        f"  {'Model':<{model_w}}  {'Think':>5}"
        f"  {'Correct':>7}  {'Wall(sum)':>9}  {'TTFT':>6}  {'dec tok/s':>9}"
        f"  {'Score':>5}"
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
        comp_str = f"{r['composite']:.2f}"

        print(
            f"  {r['model']:<{model_w}}  {think_str:>5}"
            f"  {corr_str}  {wall_str}  {ttft_str}  {dec_str}"
            f"  {comp_str:>5}"
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

    # Compute composite score
    _compute_composite(agg_rows)

    # Sort by composite descending
    agg_rows.sort(key=lambda r: -r["composite"])

    model_w = max(len("Model"), max(len(str(r["model"])) for r in agg_rows))

    has_std = any(r["dec_std"] is not None for r in agg_rows)

    header = (
        f"  {'Model':<{model_w}}  {'Think':>5}"
        f"  {'Correct':>7}  {'Wall(sum)':>9}  {'TTFT':>6}  {'dec tok/s':>9}"
    )
    if has_std:
        header += f"  {'+-std':>7}"
    header += f"  {'Score':>5}"
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
        comp_str = f"{r['composite']:.2f}"

        line = (
            f"  {r['model']:<{model_w}}  {think_str:>5}"
            f"  {corr_str}  {wall_str}  {ttft_str}  {dec_str}"
        )
        if has_std:
            std_val = r["dec_std"]
            std_str = f"  {std_val:7.1f}" if std_val is not None else f"  {'--':>7}"
            line += std_str
        line += f"  {comp_str:>5}"
        print(line)

    print("=" * 79)
