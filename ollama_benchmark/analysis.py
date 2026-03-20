"""Analysis, summary statistics, and interpretation reporting."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

GROUP_COLS = ["model", "think", "prompt_name"]

AGG_COLS = [
    "wall_latency_s",
    "ttft_s",
    "total_duration_s",
    "load_duration_s",
    "prompt_eval_count",
    "prompt_eval_duration_s",
    "eval_count",
    "eval_duration_s",
    "prompt_tokens_per_s",
    "decode_tokens_per_s",
    "end_to_end_tokens_per_s",
    "answer_chars",
    "thinking_chars",
    "judge_score",
    "gpu_util_avg_pct",
    "gpu_util_peak_pct",
    "gpu_mem_used_avg_mb",
    "gpu_mem_used_peak_mb",
    "gpu_power_avg_w",
    "gpu_power_peak_w",
]

REPORT_METRICS = [
    ("wall_latency_s", "Wall Latency"),
    ("ttft_s", "TTFT"),
    ("prompt_tokens_per_s", "Prompt tok/s"),
    ("decode_tokens_per_s", "Decode tok/s"),
    ("end_to_end_tokens_per_s", "E2E tok/s"),
    ("eval_count", "Output Tokens"),
    ("thinking_chars", "Thinking Chars"),
    ("judge_score", "Judge Score"),
]

# Weights for the composite ranking score.
_RANKING_WEIGHTS = {
    "judge_pass_rate": 0.55,
    "decode_tokens_per_s": 0.20,
    "end_to_end_tokens_per_s": 0.10,
    "ttft_s": 0.075,
    "wall_latency_s": 0.075,
}
# Metrics where lower is better (will be inverted during normalisation).
_LOWER_IS_BETTER = {"ttft_s", "wall_latency_s"}

# Fallback weights when judge data is unavailable.
_RANKING_WEIGHTS_NO_JUDGE = {
    "decode_tokens_per_s": 0.40,
    "end_to_end_tokens_per_s": 0.25,
    "ttft_s": 0.20,
    "wall_latency_s": 0.15,
}


def _fmt(x, digits=2, suffix=""):
    return f"{x:.{digits}f}{suffix}" if pd.notna(x) else "n/a"


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute multi-aggregation of AGG_COLS grouped by GROUP_COLS.

    Returns columns: <metric>_mean, <metric>_std, <metric>_median,
    <metric>_min, <metric>_max for every metric plus judge_pass_rate,
    judge_score_sum, and judge_total_count.
    Un-suffixed column names are kept as aliases for _mean for backward compat.
    """
    existing_agg = [c for c in AGG_COLS if c in df.columns]
    df = df.copy()
    for col in existing_agg:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = df.groupby(GROUP_COLS, dropna=False)[existing_agg]

    agg_mean = grouped.mean(numeric_only=True).reset_index()
    agg_std = grouped.std(numeric_only=True).reset_index()
    agg_median = grouped.median(numeric_only=True).reset_index()
    agg_min = grouped.min(numeric_only=True).reset_index()
    agg_max = grouped.max(numeric_only=True).reset_index()

    # Start with group columns + un-suffixed means (backward compat)
    summary = agg_mean.copy()

    # Add suffixed columns via single concat to avoid fragmentation warning
    suffixed_parts = []
    for label, agg_df in [
        ("mean", agg_mean),
        ("std", agg_std),
        ("median", agg_median),
        ("min", agg_min),
        ("max", agg_max),
    ]:
        renamed = agg_df[existing_agg].rename(
            columns={c: f"{c}_{label}" for c in existing_agg}
        )
        suffixed_parts.append(renamed)
    summary = pd.concat([summary] + suffixed_parts, axis=1)

    # Judge pass-rate columns (score-sum based: 0.5 scores count as 0.5)
    if "judge_score" in df.columns:
        judge_agg = (
            df.dropna(subset=["judge_score"])
            .groupby(GROUP_COLS, dropna=False)["judge_score"]
            .agg(
                judge_score_sum=lambda s: s.sum(),
                judge_total_count="count",
            )
            .reset_index()
        )
        judge_agg["judge_pass_rate"] = (
            judge_agg["judge_score_sum"] / judge_agg["judge_total_count"]
        )
        summary = summary.merge(judge_agg, on=GROUP_COLS, how="left")
    else:
        summary["judge_pass_rate"] = float("nan")
        summary["judge_score_sum"] = 0
        summary["judge_total_count"] = 0

    logger.info("Summary built: %d rows", len(summary))
    return summary


def print_prompt_comparison(summary: pd.DataFrame):
    """Print a compact per-prompt table: one row per (model, think) combo."""
    if summary.empty:
        return

    print("\n" + "=" * 60)
    print("  RESULTS BY PROMPT")
    print("=" * 60)

    has_judge = (
        "judge_score" in summary.columns and summary["judge_score"].notna().any()
    )
    model_w = max(len("Model"), summary["model"].astype(str).str.len().max())

    for prompt_name, sub in summary.groupby("prompt_name", dropna=False, sort=True):
        print(f"\n  [{prompt_name}]")
        sorted_sub = sub.sort_values(
            by=["judge_score", "wall_latency_s"],
            ascending=[False, True],
            na_position="last",
        )
        header = f"  {'Model':<{model_w}}  {'Think':>5}"
        if has_judge:
            header += f"  {'Score':>5}"
        header += f"  {'dec tok/s':>9}  {'Wall(s)':>7}  {'TTFT(s)':>7}"
        print(header)

        for _, row in sorted_sub.iterrows():
            think_str = "T" if row["think"] else "F"
            line = f"  {row['model']:<{model_w}}  {think_str:>5}"
            if has_judge:
                line += f"  {_fmt(row.get('judge_score'), digits=2):>5}"
            line += f"  {_fmt(row.get('decode_tokens_per_s'), digits=1):>9}"
            line += f"  {_fmt(row.get('wall_latency_s'), digits=1):>7}"
            line += f"  {_fmt(row.get('ttft_s'), digits=2):>7}"
            print(line)


def print_results(df: pd.DataFrame, summary: pd.DataFrame):
    """Print raw results and summary tables."""
    print("\n=== RAW RESULTS ===")
    print(df.to_string())
    print("\n=== SUMMARY (averaged over runs) ===")
    print(summary.to_string())


def print_interpretation(summary: pd.DataFrame):
    """Print per-model x per-prompt think=False vs think=True comparison."""
    print("\n" + "=" * 60)
    print("  INTERPRETATION  (think=False vs think=True per model x prompt)")
    print("=" * 60)

    for key, sub in summary.groupby(["model", "prompt_name"], dropna=False):
        assert isinstance(key, tuple)
        model_name, prompt_name = key
        think_vals = set(sub["think"].dropna().tolist())
        print(f"\n--- {model_name}  [{prompt_name}] ---")
        if {False, True} <= think_vals:
            rf = sub[sub["think"].eq(False)].iloc[0]
            rt = sub[sub["think"].eq(True)].iloc[0]
            for col, label in REPORT_METRICS:
                std_col = f"{col}_std"
                std_f = rf.get(std_col)
                std_t = rt.get(std_col)
                std_f_str = f" +/-{_fmt(std_f)}" if pd.notna(std_f) else ""
                std_t_str = f" +/-{_fmt(std_t)}" if pd.notna(std_t) else ""
                print(
                    f"  {label:20s}: think=F {_fmt(rf.get(col))}{std_f_str}"
                    f"  |  think=T {_fmt(rt.get(col))}{std_t_str}"
                )
        else:
            row = sub.iloc[0]
            print(f"  (only think={row['think']} available)")
            for col, label in REPORT_METRICS:
                std_col = f"{col}_std"
                std_val = row.get(std_col)
                std_str = f" +/-{_fmt(std_val)}" if pd.notna(std_val) else ""
                print(f"  {label:20s}: {_fmt(row.get(col))}{std_str}")


# ---------------------------------------------------------------------------
# Category analysis
# ---------------------------------------------------------------------------


def _parse_prompt_name(name: str) -> tuple[str, str, str]:
    """Parse 'category--difficulty--short_name' into (category, difficulty, short_name).

    Falls back to ('other', 'unknown', name) when the name doesn't match.
    """
    parts = name.split("--")
    if len(parts) >= 3:
        return parts[0], parts[1], "--".join(parts[2:])
    if len(parts) == 2:
        return parts[0], "unknown", parts[1]
    logger.debug("Prompt name parse fallback for '%s' → category='other'", name)
    return "other", "unknown", name


def build_category_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Average key metrics per (model, think, category)."""
    df = summary.copy()
    df["category"] = df["prompt_name"].apply(lambda n: _parse_prompt_name(n)[0])

    metric_cols = [
        c
        for c in [
            "judge_score",
            "judge_pass_rate",
            "decode_tokens_per_s",
            "end_to_end_tokens_per_s",
            "wall_latency_s",
            "ttft_s",
        ]
        if c in df.columns
    ]
    if not metric_cols:
        return pd.DataFrame()

    cat_summary = (
        df.groupby(["model", "think", "category"], dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    logger.info("Category summary built: %d rows", len(cat_summary))
    return cat_summary


def print_category_summary(cat_summary: pd.DataFrame):
    """Print a table of metrics per category."""
    if cat_summary.empty:
        return

    print("\n" + "=" * 60)
    print("  CATEGORY SUMMARY")
    print("=" * 60)

    for key, sub in cat_summary.groupby(["model", "think"], dropna=False):
        assert isinstance(key, tuple)
        model, think = key
        think_str = "T" if think else "F"
        print(f"\n  {model}  think={think_str}")

        header_parts = [f"  {'Category':<20s}"]
        cols_present = []
        col_labels = [
            ("judge_pass_rate", "PassRate"),
            ("judge_score", "JScore"),
            ("decode_tokens_per_s", "dec tok/s"),
            ("wall_latency_s", "Wall(s)"),
            ("ttft_s", "TTFT(s)"),
        ]
        for col, label in col_labels:
            if col in sub.columns and sub[col].notna().any():
                header_parts.append(f"{label:>10s}")
                cols_present.append(col)
        print("".join(header_parts))

        for _, row in sub.sort_values("category").iterrows():
            parts = [f"  {row['category']:<20s}"]
            for col in cols_present:
                parts.append(f"{_fmt(row[col]):>10s}")
            print("".join(parts))


# ---------------------------------------------------------------------------
# Model ranking
# ---------------------------------------------------------------------------


def _rank_norm(series: pd.Series, invert: bool = False) -> pd.Series:
    """Rank-based (percentile) normalisation to [0, 1].

    Each value is scored by its ordinal position among all values,
    which eliminates sensitivity to outliers that distorts min-max scaling.
    Ties receive the average of their shared ranks.
    Returns 0.5 for a single-element series.
    """
    n = len(series)
    if n <= 1:
        return pd.Series(0.5, index=series.index)
    ranked = series.rank(method="average", ascending=True)
    normed = (ranked - 1) / (n - 1)
    return 1.0 - normed if invert else normed


def build_model_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute a weighted composite score per (model, think).

    Returns a DataFrame with columns: model, think, composite_score, rank
    plus the per-metric normalised scores.
    """
    # Aggregate across prompts: mean per (model, think)
    metric_candidates = list(_RANKING_WEIGHTS.keys())
    available = [c for c in metric_candidates if c in summary.columns]

    agg = (
        summary.groupby(["model", "think"], dropna=False)[available]
        .mean(numeric_only=True)
        .reset_index()
    )

    has_judge = (
        "judge_pass_rate" in agg.columns and agg["judge_pass_rate"].notna().any()
    )
    weights = _RANKING_WEIGHTS if has_judge else _RANKING_WEIGHTS_NO_JUDGE
    logger.info("Ranking weights: %s", "full" if has_judge else "speed-only")

    composite = pd.Series(0.0, index=agg.index)
    norm_cols: dict[str, pd.Series] = {}

    for metric, weight in weights.items():
        if metric not in agg.columns or agg[metric].isna().all():
            continue
        filled = agg[metric].fillna(agg[metric].median())
        normed = _rank_norm(filled, invert=metric in _LOWER_IS_BETTER)
        norm_cols[f"{metric}_norm"] = normed
        composite += weight * normed

    agg["composite_score"] = composite
    for col_name, col_data in norm_cols.items():
        agg[col_name] = col_data

    agg = agg.sort_values("composite_score", ascending=False).reset_index(drop=True)
    agg["rank"] = range(1, len(agg) + 1)
    return agg


def print_model_ranking(ranking: pd.DataFrame):
    """Print the ranked leaderboard."""
    if ranking.empty:
        return

    print("\n" + "=" * 60)
    print("  MODEL RANKING (composite score)")
    print("=" * 60)

    model_w = max(len("Model"), ranking["model"].astype(str).str.len().max())
    print(f"  {'Rank':>4}  {'Model':<{model_w}}  {'Think':>5}  {'Score':>7}")
    for _, row in ranking.iterrows():
        think_str = "T" if row["think"] else "F"
        print(
            f"  {int(row['rank']):4d}  {row['model']:<{model_w}}"
            f"  {think_str:>5}  {row['composite_score']:7.3f}"
        )


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

_CROSS_METRICS = [
    ("decode_tokens_per_s", "Decode tok/s", False),
    ("ttft_s", "TTFT (s)", True),
    ("wall_latency_s", "Wall Latency (s)", True),
    ("judge_score", "Judge Score", False),
    ("end_to_end_tokens_per_s", "E2E tok/s", False),
]


def print_cross_model_comparison(summary: pd.DataFrame):
    """Print per-prompt best model for each key metric."""
    print("\n" + "=" * 60)
    print("  CROSS-MODEL COMPARISON (best per prompt)")
    print("=" * 60)

    for prompt_name, sub in summary.groupby("prompt_name", dropna=False):
        print(f"\n  [{prompt_name}]")
        for col, label, lower_better in _CROSS_METRICS:
            if col not in sub.columns or sub[col].dropna().empty:
                continue
            valid = sub.dropna(subset=[col])
            if valid.empty:
                continue
            if lower_better:
                best = valid.loc[valid[col].idxmin()]
            else:
                best = valid.loc[valid[col].idxmax()]
            think_str = "T" if best["think"] else "F"
            print(
                f"    {label:20s}: {best['model']} (think={think_str})"
                f"  = {_fmt(best[col])}"
            )
