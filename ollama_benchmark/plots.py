"""Chart generation and CSV export for benchmark results."""

import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

PLOT_METRICS = [
    ("ttft_s", "TTFT (s)"),
    ("wall_latency_s", "Wall Latency (s)"),
    ("prompt_tokens_per_s", "Prompt Tokens/s"),
    ("decode_tokens_per_s", "Decode Tokens/s"),
    ("end_to_end_tokens_per_s", "End-to-End Tokens/s"),
    ("thinking_chars", "Thinking Chars"),
    ("judge_score", "Judge Score"),
    ("gpu_util_avg_pct", "GPU Utilization Avg %"),
    ("gpu_mem_used_peak_mb", "GPU Memory Peak MB"),
    ("gpu_power_avg_w", "GPU Power Avg W"),
]


def generate_plots(
    summary: pd.DataFrame,
    prompts: list[dict],
    n_models: int,
    results_dir: str,
    ts_str: str,
    quiet: bool = False,
) -> list[str]:
    """Create per-prompt bar charts for each metric and save as PNG.

    Returns a list of saved file paths.
    """
    saved: list[str] = []
    plot_df = summary.assign(
        think_label=summary["think"].map({True: "think=True", False: "think=False"})
    )

    for prompt_entry in prompts:
        pname = prompt_entry["name"]
        pdf = plot_df[plot_df["prompt_name"] == pname]
        if pdf.empty:
            continue

        for metric_col, metric_title in PLOT_METRICS:
            if metric_col not in pdf.columns or pdf[metric_col].dropna().empty:
                continue

            try:
                pivot = pdf.pivot(
                    index="model",
                    columns="think_label",
                    values=metric_col,
                )
            except ValueError:
                # Fallback if duplicates exist
                pivot = pdf.pivot_table(
                    index="model",
                    columns="think_label",
                    values=metric_col,
                    aggfunc="mean",
                )
            if pivot.empty or pivot.dropna(how="all").empty:
                continue
            pivot = pivot.sort_index()

            fig, ax = plt.subplots(figsize=(max(10.0, 1.2 * n_models), 5))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"{metric_title}  [{pname}]", fontsize=14)
            ax.set_xlabel("")
            ax.set_ylabel(metric_title)
            ax.legend(title="Mode", fontsize="small", loc="best")
            plt.xticks(rotation=35, ha="right", fontsize=9)
            plt.tight_layout()
            plot_path = os.path.join(
                results_dir, f"ollama_benchmark_{metric_col}_{pname}_{ts_str}.png"
            )
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            saved.append(plot_path)
            logger.debug("Saved plot: %s", plot_path)
            if not quiet:
                print(f"Saved plot: {plot_path}")

    logger.info("Generated %d metric plots", len(saved))
    return saved


def generate_accuracy_chart(
    summary: pd.DataFrame,
    results_dir: str,
    ts_str: str,
    quiet: bool = False,
) -> str | None:
    """Create a grouped bar chart of judge_pass_rate per model by think-mode.

    Returns the saved file path, or None if no judge data is available.
    """
    if "judge_pass_rate" not in summary.columns:
        return None

    agg = (
        summary.groupby(["model", "think"], dropna=False)["judge_pass_rate"]
        .mean()
        .reset_index()
    )
    agg = agg.dropna(subset=["judge_pass_rate"])
    if agg.empty:
        return None

    agg["think_label"] = agg["think"].map({True: "think=True", False: "think=False"})
    pivot = agg.pivot(index="model", columns="think_label", values="judge_pass_rate")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(max(10.0, 1.2 * len(pivot)), 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Judge Pass Rate by Model", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Pass Rate")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Mode", fontsize="small", loc="best")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.tight_layout()

    plot_path = os.path.join(results_dir, f"ollama_benchmark_accuracy_{ts_str}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Accuracy chart saved: %s", plot_path)
    if not quiet:
        print(f"Saved plot: {plot_path}")
    return plot_path


def generate_category_chart(
    cat_summary: pd.DataFrame,
    results_dir: str,
    ts_str: str,
    quiet: bool = False,
) -> str | None:
    """Create a grouped bar chart of judge_score per category per model.

    Returns the saved file path, or None if no data is available.
    """
    if cat_summary.empty or "judge_score" not in cat_summary.columns:
        return None
    data = cat_summary.dropna(subset=["judge_score"])
    if data.empty:
        return None

    data = data.copy()
    data["label"] = (
        data["model"]
        .astype(str)
        .str.cat(data["think"].map({True: "(T)", False: "(F)"}), sep=" ")
    )

    pivot = data.pivot_table(
        index="label", columns="category", values="judge_score", aggfunc="mean"
    )
    if pivot.empty:
        return None
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(max(10.0, 1.2 * len(pivot)), 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Judge Score by Category & Model", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Judge Score")
    ax.legend(title="Category", fontsize="small", loc="best")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.tight_layout()

    plot_path = os.path.join(results_dir, f"ollama_benchmark_category_{ts_str}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Category chart saved: %s", plot_path)
    if not quiet:
        print(f"Saved plot: {plot_path}")
    return plot_path


def generate_walltime_vs_accuracy_chart(
    summary: pd.DataFrame,
    results_dir: str,
    ts_str: str,
    quiet: bool = False,
) -> str | None:
    """Scatter plot of wall latency vs judge pass rate per (model, think).

    Returns the saved file path, or None if no judge data is available.
    """
    if (
        "judge_pass_rate" not in summary.columns
        or "wall_latency_s" not in summary.columns
    ):
        return None

    agg = (
        summary.groupby(["model", "think"], dropna=False)
        .agg(
            wall_latency_s=("wall_latency_s", "sum"),
            judge_pass_rate=("judge_pass_rate", "mean"),
        )
        .reset_index()
    )
    agg = agg.dropna(subset=["judge_pass_rate", "wall_latency_s"])
    if agg.empty:
        return None

    agg["label"] = (
        agg["model"]
        .astype(str)
        .str.cat(agg["think"].map({True: "(T)", False: "(F)"}), sep=" ")
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    for think_val, marker, color in [(True, "^", "#1f77b4"), (False, "o", "#ff7f0e")]:
        subset = agg[agg["think"] == think_val]
        if subset.empty:
            continue
        ax.scatter(
            subset["wall_latency_s"],
            subset["judge_pass_rate"] * 100,
            marker=marker,
            color=color,
            s=80,
            label=f"think={'True' if think_val else 'False'}",
            zorder=3,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["label"],
                (row["wall_latency_s"], row["judge_pass_rate"] * 100),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )

    ax.set_title("Wall Time vs Accuracy", fontsize=14)
    ax.set_xlabel("Total Wall Latency (s)")
    ax.set_ylabel("Judge Pass Rate (%)")
    ax.set_ylim(-5, 105)
    ax.legend(title="Mode", fontsize="small", loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(
        results_dir, f"ollama_benchmark_walltime_vs_accuracy_{ts_str}.png"
    )
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Wall-time vs accuracy chart saved: %s", plot_path)
    if not quiet:
        print(f"Saved plot: {plot_path}")
    return plot_path


def generate_throughput_scatter_chart(
    summary: pd.DataFrame,
    results_dir: str,
    ts_str: str,
    quiet: bool = False,
) -> str | None:
    """Scatter plot of prompt tok/s vs decode tok/s per model.

    Returns the saved file path, or None if no data is available.
    """
    if (
        "prompt_tokens_per_s" not in summary.columns
        or "decode_tokens_per_s" not in summary.columns
    ):
        return None

    agg = (
        summary.groupby("model", dropna=False)
        .agg(
            prompt_tokens_per_s=("prompt_tokens_per_s", "mean"),
            decode_tokens_per_s=("decode_tokens_per_s", "mean"),
        )
        .reset_index()
    )
    agg = agg.dropna(subset=["prompt_tokens_per_s", "decode_tokens_per_s"])
    if agg.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        agg["prompt_tokens_per_s"],
        agg["decode_tokens_per_s"],
        marker="o",
        color="#1f77b4",
        s=80,
        zorder=3,
    )
    for _, row in agg.iterrows():
        ax.annotate(
            row["model"],
            (row["prompt_tokens_per_s"], row["decode_tokens_per_s"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.set_title("Encoding vs Decoding Throughput", fontsize=14)
    ax.set_xlabel("Prompt Tokens/s (Encoding)")
    ax.set_ylabel("Decode Tokens/s (Decoding)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(
        results_dir, f"ollama_benchmark_throughput_scatter_{ts_str}.png"
    )
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Throughput scatter chart saved: %s", plot_path)
    if not quiet:
        print(f"Saved plot: {plot_path}")
    return plot_path


def export_csvs(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    results_dir: str,
    ts_str: str,
    cat_summary: pd.DataFrame | None = None,
    ranking: pd.DataFrame | None = None,
    quiet: bool = False,
) -> list[str]:
    """Save raw, summary, and optional category/ranking DataFrames to CSV.

    Returns a list of saved file paths.
    """
    saved: list[str] = []

    raw_path = os.path.join(results_dir, f"ollama_benchmark_raw_{ts_str}.csv")
    summary_path = os.path.join(results_dir, f"ollama_benchmark_summary_{ts_str}.csv")
    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    saved.extend([raw_path, summary_path])

    if cat_summary is not None and not cat_summary.empty:
        cat_path = os.path.join(
            results_dir, f"ollama_benchmark_category_summary_{ts_str}.csv"
        )
        cat_summary.to_csv(cat_path, index=False)
        saved.append(cat_path)

    if ranking is not None and not ranking.empty:
        rank_path = os.path.join(results_dir, f"ollama_benchmark_ranking_{ts_str}.csv")
        ranking.to_csv(rank_path, index=False)
        saved.append(rank_path)

    logger.info("Exported %d CSV files", len(saved))
    if not quiet:
        for p in saved:
            print(f"Saved: {p}")

    return saved
