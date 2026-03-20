"""Ollama benchmark toolkit — importable package."""

from .logging_config import setup_logging
from .gpu import GPUSampler, query_gpu_once
from .cleanup import (
    kill_ollama_runners,
    unload_all_models_api,
    drop_caches,
    wait_for_vram_release,
    force_cleanup,
)
from .warmup import warmup, WARMUP_SCRIPT_TEMPLATE
from .runner import run_one, ns_to_s
from .judge import judge_response
from .config import BenchmarkConfig, load_config
from .wip import WIPTracker
from .analysis import (
    build_summary,
    print_results,
    print_interpretation,
    print_cross_model_comparison,
    build_category_summary,
    print_category_summary,
    build_model_ranking,
    print_model_ranking,
    print_prompt_comparison,
)
from .plots import (
    generate_plots,
    generate_accuracy_chart,
    generate_category_chart,
    export_csvs,
)
from .display import fmt_elapsed, print_leaderboard, print_final_leaderboard
from .orchestrator import print_config, run_benchmark, save_results

__all__ = [
    "setup_logging",
    "GPUSampler",
    "query_gpu_once",
    "kill_ollama_runners",
    "unload_all_models_api",
    "drop_caches",
    "wait_for_vram_release",
    "force_cleanup",
    "warmup",
    "WARMUP_SCRIPT_TEMPLATE",
    "run_one",
    "ns_to_s",
    "judge_response",
    "BenchmarkConfig",
    "load_config",
    "WIPTracker",
    "build_summary",
    "print_results",
    "print_interpretation",
    "print_cross_model_comparison",
    "build_category_summary",
    "print_category_summary",
    "build_model_ranking",
    "print_model_ranking",
    "print_prompt_comparison",
    "generate_plots",
    "generate_accuracy_chart",
    "generate_category_chart",
    "export_csvs",
    "fmt_elapsed",
    "print_leaderboard",
    "print_final_leaderboard",
    "print_config",
    "run_benchmark",
    "save_results",
]
