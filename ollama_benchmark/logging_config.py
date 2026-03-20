"""Centralised file-based logging configuration for the benchmark."""

import logging
import os
from datetime import datetime


def setup_logging(
    log_dir: str = "results",
    level: int = logging.DEBUG,
) -> str:
    """Configure file-only logging for the ``ollama_benchmark`` package.

    Creates a timestamped log file under *log_dir*.  Only a
    :class:`~logging.FileHandler` is attached — console output via
    ``print()`` is **not** affected.

    Returns the absolute path of the log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"benchmark_{ts}.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(formatter)

    # Package logger — all child loggers (ollama_benchmark.*) inherit this handler.
    pkg_logger = logging.getLogger("ollama_benchmark")
    pkg_logger.setLevel(level)
    pkg_logger.addHandler(handler)

    # Entry-point logger (benchmark.py uses __name__ == "__main__" → "benchmark").
    bench_logger = logging.getLogger("benchmark")
    bench_logger.setLevel(level)
    bench_logger.addHandler(handler)

    return os.path.abspath(log_path)
