"""Ollama Multi-Model Benchmark — script version of Benchmark.ipynb."""

import logging
import platform
import subprocess
import sys

for pkg in ["httpx", "psutil", "ollama"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import ollama_benchmark as ob  # noqa: E402

logger = logging.getLogger("benchmark")

log_path = ob.setup_logging()
print(f"Log: {log_path}")

logger.info("Benchmark starting — Python %s on %s", sys.version, platform.platform())

try:
    cfg = ob.load_config()
    wip = ob.WIPTracker(quiet=True)

    ob.print_config(cfg, wip)
    ob.run_benchmark(cfg, wip)
    ob.save_results(cfg, wip)
except Exception:
    logger.critical("Fatal error in benchmark run", exc_info=True)
    raise
