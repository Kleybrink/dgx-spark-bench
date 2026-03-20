"""GPU monitoring: background sampler and one-shot query via nvidia-smi."""

import logging
import subprocess
import threading
import time
from statistics import mean

logger = logging.getLogger(__name__)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def query_gpu_once() -> list[dict]:
    """Query nvidia-smi for current GPU state. Returns list of per-GPU dicts."""
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.STDOUT,
                timeout=5,
            )
            .decode("utf-8")
            .strip()
        )
        rows = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                continue
            rows.append(
                {
                    "gpu_index": int(parts[0]),
                    "gpu_name": parts[1],
                    "gpu_util_pct": _safe_float(parts[2]),
                    "gpu_mem_used_mb": _safe_float(parts[3]),
                    "gpu_mem_total_mb": _safe_float(parts[4]),
                    "gpu_power_w": _safe_float(parts[5]),
                }
            )
        return rows
    except Exception as e:
        logger.debug("nvidia-smi query failed: %s", e)
        return []


class GPUSampler:
    """Background thread that samples GPU metrics at regular intervals."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.debug("GPUSampler started (interval=%.1fs)", self.interval_s)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        logger.debug("GPUSampler stopped (%d samples collected)", len(self.samples))

    def _run(self):
        while not self._stop.is_set():
            ts = time.time()
            for row in query_gpu_once():
                self.samples.append({**row, "ts": ts})
            time.sleep(self.interval_s)

    def summary(self) -> dict:
        if not self.samples:
            return {
                "gpu_seen": False,
                "gpu_count_seen": 0,
                "gpu_util_avg_pct": None,
                "gpu_util_peak_pct": None,
                "gpu_mem_used_avg_mb": None,
                "gpu_mem_used_peak_mb": None,
                "gpu_power_avg_w": None,
                "gpu_power_peak_w": None,
            }
        utils = [
            s["gpu_util_pct"] for s in self.samples if s["gpu_util_pct"] is not None
        ]
        mems = [
            s["gpu_mem_used_mb"]
            for s in self.samples
            if s["gpu_mem_used_mb"] is not None
        ]
        powers = [
            s["gpu_power_w"] for s in self.samples if s["gpu_power_w"] is not None
        ]
        return {
            "gpu_seen": True,
            "gpu_count_seen": len({s["gpu_index"] for s in self.samples}),
            "gpu_util_avg_pct": mean(utils) if utils else None,
            "gpu_util_peak_pct": max(utils) if utils else None,
            "gpu_mem_used_avg_mb": mean(mems) if mems else None,
            "gpu_mem_used_peak_mb": max(mems) if mems else None,
            "gpu_power_avg_w": mean(powers) if powers else None,
            "gpu_power_peak_w": max(powers) if powers else None,
        }
