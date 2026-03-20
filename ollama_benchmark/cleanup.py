"""Ollama cleanup utilities: process management, VRAM release, cache flushing."""

import logging
import signal
import subprocess
import time
from typing import Any

import psutil

from .gpu import query_gpu_once

logger = logging.getLogger(__name__)

_ollama: Any = None
try:
    import ollama as _ollama
except ImportError:
    pass


def kill_ollama_runners() -> list[int]:
    """Kill orphaned 'ollama runner' sub-processes (NOT the main serve process).

    Ollama spawns a separate ``ollama runner`` process per loaded model.
    These can survive after the Python client disconnects or times out,
    keeping VRAM allocated indefinitely.
    """
    killed = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline)
            if "ollama" not in cmdline_str:
                continue
            if "runner" in cmdline_str:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except psutil.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                killed.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        except Exception as e:
            logger.warning("Could not kill PID %s: %s", proc.info.get("pid"), e)
            print(f"    warn: could not kill PID {proc.info.get('pid')}: {e}")
    if killed:
        logger.info("Killed runner PIDs: %s", killed)
    else:
        logger.debug("No runner processes found to kill")
    return killed


def unload_all_models_api(quiet: bool = False):
    """Ask Ollama to unload every running model via keep_alive=0."""
    if _ollama is None:
        if not quiet:
            print("    (ollama package not available)")
        return
    try:
        running = _ollama.ps()
        models_list = getattr(running, "models", None) or []
        if not models_list:
            if not quiet:
                print("    (no models reported by ollama ps)")
            return
        for entry in models_list:
            name = getattr(entry, "model", None) or getattr(entry, "name", None)
            if not name:
                continue
            if not quiet:
                print(f"    Unloading {name} via API …", end=" ", flush=True)
            try:
                _ollama.generate(model=name, prompt="", keep_alive=0)
                logger.info("Unloaded model %s via API", name)
                if not quiet:
                    print("ok")
            except Exception as e:
                logger.warning("Failed to unload model %s: %s", name, e)
                if not quiet:
                    print(f"failed ({e})")
    except Exception as e:
        logger.warning("Could not list running models: %s", e)
        print(f"    ⚠  Could not list running models: {e}")


def drop_caches():
    """Flush Linux page/dentry/inode caches to free shared-memory residue."""
    try:
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            timeout=10,
            capture_output=True,
        )
    except Exception as e:
        logger.debug("drop_caches failed (non-critical): %s", e)


def wait_for_vram_release(
    max_wait_s: float = 30, target_free_pct: float = 0.70, quiet: bool = False
) -> tuple[bool, str]:
    """Poll nvidia-smi until VRAM usage drops below threshold.

    Falls back to checking for ollama runner processes if the GPU does not
    report memory usage (e.g. NVIDIA GB10 / DGX Spark shows 'Not Supported').

    Returns (success, status_message).
    """
    rows = query_gpu_once()
    has_mem_reporting = False
    if rows:
        total = sum(r.get("gpu_mem_total_mb") or 0 for r in rows)
        has_mem_reporting = total > 0

    if has_mem_reporting:
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            rows = query_gpu_once()
            if rows:
                total = sum(r.get("gpu_mem_total_mb") or 0 for r in rows)
                used = sum(r.get("gpu_mem_used_mb") or 0 for r in rows)
                if total > 0:
                    free_pct = 1.0 - (used / total)
                    logger.debug(
                        "VRAM poll: %.0f/%.0f MB (%.0f%% free)",
                        used,
                        total,
                        free_pct * 100,
                    )
                    if free_pct >= target_free_pct:
                        msg = f"ok ({free_pct * 100:.0f}% free)"
                        if not quiet:
                            print(
                                f"    VRAM OK: {used:.0f}/{total:.0f} MB ({free_pct * 100:.0f}% free)"
                            )
                        return True, msg
            time.sleep(2)
        rows = query_gpu_once()
        if rows:
            total = sum(r.get("gpu_mem_total_mb") or 0 for r in rows)
            used = sum(r.get("gpu_mem_used_mb") or 0 for r in rows)
            free_pct = 1.0 - (used / total) if total > 0 else 0
            msg = f"warn: VRAM {100 - free_pct * 100:.0f}% used after {max_wait_s}s"
            logger.warning(
                "VRAM not released after %ss: %.0f/%.0f MB", max_wait_s, used, total
            )
            if not quiet:
                print(
                    f"    ⚠  VRAM not fully released after {max_wait_s}s: {used:.0f}/{total:.0f} MB"
                )
            return False, msg
        return False, f"warn: no GPU data after {max_wait_s}s"
    else:
        if not quiet:
            print(
                "    (GPU does not report memory — checking runner processes instead)"
            )
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            runners = []
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    if "ollama" in cmdline and "runner" in cmdline:
                        runners.append(proc.info["pid"])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if not runners:
                if not quiet:
                    print("    OK: no runner processes found")
                return True, "ok (no runners)"
            time.sleep(2)
        msg = f"warn: runners still active after {max_wait_s}s"
        if not quiet:
            print(
                f"    ⚠  Runner processes still active after {max_wait_s}s: {runners}"
            )
        return False, msg


def force_cleanup(
    vram_settle_timeout_s: float = 30,
    vram_free_target_pct: float = 0.70,
    quiet: bool = False,
) -> str:
    """Full cleanup sequence: API unload → kill runners → drop caches → verify VRAM.

    Returns a short status string (e.g. ``'ok (91% free)'``).
    When *quiet* is True, routine prints are suppressed.
    """
    logger.info("Cleanup [1/4] API-based model unload")
    if not quiet:
        print("  [1/4] API-based model unload …")
    unload_all_models_api(quiet=quiet)
    time.sleep(2)

    logger.info("Cleanup [2/4] Killing orphaned runner processes")
    if not quiet:
        print("  [2/4] Killing orphaned runner processes …")
    killed = kill_ollama_runners()
    if not quiet:
        if killed:
            print(f"    Killed PIDs: {killed}")
        else:
            print("    (none found)")
    time.sleep(2)

    logger.info("Cleanup [3/4] Flushing page caches")
    if not quiet:
        print("  [3/4] Flushing page caches …")
    drop_caches()

    logger.info("Cleanup [4/4] Waiting for VRAM release")
    if not quiet:
        print("  [4/4] Waiting for VRAM release …")
    _ok, status = wait_for_vram_release(
        max_wait_s=vram_settle_timeout_s,
        target_free_pct=vram_free_target_pct,
        quiet=quiet,
    )
    logger.info("Cleanup result: %s", status)
    return status
