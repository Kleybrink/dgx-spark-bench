"""Model warmup via subprocess with hard timeout."""

import json
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

WARMUP_SCRIPT_TEMPLATE = """\
import sys, json
try:
    import ollama
    from ollama import ResponseError
    model, think = sys.argv[1], sys.argv[2] == "True"
    ollama.chat(
        model=model,
        messages=[{{"role": "user", "content": "Reply with exactly: ok"}}],
        think=think,
        stream=False,
        keep_alive="{keep_alive}",
    )
    print(json.dumps({{"status": "ok"}}))
except ResponseError as e:
    msg = str(e).lower()
    if e.status_code == 404 or "not found" in msg:
        print(json.dumps({{"status": "not_found"}}))
    elif e.status_code == 400 or "does not support thinking" in msg:
        print(json.dumps({{"status": "no_think"}}))
    else:
        print(json.dumps({{"status": "error", "msg": str(e)}}))
except Exception as e:
    print(json.dumps({{"status": "error", "msg": str(e)}}))
"""


def warmup(
    model: str, think: bool, keep_alive: str = "10m", timeout_s: float = 600
) -> str:
    """Run warmup in a subprocess with a hard timeout.

    Returns one of: 'ok', 'no_think', 'not_found', 'timeout', 'error'.
    Subprocess guarantees the HTTP connection is fully closed on kill.
    """
    logger.info("Warmup start: model=%s think=%s timeout=%ss", model, think, timeout_s)
    script = WARMUP_SCRIPT_TEMPLATE.format(keep_alive=keep_alive)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, model, str(think)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[:200]
            logger.warning("Warmup subprocess error for %s: %s", model, stderr)
            print(f"    warmup stderr: {stderr}")
            return "error"
        out = result.stdout.strip()
        if not out:
            logger.warning("Warmup returned empty output for %s", model)
            return "error"
        data = json.loads(out)
        status = data.get("status", "error")
        logger.info("Warmup result: model=%s think=%s status=%s", model, think, status)
        return status
    except subprocess.TimeoutExpired:
        logger.warning(
            "Warmup timeout after %ss for model=%s think=%s", timeout_s, model, think
        )
        return "timeout"
    except Exception as e:
        logger.error("Warmup exception for %s: %s", model, e, exc_info=True)
        print(f"    warmup exception: {e}")
        return "error"
