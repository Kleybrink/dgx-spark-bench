"""Core benchmark runner: single httpx streaming request with metrics collection."""

import json
import logging
import time

import httpx

from .gpu import GPUSampler

logger = logging.getLogger(__name__)


def ns_to_s(ns) -> float | None:
    return ns / 1e9 if ns is not None else None


def run_one(
    model: str,
    prompt: str,
    think: bool,
    run_idx: int,
    base_url: str = "http://localhost:11434",
    keep_alive: str = "10m",
    timeout_s: float = 600,
    gpu_sample_interval_s: float = 0.5,
) -> dict | None:
    """Run a single benchmark using direct httpx streaming.

    httpx enforces read-timeout at the socket level. When the timeout fires,
    the connection is closed immediately, which signals Ollama to stop
    generation and begin runner teardown. No zombie threads possible.

    Returns a metrics dict, or None if the run timed out.
    """
    logger.info(
        "run_one start: model=%s think=%s run=%d timeout=%ss",
        model,
        think,
        run_idx,
        timeout_s,
    )

    sampler = GPUSampler(interval_s=gpu_sample_interval_s)
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "think": think,
        "stream": True,
        "keep_alive": keep_alive,
    }

    answer_parts: list[str] = []
    thinking_parts: list[str] = []
    final_chunk: dict | None = None
    first_visible_token_time: float | None = None
    timed_out = False

    sampler.start()
    wall_start = time.time()

    try:
        with httpx.Client(
            timeout=httpx.Timeout(
                connect=60.0,
                read=timeout_s,
                write=30.0,
                pool=30.0,
            )
        ) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    now = time.time()
                    if (now - wall_start) > timeout_s:
                        timed_out = True
                        break

                    msg = chunk.get("message", {})
                    piece_thinking = msg.get("thinking") or ""
                    piece_content = msg.get("content") or ""

                    if first_visible_token_time is None and (
                        piece_thinking or piece_content
                    ):
                        first_visible_token_time = now
                        logger.debug("First token received at %.3fs", now - wall_start)

                    if piece_thinking:
                        thinking_parts.append(piece_thinking)
                    if piece_content:
                        answer_parts.append(piece_content)

                    if chunk.get("done"):
                        final_chunk = chunk
                        break

    except httpx.ReadTimeout:
        timed_out = True
        logger.warning(
            "ReadTimeout after %ss for model=%s think=%s", timeout_s, model, think
        )
    except httpx.ConnectError as e:
        logger.error("ConnectError for %s: %s", base_url, e)
        raise ConnectionError(f"Cannot connect to Ollama at {base_url}: {e}") from e
    except httpx.HTTPStatusError as e:
        logger.error(
            "HTTPStatusError %d: %s", e.response.status_code, e.response.text[:200]
        )
        raise RuntimeError(
            f"Ollama HTTP error: {e.response.status_code} — {e.response.text[:200]}"
        ) from e
    finally:
        wall_end = time.time()
        sampler.stop()

    if timed_out:
        return None

    wall_s = wall_end - wall_start
    ttft_s = (
        (first_visible_token_time - wall_start) if first_visible_token_time else None
    )

    fc = final_chunk or {}
    total_ns = fc.get("total_duration")
    load_ns = fc.get("load_duration")
    pe_count = fc.get("prompt_eval_count")
    pe_dur_ns = fc.get("prompt_eval_duration")
    ev_count = fc.get("eval_count")
    ev_dur_ns = fc.get("eval_duration")

    pe_dur_s = ns_to_s(pe_dur_ns)
    ev_dur_s = ns_to_s(ev_dur_ns)

    prompt_tps = (
        (pe_count / pe_dur_s) if pe_count and pe_dur_s and pe_dur_s > 0 else None
    )
    decode_tps = (
        (ev_count / ev_dur_s) if ev_count and ev_dur_s and ev_dur_s > 0 else None
    )
    e2e_tps = (ev_count / wall_s) if ev_count and wall_s > 0 else None

    answer_text = "".join(answer_parts)
    thinking_text = "".join(thinking_parts)

    result = {
        "model": model,
        "think": think,
        "run": run_idx,
        "wall_latency_s": wall_s,
        "ttft_s": ttft_s,
        "total_duration_s": ns_to_s(total_ns),
        "load_duration_s": ns_to_s(load_ns),
        "prompt_eval_count": pe_count,
        "prompt_eval_duration_s": pe_dur_s,
        "eval_count": ev_count,
        "eval_duration_s": ev_dur_s,
        "prompt_tokens_per_s": prompt_tps,
        "decode_tokens_per_s": decode_tps,
        "end_to_end_tokens_per_s": e2e_tps,
        "done_reason": fc.get("done_reason"),
        "answer_chars": len(answer_text),
        "thinking_chars": len(thinking_text),
        "answer_text": answer_text,
        "answer_preview": answer_text[:200],
        "thinking_preview": thinking_text[:200],
    }
    result.update(sampler.summary())

    logger.info(
        "run_one done: model=%s think=%s run=%d wall=%.1fs ttft=%s decode_tps=%s",
        model,
        think,
        run_idx,
        wall_s,
        f"{ttft_s:.2f}s" if ttft_s is not None else "n/a",
        f"{decode_tps:.1f}" if decode_tps is not None else "n/a",
    )

    return result
