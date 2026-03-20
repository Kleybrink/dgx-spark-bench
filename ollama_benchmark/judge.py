"""LLM-as-a-Judge: score a model's answer against a golden answer."""

import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)


_SYSTEM = """\
You are an impartial evaluator. Your task is to compare a model's response \
against the expected (golden) answer and score how correct it is.

Rubric:
  1.0 — Response is fully correct (matches the golden answer exactly or semantically equivalent)
  0.5 — Response is partially correct (contains some correct elements but is incomplete or has errors)
  0.0 — Response is incorrect, irrelevant, or missing

Respond with ONLY valid JSON (no markdown fences, no prose outside the JSON):
{
  "reasoning": "<brief explanation of why the score was assigned>",
  "score": <0.0 | 0.5 | 1.0>,
  "pass": <true if score == 1.0, else false>
}
"""

_USER_TMPL = """\
ORIGINAL QUESTION:
{prompt}

EXPECTED ANSWER (golden):
{golden_answer}

MODEL RESPONSE:
{answer_text}

Compare the model response to the expected answer and assign a score.
"""


def judge_response(
    answer_text: str,
    golden_answer: str,
    judge_model: str,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.1,
    think: bool = False,
    timeout_s: float = 120,
    prompt: str = "",
) -> dict:
    """Call the judge LLM and return a scoring dict.

    Returns a dict with keys: judge_score, judge_pass, judge_reasoning, judge_error.
    On any failure, judge_error is set and other fields are None.
    """
    user_msg = _USER_TMPL.format(
        prompt=prompt.strip(),
        golden_answer=golden_answer.strip(),
        answer_text=answer_text.strip() or "(empty response)",
    )
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "think": think,
        "stream": False,
        "options": {"temperature": temperature},
    }
    logger.info(
        "Judge request: model=%s answer_len=%d",
        judge_model,
        len(answer_text),
    )
    try:
        with httpx.Client(
            timeout=httpx.Timeout(connect=30.0, read=timeout_s, write=30.0, pool=30.0)
        ) as client:
            resp = client.post(f"{base_url}/api/chat", json=payload)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
    except Exception as exc:
        logger.error("Judge HTTP error: %s", exc)
        return _error_row(str(exc))

    # Strip optional markdown code fences
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("Judge JSON parse failed, trying fallback extraction")
        # Try to extract first {...} block
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
            except json.JSONDecodeError:
                logger.error("Judge JSON parse failed after retry: %s", clean[:200])
                return _error_row(f"JSON parse failed: {clean[:200]}")
        else:
            logger.error("No JSON found in judge response: %s", clean[:200])
            return _error_row(f"No JSON found in judge response: {clean[:200]}")

    result = {
        "judge_score": parsed.get("score"),
        "judge_pass": parsed.get("pass"),
        "judge_reasoning": parsed.get("reasoning", ""),
        "judge_error": None,
    }
    logger.info(
        "Judge result: score=%s pass=%s", result["judge_score"], result["judge_pass"]
    )
    return result


def _error_row(msg: str) -> dict:
    return {
        "judge_score": None,
        "judge_pass": None,
        "judge_reasoning": None,
        "judge_error": msg,
    }
