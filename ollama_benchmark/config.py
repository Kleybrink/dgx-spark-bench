"""Load and validate benchmark configuration from config.yaml."""

import logging
import os
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """All benchmark settings parsed from config.yaml."""

    models: list[str]
    prompts: list[dict]
    ollama_base_url: str
    runs_per_mode: int
    keep_alive: str
    timeout_s: float
    warmup_timeout_s: float
    vram_settle_s: float
    vram_free_target: float
    judge_model: str | None
    judge_temperature: float
    judge_think: bool


def load_config(path: str = "config.yaml") -> BenchmarkConfig:
    """Read config.yaml and return a validated BenchmarkConfig."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Parse prompts list (backward compat: single `prompt` key → list of one)
    if "prompts" in cfg:
        prompts = cfg["prompts"]
    else:
        prompts = [
            {
                "name": "default",
                "prompt": cfg["prompt"],
                "golden_answer": cfg.get("golden_answer"),
            }
        ]

    for p in prompts:
        p["prompt"] = p["prompt"].strip()
        if not p.get("name"):
            raise ValueError("Each prompt entry must have a 'name' field")

    config = BenchmarkConfig(
        models=cfg["models"],
        prompts=prompts,
        ollama_base_url=os.environ.get(
            "OLLAMA_HOST", cfg.get("ollama_base_url", "http://localhost:11434")
        ),
        runs_per_mode=cfg.get("runs_per_mode", 1),
        keep_alive=cfg.get("keep_alive", "10m"),
        timeout_s=cfg.get("timeout_s", 600),
        warmup_timeout_s=cfg.get("warmup_timeout_s", 600),
        vram_settle_s=cfg.get("vram_settle_s", 30),
        vram_free_target=cfg.get("vram_free_target", 0.70),
        judge_model=cfg.get("judge_model"),
        judge_temperature=cfg.get("judge_temperature", 0.1),
        judge_think=cfg.get("judge_think", False),
    )

    logger.info(
        "Config loaded from %s: %d models, %d prompts, timeout=%ss, warmup=%ss, judge=%s",
        path,
        len(config.models),
        len(config.prompts),
        config.timeout_s,
        config.warmup_timeout_s,
        config.judge_model or "disabled",
    )

    return config
