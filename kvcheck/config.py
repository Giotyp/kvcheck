"""Declarative run configuration, loaded from YAML.

A RunConfig describes one comparison: a golden engine config (the trusted
baseline, caching off) vs. a test engine config (the KV-cache behavior under
test), sharing one model, one sampling setup, and one prompt suite.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class SamplingConfig(BaseModel):
    """Deterministic-by-default sampling shared by golden and test runs."""

    temperature: float = 0.0
    seed: int = 1234
    max_tokens: int = 256
    top_logprobs: int = 5


class EngineConfig(BaseModel):
    adapter: Literal["vllm", "openai_server", "fake"]
    enable_prefix_caching: bool = False
    kv_cache_dtype: str = "auto"
    extra: dict[str, Any] = Field(default_factory=dict)  # passthrough engine args


class SuiteConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Thresholds(BaseModel):
    """Pass/fail policy knobs. The policy itself lives in report.py."""

    max_divergence_rate_above_floor: float = 0.05
    max_accuracy_drop: float = 0.02


class RunConfig(BaseModel):
    model: str
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    golden: EngineConfig
    test: EngineConfig
    suite: SuiteConfig
    thresholds: Thresholds = Field(default_factory=Thresholds)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    def golden_key(self) -> str:
        """Content key identifying the golden run for the on-disk result cache.

        Two RunConfigs must produce the same key exactly when their golden
        runs would produce the same completions. Including too little risks
        reusing a stale golden (silently wrong verdicts); including too much
        defeats the cache (golden re-runs on every unrelated config tweak).
        Note: the runner separately mixes in the adapter-reported engine
        version at lookup time, so this key only covers config identity.
        """
        gkey = {
            "model": self.model,
            "golden": self.golden.model_dump(),
            "sampling": self.sampling.model_dump(),
            "suite": self.suite.model_dump(),
        }

        json_str = json.dumps(gkey)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
