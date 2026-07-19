"""Orchestration: golden (cached) + calibration + test -> divergence records.

The runner is deliberately engine-agnostic. It takes *factories* (not built
engines) so that on a golden cache hit it never even constructs — let alone
loads — the expensive golden model. Three logical passes:

  reference   : golden config, pass 1  (the trusted baseline)
  calibration : golden config, pass 2  (identical config, re-run -> noise floor)
  test        : test config,   pass 1  (the KV-cache behavior under test)

Signal   = divergence(reference, test).
Noise floor = divergence(reference, calibration)  -- how much golden differs
from *itself* purely from run-to-run nondeterminism. The report judges the
signal relative to this floor.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kvcheck.config import RunConfig
from kvcheck.engines.base import EngineAdapter
from kvcheck.metrics.token_diff import token_divergence
from kvcheck.suites.base import PromptSuite
from kvcheck.types import Completion, DivergenceRecord

EngineFactory = Callable[[], EngineAdapter]


@dataclass
class RunResult:
    reference: list[Completion]
    calibration: list[Completion]
    test: list[Completion]
    records: list[DivergenceRecord]  # test vs reference, scored requests only
    floor_records: list[DivergenceRecord]  # calibration vs reference, scored only
    engine_versions: dict[str, str]
    golden_accuracy: float | None = None  # None for ungraded suites
    test_accuracy: float | None = None


# ---- Completion (de)serialization for the on-disk golden cache -----------
# JSON object keys are always strings, so top_logprobs' int token-id keys are
# stringified on write and restored to int on read.


def _completion_to_dict(c: Completion) -> dict:
    return {
        "request_id": c.request_id,
        "text": c.text,
        "token_ids": list(c.token_ids),
        "top_logprobs": [{str(k): v for k, v in d.items()} for d in c.top_logprobs],
        "finish_reason": c.finish_reason,
    }


def _completion_from_dict(d: dict) -> Completion:
    return Completion(
        request_id=d["request_id"],
        text=d["text"],
        token_ids=tuple(d["token_ids"]),
        top_logprobs=tuple({int(k): v for k, v in lp.items()} for lp in d["top_logprobs"]),
        finish_reason=d.get("finish_reason", "stop"),
    )


def _compare(
    reference: list[Completion], other: list[Completion], scored_ids: set[str]
) -> list[DivergenceRecord]:
    ref = {c.request_id: c for c in reference}
    oth = {c.request_id: c for c in other}
    return [token_divergence(ref[rid], oth[rid]) for rid in sorted(scored_ids)]


def _accuracy(
    suite: PromptSuite, completions: list[Completion], scored_ids: set[str]
) -> float | None:
    """Mean grade over scored requests, or None if the suite grades nothing."""
    by_id = {c.request_id: c for c in completions}
    grades = [
        g
        for rid in scored_ids
        if (g := suite.grade(rid, by_id[rid])) is not None
    ]
    if not grades:
        return None
    return sum(grades) / len(grades)


def _load_or_run_golden(
    config: RunConfig, schedule, make_golden: EngineFactory, cache_dir: Path
) -> tuple[list[Completion], list[Completion], str]:
    engine = make_golden()  # cheap: construction must not load the model
    version = engine.version()
    key = f"{config.golden_key()}-{version}"
    path = Path(cache_dir) / f"{key}.json"

    if path.exists():
        payload = json.loads(path.read_text())
        ref = [_completion_from_dict(d) for d in payload["reference"]]
        cal = [_completion_from_dict(d) for d in payload["calibration"]]
        return ref, cal, version

    with engine:  # start() loads the model; stop() frees it
        ref = engine.generate(schedule, config.sampling)
        cal = engine.generate(schedule, config.sampling)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "reference": [_completion_to_dict(c) for c in ref],
                "calibration": [_completion_to_dict(c) for c in cal],
            }
        )
    )
    return ref, cal, version


def run(
    config: RunConfig,
    suite: PromptSuite,
    make_golden: EngineFactory,
    make_test: EngineFactory,
    cache_dir: str | Path = ".kvcheck/cache",
) -> RunResult:
    schedule = suite.requests()
    scored_ids = {r.request_id for r in schedule if r.scored}

    reference, calibration, gver = _load_or_run_golden(
        config, schedule, make_golden, cache_dir
    )

    test_engine = make_test()
    with test_engine:
        tver = test_engine.version()
        test_out = test_engine.generate(schedule, config.sampling)

    return RunResult(
        reference=reference,
        calibration=calibration,
        test=test_out,
        records=_compare(reference, test_out, scored_ids),
        floor_records=_compare(reference, calibration, scored_ids),
        engine_versions={"golden": gver, "test": tver},
        golden_accuracy=_accuracy(suite, reference, scored_ids),
        test_accuracy=_accuracy(suite, test_out, scored_ids),
    )
