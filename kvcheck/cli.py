"""Command-line entry point: `kvcheck run config.yaml`.

Wires config -> suite + engine factories -> runner -> summary -> verdict, then
prints a report and returns an exit code (0 pass / 1 fail) for CI.
"""

import argparse
from pathlib import Path

from kvcheck.config import EngineConfig, RunConfig, SamplingConfig, SuiteConfig
from kvcheck.report import render_console, summarize, verdict, write_json
from kvcheck.runner import EngineFactory, run
from kvcheck.suites.base import PromptSuite
from kvcheck.suites.gsm8k import GSM8KSuite
from kvcheck.suites.synthetic import SyntheticSuite

# Suite registry. New suites register here.
SUITES: dict[str, type[PromptSuite]] = {
    "synthetic": SyntheticSuite,
    "gsm8k": GSM8KSuite,
}


def resolve_model(config: RunConfig, engine_cfg: EngineConfig) -> str:
    """Model for one side: the engine's own override, else the run's model."""
    return engine_cfg.model or config.model


def build_suite(suite_cfg: SuiteConfig) -> PromptSuite:
    try:
        cls = SUITES[suite_cfg.name]
    except KeyError as e:
        raise SystemExit(f"unknown suite: {suite_cfg.name!r}") from e
    return cls(**suite_cfg.params)


def build_engine_factory(
    model: str, engine_cfg: EngineConfig, sampling: SamplingConfig
) -> EngineFactory:
    """Return a zero-arg factory that constructs (but does not start) an engine.

    Heavy backends are imported lazily so that `kvcheck` runs without vLLM
    installed until an actual vLLM run is requested.
    """
    if engine_cfg.adapter == "vllm":

        def factory():
            from kvcheck.engines.vllm_inprocess import VLLMInProcess

            return VLLMInProcess(model=model, engine=engine_cfg, sampling=sampling)

        return factory

    if engine_cfg.adapter == "openai_server":

        def factory():
            from kvcheck.engines.openai_server import OpenAIServerAdapter

            # port from extra["port"] if present, else default; base_url attaches
            # to an already-running server instead of launching one.
            port = int(engine_cfg.extra.get("port", 8000))
            base_url = engine_cfg.extra.get("base_url")
            return OpenAIServerAdapter(
                model=model, engine=engine_cfg, sampling=sampling,
                port=port, base_url=base_url,
            )

        return factory

    raise SystemExit(
        f"adapter {engine_cfg.adapter!r} cannot be constructed from the CLI "
        "(use the Python API to inject a FakeEngine for tests)"
    )


def execute(
    config: RunConfig,
    cache_dir: str | Path,
    json_path: str | Path | None,
    make_golden: EngineFactory,
    make_test: EngineFactory,
) -> int:
    suite = build_suite(config.suite)
    result = run(config, suite, make_golden, make_test, cache_dir)
    summary = summarize(result)
    passed = verdict(summary, config.thresholds)
    render_console(summary, passed)
    if json_path is not None:
        write_json(summary, passed, json_path)
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kvcheck")
    sub = parser.add_subparsers(dest="command", required=True)

    runp = sub.add_parser("run", help="run a golden-vs-test divergence check")
    runp.add_argument("config", help="path to a run config YAML")
    runp.add_argument("--cache-dir", default=".kvcheck/cache")
    runp.add_argument("--json", dest="json_path", default=None, help="write report JSON here")

    args = parser.parse_args(argv)

    config = RunConfig.from_yaml(args.config)
    make_golden = build_engine_factory(
        resolve_model(config, config.golden), config.golden, config.sampling
    )
    make_test = build_engine_factory(
        resolve_model(config, config.test), config.test, config.sampling
    )
    return execute(config, args.cache_dir, args.json_path, make_golden, make_test)


if __name__ == "__main__":
    raise SystemExit(main())
