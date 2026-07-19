import pytest

from kvcheck.cli import build_suite, execute
from kvcheck.config import EngineConfig, RunConfig, SamplingConfig, SuiteConfig, Thresholds
from kvcheck.engines.fake import FakeEngine
from kvcheck.suites.synthetic import SyntheticSuite
from kvcheck.types import Completion


def cfg(**kw) -> RunConfig:
    base = dict(
        model="fake/m",
        sampling=SamplingConfig(),
        golden=EngineConfig(adapter="fake"),
        test=EngineConfig(adapter="fake"),
        suite=SuiteConfig(name="synthetic", params={"num_groups": 1, "questions_per_group": 2}),
        thresholds=Thresholds(),
    )
    base.update(kw)
    return RunConfig(**base)


def scripts_for(suite, diverge=False):
    out = {}
    for r in suite.requests():
        ids = (1, 2, 9) if (diverge and r.scored) else (1, 2, 3)
        out[r.request_id] = Completion(r.request_id, "", tuple(ids), tuple({t: 0.0} for t in ids))
    return out


def test_build_suite_from_config():
    suite = build_suite(SuiteConfig(name="synthetic", params={"num_groups": 2}))
    assert isinstance(suite, SyntheticSuite)
    assert suite.num_groups == 2


def test_unknown_suite_raises():
    with pytest.raises(SystemExit):
        build_suite(SuiteConfig(name="nope"))


def test_execute_returns_zero_when_passing(tmp_path):
    config = cfg()
    scripts = scripts_for(build_suite(config.suite))  # golden == test -> pass
    code = execute(
        config, cache_dir=tmp_path, json_path=None,
        make_golden=lambda: FakeEngine(scripts, "v1"),
        make_test=lambda: FakeEngine(scripts, "v1"),
    )
    assert code == 0


def test_execute_returns_one_when_failing(tmp_path):
    config = cfg()
    suite = build_suite(config.suite)
    golden = scripts_for(suite)
    test = scripts_for(suite, diverge=True)  # every scored request diverges
    code = execute(
        config, cache_dir=tmp_path, json_path=None,
        make_golden=lambda: FakeEngine(golden, "v1"),
        make_test=lambda: FakeEngine(test, "v1"),
    )
    assert code == 1


def test_execute_writes_json_artifact(tmp_path):
    config = cfg()
    scripts = scripts_for(build_suite(config.suite))
    out = tmp_path / "report.json"
    execute(
        config, cache_dir=tmp_path, json_path=out,
        make_golden=lambda: FakeEngine(scripts, "v1"),
        make_test=lambda: FakeEngine(scripts, "v1"),
    )
    assert out.exists()
