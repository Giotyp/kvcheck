import json

import pytest

from kvcheck.cli import build_suite, execute, main, run_report
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


def _report(path, divergence_rate, floor=0.0, test_acc=0.6, drop=0.0):
    path.write_text(json.dumps({
        "passed": True,
        "summary": {
            "n_scored": 50, "divergence_rate": divergence_rate,
            "argmax_flip_rate": 0.01, "mean_kl": 0.005,
            "floor_divergence_rate": floor, "floor_mean_kl": 0.001,
            "golden_accuracy": 0.6, "test_accuracy": test_acc, "accuracy_drop": drop,
        },
    }))


def test_report_single_returns_zero(tmp_path):
    r = tmp_path / "r.json"
    _report(r, 0.2)
    assert run_report(str(r), None, tol=0.05) == 0


def test_report_compare_flags_regression(tmp_path):
    base, cur = tmp_path / "base.json", tmp_path / "cur.json"
    _report(base, 0.2)
    _report(cur, 0.5)  # divergence excess jumped
    assert run_report(str(cur), str(base), tol=0.05) == 1


def test_report_compare_ok_when_stable(tmp_path):
    base, cur = tmp_path / "base.json", tmp_path / "cur.json"
    _report(base, 0.2)
    _report(cur, 0.21)
    assert run_report(str(cur), str(base), tol=0.05) == 0


def test_main_report_compare_exit_code(tmp_path):
    base, cur = tmp_path / "base.json", tmp_path / "cur.json"
    _report(base, 0.2)
    _report(cur, 0.5)
    assert main(["report", str(cur), "--compare", str(base)]) == 1
