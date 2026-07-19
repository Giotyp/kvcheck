import pytest

from kvcheck.config import EngineConfig, RunConfig, SamplingConfig, SuiteConfig
from kvcheck.engines.fake import FakeEngine
from kvcheck.runner import run
from kvcheck.suites.synthetic import SyntheticSuite
from kvcheck.types import Completion


def make_config() -> RunConfig:
    return RunConfig(
        model="fake/model",
        sampling=SamplingConfig(max_tokens=8),
        golden=EngineConfig(adapter="fake", enable_prefix_caching=False),
        test=EngineConfig(adapter="fake", enable_prefix_caching=True),
        suite=SuiteConfig(name="synthetic"),
    )


def suite() -> SyntheticSuite:
    return SyntheticSuite(num_groups=2, questions_per_group=2, prefix_words=10, warmup=True)


def scripts_for(schedule, *, diverge_ids=()) -> dict[str, Completion]:
    """One completion per request; diverge_ids get a different final token."""
    out = {}
    for r in schedule:
        base = (10, 11, 12)
        ids = (10, 11, 99) if r.request_id in diverge_ids else base
        out[r.request_id] = Completion(
            request_id=r.request_id,
            text="",
            token_ids=ids,
            top_logprobs=tuple({t: 0.0} for t in ids),
        )
    return out


class ExplodingEngine(FakeEngine):
    def start(self):
        raise AssertionError("golden engine started on a cache hit")


def test_only_scored_requests_are_compared(tmp_path):
    s = suite()
    sched = s.requests()
    scripts = scripts_for(sched)
    result = run(
        make_config(), s,
        make_golden=lambda: FakeEngine(scripts, "v1"),
        make_test=lambda: FakeEngine(scripts, "v1"),
        cache_dir=tmp_path,
    )
    scored = sum(1 for r in sched if r.scored)
    assert len(result.records) == scored == 4


def test_noise_floor_is_zero_for_deterministic_golden(tmp_path):
    s = suite()
    scripts = scripts_for(s.requests())
    result = run(
        make_config(), s,
        make_golden=lambda: FakeEngine(scripts, "v1"),
        make_test=lambda: FakeEngine(scripts, "v1"),
        cache_dir=tmp_path,
    )
    assert all(r.exact_match for r in result.floor_records)


def test_test_divergence_is_detected(tmp_path):
    s = suite()
    sched = s.requests()
    scored_ids = [r.request_id for r in sched if r.scored]
    golden = scripts_for(sched)
    test = scripts_for(sched, diverge_ids={scored_ids[0]})
    result = run(
        make_config(), s,
        make_golden=lambda: FakeEngine(golden, "v1"),
        make_test=lambda: FakeEngine(test, "v1"),
        cache_dir=tmp_path,
    )
    diverged = [r for r in result.records if not r.exact_match]
    assert len(diverged) == 1
    assert diverged[0].request_id == scored_ids[0]


def test_same_schedule_replayed_to_golden_and_test(tmp_path):
    s = suite()
    sched = s.requests()
    scripts = scripts_for(sched)
    golden_eng = FakeEngine(scripts, "v1")
    test_eng = FakeEngine(scripts, "v1")
    run(
        make_config(), s,
        make_golden=lambda: golden_eng,
        make_test=lambda: test_eng,
        cache_dir=tmp_path,
    )
    expected = [r.request_id for r in sched]
    assert golden_eng.calls[0] == expected  # golden reference pass
    assert test_eng.calls[0] == expected  # test pass


def test_golden_result_is_cached_across_runs(tmp_path):
    s = suite()
    scripts = scripts_for(s.requests())
    cfg = make_config()

    first = run(
        cfg, s,
        make_golden=lambda: FakeEngine(scripts, "v1"),
        make_test=lambda: FakeEngine(scripts, "v1"),
        cache_dir=tmp_path,
    )
    # second run: golden factory would explode if started -> proves cache hit
    second = run(
        cfg, s,
        make_golden=lambda: ExplodingEngine(scripts, "v1"),
        make_test=lambda: FakeEngine(scripts, "v1"),
        cache_dir=tmp_path,
    )
    assert [c.token_ids for c in first.reference] == [c.token_ids for c in second.reference]


def test_engine_version_busts_the_cache(tmp_path):
    s = suite()
    scripts = scripts_for(s.requests())
    cfg = make_config()
    run(
        cfg, s,
        make_golden=lambda: FakeEngine(scripts, "v1"),
        make_test=lambda: FakeEngine(scripts, "v1"),
        cache_dir=tmp_path,
    )
    # different engine version must re-run golden, not reuse v1's artifact
    with pytest.raises(AssertionError):
        run(
            cfg, s,
            make_golden=lambda: ExplodingEngine(scripts, "v2"),
            make_test=lambda: FakeEngine(scripts, "v1"),
            cache_dir=tmp_path,
        )
