from kvcheck.config import EngineConfig, RunConfig, SamplingConfig, SuiteConfig
from kvcheck.engines.fake import FakeEngine
from kvcheck.runner import run
from kvcheck.suites.base import PromptSuite
from kvcheck.types import Completion, GenerationRequest


class GradedSuite(PromptSuite):
    """Two scored questions with gold answers 1 and 2 (plus one warmup)."""

    def requests(self):
        return [
            GenerationRequest("warm", "prefix q0", group_id="g", scored=False),
            GenerationRequest("q0", "prefix q0", group_id="g", scored=True),
            GenerationRequest("q1", "prefix q1", group_id="g", scored=True),
        ]

    def grade(self, request_id, completion):
        gold = {"q0": (1,), "q1": (2,)}.get(request_id)
        if gold is None:
            return None
        return 1.0 if completion.token_ids == gold else 0.0


def cfg():
    return RunConfig(
        model="fake/m", sampling=SamplingConfig(),
        golden=EngineConfig(adapter="fake"), test=EngineConfig(adapter="fake"),
        suite=SuiteConfig(name="graded"),
    )


def scripts(mapping):
    return {rid: Completion(rid, "", tuple(ids), ()) for rid, ids in mapping.items()}


def test_accuracy_computed_for_graded_suite(tmp_path):
    # golden gets both right (acc 1.0); test gets q1 wrong (acc 0.5)
    golden = scripts({"warm": (0,), "q0": (1,), "q1": (2,)})
    test = scripts({"warm": (0,), "q0": (1,), "q1": (9,)})
    result = run(
        cfg(), GradedSuite(),
        make_golden=lambda: FakeEngine(golden, "v1"),
        make_test=lambda: FakeEngine(test, "v1"),
        cache_dir=tmp_path,
    )
    assert result.golden_accuracy == 1.0
    assert result.test_accuracy == 0.5


def test_accuracy_is_none_for_ungraded_suite(tmp_path):
    from kvcheck.suites.synthetic import SyntheticSuite

    s = SyntheticSuite(num_groups=1, questions_per_group=2, prefix_words=5)
    sc = {r.request_id: Completion(r.request_id, "", (1, 2, 3), ()) for r in s.requests()}
    result = run(
        cfg(), s,
        make_golden=lambda: FakeEngine(sc, "v1"),
        make_test=lambda: FakeEngine(sc, "v1"),
        cache_dir=tmp_path,
    )
    assert result.golden_accuracy is None
    assert result.test_accuracy is None
