import json

import pytest

from kvcheck.config import Thresholds
from kvcheck.report import Summary, summarize, verdict, write_json
from kvcheck.runner import RunResult
from kvcheck.types import Completion, DivergenceRecord


def comp(rid, ids) -> Completion:
    return Completion(request_id=rid, text="", token_ids=tuple(ids),
                      top_logprobs=tuple({t: 0.0} for t in ids))


def rec(rid, *, exact, flips=0, n=3, kl=0.0) -> DivergenceRecord:
    return DivergenceRecord(
        request_id=rid, exact_match=exact, first_divergence_index=(3 if exact else 1),
        n_comparable=n, argmax_flips=flips, mean_kl=kl, golden_len=3, test_len=3,
    )


def make_result(records, floor_records, golden_acc=None, test_acc=None) -> RunResult:
    return RunResult(
        reference=[], calibration=[], test=[],
        records=records, floor_records=floor_records,
        engine_versions={"golden": "v1", "test": "v1"},
        golden_accuracy=golden_acc, test_accuracy=test_acc,
    )


def test_summarize_accuracy_drop():
    records = [rec("a", exact=True)]
    s = summarize(make_result(records, records, golden_acc=0.8, test_acc=0.7))
    assert s.golden_accuracy == 0.8
    assert s.test_accuracy == 0.7
    assert s.accuracy_drop == pytest.approx(0.1)


def test_summarize_accuracy_drop_none_when_ungraded():
    records = [rec("a", exact=True)]
    s = summarize(make_result(records, records))
    assert s.accuracy_drop is None


# ---- summarize (implemented) -------------------------------------------


def test_summarize_divergence_rate():
    records = [rec("a", exact=True), rec("b", exact=False), rec("c", exact=True), rec("d", exact=False)]
    floor = [rec("a", exact=True), rec("b", exact=True), rec("c", exact=True), rec("d", exact=True)]
    s = summarize(make_result(records, floor))
    assert s.n_scored == 4
    assert s.divergence_rate == 0.5
    assert s.floor_divergence_rate == 0.0


def test_summarize_argmax_flip_rate_is_position_weighted():
    # 2 flips out of 6 comparable positions
    records = [rec("a", exact=False, flips=2, n=4), rec("b", exact=True, flips=0, n=2)]
    s = summarize(make_result(records, records))
    assert s.argmax_flip_rate == 2 / 6


# ---- verdict (TODO(human)) ---------------------------------------------


def base_thresholds():
    return Thresholds(max_divergence_rate_above_floor=0.05)


def test_identical_config_passes():
    # divergence exactly equals the floor -> excess is zero -> PASS
    records = [rec("a", exact=False), rec("b", exact=True)]
    s = summarize(make_result(records, records))
    assert verdict(s, base_thresholds()) is True


def test_divergence_far_above_floor_fails():
    records = [rec("a", exact=False), rec("b", exact=False)]  # 100% diverge
    floor = [rec("a", exact=True), rec("b", exact=True)]  # 0% floor
    s = summarize(make_result(records, floor))
    assert verdict(s, base_thresholds()) is False


def test_high_divergence_all_explained_by_floor_passes():
    # golden is itself nondeterministic: floor is also high. Excess ~ 0 -> PASS.
    records = [rec("a", exact=False), rec("b", exact=False), rec("c", exact=True), rec("d", exact=True)]
    floor = [rec("a", exact=False), rec("b", exact=False), rec("c", exact=True), rec("d", exact=True)]
    s = summarize(make_result(records, floor))
    assert verdict(s, base_thresholds()) is True


def test_small_excess_within_margin_passes():
    # 0.25 divergence vs 0.25 floor... excess 0 -> pass; bump one to test margin
    records = [rec(f"r{i}", exact=(i != 0)) for i in range(20)]  # 1/20 = 0.05 diverge
    floor = [rec(f"r{i}", exact=True) for i in range(20)]  # 0 floor
    s = summarize(make_result(records, floor))
    assert verdict(s, base_thresholds()) is True  # excess 0.05 == threshold


def thresholds_both():
    return Thresholds(max_divergence_rate_above_floor=0.05, max_accuracy_drop=0.02)


def test_accuracy_drop_above_threshold_fails():
    # divergence is within the floor (would pass on its own) but accuracy fell 10pts
    records = [rec("a", exact=True)]
    s = summarize(make_result(records, records, golden_acc=0.80, test_acc=0.70))
    assert verdict(s, thresholds_both()) is False


def test_small_accuracy_drop_passes():
    records = [rec("a", exact=True)]
    s = summarize(make_result(records, records, golden_acc=0.80, test_acc=0.79))
    assert verdict(s, thresholds_both()) is True


def test_none_accuracy_drop_is_ignored():
    # ungraded suite: accuracy_drop is None -> verdict decided by divergence alone
    records = [rec("a", exact=True)]
    s = summarize(make_result(records, records))  # no accuracy
    assert verdict(s, thresholds_both()) is True


def test_accuracy_improvement_never_fails():
    # test config happened to score higher; negative drop must not fail the run
    records = [rec("a", exact=True)]
    s = summarize(make_result(records, records, golden_acc=0.70, test_acc=0.90))
    assert verdict(s, thresholds_both()) is True


# ---- write_json (implemented) ------------------------------------------


def test_write_json_roundtrips(tmp_path):
    records = [rec("a", exact=False)]
    s = summarize(make_result(records, records))
    path = tmp_path / "report.json"
    write_json(s, verdict_passed=True, path=path)
    loaded = json.loads(path.read_text())
    assert loaded["passed"] is True
    assert loaded["summary"]["n_scored"] == 1
