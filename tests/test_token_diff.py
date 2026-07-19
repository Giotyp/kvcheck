import math

import pytest

from kvcheck.metrics.token_diff import (
    argmax,
    first_divergence_index,
    token_divergence,
    topk_kl,
)
from kvcheck.types import Completion

LN = math.log


def comp(ids, tlp=None) -> Completion:
    ids = tuple(ids)
    if tlp is None:  # default: each emitted token certain (p=1)
        tlp = tuple({tid: 0.0} for tid in ids)
    return Completion(request_id="r", text="", token_ids=ids, top_logprobs=tuple(tlp))


# ---- primitives (already implemented) ----------------------------------


def test_argmax_picks_highest_logprob():
    assert argmax({1: -0.5, 2: -0.1, 3: -2.0}) == 2
    assert argmax({}) is None


def test_topk_kl_zero_for_identical():
    d = {1: LN(0.5), 2: LN(0.5)}
    assert topk_kl(d, d) == pytest.approx(0.0)


def test_topk_kl_floors_missing_token():
    # P puts all mass on token 1; Q's top-k omits token 1 -> floored logprob -30
    assert topk_kl({1: 0.0}, {2: 0.0}) == pytest.approx(30.0)


def test_first_divergence_index_variants():
    assert first_divergence_index((1, 2, 3), (1, 2, 3)) == 3  # identical
    assert first_divergence_index((1, 2, 3), (1, 9, 3)) == 1  # mid diff
    assert first_divergence_index((1, 2), (1, 2, 3)) == 2  # prefix
    assert first_divergence_index((), (1,)) == 0  # empty


# ---- token_divergence (TODO(human)) ------------------------------------


def test_identical_completions():
    c = comp([1, 2])
    r = token_divergence(c, c)
    assert r.exact_match is True
    assert r.first_divergence_index == 2
    assert r.n_comparable == 2
    assert r.argmax_flips == 0
    assert r.mean_kl == pytest.approx(0.0)
    assert (r.golden_len, r.test_len) == (2, 2)


def test_divergence_at_position_one_with_distribution_flip():
    golden = comp([5, 7], tlp=[{5: 0.0}, {7: LN(0.6), 8: LN(0.4)}])
    test = comp([5, 8], tlp=[{5: 0.0}, {8: LN(0.6), 7: LN(0.4)}])
    r = token_divergence(golden, test)
    assert r.exact_match is False
    assert r.first_divergence_index == 1
    assert r.n_comparable == 2  # positions 0 and 1 share identical context
    assert r.argmax_flips == 1  # only position 1 flips
    # pos 0 KL = 0; pos 1 KL = 0.2*ln(1.5); mean over 2 positions
    expected = (0.0 + 0.2 * LN(1.5)) / 2
    assert r.mean_kl == pytest.approx(expected)


def test_prefix_with_length_difference():
    golden = comp([1, 2, 3])
    test = comp([1, 2])
    r = token_divergence(golden, test)
    assert r.exact_match is False
    assert r.first_divergence_index == 2
    assert r.n_comparable == 2  # position 2 exists only in golden -> not comparable
    assert r.argmax_flips == 0
    assert r.mean_kl == pytest.approx(0.0)
    assert (r.golden_len, r.test_len) == (3, 2)
