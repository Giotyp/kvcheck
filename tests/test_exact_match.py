from kvcheck.metrics.exact_match import exact_match_rate
from kvcheck.types import Completion


def comp(rid, ids) -> Completion:
    return Completion(request_id=rid, text="", token_ids=tuple(ids))


def test_all_match_gives_rate_one():
    golden = [comp("a", [1, 2]), comp("b", [3])]
    test = [comp("a", [1, 2]), comp("b", [3])]
    assert exact_match_rate(golden, test) == 1.0


def test_partial_match_rate():
    golden = [comp("a", [1, 2]), comp("b", [3, 4])]
    test = [comp("a", [1, 2]), comp("b", [3, 9])]
    assert exact_match_rate(golden, test) == 0.5


def test_pairs_by_request_id_not_position():
    golden = [comp("a", [1]), comp("b", [2])]
    test = [comp("b", [2]), comp("a", [1])]  # reversed order
    assert exact_match_rate(golden, test) == 1.0


def test_empty_returns_one():
    assert exact_match_rate([], []) == 1.0
