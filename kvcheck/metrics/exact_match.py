"""Exact-match rate: fraction of requests whose test tokens equal golden tokens."""

from kvcheck.types import Completion


def _by_id(completions: list[Completion]) -> dict[str, Completion]:
    return {c.request_id: c for c in completions}


def exact_match_rate(golden: list[Completion], test: list[Completion]) -> float:
    """Fraction of shared request IDs whose token_ids match exactly.

    Pairs by request_id (not list position), so the two runs may return
    completions in different orders. Returns 1.0 when there is nothing to
    compare.
    """
    g = _by_id(golden)
    t = _by_id(test)
    shared = g.keys() & t.keys()
    if not shared:
        return 1.0
    matches = sum(1 for rid in shared if g[rid].token_ids == t[rid].token_ids)
    return matches / len(shared)
