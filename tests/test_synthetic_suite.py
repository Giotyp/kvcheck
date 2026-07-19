from kvcheck.suites.base import PromptSuite
from kvcheck.suites.synthetic import SyntheticSuite


def make(**kw) -> SyntheticSuite:
    params = dict(num_groups=3, questions_per_group=4, prefix_words=50, seed=7, warmup=True)
    params.update(kw)
    return SyntheticSuite(**params)


def group_of(rid: str) -> str:
    # request_id format is "<group_id>/..."; group_id is the part before the slash
    return rid.split("/", 1)[0]


def test_is_a_prompt_suite():
    assert isinstance(make(), PromptSuite)


def test_scored_count_matches_groups_times_questions():
    reqs = make(num_groups=3, questions_per_group=4).requests()
    scored = [r for r in reqs if r.scored]
    assert len(scored) == 12


def test_every_prompt_starts_with_its_group_prefix():
    suite = make()
    reqs = suite.requests()
    for r in reqs:
        assert r.group_id is not None
        assert r.prompt.startswith(suite.group_prefix(r.group_id))


def test_request_ids_are_unique():
    reqs = make().requests()
    assert len({r.request_id for r in reqs}) == len(reqs)


def test_group_requests_are_contiguous():
    """All requests of one group must sit back-to-back, or the test engine may
    evict the shared prefix before it can be reused."""
    order = [group_of(r.request_id) for r in make().requests()]
    # each group id appears as a single unbroken run
    seen_runs = [g for i, g in enumerate(order) if i == 0 or order[i - 1] != g]
    assert len(seen_runs) == len(set(seen_runs))


def test_warmup_request_is_unscored_and_first_in_its_group():
    reqs = make(warmup=True).requests()
    # group the requests by their group id, preserving order
    by_group: dict[str, list] = {}
    for r in reqs:
        by_group.setdefault(r.group_id, []).append(r)
    for group_reqs in by_group.values():
        assert group_reqs[0].scored is False  # cold warmup pass leads
        assert all(r.scored for r in group_reqs[1:])  # warm scored requests follow


def test_no_warmup_when_disabled():
    reqs = make(warmup=False).requests()
    assert all(r.scored for r in reqs)


def test_deterministic_for_same_seed():
    a = [(r.request_id, r.prompt) for r in make(seed=42).requests()]
    b = [(r.request_id, r.prompt) for r in make(seed=42).requests()]
    assert a == b


def test_different_seed_changes_content():
    a = {r.prompt for r in make(seed=1).requests()}
    b = {r.prompt for r in make(seed=2).requests()}
    assert a != b
