from kvcheck.suites.base import PromptSuite
from kvcheck.suites.lm_eval import LMEvalSuite, pick_metric
from kvcheck.types import Completion

DOCS = [{"q": "2+2?", "a": "4"}, {"q": "3+5?", "a": "8"}, {"q": "10-1?", "a": "9"}]


class FakeTask:
    """Minimal stand-in for an lm-eval generate_until Task."""

    def doc_to_text(self, doc):
        return f"Question: {doc['q']}\nAnswer:"

    def doc_to_target(self, doc):
        return f" {doc['a']}"

    def process_results(self, doc, results):
        pred = results[0].strip()
        return {"exact_match": 1.0 if pred == doc["a"] else 0.0}


def make(**kw) -> LMEvalSuite:
    params = dict(task=FakeTask(), docs=DOCS, prefix="PREFIX\n\n", warmup=True)
    params.update(kw)
    return LMEvalSuite(**params)


def comp(rid, text) -> Completion:
    return Completion(request_id=rid, text=text, token_ids=())


# ---- pick_metric (implemented) -----------------------------------------


def test_pick_metric_prefers_exact_match():
    assert pick_metric({"acc": 0.0, "exact_match": 1.0}, None) == 1.0


def test_pick_metric_respects_explicit_key():
    assert pick_metric({"acc": 1.0, "acc_norm": 0.0}, "acc_norm") == 0.0


def test_pick_metric_falls_back_to_first_value():
    assert pick_metric({"rougeL": 0.42}, None) == 0.42


# ---- suite behavior ----------------------------------------------------


def test_is_a_prompt_suite():
    assert isinstance(make(), PromptSuite)


def test_one_scored_request_per_doc():
    reqs = make().requests()
    assert sum(1 for r in reqs if r.scored) == len(DOCS)


def test_prompts_share_prefix_and_include_question():
    suite = make()
    for r in suite.requests():
        assert r.prompt.startswith("PREFIX\n\n")
        assert "Question:" in r.prompt


def test_warmup_leads_and_is_unscored():
    reqs = make(warmup=True).requests()
    assert reqs[0].scored is False
    assert sum(1 for r in reqs if r.scored) == len(DOCS)


def test_no_warmup_when_disabled():
    assert all(r.scored for r in make(warmup=False).requests())


def test_request_ids_unique():
    reqs = make().requests()
    assert len({r.request_id for r in reqs}) == len(reqs)


def test_grade_uses_task_process_results():
    suite = make()
    scored = [r for r in suite.requests() if r.scored]
    q0 = scored[0]  # DOCS[0] gold "4"
    # exact grading: process_results compares stripped text to gold
    assert suite.grade(q0.request_id, comp(q0.request_id, "4")) == 1.0
    assert suite.grade(q0.request_id, comp(q0.request_id, "5")) == 0.0


def test_grade_unknown_request_returns_none():
    assert make().grade("nope", comp("nope", "x")) is None
