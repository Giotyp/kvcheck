from kvcheck.suites.base import PromptSuite
from kvcheck.suites.gsm8k import GSM8KSuite, extract_final_number, extract_gold
from kvcheck.types import Completion

# minimal stand-in for the HF dataset rows
EXAMPLES = [
    {"question": "2+2?", "answer": "Adding.\n#### 4"},
    {"question": "3+5?", "answer": "Sum.\n#### 8"},
    {"question": "10-1?", "answer": "Minus.\n#### 9"},
    {"question": "6*7?", "answer": "Product.\n#### 42"},
]


def make(**kw):
    params = dict(num_questions=2, num_fewshot=1, examples=EXAMPLES, warmup=True)
    params.update(kw)
    return GSM8KSuite(**params)


def test_extract_gold_takes_number_after_hashes():
    assert extract_gold("blah\n#### 1,234") == "1234"


def test_extract_final_number_from_model_output():
    assert extract_final_number("So the answer is 42.") == "42"
    assert extract_final_number("no digits here") is None


def test_is_a_prompt_suite():
    assert isinstance(make(), PromptSuite)


def test_scored_question_count():
    reqs = make(num_questions=2).requests()
    assert sum(1 for r in reqs if r.scored) == 2


def test_all_questions_share_the_fewshot_prefix():
    suite = make()
    prompts = [r.prompt for r in suite.requests()]
    prefix = suite.fewshot_prefix
    assert prefix  # non-empty
    assert all(p.startswith(prefix) for p in prompts)


def test_grade_matches_gold():
    suite = make(num_questions=2)
    scored = [r for r in suite.requests() if r.scored]
    # first scored question is examples[num_fewshot=1] -> "3+5?" gold 8
    q = scored[0]
    assert suite.grade(q.request_id, Completion(q.request_id, "answer: 8", (), ())) == 1.0
    assert suite.grade(q.request_id, Completion(q.request_id, "answer: 7", (), ())) == 0.0


def test_warmup_leads_and_is_unscored():
    reqs = make(warmup=True).requests()
    assert reqs[0].scored is False
    assert sum(1 for r in reqs if r.scored) == 2
