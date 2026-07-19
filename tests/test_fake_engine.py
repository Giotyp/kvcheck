import pytest

from kvcheck.config import SamplingConfig
from kvcheck.engines.base import EngineAdapter
from kvcheck.engines.fake import FakeEngine
from kvcheck.types import Completion, GenerationRequest

SAMPLING = SamplingConfig()


def req(rid: str, prompt: str = "p") -> GenerationRequest:
    return GenerationRequest(request_id=rid, prompt=prompt)


def comp(rid: str, token_ids=(1, 2, 3)) -> Completion:
    return Completion(request_id=rid, text="x", token_ids=tuple(token_ids))


def test_fake_engine_is_an_engine_adapter():
    assert isinstance(FakeEngine(scripts={}), EngineAdapter)


def test_returns_scripted_completions_in_request_order():
    scripts = {"a": comp("a"), "b": comp("b")}
    with FakeEngine(scripts=scripts) as eng:
        out = eng.generate([req("b"), req("a")], SAMPLING)
    assert [c.request_id for c in out] == ["b", "a"]


def test_unknown_request_id_raises():
    with FakeEngine(scripts={}) as eng:
        with pytest.raises(KeyError):
            eng.generate([req("missing")], SAMPLING)


def test_generate_before_start_raises():
    eng = FakeEngine(scripts={"a": comp("a")})
    with pytest.raises(RuntimeError):
        eng.generate([req("a")], SAMPLING)


def test_version_is_stable_and_nonempty():
    eng = FakeEngine(scripts={}, version_tag="fake-1")
    assert eng.version() == "fake-1"
    assert eng.version() == eng.version()


def test_records_calls_for_schedule_assertions():
    """Runner tests need to prove the same schedule was replayed to each engine."""
    scripts = {"a": comp("a"), "b": comp("b")}
    with FakeEngine(scripts=scripts) as eng:
        eng.generate([req("a"), req("b")], SAMPLING)
        eng.generate([req("b")], SAMPLING)
    assert eng.calls == [["a", "b"], ["b"]]
