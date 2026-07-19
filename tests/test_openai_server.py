from kvcheck.config import EngineConfig, SamplingConfig
from kvcheck.engines.openai_server import (
    build_server_command,
    intern_token,
    parse_completion,
)
from kvcheck.types import GenerationRequest


def test_intern_token_is_deterministic_and_distinct():
    assert intern_token("hello") == intern_token("hello")
    assert intern_token("hello") != intern_token("world")
    assert isinstance(intern_token("x"), int)


def test_build_command_reflects_prefix_caching_flag():
    on = build_server_command(
        "py", "m", EngineConfig(adapter="openai_server", enable_prefix_caching=True),
        SamplingConfig(), port=8001,
    )
    off = build_server_command(
        "py", "m", EngineConfig(adapter="openai_server", enable_prefix_caching=False),
        SamplingConfig(), port=8001,
    )
    assert "--enable-prefix-caching" in on
    assert "--no-enable-prefix-caching" in off
    assert "8001" in on
    assert "m" in on


def test_build_command_passes_extra_args_as_flags():
    cmd = build_server_command(
        "py", "m",
        EngineConfig(adapter="openai_server", extra={"gpu_memory_utilization": 0.8}),
        SamplingConfig(), port=8000,
    )
    assert "--gpu-memory-utilization" in cmd
    assert "0.8" in cmd


def test_parse_completion_maps_strings_to_interned_ids():
    req = GenerationRequest("r0", "prompt")
    choice = {
        "text": " 42",
        "finish_reason": "stop",
        "logprobs": {
            "tokens": ["4", "2"],
            "token_logprobs": [-0.1, -0.2],
            "top_logprobs": [{"4": -0.1, "5": -2.0}, {"2": -0.2, "3": -1.9}],
        },
    }
    comp = parse_completion(req, choice)
    assert comp.request_id == "r0"
    assert comp.text == " 42"
    assert comp.token_ids == (intern_token("4"), intern_token("2"))
    assert comp.top_logprobs[0][intern_token("4")] == -0.1
    assert comp.top_logprobs[1][intern_token("3")] == -1.9


def test_parse_completion_handles_missing_logprobs():
    req = GenerationRequest("r0", "p")
    comp = parse_completion(req, {"text": "hi", "finish_reason": "stop", "logprobs": None})
    assert comp.token_ids == ()
    assert comp.top_logprobs == ()
