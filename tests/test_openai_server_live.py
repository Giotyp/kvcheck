"""Live HTTP validation of OpenAIServerAdapter against a real (mock) server.

The adapter's contract is 'speak the OpenAI completions API correctly over HTTP'.
This exercises that contract over real sockets — health polling, the POST body,
response parsing, and the full runner path — without needing a GPU or a working
`vllm serve`. Attach mode (base_url set) is used, so no subprocess is launched.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kvcheck.config import EngineConfig, RunConfig, SamplingConfig, SuiteConfig
from kvcheck.engines.openai_server import OpenAIServerAdapter, intern_token
from kvcheck.runner import run
from kvcheck.suites.synthetic import SyntheticSuite
from kvcheck.types import GenerationRequest

SAMPLING = SamplingConfig(max_tokens=8)


class _MockOpenAI(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible server: /health, /version, /v1/completions."""

    received_bodies: list[dict] = []

    def log_message(self, *args):  # silence access logs
        pass

    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b"ok")
        elif self.path == "/version":
            self._send(200, json.dumps({"version": "mock"}).encode())
        else:
            self._send(404, b"{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        type(self).received_bodies.append(body)
        # Echo a deterministic two-token completion with top-k logprobs.
        resp = {
            "choices": [
                {
                    "text": " 42",
                    "finish_reason": "stop",
                    "logprobs": {
                        "tokens": ["4", "2"],
                        "token_logprobs": [-0.1, -0.2],
                        "top_logprobs": [{"4": -0.1, "5": -2.0}, {"2": -0.2, "3": -1.9}],
                    },
                }
            ]
        }
        self._send(200, json.dumps(resp).encode())


@pytest.fixture
def server_url():
    _MockOpenAI.received_bodies = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockOpenAI)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()


def make_adapter(base_url: str) -> OpenAIServerAdapter:
    return OpenAIServerAdapter(
        model="mock/model",
        engine=EngineConfig(adapter="openai_server"),
        sampling=SAMPLING,
        base_url=base_url,
        startup_timeout=10,
    )


def test_attach_health_and_generate(server_url):
    with make_adapter(server_url) as eng:
        out = eng.generate([GenerationRequest("r0", "the prompt")], SAMPLING)
    assert out[0].request_id == "r0"
    assert out[0].text == " 42"
    assert out[0].token_ids == (intern_token("4"), intern_token("2"))
    assert out[0].top_logprobs[0][intern_token("5")] == -2.0


def test_request_body_is_well_formed(server_url):
    sampling = SamplingConfig(max_tokens=8, stop=["\nQ:"])
    with make_adapter(server_url) as eng:
        eng.generate([GenerationRequest("r0", "hello world")], sampling)
    body = _MockOpenAI.received_bodies[-1]
    assert body["prompt"] == "hello world"
    assert body["max_tokens"] == 8
    assert body["temperature"] == 0.0
    assert body["stop"] == ["\nQ:"]
    assert body["logprobs"] == SAMPLING.top_logprobs


def test_version_queried_over_http(server_url):
    assert make_adapter(server_url).version() == "vllm-server-mock"


def test_full_runner_through_http(server_url, tmp_path):
    suite = SyntheticSuite(num_groups=2, questions_per_group=2, prefix_words=5)
    config = RunConfig(
        model="mock/model",
        sampling=SAMPLING,
        golden=EngineConfig(adapter="openai_server", enable_prefix_caching=False),
        test=EngineConfig(adapter="openai_server", enable_prefix_caching=True),
        suite=SuiteConfig(name="synthetic"),
    )
    result = run(
        config, suite,
        make_golden=lambda: make_adapter(server_url),
        make_test=lambda: make_adapter(server_url),
        cache_dir=tmp_path,
    )
    # mock returns identical completions -> zero divergence, zero floor
    assert len(result.records) == 4
    assert all(r.exact_match for r in result.records)
    assert all(r.exact_match for r in result.floor_records)


def test_unhealthy_server_fails_fast_with_reason():
    # nothing listening on port 1 -> should raise with the last connection error
    eng = OpenAIServerAdapter(
        model="m", engine=EngineConfig(adapter="openai_server"),
        sampling=SAMPLING, base_url="http://127.0.0.1:1", startup_timeout=1,
    )
    with pytest.raises(RuntimeError, match="last error"):
        eng.start()
