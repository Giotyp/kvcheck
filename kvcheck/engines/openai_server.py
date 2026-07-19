"""OpenAI-compatible server adapter.

Launches (or attaches to) a `vllm serve` process and drives it over HTTP. This
proves the EngineAdapter abstraction is engine-agnostic: anything speaking the
OpenAI completions API — vLLM, SGLang, LMCache — plugs in here unchanged.

Token-id caveat: the HTTP completions API returns token *strings*, not the
integer ids the in-process adapter exposes and the metrics compare. We intern
each string to a stable pseudo-id (a hash), so golden and test — which share a
tokenizer — map identical strings to identical ids. These ids are only
meaningful for equality/KL comparison, never as real vocab indices.
"""

import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

from kvcheck.config import EngineConfig, SamplingConfig
from kvcheck.engines.base import EngineAdapter
from kvcheck.types import Completion, GenerationRequest


def intern_token(token: str) -> int:
    """Stable pseudo-id for a token string (same string -> same id everywhere)."""
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=6).digest(), "big")


def build_server_command(
    python: str, model: str, engine: EngineConfig, sampling: SamplingConfig, port: int
) -> list[str]:
    cmd = [
        python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--seed", str(sampling.seed),
        "--kv-cache-dtype", engine.kv_cache_dtype,
    ]
    cmd.append(
        "--enable-prefix-caching" if engine.enable_prefix_caching
        else "--no-enable-prefix-caching"
    )
    # port/base_url are adapter-level, not vllm-serve flags.
    reserved = {"port", "base_url"}
    for key, value in engine.extra.items():
        if key in reserved:
            continue
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    return cmd


def parse_completion(request: GenerationRequest, choice: dict) -> Completion:
    lp = choice.get("logprobs")
    token_ids: tuple[int, ...] = ()
    top: tuple[dict[int, float], ...] = ()
    if lp:
        token_ids = tuple(intern_token(t) for t in lp.get("tokens", []))
        top = tuple(
            {intern_token(k): v for k, v in pos.items()}
            for pos in lp.get("top_logprobs", []) or []
        )
    return Completion(
        request_id=request.request_id,
        text=choice.get("text", ""),
        token_ids=token_ids,
        top_logprobs=top,
        finish_reason=choice.get("finish_reason") or "stop",
    )


class OpenAIServerAdapter(EngineAdapter):
    def __init__(
        self,
        model: str,
        engine: EngineConfig,
        sampling: SamplingConfig,
        host: str = "127.0.0.1",
        port: int = 8000,
        base_url: str | None = None,
        startup_timeout: float = 300.0,
    ):
        self.model = model
        self.engine = engine
        self._sampling = sampling
        self.port = port
        # If base_url is given we attach to an already-running server and never
        # launch/kill a subprocess.
        self._attach = base_url is not None
        self.base_url = base_url or f"http://{host}:{port}"
        self.startup_timeout = startup_timeout
        self._proc: subprocess.Popen | None = None

    def version(self) -> str:
        try:
            with urllib.request.urlopen(f"{self.base_url}/version", timeout=5) as r:
                return f"vllm-server-{json.load(r).get('version', '?')}"
        except Exception:
            return "vllm-server"

    def start(self) -> None:
        if self._attach:
            self._wait_healthy()
            return
        cmd = build_server_command(
            sys.executable, self.model, self.engine, self._sampling, self.port
        )
        self._proc = subprocess.Popen(cmd)
        self._wait_healthy()

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_error = "no response"
        while time.monotonic() < deadline:
            # A dead subprocess will never become healthy — fail immediately.
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"vllm server exited (code {self._proc.returncode}) before becoming healthy"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as r:
                    if r.status == 200:
                        return
                    last_error = f"HTTP {r.status}"
            except urllib.error.HTTPError as e:
                # Server is up but erroring (e.g. broken router) — record why.
                last_error = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
                time.sleep(2.0)
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_error = str(e)  # not listening yet
                time.sleep(2.0)
        raise RuntimeError(
            f"server at {self.base_url} not healthy within {self.startup_timeout}s; "
            f"last error: {last_error}"
        )

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def generate(
        self, requests: list[GenerationRequest], sampling: SamplingConfig
    ) -> list[Completion]:
        # Sequential submission (one HTTP call per request) faithfully replays
        # the suite's warm-cache schedule server-side.
        return [self._one(r, sampling) for r in requests]

    def _one(self, request: GenerationRequest, sampling: SamplingConfig) -> Completion:
        body = {
            "model": self.model,
            "prompt": request.prompt,
            "max_tokens": sampling.max_tokens,
            "temperature": sampling.temperature,
            "seed": sampling.seed,
            "logprobs": sampling.top_logprobs,
        }
        if sampling.stop:
            body["stop"] = sampling.stop
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/completions", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.load(r)
        return parse_completion(request, payload["choices"][0])
