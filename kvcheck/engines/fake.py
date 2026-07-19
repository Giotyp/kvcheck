"""A GPU-free engine that returns pre-scripted completions.

FakeEngine lets us test the whole pipeline — schedule, runner, metrics,
report — without loading a model. To simulate cache-induced divergence, build
two FakeEngines with different scripts for the same request IDs: one plays the
"golden" role, the other the "test" role.
"""

from kvcheck.config import SamplingConfig
from kvcheck.engines.base import EngineAdapter
from kvcheck.types import Completion, GenerationRequest


class FakeEngine(EngineAdapter):
    def __init__(self, scripts: dict[str, Completion], version_tag: str = "fake-0"):
        self._scripts = scripts
        self._version = version_tag
        self._started = False
        self.calls: list[list[str]] = []  # request-id order of each generate() call

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def generate(
        self, requests: list[GenerationRequest], sampling: SamplingConfig
    ) -> list[Completion]:
        if not self._started:
            raise RuntimeError("FakeEngine.generate() called before start()")
        self.calls.append([r.request_id for r in requests])
        return [self._scripts[r.request_id] for r in requests]

    def version(self) -> str:
        return self._version
