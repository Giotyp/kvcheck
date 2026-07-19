"""The engine adapter contract.

Every backend — in-process vLLM, an OpenAI-compatible server, or the test
FakeEngine — hides behind this one interface so the runner never depends on a
specific engine. An adapter instance *is* one engine configuration: the golden
and test configs are two separate adapter instances, constructed differently
(e.g. prefix caching off vs. on).
"""

from abc import ABC, abstractmethod

from kvcheck.config import SamplingConfig
from kvcheck.types import Completion, GenerationRequest


class EngineAdapter(ABC):
    @abstractmethod
    def start(self) -> None:
        """Load the model / spin up the server. Called once before generate()."""

    @abstractmethod
    def stop(self) -> None:
        """Release GPU memory / tear down the server."""

    @abstractmethod
    def generate(
        self, requests: list[GenerationRequest], sampling: SamplingConfig
    ) -> list[Completion]:
        """Run requests **in the given order** and return completions in that
        same order. Order is load-bearing: it is what exercises the KV cache."""

    @abstractmethod
    def version(self) -> str:
        """Stable identifier for this backend build, mixed into the golden
        cache key so results from a different engine version aren't reused."""

    def __enter__(self) -> "EngineAdapter":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
