"""The prompt-suite contract.

A suite owns two things: *what* prompts to run and — crucially — *what order*
to run them in. Order is what exercises the KV cache: requests that share a
long prefix must sit close enough together that the test engine actually reuses
the cached prefix instead of evicting it. A suite may also grade completions
for task-accuracy metrics; ungraded suites (like the synthetic one) return None.
"""

from abc import ABC, abstractmethod

from kvcheck.types import Completion, GenerationRequest


class PromptSuite(ABC):
    @abstractmethod
    def requests(self) -> list[GenerationRequest]:
        """The ordered request schedule. The runner replays this exact order
        (same list, same order) to golden, calibration, and test engines."""

    def grade(self, request_id: str, completion: Completion) -> float | None:
        """Task-accuracy score in [0, 1] for a graded suite, else None."""
        return None
