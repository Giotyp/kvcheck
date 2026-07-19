"""Core runtime data types that flow between suites, engines, and metrics."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GenerationRequest:
    """A single prompt to run, positioned inside a suite's request schedule.

    The suite controls request *order* — that order is what exercises (or
    doesn't exercise) the KV cache. Warm-up requests carry scored=False:
    they are executed to populate the cache but excluded from comparison.
    """

    request_id: str
    prompt: str
    group_id: str | None = None  # shared-prefix group this request belongs to
    scored: bool = True


@dataclass(frozen=True)
class Completion:
    """One engine response.

    top_logprobs holds, for each generated position, the engine's top-k
    candidate token IDs mapped to their logprobs. Keyed by token ID (not
    string): golden and test configs share a tokenizer, so IDs align
    exactly across runs; decoded strings can collide.
    """

    request_id: str
    text: str
    token_ids: tuple[int, ...]
    top_logprobs: tuple[dict[int, float], ...] = field(default=())
    finish_reason: str = "stop"
