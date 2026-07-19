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


@dataclass(frozen=True)
class DivergenceRecord:
    """Per-request comparison of a test completion against its golden completion.

    Token-level fields only cover *comparable* positions — those whose entire
    preceding context is identical in both runs. Once the emitted tokens differ
    at `first_divergence_index`, every later position has a different context,
    so comparing distributions there is meaningless.
    """

    request_id: str
    exact_match: bool
    first_divergence_index: int  # first position where token_ids differ; == common length if none
    n_comparable: int  # positions with identical preceding context
    argmax_flips: int  # comparable positions whose top-1 token differs
    mean_kl: float  # mean top-k KL over comparable positions (0.0 if none)
    golden_len: int
    test_len: int
