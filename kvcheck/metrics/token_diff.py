"""Token-level divergence between a golden and a test completion.

The subtlety this module encodes: decoding is sequential, so a position's
output depends on every token before it. The moment golden and test emit a
different token, their contexts diverge and no later position is comparable.
We therefore measure distribution-level divergence (argmax flips, KL) only over
positions whose entire preceding context is identical in both runs.
"""

import math

from kvcheck.types import Completion, DivergenceRecord

# A token in one run's top-k may be absent from the other's. We know only that
# its logprob is below the smallest reported one; floor it well below any real
# top-k logprob so its contribution to KL is large but finite.
_KL_FLOOR_LOGPROB = -30.0


def argmax(logprobs: dict[int, float]) -> int | None:
    """Token id with the highest logprob at a position, or None if empty."""
    if not logprobs:
        return None
    return max(logprobs, key=logprobs.__getitem__)


def topk_kl(p_logprobs: dict[int, float], q_logprobs: dict[int, float]) -> float:
    """Truncated KL(P || Q) over P's top-k support, in nats.

    P and Q are {token_id: logprob} at one position (golden and test). Engine
    logprobs are already normalized over the full vocabulary, so we sum
    p * (log p - log q) over the tokens P reports. A token missing from Q's
    top-k gets a floored logprob. Identical distributions give 0.0; an empty P
    gives 0.0.
    """
    kl = 0.0
    for token_id, p_lp in p_logprobs.items():
        p = math.exp(p_lp)
        q_lp = q_logprobs.get(token_id, _KL_FLOOR_LOGPROB)
        kl += p * (p_lp - q_lp)
    return kl


def first_divergence_index(
    golden_ids: tuple[int, ...], test_ids: tuple[int, ...]
) -> int:
    """First position where the token sequences differ.

    If one sequence is a prefix of the other (or they are equal), returns the
    length of the shorter — there is no *differing* position within the overlap.
    """
    for i, (g, t) in enumerate(zip(golden_ids, test_ids, strict=False)):
        if g != t:
            return i
    return min(len(golden_ids), len(test_ids))


def token_divergence(golden: Completion, test: Completion) -> DivergenceRecord:
    """Compare one test completion against its golden completion.

    Build and return a DivergenceRecord. The token-level fields must be
    computed only over *comparable* positions: those whose entire preceding
    context is identical in both runs.
    """
    assert golden.request_id == test.request_id
    request_id = golden.request_id
    exact_match = golden.token_ids == test.token_ids

    fdi = first_divergence_index(golden.token_ids, test.token_ids)
    n_comparable = min(fdi + 1, len(golden.top_logprobs), len(test.top_logprobs))

    argmax_flips = 0
    kl = []

    for i in range(n_comparable):
        if argmax(golden.top_logprobs[i]) != argmax(test.top_logprobs[i]):
            argmax_flips += 1

        kl.append(topk_kl(golden.top_logprobs[i], test.top_logprobs[i]))

    mean_kl = sum(kl) / len(kl) if len(kl) > 0 else 0.0

    golden_len = len(golden.token_ids)
    test_len = len(test.token_ids)

    return DivergenceRecord(
        request_id,
        exact_match,
        fdi,
        n_comparable,
        argmax_flips,
        mean_kl,
        golden_len,
        test_len,
    )
