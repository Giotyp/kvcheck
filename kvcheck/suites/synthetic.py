"""Synthetic shared-prefix suite: controlled conditions for triggering cache reuse.

Each *group* is a long shared document prefix plus several short questions that
all reuse that prefix. When the group's requests run back-to-back, a prefix-
caching engine computes the long prefix once and reuses its KV for the rest —
which is exactly the code path we want to stress. The content here is
deterministic filler; the interesting part (the request *schedule*) is
requests().
"""

import random
from dataclasses import dataclass

from kvcheck.suites.base import PromptSuite
from kvcheck.types import GenerationRequest

_WORDS = (
    "system latency cache token vector kernel batch tensor memory prefix "
    "attention decode prompt weight logit sample entropy context window buffer"
).split()


@dataclass(frozen=True)
class _Group:
    group_id: str
    prefix: str  # the long shared document
    questions: list[str]  # short suffixes, each reuses `prefix`


class SyntheticSuite(PromptSuite):
    def __init__(
        self,
        num_groups: int = 4,
        questions_per_group: int = 4,
        prefix_words: int = 200,
        seed: int = 1234,
        warmup: bool = True,
    ):
        self.num_groups = num_groups
        self.questions_per_group = questions_per_group
        self.prefix_words = prefix_words
        self.warmup = warmup
        self._rng = random.Random(seed)
        self.groups: list[_Group] = self._build_groups()

    def _build_groups(self) -> list[_Group]:
        groups = []
        for g in range(self.num_groups):
            body = " ".join(self._rng.choice(_WORDS) for _ in range(self.prefix_words))
            prefix = f"Document {g}: {body}\n"
            questions = [
                f"Q{q}: summarize aspect {self._rng.randint(0, 999)} in one word."
                for q in range(self.questions_per_group)
            ]
            groups.append(_Group(group_id=f"g{g}", prefix=prefix, questions=questions))
        return groups

    def group_prefix(self, group_id: str) -> str:
        return next(grp.prefix for grp in self.groups if grp.group_id == group_id)

    def requests(self) -> list[GenerationRequest]:
        """Flatten self.groups into the ordered request schedule.

        Return a list of GenerationRequest. Every request's prompt is a group's
        prefix followed by one question. The ordering and the scored/warmup
        flags are the design decision — see the failing tests for the invariants
        the schedule must satisfy.
        """
        request_sched = []
        for group in self.groups:
            if self.warmup:
                request_sched.append(
                    GenerationRequest(
                        f"{group.group_id}/warmup",
                        group.prefix + group.questions[0],
                        group.group_id,
                        scored=False,
                    )
                )
            for i, question in enumerate(group.questions):
                request_sched.append(
                    GenerationRequest(
                        f"{group.group_id}/{question[i]}",
                        group.prefix + group.questions[0],
                        group.group_id,
                        scored=True,
                    )
                )
        return request_sched
