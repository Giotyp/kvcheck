"""GSM8K accuracy suite with a shared few-shot prefix.

The few-shot examples are prepended identically to every question, so the whole
suite is effectively one shared-prefix group: a prefix-caching engine computes
the (long) few-shot block once and reuses it. This gives us a real task-accuracy
signal *and* exercises the cache at the same time.

The dataset is loaded lazily via `datasets` (an optional dependency); tests
inject an in-memory `examples` list instead.
"""

import re

from kvcheck.suites.base import PromptSuite
from kvcheck.types import Completion, GenerationRequest

_NUMBER = re.compile(r"-?\d[\d,]*")


def extract_gold(answer: str) -> str:
    """GSM8K gold answers end with '#### <number>'."""
    return answer.split("####")[-1].strip().replace(",", "")


def extract_final_number(text: str) -> str | None:
    """The model's predicted answer: the last number it emits, comma-stripped."""
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def _load_examples(split: str, n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [ds[i] for i in range(min(n, len(ds)))]


class GSM8KSuite(PromptSuite):
    def __init__(
        self,
        num_questions: int = 100,
        num_fewshot: int = 4,
        split: str = "test",
        warmup: bool = True,
        examples: list[dict] | None = None,
    ):
        self.num_questions = num_questions
        self.num_fewshot = num_fewshot
        self.warmup = warmup
        rows = examples if examples is not None else _load_examples(
            split, num_fewshot + num_questions
        )
        shots = rows[:num_fewshot]
        self._questions = rows[num_fewshot : num_fewshot + num_questions]
        self.fewshot_prefix = self._build_prefix(shots)
        # gold answers keyed by the same request ids requests() emits, so grade()
        # works regardless of call order.
        self._gold: dict[str, str] = {
            f"gsm8k/q{i}": extract_gold(ex["answer"]) for i, ex in enumerate(self._questions)
        }

    @staticmethod
    def _build_prefix(shots: list[dict]) -> str:
        blocks = [f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots]
        return "".join(blocks)

    def requests(self) -> list[GenerationRequest]:
        reqs: list[GenerationRequest] = []
        if self.warmup and self._questions:
            first = self._questions[0]
            reqs.append(
                GenerationRequest(
                    request_id="gsm8k/warmup",
                    prompt=self._prompt(first),
                    group_id="gsm8k",
                    scored=False,
                )
            )
        for i, ex in enumerate(self._questions):
            reqs.append(
                GenerationRequest(
                    request_id=f"gsm8k/q{i}",
                    prompt=self._prompt(ex),
                    group_id="gsm8k",
                    scored=True,
                )
            )
        return reqs

    def _prompt(self, example: dict) -> str:
        return f"{self.fewshot_prefix}Question: {example['question']}\nAnswer:"

    def grade(self, request_id: str, completion: Completion) -> float | None:
        gold = self._gold.get(request_id)
        if gold is None:
            return None
        return 1.0 if extract_final_number(completion.text) == gold else 0.0
