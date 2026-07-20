"""Adapter suite that borrows lm-evaluation-harness task definitions.

lm-eval ships hundreds of benchmark tasks. Rather than hand-write a PromptSuite
per benchmark (as suites/gsm8k.py does), this pulls a task's prompt construction
(`doc_to_text`) and grading (`process_results`) from lm-eval and routes the
actual generation through kvcheck's golden/test engines.

Scope: this targets `generate_until`-style (generative) tasks — GSM8K, GPQA/MMLU
generative variants, DROP, etc. — because kvcheck compares *generated token
sequences*. Multiple-choice tasks scored by loglikelihood ranking have no
generated output to diverge and are out of scope for this adapter.

The `lm_eval` import is deferred to from_task_name(); the core logic depends only
on a small task surface (doc_to_text / process_results) so it is unit-tested with
a fake task, no lm-eval install required.
"""

from kvcheck.suites.base import PromptSuite
from kvcheck.types import Completion, GenerationRequest

# When process_results returns several metrics, prefer these accuracy-like keys.
_METRIC_PREFERENCE = ("exact_match", "acc", "acc_norm", "score", "f1")


def pick_metric(metrics: dict, metric_key: str | None) -> float:
    """Reduce an lm-eval process_results dict to a single [0, 1]-ish score."""
    if metric_key is not None:
        return float(metrics[metric_key])
    for key in _METRIC_PREFERENCE:
        if key in metrics:
            return float(metrics[key])
    return float(next(iter(metrics.values())))  # fall back to first reported metric


class LMEvalSuite(PromptSuite):
    def __init__(
        self,
        task,
        docs: list,
        prefix: str = "",
        warmup: bool = True,
        metric_key: str | None = None,
    ):
        self.task = task
        self.docs = docs
        self.prefix = prefix  # fixed few-shot preamble, shared across docs
        self.warmup = warmup
        self.metric_key = metric_key
        self._doc_by_id: dict[str, object] = {}

    def _prompt(self, doc) -> str:
        return f"{self.prefix}{self.task.doc_to_text(doc)}"

    def requests(self) -> list[GenerationRequest]:
        self._doc_by_id = {}
        reqs: list[GenerationRequest] = []
        if self.warmup and self.docs:
            reqs.append(
                GenerationRequest(
                    request_id="lmeval/warmup",
                    prompt=self._prompt(self.docs[0]),
                    group_id="lmeval",
                    scored=False,
                )
            )
        for i, doc in enumerate(self.docs):
            rid = f"lmeval/q{i}"
            self._doc_by_id[rid] = doc
            reqs.append(
                GenerationRequest(
                    request_id=rid, prompt=self._prompt(doc), group_id="lmeval", scored=True
                )
            )
        return reqs

    def grade(self, request_id: str, completion: Completion) -> float | None:
        doc = self._doc_by_id.get(request_id)
        if doc is None:
            return None
        metrics = self.task.process_results(doc, [completion.text])
        return pick_metric(metrics, self.metric_key)

    @classmethod
    def from_task_name(
        cls,
        task: str,
        num_fewshot: int = 0,
        limit: int | None = None,
        warmup: bool = True,
        metric_key: str | None = None,
    ) -> "LMEvalSuite":
        """Construct from a live lm-eval task (requires `lm_eval` installed).

        The few-shot preamble is built from the task's own examples using
        doc_to_text/doc_to_target — the only surface that is stable across
        lm-eval versions — so all eval prompts share one prefix (which also
        exercises the KV cache).
        """
        from lm_eval.tasks import get_task_dict

        t = get_task_dict([task])[task]

        eval_docs = list(t.test_docs() if t.has_test_docs() else t.validation_docs())
        if limit is not None:
            eval_docs = eval_docs[:limit]

        prefix = ""
        if num_fewshot > 0:
            shot_docs = list(t.training_docs() if t.has_training_docs() else eval_docs)
            shots = shot_docs[:num_fewshot]
            prefix = "".join(
                f"{t.doc_to_text(d)}{t.doc_to_target(d)}\n\n" for d in shots
            )

        return cls(
            task=t, docs=eval_docs, prefix=prefix, warmup=warmup, metric_key=metric_key
        )
