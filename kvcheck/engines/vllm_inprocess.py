"""In-process vLLM adapter.

Drives vLLM through its Python `LLM` API in the harness process. Construction is
cheap (stores config only); start() is what loads the model onto the GPU, so the
runner can skip it entirely on a golden cache hit. All vLLM imports are lazy so
kvcheck imports fine on machines without vLLM installed.

GPU selection is via the CUDA_VISIBLE_DEVICES environment variable, set before
launching kvcheck (e.g. `CUDA_VISIBLE_DEVICES=1 kvcheck run ...`).
"""

from kvcheck.config import EngineConfig, SamplingConfig
from kvcheck.engines.base import EngineAdapter
from kvcheck.types import Completion, GenerationRequest


class VLLMInProcess(EngineAdapter):
    def __init__(self, model: str, engine: EngineConfig, sampling: SamplingConfig):
        self.model = model
        self.engine = engine
        self._sampling = sampling
        self._llm = None  # created in start()

    def version(self) -> str:
        import vllm

        return f"vllm-{vllm.__version__}"

    def start(self) -> None:
        from vllm import LLM

        self._llm = LLM(
            model=self.model,
            enable_prefix_caching=self.engine.enable_prefix_caching,
            kv_cache_dtype=self.engine.kv_cache_dtype,
            seed=self._sampling.seed,
            **self.engine.extra,
        )

    def stop(self) -> None:
        # Release the engine and free GPU memory *synchronously*, so a second
        # engine (the test side) can be constructed on the same GPU right after
        # the golden side runs. vLLM v1 runs EngineCore in a child process:
        # dropping the Python handle and calling torch.cuda.empty_cache() only
        # touch the parent's CUDA context and do NOT free the child's memory,
        # which made the test engine fail with "No available memory for the
        # cache blocks". Shut the engine core down explicitly (terminates the
        # child), then tear down any distributed state and empty the allocator.
        llm = self._llm
        self._llm = None
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception:
                pass
        del llm
        import gc

        gc.collect()
        try:
            import torch
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(
        self, requests: list[GenerationRequest], sampling: SamplingConfig
    ) -> list[Completion]:
        if self._llm is None:
            raise RuntimeError("VLLMInProcess.generate() called before start()")
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=sampling.temperature,
            seed=sampling.seed,
            max_tokens=sampling.max_tokens,
            logprobs=sampling.top_logprobs,
            stop=sampling.stop,
        )
        prompts = [r.prompt for r in requests]
        # vLLM returns outputs in input order; we still re-key by index to be explicit.
        outputs = self._llm.generate(prompts, params)
        return [self._to_completion(requests[i], outputs[i]) for i in range(len(requests))]

    @staticmethod
    def _to_completion(request: GenerationRequest, output) -> Completion:
        out = output.outputs[0]
        token_ids = tuple(out.token_ids)
        top_logprobs: list[dict[int, float]] = []
        for pos in out.logprobs or ():
            # pos maps token_id -> Logprob(logprob=..., ...)
            top_logprobs.append({tid: lp.logprob for tid, lp in pos.items()})
        return Completion(
            request_id=request.request_id,
            text=out.text,
            token_ids=token_ids,
            top_logprobs=tuple(top_logprobs),
            finish_reason=out.finish_reason or "stop",
        )
