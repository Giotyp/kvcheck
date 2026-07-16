# Implementation of kvcheck

## Problem/evidence 
This is the best-evidenced gap. vLLM #18055 (accuracy degradation with prefix cache + recomputation, stale/unresolved), #43559 (~20% accuracy drop with prefix caching + MTP speculative decode), #33123 (ROCm cache-hit vs cache-miss output divergence changing argmax), #40896 (temperature=0 nondeterminism with prefix caching), #4670 (user asks how vLLM "ensures correctness when reusing KV cache"). SGLang #23020 (PD-disaggregation KV corruption → gibberish after ~100 requests; the reporter hand-built a GPQA Diamond eval). LMCache roadmap #574 wants an MMLU correctness CI check but only for specific MLA models. vLLM RFC #35639 admits accuracy+perf eval is fragmented. No reusable harness exists — only one-off paper evals. The magnitude is genuinely disputed: ContextPilot (arXiv:2511.03475) says approximate matching "can degrade reasoning quality by 9–11% (dropping from around 60% to approximately 50%)," while CacheBlend (arXiv:2405.16444) reports its "reduction in F1 and Rouge-L score is within 0.02" — a standard tool would settle this.

## What it does / how
A pytest-style harness that, given an engine config, runs a battery of prompts through (a) a golden config (full recompute / FP16 KV / caching off) and (b) the config under test (prefix caching on, CacheBlend approximate reuse, fp8/int4 KV), then reports divergence: exact-match rate, token-level KL/argmax-flip rate, and task accuracy on plug-in benchmarks (GPQA, MMLU, LongBench via lm-eval-harness adapters). Deterministic seeding, temperature=0 correctness checks, and multi-turn "error compounding" tests. Emits a CI-friendly pass/fail against a quality threshold and a regression report across git commits.


## Closest tools & shortfalls
lm-evaluation-harness (general model accuracy, manual before/after, not KV-reuse-aware); LMCache/lmcache-tests (functional/compatibility, not quality); LMBenchmark (throughput). vLLM vllm bench eval is only a proposed RFC.

## Scope
Multi-week for a strong v1 (vLLM + LMCache backends, GPQA/MMLU adapters); weekend for a proof-of-concept diffing two vLLM configs.

## Adoption
pip; vLLM/SGLang/LMCache contributors and inference-platform teams; a natural fit for engine CI. Highest research synergy — directly usable in the user's KV-cache papers as an evaluation artifact.