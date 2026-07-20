# kvcheck

Detect output-quality divergence caused by **KV-cache reuse and approximation**
in LLM inference engines (vLLM, SGLang, LMCache).

Engines reuse and approximate the KV cache for speed — prefix caching,
CacheBlend-style approximate reuse, fp8/int4 KV. These optimizations can
silently change a model's output, sometimes degrading task accuracy (see e.g.
vLLM #18055, #43559, #40896; SGLang #23020). kvcheck measures that change and
turns it into a CI-friendly pass/fail.

## How it works

kvcheck compares two engine configs on the same prompt battery:

- **golden** — the trusted baseline (caching off, full recompute)
- **test** — the KV-cache behavior under test (prefix caching on, fp8 KV, …)

It runs three passes and reports the *test* config's divergence **relative to a
calibrated noise floor**:

```
reference   = golden config, pass 1   (the baseline)
calibration = golden config, pass 2   → noise floor = divergence(reference, calibration)
test        = test config,   pass 1   → signal      = divergence(reference, test)
```

The noise floor matters because temperature-0 is *not* actually deterministic in
these engines (batch composition, FP reduction order). kvcheck fails a run only
when the test config diverges **beyond** what golden already diverges from
itself.

## Quickstart

```bash
pip install "kvcheck[vllm]"

# golden = prefix caching off, test = prefix caching on
CUDA_VISIBLE_DEVICES=0 kvcheck run examples/prefix_cache.yaml --json report.json
echo $?     # 0 = pass, 1 = fail (CI-friendly)
```

**Cross-commit regression tracking** — compare a new report against a committed
baseline; exit code 1 if any metric regressed beyond `--tol`:

```bash
kvcheck report report.json --compare baseline_report.json --tol 0.05
```

`kvcheck report report.json` (no `--compare`) just re-renders a saved report.

Example output (GSM8K, prefix cache on vs off, Qwen2.5-Math-1.5B):

```
              kvcheck divergence report
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ metric           ┃ test vs golden ┃    noise floor ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ divergence rate  │          0.200 │          0.000 │
│ mean top-k KL    │         0.0046 │         0.0009 │
│ argmax flip rate │          0.002 │              - │
│ task accuracy    │          0.580 │ 0.600 (golden) │
│ accuracy drop    │         +0.020 │              - │
│ scored requests  │             50 │              - │
└──────────────────┴────────────────┴────────────────┘
FAIL
```

## Metrics

| metric | meaning |
|---|---|
| **divergence rate** | fraction of requests whose test token sequence differs from golden |
| **first divergence index** | per request, the first position where tokens diverge (comparison is only valid up to here — after it the contexts differ) |
| **argmax flip rate** | over position-aligned comparable tokens, how often the top-1 token differs |
| **mean top-k KL** | position-weighted KL between golden and test next-token distributions |
| **task accuracy / drop** | for graded suites (GSM8K), golden vs test accuracy and the delta |

## Config

A run is one YAML file (`examples/prefix_cache.yaml`, `examples/gsm8k.yaml`):

```yaml
model: Qwen/Qwen2.5-Math-1.5B-Instruct
sampling: { temperature: 0.0, seed: 1234, max_tokens: 256, top_logprobs: 5 }
golden:   { adapter: vllm, enable_prefix_caching: false }
test:     { adapter: vllm, enable_prefix_caching: true }
suite:    { name: gsm8k, params: { num_questions: 50, num_fewshot: 4 } }
thresholds: { max_divergence_rate_above_floor: 0.05, max_accuracy_drop: 0.02 }
```

**Engine adapters** (`adapter:`)

| adapter | how it drives the engine | notes |
|---|---|---|
| `vllm` | in-process `vllm.LLM` | full logprobs, integer token ids |
| `openai_server` | launches/attaches to `vllm serve` over HTTP | engine-agnostic (vLLM/SGLang/LMCache); attach to a running server with `extra.base_url` |
| `fake` | scripted completions | tests only, no GPU |

**Suites** (`suite.name`)

| suite | what it does |
|---|---|
| `synthetic` | generated shared-prefix groups — controlled conditions that force cache reuse |
| `gsm8k` | GSM8K subset; few-shot examples form the shared prefix, so it grades accuracy *and* exercises the cache |
| `lm_eval` | any lm-evaluation-harness **generative** task (`generate_until`) — borrows the task's prompt + grading; `params: {task, num_fewshot, limit}`. Needs `pip install "kvcheck[lm_eval]"` |

Multiple-choice lm-eval tasks (scored by loglikelihood ranking) are out of
scope for the `lm_eval` suite — kvcheck compares generated token sequences, so it
targets generative tasks.

Golden and test may use **different model checkpoints** via a per-side
`model:` override on either engine (e.g. golden = original model, test = a
quantized checkpoint). When only the test side changes, the cached golden is
reused.

The golden run is cached on disk under a key derived from
`(model, sampling, golden config, suite, engine version)`, so iterating on the
test config skips the golden re-run.

## Recipe: does fp8-KV calibration recover accuracy?

kvcheck found that naive fp8 KV cache (and vLLM's runtime `calculate_kv_scales`)
collapse GSM8K accuracy to ~0 on Qwen2.5-Math-1.5B. To test whether *proper*
offline calibration recovers it, produce a calibrated checkpoint and compare:

```bash
pip install llmcompressor
python scripts/calibrate_fp8_kv.py \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --output ./checkpoints/qwen-math-1.5b-fp8kv

# golden = original model (auto KV, reused from cache); test = calibrated fp8 checkpoint
kvcheck run examples/fp8_kv_offline_calibrated.yaml --json fp8off_report.json
```

`scripts/calibrate_fp8_kv.py` uses llm-compressor to embed static per-layer
`k_scale`/`v_scale` into the checkpoint (the static equivalent of what
`calculate_kv_scales` only approximates at runtime).

## Development

```bash
uv sync --group dev
uv run pytest          # GPU-free: metrics, runner, suites, adapters (FakeEngine + mock HTTP)
uv run ruff check kvcheck/
```

The architecture keeps engines, suites, and metrics behind small interfaces
(`EngineAdapter`, `PromptSuite`), so the whole pipeline is testable without a
GPU via `FakeEngine` and a mock OpenAI server.
