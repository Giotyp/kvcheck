# kvcheck

Detect output-quality divergence caused by KV-cache reuse and approximation in LLM
inference engines (vLLM, SGLang, LMCache).

kvcheck runs a prompt battery through a **golden** engine config (full recompute,
caching off) and a **test** config (prefix caching on, approximate KV reuse,
quantized KV cache), then reports divergence:

- exact-match rate and first-divergence token index
- prefix-aligned argmax-flip rate and top-k KL
- task-accuracy delta on plug-in benchmarks

Every run calibrates a **noise floor** (golden vs. golden) so cache-induced
divergence is distinguished from baseline temperature-0 nondeterminism.

## Usage

```bash
pip install kvcheck[vllm]
kvcheck run examples/prefix_cache.yaml   # exit code 0/1 against thresholds
```

## Development

```bash
uv sync --group dev
uv run pytest
```
