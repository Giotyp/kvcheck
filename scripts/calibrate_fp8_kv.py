#!/usr/bin/env python
"""Produce an fp8-KV-calibrated checkpoint with embedded per-layer k/v scales.

This is the *proper* fp8 KV calibration that vLLM's runtime `calculate_kv_scales`
flag only approximates: llm-compressor runs the model over a calibration corpus,
measures per-tensor activation ranges, and writes static `k_scale`/`v_scale`
into each attention layer of an output checkpoint. vLLM then loads that
checkpoint with `kv_cache_dtype="fp8"` and reads the embedded scales.

Run this on a machine with network + llm-compressor installed:

    pip install llmcompressor
    python scripts/calibrate_fp8_kv.py \
        --model Qwen/Qwen2.5-Math-1.5B-Instruct \
        --output ./checkpoints/qwen-math-1.5b-fp8kv \
        --calib-samples 512

Then point kvcheck's test config at --output (see
examples/fp8_kv_offline_calibrated.yaml) and run:

    kvcheck run examples/fp8_kv_offline_calibrated.yaml --json fp8off_report.json

The golden side keeps the original model, so the cached golden run is reused.
"""

import argparse
import json
import os


def build_recipe():
    """KV-cache-only fp8 quantization recipe (weights/activations left alone).

    We quantize *only* the KV cache — that is the axis kvcheck is testing.
    Quantizing weights too would confound the measurement.
    """
    from llmcompressor.modifiers.quantization import QuantizationModifier

    return QuantizationModifier(
        kv_cache_scheme={
            "num_bits": 8,
            "type": "float",  # fp8 (e4m3)
            "strategy": "tensor",  # one scale per k / v tensor per layer
            "dynamic": False,  # STATIC scales — the whole point vs calculate_kv_scales
            "symmetric": True,
        }
    )


def load_calibration_dataset(name: str, num_samples: int, tokenizer, max_len: int):
    """A small text corpus to measure activation ranges over.

    GSM8K train is used by default: it is already cached offline and matches the
    eval domain. Calibration only needs representative activations, not labels.
    """
    from datasets import load_dataset

    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds = ds.select(range(min(num_samples, len(ds))))

        def to_text(row):
            return {"text": f"Question: {row['question']}\nAnswer: {row['answer']}"}

        ds = ds.map(to_text)
    else:
        ds = load_dataset(name, split="train")
        ds = ds.select(range(min(num_samples, len(ds))))

    def tokenize(row):
        return tokenizer(
            row["text"], truncation=True, max_length=max_len, padding=False
        )

    return ds.map(tokenize, remove_columns=ds.column_names)


def repair_saved_config(output_dir: str, base_model: str) -> None:
    """Undo config drift from oneshot's save so the checkpoint actually loads.

    llm-compressor re-serializes the config with whatever transformers version
    is installed, which may be much newer than the one the base model was
    authored under. For tied-embedding models (Qwen2.5, Llama-3.2, ...) that
    save can flip `tie_word_embeddings` to False *without* writing a separate
    `lm_head.weight` tensor. On load, the LM head is then randomly initialised
    and the model emits pure garbage (repeated "!" / multilingual token soup) —
    even though every transformer weight is byte-identical to the original. The
    KV scales and all other weights are fine; only this one config field is
    wrong. Restore the base model's tie setting.
    """
    from safetensors import safe_open
    from transformers import AutoConfig

    base_cfg = AutoConfig.from_pretrained(base_model)
    cfg_path = os.path.join(output_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    st = os.path.join(output_dir, "model.safetensors")
    with safe_open(st, "pt") as st_f:
        has_lm_head = "lm_head.weight" in st_f.keys()

    if base_cfg.tie_word_embeddings and not has_lm_head:
        cfg["tie_word_embeddings"] = True

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("[calibrate] repaired config.json (tie_word_embeddings preserved)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="base model id or path")
    p.add_argument("--output", required=True, help="output checkpoint directory")
    p.add_argument(
        "--calib-dataset", default="gsm8k", help="'gsm8k' or an HF dataset id"
    )
    p.add_argument("--calib-samples", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=1024)
    args = p.parse_args()

    # Imports are deferred so `--help` works without the heavy deps installed.
    try:
        from llmcompressor import oneshot
    except ImportError:  # older llm-compressor layout
        from llmcompressor.transformers import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[calibrate] loading {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(
        f"[calibrate] building calibration set: {args.calib_dataset} x{args.calib_samples}"
    )
    calib = load_calibration_dataset(
        args.calib_dataset, args.calib_samples, tokenizer, args.max_seq_len
    )

    print("[calibrate] running oneshot fp8-KV calibration ...")
    oneshot(
        model=model,
        dataset=calib,
        recipe=build_recipe(),
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.calib_samples,
        output_dir=args.output,
    )
    tokenizer.save_pretrained(args.output)
    repair_saved_config(args.output, args.model)
    print(f"[calibrate] done -> {args.output}")
    print("[calibrate] point a kvcheck test config's `model:` at this dir with")
    print("            kv_cache_dtype: fp8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
