#!/usr/bin/env python
"""Sweep a *mixed-precision* fp8 KV cache: keep chosen layers full-precision,
quantize the rest to fp8.

vLLM's `kv_cache_dtype_skip_layers` forces the listed layer indices back to
`kv_cache_dtype="auto"` (full-precision KV) while every other layer stays fp8.
That makes it a zero-calibration probe of *where* fp8 KV does its damage: if the
accuracy collapse is driven by a few outlier layers (the massive-activation /
attention-sink layers 0/1/7 identified earlier), keeping just those at full
precision should recover most of the 60% GSM8K accuracy while still fp8-quantizing
the other ~25 layers.

Each skip-set runs in its own subprocess (clean GPU teardown by process exit) and
reuses the cached golden (this shares examples/gsm8k.yaml's golden_key), so only
the ~30 s test engine loads per run. Results print as one comparison table.

    CUDA_VISIBLE_DEVICES=x python scripts/sweep_kv_skip_layers.py
"""

import json
import os
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
BASE_CONFIG = REPO / "examples" / "fp8_kv.yaml"  # test side is kv_cache_dtype: fp8
PIE_PYTHON = Path.home() / ".pie" / "venvs" / "vllm" / "bin" / "python"
SCRATCH = Path(
    os.environ.get(
        "SWEEP_SCRATCH",
        "/tmp/claude-1005/-home-george-Tools-kvcheck/"
        "7a2cac2c-74b7-42c2-b676-57184ccc7230/scratchpad",
    )
)
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "1")


# Each entry is (label, [layer-index strings kept at FULL precision]); every
# other layer is fp8. An empty list [] means "skip nothing" = the pure-fp8
# baseline (expected ~0% accuracy).
# The massive-activation / attention-sink layers found earlier are 0, 1, 7.
SKIP_SETS: list[tuple[str, list[str]]] = [
    # Minimal-set decomposition: which single layer(s) carry the recovery?
    ("skip-0-1", ["0", "1"]),
    ("skip-0", ["0"]),
    ("skip-1", ["1"]),
    ("skip-7", ["7"]),
]


def run_one(label: str, skip_layers: list[str]) -> dict | None:
    """Run one skip-set via the CLI in a fresh subprocess; return its summary."""
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cfg["test"].setdefault("extra", {})
    cfg["test"]["extra"]["kv_cache_dtype_skip_layers"] = list(skip_layers)

    cfg_path = SCRATCH / f"kvskip_{label}.yaml"
    report_path = SCRATCH / f"kvskip_{label}.json"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": GPU,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": str(REPO),
    }
    print(
        f"\n=== [{label}] full-precision layers: "
        f"{skip_layers or 'none (pure fp8)'} ===",
        flush=True,
    )
    proc = subprocess.run(
        [
            str(PIE_PYTHON),
            "-m",
            "kvcheck.cli",
            "run",
            str(cfg_path),
            "--json",
            str(report_path),
        ],
        env=env,
    )
    if not report_path.exists():
        print(f"[{label}] FAILED — no report written (exit {proc.returncode})")
        return None
    summary = json.loads(report_path.read_text())["summary"]
    return {"label": label, "skip": skip_layers, **summary}


def print_table(rows: list[dict]) -> None:
    golden = next(
        (r["golden_accuracy"] for r in rows if r.get("golden_accuracy") is not None),
        None,
    )
    print("\n" + "=" * 72)
    print(
        f"KV skip-layer sweep — golden accuracy: " f"{golden:.3f}"
        if golden is not None
        else "KV skip-layer sweep"
    )
    print("=" * 72)
    print(
        f"{'skip-set':<20} {'test_acc':>9} {'acc_drop':>9} {'flip':>7} {'mean_kl':>9}"
    )
    print("-" * 72)
    for r in rows:
        acc = r.get("test_accuracy")
        drop = r.get("accuracy_drop")
        print(
            f"{r['label']:<20} "
            f"{acc if acc is None else f'{acc:9.3f}':>9} "
            f"{drop if drop is None else f'{drop:+9.3f}':>9} "
            f"{r['argmax_flip_rate']:7.3f} "
            f"{r['mean_kl']:9.4f}"
        )
    print("=" * 72)


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rows = [r for st in SKIP_SETS if (r := run_one(*st)) is not None]
    if rows:
        print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
