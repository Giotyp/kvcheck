"""Golden and test may point at different model checkpoints.

Needed for the calibrated-fp8 experiment: golden runs the original model, test
runs a separately-quantized checkpoint. EngineConfig.model overrides the
top-level RunConfig.model for that side only.
"""

from kvcheck.cli import resolve_model
from kvcheck.config import EngineConfig, RunConfig, SamplingConfig, SuiteConfig


def cfg(golden_model=None, test_model=None) -> RunConfig:
    return RunConfig(
        model="base/model",
        sampling=SamplingConfig(),
        golden=EngineConfig(adapter="vllm", model=golden_model),
        test=EngineConfig(adapter="vllm", model=test_model),
        suite=SuiteConfig(name="synthetic"),
    )


def test_defaults_to_top_level_model():
    c = cfg()
    assert resolve_model(c, c.golden) == "base/model"
    assert resolve_model(c, c.test) == "base/model"


def test_per_side_override_wins():
    c = cfg(test_model="calibrated/checkpoint")
    assert resolve_model(c, c.golden) == "base/model"  # golden unchanged
    assert resolve_model(c, c.test) == "calibrated/checkpoint"


def test_golden_key_ignores_test_side_model():
    """Swapping only the test checkpoint must not invalidate the cached golden."""
    base = cfg()
    swapped = cfg(test_model="calibrated/checkpoint")
    assert base.golden_key() == swapped.golden_key()


def test_golden_key_changes_with_golden_side_model():
    base = cfg()
    swapped = cfg(golden_model="other/checkpoint")
    assert base.golden_key() != swapped.golden_key()
