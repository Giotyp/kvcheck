import textwrap

from kvcheck.config import EngineConfig, RunConfig, SamplingConfig, SuiteConfig


def make_config(**overrides) -> RunConfig:
    base = dict(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        golden=EngineConfig(adapter="vllm", enable_prefix_caching=False),
        test=EngineConfig(adapter="vllm", enable_prefix_caching=True),
        suite=SuiteConfig(name="synthetic", params={"num_groups": 4}),
    )
    base.update(overrides)
    return RunConfig(**base)


def test_run_config_loads_from_yaml(tmp_path):
    yaml_text = textwrap.dedent("""
        model: Qwen/Qwen2.5-1.5B-Instruct
        sampling:
          max_tokens: 128
        golden:
          adapter: vllm
          enable_prefix_caching: false
        test:
          adapter: vllm
          enable_prefix_caching: true
        suite:
          name: synthetic
          params:
            num_groups: 4
    """)
    path = tmp_path / "run.yaml"
    path.write_text(yaml_text)

    cfg = RunConfig.from_yaml(path)

    assert cfg.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.sampling.max_tokens == 128
    assert cfg.golden.enable_prefix_caching is False
    assert cfg.test.enable_prefix_caching is True
    assert cfg.suite.params == {"num_groups": 4}


def test_sampling_defaults_are_deterministic():
    cfg = make_config()
    assert cfg.sampling.temperature == 0.0
    assert cfg.sampling.seed is not None
    assert cfg.sampling.top_logprobs >= 1


def test_golden_key_is_stable():
    assert make_config().golden_key() == make_config().golden_key()


def test_golden_key_ignores_test_engine_and_thresholds():
    """The golden run's identity must not depend on what we compare it against."""
    a = make_config()
    b = make_config(test=EngineConfig(adapter="vllm", kv_cache_dtype="fp8"))
    assert a.golden_key() == b.golden_key()


def test_golden_key_changes_with_golden_inputs():
    base = make_config()
    diff_model = make_config(model="other/model")
    diff_sampling = make_config(sampling=SamplingConfig(max_tokens=64))
    diff_suite = make_config(suite=SuiteConfig(name="synthetic", params={"num_groups": 8}))
    diff_golden = make_config(golden=EngineConfig(adapter="vllm", kv_cache_dtype="fp8"))

    keys = {
        base.golden_key(),
        diff_model.golden_key(),
        diff_sampling.golden_key(),
        diff_suite.golden_key(),
        diff_golden.golden_key(),
    }
    assert len(keys) == 5, "every golden-relevant field must change the key"
