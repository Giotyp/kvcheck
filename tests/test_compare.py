from kvcheck.compare import MetricDelta, build_deltas, compare_reports, is_regression


def summary(**kw) -> dict:
    base = dict(
        n_scored=50,
        divergence_rate=0.2,
        argmax_flip_rate=0.01,
        mean_kl=0.005,
        floor_divergence_rate=0.0,
        floor_mean_kl=0.001,
        golden_accuracy=0.6,
        test_accuracy=0.58,
        accuracy_drop=0.02,
    )
    base.update(kw)
    return base


def report(**kw) -> dict:
    return {"passed": kw.pop("passed", True), "summary": summary(**kw)}


def delta_named(deltas, name) -> MetricDelta:
    return next(d for d in deltas if d.name == name)


# ---- build_deltas (implemented) ----------------------------------------


def test_build_deltas_computes_excess_divergence():
    d = build_deltas(summary(divergence_rate=0.2, floor_divergence_rate=0.05),
                     summary(divergence_rate=0.3, floor_divergence_rate=0.05))
    ex = delta_named(d, "divergence_excess")
    assert ex.baseline == pytest_approx(0.15)
    assert ex.current == pytest_approx(0.25)
    assert ex.delta == pytest_approx(0.10)
    assert ex.higher_is_worse is True


def test_build_deltas_marks_accuracy_as_higher_is_better():
    d = build_deltas(summary(test_accuracy=0.60), summary(test_accuracy=0.55))
    acc = delta_named(d, "test_accuracy")
    assert acc.delta == pytest_approx(-0.05)
    assert acc.higher_is_worse is False


def test_build_deltas_handles_missing_accuracy():
    d = build_deltas(summary(accuracy_drop=None, test_accuracy=None),
                     summary(accuracy_drop=None, test_accuracy=None))
    acc = delta_named(d, "accuracy_drop")
    assert acc.delta is None  # no signal -> no delta


# ---- is_regression (TODO(human)) ---------------------------------------


def test_no_change_is_not_a_regression():
    d = build_deltas(summary(), summary())
    assert is_regression(d, tol=0.01) is False


def test_worse_divergence_beyond_tol_is_a_regression():
    d = build_deltas(summary(divergence_rate=0.2), summary(divergence_rate=0.4))
    assert is_regression(d, tol=0.05) is True


def test_improvement_is_not_a_regression():
    # divergence dropped and accuracy rose -> better, not a regression
    d = build_deltas(summary(divergence_rate=0.4, test_accuracy=0.5),
                     summary(divergence_rate=0.2, test_accuracy=0.6))
    assert is_regression(d, tol=0.05) is False


def test_accuracy_drop_beyond_tol_is_a_regression():
    # divergence unchanged, but test accuracy fell 10 points
    d = build_deltas(summary(test_accuracy=0.60, accuracy_drop=0.02),
                     summary(test_accuracy=0.50, accuracy_drop=0.12))
    assert is_regression(d, tol=0.05) is True


def test_change_within_tol_is_not_a_regression():
    d = build_deltas(summary(divergence_rate=0.20), summary(divergence_rate=0.23))
    assert is_regression(d, tol=0.05) is False  # 0.03 <= 0.05


def test_missing_accuracy_never_regresses_on_that_axis():
    d = build_deltas(summary(accuracy_drop=None, test_accuracy=None),
                     summary(accuracy_drop=None, test_accuracy=None))
    assert is_regression(d, tol=0.01) is False


# ---- compare_reports (implemented) -------------------------------------


def test_compare_reports_wraps_deltas_and_verdict():
    result = compare_reports(report(divergence_rate=0.2), report(divergence_rate=0.5), tol=0.05)
    assert result.regressed is True
    assert any(d.name == "divergence_excess" for d in result.deltas)


# tiny local approx to avoid importing pytest at module top for helpers
def pytest_approx(x, eps=1e-9):
    class _A:
        def __eq__(self, other):
            return abs(other - x) < eps

    return _A()
