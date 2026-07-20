"""Compare two saved report.json files for cross-commit regression tracking.

Each `kvcheck run --json` writes {passed, summary}. Persisting those under
version control (or CI artifacts) lets you ask: did this commit make the
KV-cache quality *worse* than a known-good baseline? build_deltas() turns two
summaries into per-metric deltas; is_regression() decides whether any delta is
bad enough to flag.
"""

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table


@dataclass
class MetricDelta:
    name: str
    baseline: float | None
    current: float | None
    delta: float | None  # current - baseline, or None if either side is missing
    higher_is_worse: bool


@dataclass
class Comparison:
    deltas: list[MetricDelta]
    regressed: bool


def _sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def build_deltas(baseline: dict, current: dict) -> list[MetricDelta]:
    """Per-metric comparison of two run summaries.

    'divergence_excess' is the headline: divergence above the run's own noise
    floor (each run brings its own floor, so raw divergence_rate isn't
    comparable across runs — the excess is).
    """

    def excess(s: dict) -> float:
        return s["divergence_rate"] - s["floor_divergence_rate"]

    specs = [
        ("divergence_excess", excess(baseline), excess(current), True),
        ("mean_kl", baseline["mean_kl"], current["mean_kl"], True),
        (
            "argmax_flip_rate",
            baseline["argmax_flip_rate"],
            current["argmax_flip_rate"],
            True,
        ),
        (
            "accuracy_drop",
            baseline.get("accuracy_drop"),
            current.get("accuracy_drop"),
            True,
        ),
        (
            "test_accuracy",
            baseline.get("test_accuracy"),
            current.get("test_accuracy"),
            False,
        ),
    ]
    return [
        MetricDelta(name, base, curr, _sub(curr, base), higher_is_worse)
        for name, base, curr, higher_is_worse in specs
    ]


def is_regression(deltas: list[MetricDelta], tol: float) -> bool:
    """True if any metric moved in its 'worse' direction by more than `tol`.

    Each MetricDelta carries `delta` (current - baseline, or None) and
    `higher_is_worse`. A metric where higher is worse regresses when it went UP
    past tol; a metric where higher is better regresses when it went DOWN past
    tol. Missing deltas (None) carry no signal. The run regresses overall if any
    single metric regresses.
    """
    for delta in deltas:
        if delta.delta is None:
            continue
        if delta.higher_is_worse and delta.delta > tol:
            return True
        elif not delta.higher_is_worse and -delta.delta > tol:
            return True
    return False


def compare_reports(baseline: dict, current: dict, tol: float = 0.05) -> Comparison:
    deltas = build_deltas(baseline["summary"], current["summary"])
    return Comparison(deltas=deltas, regressed=is_regression(deltas, tol))


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.4f}"


def render_comparison(
    comparison: Comparison, tol: float, console: Console | None = None
) -> None:
    console = console or Console()
    table = Table(title=f"kvcheck regression report (tol={tol})")
    table.add_column("metric")
    table.add_column("baseline", justify="right")
    table.add_column("current", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("dir")
    for d in comparison.deltas:
        arrow = "↑worse" if d.higher_is_worse else "↑better"
        delta_str = _fmt(d.delta)
        if d.delta is not None:
            delta_str = f"{d.delta:+.4f}"
        table.add_row(d.name, _fmt(d.baseline), _fmt(d.current), delta_str, arrow)
    console.print(table)
    if comparison.regressed:
        console.print("[bold]REGRESSED[/bold]", style="red")
    else:
        console.print("[bold]OK[/bold] (no regression)", style="green")
