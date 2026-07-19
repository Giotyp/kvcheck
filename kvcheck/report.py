"""Aggregate divergence records into a summary, a verdict, and CI artifacts.

The one judgment call here is verdict(): given how much the test config diverged
from golden (the signal) AND how much golden diverged from itself (the noise
floor), decide pass/fail.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from kvcheck.config import Thresholds
from kvcheck.runner import RunResult
from kvcheck.types import DivergenceRecord


@dataclass
class Summary:
    n_scored: int
    divergence_rate: float  # fraction of requests where test != reference exactly
    argmax_flip_rate: float  # flips per comparable position
    mean_kl: float  # position-weighted mean top-k KL
    floor_divergence_rate: float  # same fraction, calibration vs reference
    floor_mean_kl: float


def _divergence_rate(records: list[DivergenceRecord]) -> float:
    if not records:
        return 0.0
    return sum(0 if r.exact_match else 1 for r in records) / len(records)


def _flip_rate(records: list[DivergenceRecord]) -> float:
    positions = sum(r.n_comparable for r in records)
    if positions == 0:
        return 0.0
    return sum(r.argmax_flips for r in records) / positions


def _weighted_kl(records: list[DivergenceRecord]) -> float:
    positions = sum(r.n_comparable for r in records)
    if positions == 0:
        return 0.0
    return sum(r.mean_kl * r.n_comparable for r in records) / positions


def summarize(result: RunResult) -> Summary:
    return Summary(
        n_scored=len(result.records),
        divergence_rate=_divergence_rate(result.records),
        argmax_flip_rate=_flip_rate(result.records),
        mean_kl=_weighted_kl(result.records),
        floor_divergence_rate=_divergence_rate(result.floor_records),
        floor_mean_kl=_weighted_kl(result.floor_records),
    )


def verdict(summary: Summary, thresholds: Thresholds) -> bool:
    """Return True if the run PASSES, False if it FAILS.

    The signal to judge is how much the test config diverged *beyond* what
    golden already diverges from itself (the noise floor). A run should not
    fail for divergence that the floor already explains. Compare that excess
    against thresholds.max_divergence_rate_above_floor.
    """
    excess = summary.divergence_rate - summary.floor_divergence_rate

    if excess <= thresholds.max_divergence_rate_above_floor:
        return True
    else:
        return False


def render_console(
    summary: Summary, passed: bool, console: Console | None = None
) -> None:
    console = console or Console()
    table = Table(title="kvcheck divergence report")
    table.add_column("metric")
    table.add_column("test vs golden", justify="right")
    table.add_column("noise floor", justify="right")
    table.add_row(
        "divergence rate",
        f"{summary.divergence_rate:.3f}",
        f"{summary.floor_divergence_rate:.3f}",
    )
    table.add_row(
        "mean top-k KL", f"{summary.mean_kl:.4f}", f"{summary.floor_mean_kl:.4f}"
    )
    table.add_row("argmax flip rate", f"{summary.argmax_flip_rate:.3f}", "-")
    table.add_row("scored requests", str(summary.n_scored), "-")
    console.print(table)
    console.print(
        f"[bold]{'PASS' if passed else 'FAIL'}[/bold]",
        style="green" if passed else "red",
    )


def write_json(summary: Summary, verdict_passed: bool, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps({"passed": verdict_passed, "summary": asdict(summary)}, indent=2)
    )
