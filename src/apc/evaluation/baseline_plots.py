"""Cross-baseline comparison plots for B0-B4 (`docs/EXPERIMENT_PLAN.md`
section 9's "lifetime compute proxy by baseline", plus two more comparisons
directly tied to H4/H5 -- Task 013).

Kept separate from `apc.evaluation.baselines` for the same reason
`apc.evaluation.sequential_benchmark_plots` is kept separate from
`apc.evaluation.sequential_benchmark`: importing the benchmark/baseline
modules (or running their tests) should never require matplotlib.

Consumes the `dict[str, BaselineReport | SequentialBenchmarkReport]`
`apc.evaluation.baselines.run_all_baselines` returns -- `SequentialBenchmarkReport`
(B4) and `BaselineReport` (B0-B3) expose different per-event report types
(`EventReport` vs `BaselineEventReport`), so each plot reads only the
fields both share (`persistent_param_count`/`train_steps`-equivalents),
duck-typed rather than requiring a common base class neither report
otherwise needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_BASELINE_ORDER = ("B0", "B1", "B2", "B3", "B4")


def _ordered(results: dict[str, Any]) -> list[tuple[str, Any]]:
    return [(name, results[name]) for name in _BASELINE_ORDER if name in results]


def _cumulative_train_steps(name: str, report: Any) -> list[int]:
    """Pretrain steps, then running total after each event. B4's `EventReport`
    additionally spends consolidation steps per shadow attempt, which
    `BaselineEventReport` (B0-B3) has no equivalent field for. B4's report
    holds a bare `SequentialBenchmarkConfig`; B0-B3's `BaselineReport.config`
    wraps one under `.sequential` (see `apc.evaluation.baselines.
    BaselineConfig`) -- both are read here to reach the same
    `pretrain.steps` field."""
    pretrain_config = report.config.pretrain if name == "B4" else report.config.sequential.pretrain
    running = pretrain_config.steps
    cumulative = [running]
    for e in report.events:
        if name == "B4":
            running += e.search_steps + e.plastic_steps
            if e.consolidation is not None:
                running += e.consolidation.steps * e.shadow_attempts
        else:
            running += e.train_steps
        cumulative.append(running)
    return cumulative


def plot_lifetime_compute_comparison(results: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, report in _ordered(results):
        cumulative = _cumulative_train_steps(name, report)
        ax.plot(range(len(cumulative)), cumulative, "o-", label=name)
    ax.set_xlabel("pretrain, then event index")
    ax.set_ylabel("cumulative train steps (compute proxy)")
    ax.set_title("Lifetime compute proxy by baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_persistent_growth_comparison(results: dict[str, Any], out_path: Path) -> None:
    """Final persistent parameter count per baseline -- Experiment Plan H4
    ("persistent parameter growth is lower [for APC] than ... an
    expansion-without-consolidation baseline")."""
    import matplotlib.pyplot as plt

    ordered = _ordered(results)
    names = [name for name, _ in ordered]
    counts = [report.persistent_parameter_count_final for _, report in ordered]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(names, counts)
    ax.set_ylabel("persistent parameter count (final)")
    ax.set_title("Final persistent capacity by baseline")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_retention_comparison(results: dict[str, Any], out_path: Path) -> None:
    """Max forgetting and mean backward transfer per baseline --
    Experiment Plan H5 ("retains earlier tasks better than unconstrained
    fine-tuning of a comparable dynamic model")."""
    import matplotlib.pyplot as plt

    ordered = _ordered(results)
    names = [name for name, _ in ordered]
    max_forgetting = [report.max_forgetting for _, report in ordered]
    backward_transfer = [report.mean_backward_transfer for _, report in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(names, max_forgetting)
    axes[0].set_ylabel("max forgetting")
    axes[0].set_title("Max forgetting by baseline")
    axes[1].bar(names, backward_transfer)
    axes[1].set_ylabel("mean backward transfer")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Mean backward transfer by baseline")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_all_baselines(results: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Render every cross-baseline comparison plot into `out_dir`, returning
    the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = {
        "lifetime_compute_comparison.png": plot_lifetime_compute_comparison,
        "persistent_growth_comparison.png": plot_persistent_growth_comparison,
        "retention_comparison.png": plot_retention_comparison,
    }
    written = []
    for filename, fn in jobs.items():
        path = out_dir / filename
        fn(results, path)
        written.append(path)
    return written
