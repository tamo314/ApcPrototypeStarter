"""Required plots for the Phase A report (`docs/EXPERIMENT_PLAN.md` section
9), rendered from one `apc.evaluation.sequential_benchmark.
SequentialBenchmarkReport`.

Kept separate from `apc.evaluation.sequential_benchmark` so importing that
module (or running its tests) never requires matplotlib -- only callers
that actually want plots (the CLI script) pay for the dependency.

Only the single APC configuration Task 012 runs is available here, so
"primitive reuse heatmap by task family" is necessarily a single-run view
and `plot_lifetime_compute_proxy` below only plots this one run's compute.
For the actual cross-baseline "lifetime compute proxy by baseline" plot
section 9 asks for (B0-B4), see `apc.evaluation.baseline_plots.
plot_lifetime_compute_comparison` (Task 013).
"""

from __future__ import annotations

from pathlib import Path

from apc.evaluation.sequential_benchmark import SequentialBenchmarkReport

_LABEL_ORDER = ("K", "C", "N", "R")


def _event_ticks(report: SequentialBenchmarkReport) -> list[str]:
    return [
        f"{i}:{e.label}" + (f"/{e.operation_name}" if e.operation_name else "")
        for i, e in enumerate(report.events)
    ]


def plot_performance_over_stream(report: SequentialBenchmarkReport, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    ticks = _event_ticks(report)
    pre = [e.pre_exact_match for e in report.events]
    post = [e.post_exact_match for e in report.events]
    fig, ax = plt.subplots(figsize=(max(6, len(ticks) * 0.8), 4))
    x = range(len(ticks))
    ax.plot(x, pre, "o--", label="pre-event exact match")
    ax.plot(x, post, "o-", label="post-event exact match")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_ylabel("exact match")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Performance over the task stream")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_parameter_counts_over_stream(report: SequentialBenchmarkReport, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    ticks = _event_ticks(report)
    persistent = [e.persistent_param_count for e in report.events]
    temporary_peak = [e.temporary_peak_param_count for e in report.events]
    fig, ax = plt.subplots(figsize=(max(6, len(ticks) * 0.8), 4))
    x = range(len(ticks))
    ax.plot(x, persistent, "o-", label="persistent (bank) parameters")
    ax.plot(x, temporary_peak, "o-", label="temporary parameters (at event end)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_ylabel("parameter count")
    ax.set_title("Persistent vs. temporary parameter count over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_controller_state_over_time(report: SequentialBenchmarkReport, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    states = ["stable", "search", "plastic", "consolidate", "shadow"]
    state_index = {s: i for i, s in enumerate(states)}
    steps = [0]
    values = [state_index["stable"]]
    for t in report.controller_transitions:
        steps.append(t["step"])
        values.append(state_index.get(t["new_state"], -1))
    fig, ax = plt.subplots(figsize=(max(6, len(steps) * 0.3), 3))
    ax.step(steps, values, where="post")
    ax.set_yticks(list(state_index.values()))
    ax.set_yticklabels(states)
    ax.set_xlabel("controller step")
    ax.set_title("Controller state over time")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_adaptation_steps_by_label(report: SequentialBenchmarkReport, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    totals: dict[str, list[int]] = {label: [] for label in _LABEL_ORDER}
    for e in report.events:
        totals[e.label].append(e.search_steps + e.plastic_steps)
    labels = [label for label in _LABEL_ORDER if totals[label]]
    means = [sum(totals[label]) / len(totals[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, means)
    ax.set_ylabel("mean adaptation steps (search + plastic)")
    ax.set_title("Adaptation steps per event type")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_primitive_reuse_heatmap(report: SequentialBenchmarkReport, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    primitive_ids = sorted(
        {e.candidate_primitive_id for e in report.events if e.candidate_primitive_id is not None}
    )
    ticks = _event_ticks(report)
    if not primitive_ids:
        primitive_ids = [-1]
    grid = np.zeros((len(primitive_ids), len(report.events)))
    for col, e in enumerate(report.events):
        if e.candidate_primitive_id is not None:
            row = primitive_ids.index(e.candidate_primitive_id)
            grid[row, col] = 2  # newly consolidated this event
        elif e.label == "R" and not e.had_cycle:
            # Reused: mark every primitive row as a faint "was available" hit
            # is not derivable without per-event router selections, so mark
            # the most recently consolidated primitive up to this point.
            available = [pid for pid in primitive_ids if pid != -1]
            if available:
                grid[primitive_ids.index(available[-1]), col] = 1
    fig, ax = plt.subplots(figsize=(max(6, len(ticks) * 0.6), max(2, len(primitive_ids) * 0.6)))
    im = ax.imshow(grid, aspect="auto", cmap="Greens", vmin=0, vmax=2)
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_yticks(range(len(primitive_ids)))
    ax.set_yticklabels([f"primitive {pid}" for pid in primitive_ids])
    ax.set_title("Primitive reuse by event (0=absent, 1=reused, 2=consolidated here)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_forgetting_per_task(report: SequentialBenchmarkReport, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    ticks = [
        f"{s.index}:{s.label}" + (f"/{s.operation_name}" if s.operation_name else "")
        for s in report.retention
    ]
    forgetting = [s.forgetting for s in report.retention]
    fig, ax = plt.subplots(figsize=(max(6, len(ticks) * 0.6), 4))
    ax.bar(range(len(ticks)), forgetting)
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_ylabel("forgetting (post-event minus final exact match)")
    ax.set_title("Forgetting per task")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_lifetime_compute_proxy(report: SequentialBenchmarkReport, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    cumulative = []
    running = 0
    running += report.config.pretrain.steps
    cumulative.append(running)
    for e in report.events:
        running += e.search_steps + e.plastic_steps
        if e.consolidation is not None:
            running += e.consolidation.steps * e.shadow_attempts
        cumulative.append(running)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(cumulative)), cumulative, "o-", label="APC (this run)")
    ax.set_xlabel("pretrain, then event index")
    ax.set_ylabel("cumulative train steps (compute proxy)")
    ax.set_title(
        "Lifetime compute proxy (this run only -- see "
        "apc.evaluation.baseline_plots for the B0-B4 comparison)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_compression_ratio_per_novel_operation(
    report: SequentialBenchmarkReport, out_path: Path
) -> None:
    import matplotlib.pyplot as plt

    entries = [
        (e.operation_name or f"event{e.index}", e.consolidation.compression_ratio)
        for e in report.events
        if e.label == "N" and e.consolidation is not None
    ]
    fig, ax = plt.subplots(figsize=(5, 4))
    if entries:
        names, ratios = zip(*entries, strict=True)
        ax.bar(names, ratios)
    ax.set_ylabel("compression ratio (candidate / temporary params)")
    ax.set_title("Consolidation compression ratio per novel operation")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_all(report: SequentialBenchmarkReport, out_dir: str | Path) -> list[Path]:
    """Render every required plot into `out_dir`, returning the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = {
        "performance_over_stream.png": plot_performance_over_stream,
        "parameter_counts_over_stream.png": plot_parameter_counts_over_stream,
        "controller_state_over_time.png": plot_controller_state_over_time,
        "adaptation_steps_by_label.png": plot_adaptation_steps_by_label,
        "primitive_reuse_heatmap.png": plot_primitive_reuse_heatmap,
        "forgetting_per_task.png": plot_forgetting_per_task,
        "lifetime_compute_proxy.png": plot_lifetime_compute_proxy,
        "compression_ratio_per_novel_operation.png": plot_compression_ratio_per_novel_operation,
    }
    written = []
    for filename, fn in jobs.items():
        path = out_dir / filename
        fn(report, path)
        written.append(path)
    return written
