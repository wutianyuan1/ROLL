import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 14
matplotlib.rcParams["ytick.labelsize"] = 14
matplotlib.rcParams["legend.fontsize"] = 13


SCENARIO_ORDER = ["increasing", "decreasing", "mixed"]
SCENARIO_LABELS = {
    "increasing": "Increasing",
    "decreasing": "Decreasing",
    "mixed": "Mixed",
}
COST_METHOD_ORDER = ["static_weave", "random", "most_idle", "dynamic_regroup"]
METHOD_LABELS = {
    "static_weave": "Weave",
    "dynamic_regroup": "Regroup (Opt)",
    "most_idle": "Most Idle",
    "random": "Random",
}
UTIL_METHOD_ORDER = ["static_weave", "dynamic_regroup"]


def load_results(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)

    missing_scenarios = [scenario for scenario in SCENARIO_ORDER if scenario not in data]
    if missing_scenarios:
        raise ValueError(f"Missing scenarios in result file: {missing_scenarios}")

    return data


def merge_results(base_results: dict, extra_results: list[dict]) -> dict:
    merged = {scenario: dict(values) for scenario, values in base_results.items()}
    for extra in extra_results:
        for scenario in SCENARIO_ORDER:
            merged.setdefault(scenario, {})
            merged[scenario].update(extra.get(scenario, {}))
    return merged


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", linewidth=1, color="#b0b0b0", alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def get_bar_style():
    util_colors = sns.color_palette("Blues")[::-1][2:]
    cost_colors = sns.color_palette("Blues")[::-1][::2] + ["white"]
    return {
        "static_weave": {"cost_color": cost_colors[0], "util_color": util_colors[0], "util_hatch": None},
        "random": {"cost_color": cost_colors[1], "util_color": cost_colors[3], "util_hatch": None},
        "most_idle": {"cost_color": cost_colors[2], "util_color": cost_colors[2], "util_hatch": None},
        "dynamic_regroup": {"cost_color": cost_colors[3], "util_color": util_colors[3], "util_hatch": "..."},
    }


def annotate_bars(ax, bars, fmt: str = "{:.2f}", y_offset: float = 0.015) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + span * y_offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=12,
        )


def _compute_cost_break_ranges(results: dict, methods: list[str]) -> tuple[tuple[float, float], tuple[float, float]]:
    values = []
    for method in methods:
        for scenario in SCENARIO_ORDER:
            baseline = results[scenario]["dynamic_regroup"]["total_cost"]
            values.append(results[scenario][method]["total_cost"] / baseline)

    low_candidates = [v for v in values if v <= 1.5]
    high_candidates = [v for v in values if v > 1.5]
    if not low_candidates or not high_candidates:
        return (0.0, max(values) * 1.12), (0.0, max(values) * 1.12)

    low_top = max(low_candidates) * 1.15
    high_bottom = min(high_candidates) * 0.94
    high_top = max(high_candidates) * 1.08
    if high_bottom <= low_top:
        midpoint = (max(low_candidates) + min(high_candidates)) / 2
        low_top = midpoint * 0.92
        high_bottom = midpoint * 1.08
    return (0.0, low_top), (high_bottom, high_top)


def annotate_broken_bars(ax_low, ax_high, bars, values, low_ylim, high_ylim, fmt: str = "{:.2f}") -> None:
    low_span = low_ylim[1] - low_ylim[0]
    high_span = high_ylim[1] - high_ylim[0]
    for bar, value in zip(bars, values):
        target_ax = ax_high if value >= high_ylim[0] else ax_low
        offset = high_span * 0.03 if target_ax is ax_high else low_span * 0.02
        target_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=12,
        )


def plot_normalized_total_cost(results: dict, output_path: Path) -> None:
    methods = [m for m in COST_METHOD_ORDER if m in results[SCENARIO_ORDER[0]]]
    low_ylim, high_ylim = _compute_cost_break_ranges(results, methods)
    fig, (ax_high, ax_low) = plt.subplots(
        2,
        1,
        figsize=(8, 1.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.8], "hspace": 0.05},
    )
    style_axes(ax_high)
    style_axes(ax_low)
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.18
    bar_style = get_bar_style()

    legend_handles = None
    legend_labels = None
    for idx, method in enumerate(methods):
        values = []
        for scenario in SCENARIO_ORDER:
            baseline = results[scenario]["dynamic_regroup"]["total_cost"]
            values.append(results[scenario][method]["total_cost"] / baseline)
        offset = (idx - (len(methods) - 1) / 2) * width
        bars_low = ax_low.bar(
            x + offset,
            values,
            width=width,
            color=bar_style[method]["cost_color"],
            edgecolor="black",
            linewidth=1,
            hatch=None,
            label=METHOD_LABELS[method],
            zorder=2,
        )
        ax_high.bar(
            x + offset,
            values,
            width=width,
            color=bar_style[method]["cost_color"],
            edgecolor="black",
            linewidth=1,
            hatch=None,
            label=METHOD_LABELS[method],
            zorder=2,
        )
        annotate_broken_bars(ax_low, ax_high, bars_low, values, low_ylim, high_ylim, fmt="{:.2f}")
        legend_handles, legend_labels = ax_low.get_legend_handles_labels()

    ax_low.plot(
        [-0.6, len(SCENARIO_ORDER) - 0.4],
        [1.0, 1.0],
        linestyle=":",
        c="red",
        linewidth=1,
    )
    ax_low.set_ylabel("Total Cost\n(Normalized)", labelpad=0)
    ax_low.yaxis.set_label_coords(-0.09, 0.72)
    ax_low.set_xticks(x)
    ax_low.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax_low.set_xlim(-0.6, len(SCENARIO_ORDER) - 0.4)
    ax_low.set_ylim(*low_ylim)
    ax_high.set_ylim(*high_ylim)
    ax_high.spines["bottom"].set_visible(False)
    ax_low.spines["top"].set_visible(False)
    ax_high.tick_params(labeltop=False, bottom=False)
    ax_low.tick_params(top=False)

    d = 0.012
    kwargs = dict(transform=ax_high.transAxes, color="k", clip_on=False, linewidth=1)
    ax_high.plot((-d, +d), (-d, +d), **kwargs)
    ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_low.transAxes)
    ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    fig.legend(
        legend_handles,
        legend_labels,
        ncols=4,
        frameon=False,
        columnspacing=0.8,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.05),
    )
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.135, right=0.98, hspace=0.05)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_utilization(results: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 2), sharey=True)
    bar_style = get_bar_style()
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.32

    metric_specs = [
        ("rollout_utilization", "Rollout Utilization"),
        ("train_utilization", "Train Utilization"),
    ]

    for ax, (metric_key, title) in zip(axes, metric_specs):
        style_axes(ax)
        for idx, method in enumerate(UTIL_METHOD_ORDER):
            values = [results[scenario][method][metric_key] for scenario in SCENARIO_ORDER]
            offset = (idx - 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=bar_style[method]["util_color"],
                edgecolor="black",
                linewidth=1,
                hatch=bar_style[method]["util_hatch"],
                label=METHOD_LABELS[method],
                zorder=2,
            )
            annotate_bars(ax, bars, fmt="{:.2f}", y_offset=0.012)
        ax.set_title(title, pad=8, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
        ax.set_ylim(0.0, 1.0)

    axes[0].set_ylabel("Utilization")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncols=2,
        frameon=False,
        columnspacing=0.8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_default_output_paths(input_path: Path) -> tuple[Path, Path]:
    stem = input_path.stem
    parent = input_path.parent
    return (
        parent / f"{stem}_normalized_cost.pdf",
        parent / f"{stem}_cluster_utilization.pdf",
    )


def main():
    parser = argparse.ArgumentParser(description="Plot drift experiment results as publication-style bar charts.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/wuty/Desktop/ROLL/global_scheduler/results/wild_mixed_results.json"),
        help="Path to the result JSON file.",
    )
    parser.add_argument(
        "--cost-output",
        type=Path,
        default=None,
        help="Output PDF for normalized total cost chart.",
    )
    parser.add_argument(
        "--util-output",
        type=Path,
        default=None,
        help="Output PDF for rollout/train utilization chart.",
    )
    parser.add_argument(
        "--extra-inputs",
        default="",
        help="Comma-separated extra result JSON files to merge for the cost plot.",
    )
    args = parser.parse_args()

    base_results = load_results(args.input)
    extra_paths = [Path(item.strip()) for item in args.extra_inputs.split(",") if item.strip()]
    extra_results = [load_results(path) for path in extra_paths]
    results = merge_results(base_results, extra_results)
    default_cost_output, default_util_output = build_default_output_paths(args.input)
    cost_output = args.cost_output or default_cost_output
    util_output = args.util_output or default_util_output

    cost_output.parent.mkdir(parents=True, exist_ok=True)
    util_output.parent.mkdir(parents=True, exist_ok=True)

    plot_normalized_total_cost(results, cost_output)
    plot_cluster_utilization(results, util_output)

    print(f"Saved normalized cost plot to {cost_output}")
    print(f"Saved cluster utilization plot to {util_output}")


if __name__ == "__main__":
    main()
