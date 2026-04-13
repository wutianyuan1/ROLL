import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from global_scheduler.generate_drift_traces import (
    DEFAULT_REFERENCE_DIR,
    REFERENCE_FILENAMES,
    TEMPLATE_NAMES,
    build_template_rollout_steps,
    load_reference_ratios,
    read_reference_curve,
)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

MODEL_ORDER = [
    ('7B', '7b'),
    ('14B', '14b'),
    ('32B', '32b'),
]
TEMPLATE_LABELS = {
    'no_drift': 'No Drift',
    'decreasing_0.5x': 'Decreasing 0.5x',
    'decreasing_1.0x': 'Decreasing 1.0x',
    'decreasing_1.5x': 'Decreasing 1.5x',
    'increasing_0.5x': 'Increasing 0.5x',
    'increasing_1.0x': 'Increasing 1.0x',
    'increasing_1.5x': 'Increasing 1.5x',
}
TEMPLATE_COLORS = {
    'no_drift': '#111111',
    'decreasing_0.5x': '#4C78A8',
    'decreasing_1.0x': '#1F77B4',
    'decreasing_1.5x': '#0D3B66',
    'increasing_0.5x': '#F28E2B',
    'increasing_1.0x': '#E15759',
    'increasing_1.5x': '#B22222',
}
TEMPLATE_STYLES = {
    'no_drift': '-',
    'decreasing_0.5x': '--',
    'decreasing_1.0x': '-',
    'decreasing_1.5x': ':',
    'increasing_0.5x': '--',
    'increasing_1.0x': '-',
    'increasing_1.5x': ':',
}


def plot_reference_drifts(reference_dir: str, output: str):
    reference_ratios = load_reference_ratios(reference_dir)
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5), constrained_layout=True)

    legend_handles = []
    legend_labels = []

    for ax, (model_label, ref_key) in zip(axes, MODEL_ORDER):
        reference_values = read_reference_curve(str(Path(reference_dir) / REFERENCE_FILENAMES[ref_key]))
        base_rollout = reference_values[0]
        steps = list(range(len(reference_ratios[ref_key])))

        for template_name in TEMPLATE_NAMES:
            rollout_steps = build_template_rollout_steps(base_rollout, reference_ratios[ref_key], template_name)
            line, = ax.plot(
                steps,
                rollout_steps,
                label=TEMPLATE_LABELS[template_name],
                color=TEMPLATE_COLORS[template_name],
                linestyle=TEMPLATE_STYLES[template_name],
                linewidth=2.0,
            )
            if len(legend_handles) < len(TEMPLATE_NAMES):
                legend_handles.append(line)
                legend_labels.append(TEMPLATE_LABELS[template_name])

        ax.set_title(f'{model_label} Reference-Based Drift')
        ax.set_xlabel('Iteration')
        if '7' in model_label:
            ax.set_ylabel('Step Time (s)')
        ax.grid(True, alpha=0.25)

    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=False,
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the 7 reference-based drift templates for 7B/14B/32B as a 1x3 PDF.')
    parser.add_argument('--reference-dir', default=DEFAULT_REFERENCE_DIR, help='Directory containing 7b/14b/32b reference CSVs.')
    parser.add_argument(
        '--output',
        default='/root/workspace/weave/ROLL/global_scheduler/drift_reference_templates.pdf',
        help='Output PDF path.',
    )
    args = parser.parse_args()
    plot_reference_drifts(args.reference_dir, args.output)
    print(f'Wrote {args.output}')
