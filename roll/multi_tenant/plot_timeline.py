#!/usr/bin/env python3
"""
Timeline plotter for fault-tolerance emulation using paper style.
"""
import json
import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns

# Set font rendering for publication quality
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

blues = sns.color_palette('Blues')


def load_events(path: Path):
    """Load events from JSONL file."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_segments(events):
    """Build time segments from enter/exit events."""
    open_events = {}
    segments = []
    for e in events:
        if e['event'] == 'section_enter':
            open_events[(e['iter'], e['phase'])] = e
        elif e['event'] in ('section_exit', 'section_crash'):
            key = (e['iter'], e['phase'])
            if key in open_events:
                start = open_events.pop(key)
                segments.append((e['phase'], e['iter'], start['time'], e['time']))
    return segments


def plot_timeline(output_dir: Path, output_path: Path):
    """Generate timeline plot with paper styling."""
    # Load events
    job_events = {}
    for name in ('A', 'B', 'C'):
        events_path = output_dir / f'{name}.jsonl'
        if events_path.exists():
            job_events[name] = load_events(events_path)

    if not job_events:
        print(f"Error: No event files found in {output_dir}")
        return

    # Find fail/recover times
    a_events = job_events.get('A', [])
    fail_time = None
    recover_time = None
    for e in a_events:
        if e['event'] == 'section_crash' and fail_time is None:
            fail_time = e['time']
        if e['event'] == 'job_recovered' and recover_time is None:
            recover_time = e['time']

    # Get time offset
    all_times = [e['time'] for events in job_events.values() for e in events]
    t0 = min(all_times)

    # Define patterns and colors (matching paper style)
    patterns = {
        "A": {"generate": ("#F6C667", "XXXX"), "train": ("#C07A00", "XXXX")},
        "B": {"generate": ("#7FB8FF", "..."), "train": ("#1E5AA8", "...")},
        "C": {"generate": ("#C9C9C9", "---"), "train": ("#595959", "---")}
    }

    # Job name to lane mapping
    rollout_lane = {"A": 3, "B": 2, "C": 1}
    train_lane = 0

    # Create figure with paper dimensions
    fig, ax = plt.subplots(figsize=(6, 2.5))

    # Plot segments for each job
    for job_name, events in job_events.items():
        segments = build_segments(events)

        # Plot train segments
        for phase, iter_num, start, end in segments:
            x = start - t0
            duration = end - start

            if phase == 'train':
                lane = train_lane
                color = patterns[job_name]['train'][0]
                hatch_pattern = patterns[job_name]['train'][1]
            else:  # generate
                lane = rollout_lane[job_name]
                color = patterns[job_name]['generate'][0]
                hatch_pattern = patterns[job_name]['generate'][1]

            # Draw filled rectangle with hatch
            ax.add_patch(patches.Rectangle(
                [x, lane], duration, 1,
                edgecolor='white',
                hatch=hatch_pattern,
                facecolor=color,
                zorder=1
            ))
            # Draw border
            ax.add_patch(patches.Rectangle(
                [x, lane], duration, 1,
                edgecolor='black',
                facecolor='none',
                zorder=2
            ))

        # Add legend entries (invisible rectangles)
        ax.add_patch(patches.Rectangle(
            [0, 0], 0, 0,
            edgecolor='white',
            hatch=patterns[job_name]['generate'][1],
            facecolor=patterns[job_name]['generate'][0],
            zorder=1,
            label=f"{job_name}-rollout"
        ))
        ax.add_patch(patches.Rectangle(
            [0, 0], 0, 0,
            edgecolor='white',
            hatch=patterns[job_name]['train'][1],
            facecolor=patterns[job_name]['train'][0],
            zorder=1,
            label=f"{job_name}-train"
        ))

    # Add failure and recovery lines
    if fail_time is not None:
        ax.axvline(fail_time - t0, color='red', linestyle='--', linewidth=1.5, zorder=3)
        ax.text(fail_time - t0 + 0.5, 3.7, 'A fails', color='red', fontsize=12)

    if recover_time is not None:
        ax.axvline(recover_time - t0, color='blue', linestyle='--', linewidth=1.5, zorder=3)
        ax.text(recover_time - t0 + 0.5, 3.7, 'A recovers', color='blue', fontsize=12)

    # Set axis limits and labels
    max_time = max(all_times) - t0
    ax.set_xlim(0, max_time * 1.02)
    ax.set_ylim(0, 4)

    # Y-axis labels (matching paper style)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(["Trainer", "Rollout-3", "Rollout-2", "Rollout-1"], fontsize=14)

    # X-axis labels
    ax.set_xlabel("Time (s)", fontsize=14)
    plt.xticks(fontsize=14)

    # Legend
    lgd = plt.figlegend(
        ncols=3,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        fontsize=14,
        columnspacing=0.5,
        frameon=False
    )

    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=200, bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)

    print(f"Timeline saved to {output_path}")
    print(f"PDF saved to {output_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description='Plot timeline from emulation results')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory with event logs')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / 'timeline_styled.png'

    plot_timeline(output_dir, output_path)


if __name__ == '__main__':
    main()
