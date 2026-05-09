import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import sys


def parse_scores_from_lines(lines):
    steps = []
    values = []
    for line in lines:
        if line.startswith("step "):
            parts = line.strip().split(": ")
            if len(parts) != 2:
                continue
            step_part, value_part = parts
            try:
                step = int(step_part.split()[1])
                value = float(value_part)
            except (IndexError, ValueError):
                continue
            steps.append(step)
            values.append(value)
    return steps[:100], values[:100]


def parse_scores_from_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return parse_scores_from_lines(f)


def plot_critic_scores(file_paths, output_path=None):
    if not file_paths:
        steps, values = parse_scores_from_lines(sys.stdin)
        plot_items = [("stdin", steps, values)]
    else:
        plot_items = [
            (Path(file_path).stem, *parse_scores_from_file(file_path))
            for file_path in file_paths
        ]

    plt.figure(figsize=(10, 6))
    for label, steps, values in plot_items:
        if not steps:
            print(f"warning: no valid step data found for {label}", file=sys.stderr)
            continue
        plt.plot(steps, values, label=label)

    plt.xlabel("Step")
    plt.ylabel("Critic Score Mean")
    plt.title("Critic Score Mean over Steps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot step/value series from one or more input files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Paths to input files containing lines like 'step 10: 0.5'. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to save the plot instead of showing it interactively.",
    )
    args = parser.parse_args()

    plot_critic_scores(args.files, args.output)


if __name__ == "__main__":
    main()
