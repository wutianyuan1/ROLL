import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import sys

def extract_critic_score_mean(file_path):
    pattern = re.compile(r'["\']?critic/score/mean["\']?\s*:\s*([-\d.eE]+)')
    values = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            for match in pattern.finditer(line):
                try:
                    values.append(float(match.group(1)))
                except ValueError:
                    continue
    assert len(values) % 2 == 0, "expected an even number of critic/score/mean values"

    for i in range(0, len(values), 2):
        v0, v1 = values[i], values[i + 1]
        assert v0 == v1, f"pair values differ at step {i // 2}: {v0} != {v1}"
    return values[::2][:60]  # Return every other value (the first of each pair)

def smooth_values(values: list[float], alpha: float = 0.3) -> list[float]:
    if not values:
        return []
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
    return smoothed


def plot_critic_scores(model_size: str, file_paths: list[str], alpha: float):
    plot_items = []
    for file_path in file_paths:
        assert file_path.endswith(".out"), f"Expected .out file, got {file_path}"
        assert file_path.startswith(model_size), f"Expected file to start with {model_size}, got {file_path}"
        assert "_mig" in file_path or "_unmig" in file_path, f"Expected 'mig' or 'unmig' in file name, got {file_path}"
        assert not ("_mig" in file_path and "_unmig" in file_path), f"File name should not contain both 'mig' and 'unmig', got {file_path}"
        label = "Weave w/ Mig." if "_mig" in file_path else "veRL w/o Mig."
        values = extract_critic_score_mean(file_path)
        plot_items.append((label, smooth_values(values, alpha)))

    plt.figure(figsize=(6, 3))
    for label, values in plot_items:
        color = 'black' if label == 'Weave w/ Mig.' else 'blue'
        linestyle = '-' if label == 'Weave w/ Mig.' else '-.'
        plt.plot(range(len(values)), values, label=label, linewidth=3, color=color, linestyle=linestyle)

    plt.xlabel("Steps", fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel("Avg. Score", fontsize=16)
    plt.yticks(fontsize=14)
    plt.grid(linestyle='-.')
    plt.legend(loc='lower center', frameon=False, fontsize=16)
    plt.tight_layout()

    plt.savefig(f'migration_scores_{model_size}.pdf')


def main():
    parser = argparse.ArgumentParser(description="Plot migration critic score means with exponential smoothing.")
    parser.add_argument("--alpha", type=float, default=0.9,
                        help="Exponential smoothing factor in [0, 1] (default: 0.9)")
    args = parser.parse_args()

    files = {'7B': ['7B_200steps_unmig.out', '7B_100steps_mig.out'],
             '14B': ['14B_100steps_unmig.out', '14B_100steps_mig.out'],
             '32B': ['32B_100steps_unmig.out', '32B_100steps_mig.out']}

    for model_size, file_paths in files.items():
        plot_critic_scores(model_size, file_paths, args.alpha)


if __name__ == "__main__":
    main()
