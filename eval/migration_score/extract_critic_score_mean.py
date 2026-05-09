import argparse
import re
from pathlib import Path

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
    return values

def main():
    parser = argparse.ArgumentParser(description="Extract critic/score/mean values from a .out file.")
    parser.add_argument("file", type=Path, help="Path to the .out file")
    args = parser.parse_args()

    values = extract_critic_score_mean(args.file)
    assert len(values) % 2 == 0, "expected an even number of critic/score/mean values"

    for i in range(0, len(values), 2):
        v0, v1 = values[i], values[i + 1]
        assert v0 == v1, f"pair values differ at step {i // 2}: {v0} != {v1}"
        print(f"step {i // 2}: {v0}")

if __name__ == "__main__":
    main()