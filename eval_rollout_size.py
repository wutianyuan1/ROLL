#!/usr/bin/env python3
"""
Batch-run start_gen.sh over different rollout_batch_size settings,
parse Output_lens and time/step_generate from stdout, and log results.

Usage:
    python3 profile_rollout.py \
        --config examples/qwen2.5-7B-rlvr_megatron/gen_config_4gpu.yaml \
        --command "./start_gen.sh" \
        --log results.csv
"""

import argparse
import os
import re
import subprocess
import sys
import json
import csv
from shutil import copyfile

def replace_rollout_batch_size(config_path: str, new_size: int):
    """
    Read the YAML config at config_path, replace the line with rollout_batch_size: <int>
    to the new_size, in-place.
    """
    pattern = re.compile(r'^(\s*)rollout_batch_size\s*:\s*\d+')
    replaced = False
    lines = []
    with open(config_path, 'r') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                indent = m.group(1)
                line = f"{indent}rollout_batch_size: {new_size}\n"
                replaced = True
            lines.append(line)
    if not replaced:
        print(f"Warning: did not find rollout_batch_size in {config_path}", file=sys.stderr)
    with open(config_path, 'w') as f:
        f.writelines(lines)

def run_command(cmd: str, env: dict):
    """
    Run the shell command cmd (string), return stdout as a string.
    """
    proc = subprocess.run(cmd, shell=True, env=env,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          universal_newlines=True)
    return proc.stdout

def parse_output(output: str):
    """
    Parse stdout to extract:
      - Output_lens: a list of ints
      - time/step_generate: a float
    Returns (output_lens: list[int], time_per_step: float)
    or (None, None) if not found.
    """
    output_lens = None
    time_per_step = None

    # regex for Output_lens: [....]
    re_lens = re.compile(r'Output_lens\s*:\s*\[([0-9,\s]+)\]')
    # regex for "time/step_generate": float
    re_time = re.compile(r'"time/step_generate"\s*:\s*([0-9.eE+-]+)')

    for line in output.splitlines():
        if output_lens is None:
            m = re_lens.search(line)
            if m:
                nums = [int(x.strip()) for x in m.group(1).split(',') if x.strip()]
                output_lens = nums
        if time_per_step is None:
            m = re_time.search(line)
            if m:
                try:
                    time_per_step = float(m.group(1))
                except ValueError:
                    pass
        if output_lens is not None and time_per_step is not None:
            break

    return output_lens, time_per_step

def main():
    parser = argparse.ArgumentParser(description="Profile rollout_batch_size runs")
    parser.add_argument('--config', required=True,
                        help="Path to the YAML config file")
    parser.add_argument('--command', default='./start_gen.sh',
                        help="Command to run (e.g. ./start_gen.sh)")
    parser.add_argument('--log', default='results.csv',
                        help="CSV file to append results to")
    parser.add_argument('--sizes', nargs='+', type=int,
                        default=[8,16,32,64,128,256,512],
                        help="rollout_batch_size values to sweep")
    args = parser.parse_args()

    # Prepare log file
    is_new = not os.path.exists(args.log)
    log_f = open(args.log, 'a', newline='')
    writer = csv.writer(log_f)
    if is_new:
        writer.writerow(['rollout_batch_size', 'output_lens', 'time_step_generate'])
        log_f.flush()

    # Environment for subprocess
    base_env = os.environ.copy()
    base_env['RAY_DEDUP_LOGS'] = '0'

    # Backup the original config once
    backup_path = args.config + '.bak'
    if not os.path.exists(backup_path):
        copyfile(args.config, backup_path)

    try:
        for size in args.sizes:
            print(f"--- Running with rollout_batch_size = {size} ---")
            # restore original then replace
            copyfile(backup_path, args.config)
            replace_rollout_batch_size(args.config, size)

            # run the command
            out = run_command(args.command, env=base_env)

            # optionally, you can dump the full output to a file:
            # with open(f'out_{size}.log', 'w') as f: f.write(out)

            # parse
            lenses, tps = parse_output(out)
            if lenses is None or tps is None:
                print(f"Warning: parsing failed for size={size}", file=sys.stderr)
                print("stdout was:\n", out, file=sys.stderr)
            # write to CSV
            writer.writerow([size, json.dumps(lenses), tps])
            log_f.flush()
            print(f"-> Parsed Output_lens={lenses}, time/step_generate={tps}")
    finally:
        log_f.close()
        # restore the original config
        copyfile(backup_path, args.config)
        print(f"Restored original config from {backup_path}")

if __name__ == '__main__':
    main()
