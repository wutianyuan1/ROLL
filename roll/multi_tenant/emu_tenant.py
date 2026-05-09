import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns

# Set font rendering for publication quality
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


ROLL_ROOT = Path(__file__).resolve().parent.parent.parent
RLVR_PIPELINE = ROLL_ROOT / "roll" / "pipeline" / "rlvr" / "rlvr_pipeline.py"
SCHEDULER = ROLL_ROOT / "worker_scheduler" / "scheduler.py"
PYTHON = Path(sys.executable)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_mapping_file(path: Path) -> None:
    path.write_text("E 0 0\nD1 1 0\nD2 2 0\n", encoding="utf-8")


def make_env(port: int, mapping_file: Path, scheduler_log: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(ROLL_ROOT),
        "MASTER_ADDR": "127.0.0.1",
        "SCHEDULER_ADDR": "127.0.0.1",
        "SCHEDULER_PORT": str(port),
        "NG": "3",
        "NT": "1",
        "N_JOB": "3",
        "GDA": "EMU-GEN",
        "TDA": "EMU-TRAIN",
        "MAPFN": str(mapping_file),
        "SCHEDULER_OUT_PATH": str(scheduler_log),
        "RATIO": "1.0",
    })
    return env


def launch_scheduler(env: Dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        [str(PYTHON), str(SCHEDULER)],
        cwd=str(ROLL_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def launch_job(
    job_name: str,
    generate_real_sec: float,
    train_real_sec: float,
    output_dir: Path,
    failure_iter: int = None,
    failure_phase: str = None,
    failure_after_ratio: float = 0.0,
    recover: bool = False,
    max_iters: int = 8,
    shared_env: Dict[str, str] = None,
) -> subprocess.Popen:
    env = dict(shared_env or os.environ)
    env["JOB_NAME"] = job_name
    if recover:
        env["RECOVER_JOB"] = "1"
    else:
        env.pop("RECOVER_JOB", None)
    cmd = [
        str(PYTHON),
        str(RLVR_PIPELINE),
        "--emulated",
        "--job-name", job_name,
        "--gen-devices", "[0]",
        "--train-devices", "[0]",
        "--generate-real-sec", str(generate_real_sec),
        "--train-real-sec", str(train_real_sec),
        "--time-scale", "200.0",
        "--max-iters", str(max_iters),
        "--event-log-dir", str(output_dir),
    ]
    if failure_iter is not None:
        cmd.extend(["--failure-iter", str(failure_iter)])
    if failure_phase is not None:
        cmd.extend(["--failure-phase", failure_phase])
    if failure_after_ratio:
        cmd.extend(["--failure-after-ratio", str(failure_after_ratio)])
    return subprocess.Popen(
        cmd,
        cwd=str(ROLL_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def load_job_events(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def assert_failure_behavior(job_events: Dict[str, List[Dict]]) -> Tuple[float, float]:
    a_events = job_events["E"]
    fail_event = next(event for event in a_events if event["event"] == "section_crash")
    recover_event = next(event for event in a_events if event["event"] == "job_recovered")
    fail_time = fail_event["time"]
    recover_time = recover_event["time"]

    bc_progress = [
        event
        for job_name in ("D1", "D2")
        for event in job_events[job_name]
        if fail_time < event["time"] < recover_time and event["event"] == "section_exit"
    ]
    assert bc_progress, "Expected D1/D2 to keep making progress while E is down."

    a_after_recover = [
        event
        for event in a_events
        if event["time"] > recover_time and event["event"] == "section_exit"
    ]
    assert any(event["phase"] == "generate" for event in a_after_recover), "Expected E generate after recovery."
    assert any(event["phase"] == "train" for event in a_after_recover), "Expected E train after recovery."
    return fail_time, recover_time


def build_segments(events: List[Dict]) -> List[Tuple[str, int, float, float]]:
    open_events: Dict[Tuple[int, str], Dict] = {}
    segments = []
    for event in events:
        if event["event"] == "section_enter":
            open_events[(event["iter"], event["phase"])] = event
        elif event["event"] in {"section_exit", "section_crash"}:
            key = (event["iter"], event["phase"])
            if key in open_events:
                start = open_events.pop(key)
                segments.append((event["phase"], event["iter"], start["time"], event["time"]))
    return segments


def plot_timeline(job_events: Dict[str, List[Dict]], fail_time: float, recover_time: float, output_path: Path) -> None:
    # Define patterns and colors (matching paper style)
    patterns = {
        "D1": {"generate": ("#F6C667", "XXXX"), "train": ("#C07A00", "XXXX")},
        "D2": {"generate": ("#7FB8FF", "..."), "train": ("#1E5AA8", "...")},
        "E": {"generate": ("#C9C9C9", "---"), "train": ("#595959", "---")}
    }
    rollout_lane = {"E": 1, "D1": 3, "D2": 2}
    train_lane = 0

    all_times = [event["time"] for events in job_events.values() for event in events]
    t0 = min(all_times)

    fig, ax = plt.subplots(figsize=(6, 2.5))

    for job_name, events in job_events.items():
        for phase, _iter, start, end in build_segments(events):
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

        # Add legend entries
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
    ax.axvline(fail_time - t0, color='red', linestyle='--', linewidth=1.5, zorder=3)
    ax.text(fail_time - t0 + 0.2, 4.2, 'E fails', color='red', fontsize=12)
    ax.axvline(recover_time - t0, color='blue', linestyle='--', linewidth=1.5, zorder=3)
    ax.text(recover_time - t0 + 0.2, 4.2, 'E recovers', color='blue', fontsize=12)

    # Set axis limits and labels
    max_time = max(all_times) - t0
    ax.set_xlim(0, max_time * 1.02)
    ax.set_ylim(0, 4)
    ax = plt.gca()
    ax.set_xticks([i*5 for i in range(9)], [str(int(i*5*200)) for i in range(9)])
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(["Trainer", "Rollout-3", "Rollout-2", "Rollout-1"], fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=14)
    plt.xticks(fontsize=12)

    # Legend
    lgd = plt.figlegend(
        ncols=3,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        fontsize=12,
        columnspacing=0.5,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=200, bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)


def shutdown_process(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def shutdown_redis(port: int) -> None:
    subprocess.run(
        ["redis-cli", "-h", "127.0.0.1", "-p", str(port), "shutdown", "nosave"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROLL_ROOT / "roll" / "multi_tenant" / "emu_failure_demo"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_file = output_dir / "case_mapping.txt"
    scheduler_log = output_dir / "scheduler.log"
    timeline_path = output_dir / "timeline.png"
    manifest_path = output_dir / "manifest.json"
    write_mapping_file(mapping_file)

    port = find_free_port()
    env = make_env(port=port, mapping_file=mapping_file, scheduler_log=scheduler_log)
    scheduler_proc = launch_scheduler(env)
    time.sleep(2.0)

    jobs = {}
    try:
        jobs["E"] = launch_job(
            "E", 768, 130, output_dir,
            failure_iter=2,
            failure_phase="generate",
            failure_after_ratio=0.5,
            max_iters=6,
            shared_env=env,
        )
        # jobs["D1"] = launch_job("D1", 330, 120, output_dir, max_iters=18, shared_env=env)
        # jobs["D2"] = launch_job("D2", 330, 120, output_dir, max_iters=18, shared_env=env)

        # a_rc = jobs["E"].wait(timeout=40)
        # if a_rc == 0:
        #     raise RuntimeError("E should have crashed in the injected failure case.")

        # time.sleep(10.0)
        # jobs["E_recovered"] = launch_job(
        #     "E", 768, 130, output_dir,
        #     recover=True,
        #     max_iters=6,
        #     shared_env=env,
        # )

        # for name in ("D1", "D2", "E_recovered"):
        #     jobs[name].wait(timeout=80)

        job_events = {
            "E": load_job_events(output_dir / "E.jsonl"),
            "D1": load_job_events(output_dir / "D1.jsonl"),
            "D2": load_job_events(output_dir / "D2.jsonl"),
        }
        fail_time, recover_time = assert_failure_behavior(job_events)
        plot_timeline(job_events, fail_time, recover_time, timeline_path)

        manifest = {
            "scheduler_log": str(scheduler_log),
            "timeline": str(timeline_path),
            "job_logs": {job: str(output_dir / f"{job}.jsonl") for job in ("E", "D1", "D2")},
            "fail_time": fail_time,
            "recover_time": recover_time,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return 0
    finally:
        for proc in jobs.values():
            shutdown_process(proc)
        shutdown_process(scheduler_proc)
        shutdown_redis(port)


if __name__ == "__main__":
    raise SystemExit(main())
