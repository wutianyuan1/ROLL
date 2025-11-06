import time
import os
import sys
import random
import numpy as np
import json
import math
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from typing import Callable, List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor

from global_scheduler.brute_force_solver_new import BruteForceSolver
from global_scheduler.weave_scheduler import WeaveScheduler, per_time_cost
from global_scheduler.baselines import BaselineScheduler, RandomScheduler, MostIdleScheduler
from global_scheduler.structs import Job

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def read_profile():
    with open("global_scheduler/trace/profile.json", 'r') as f:
        # profile_time['colo'/'disagg']['xB_xk']['generate'/'rollout']
        profile_time: Dict[str, Dict[str, Dict[str, float]]] = json.load(f)
    return profile_time


def read_trace(trace_fn):
    traces = []
    with open(trace_fn, 'r') as f:
        content = f.read().split("\n")
    for line in content:
        if len(line) <= 1:
            continue
        items = line.split(', ')
        jid, t, event, job_type = items
        jid = jid.strip("job")
        t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        event = int(event)
        if event == 1:
            traces.append([jid, t, event, job_type])
        else:
            traces.append([jid, t, event])
    return traces


def baseline(trace_fn: str, colo: bool):
    location = 'colo' if colo else 'disagg'
    profile_time = read_profile()
    trace = read_trace(trace_fn)
    jobs = {}
    total_cost = 0
    jid_2_num_steps = {}
    # List[(# H20 nodes, # H800 nodes, timing)]
    states: List[Tuple[int, int, datetime]] = []
    curr_num_h20_nodes, curr_num_h800_nodes = 0, 0
    for items in trace:
        if len(items) > 3:
            jid, t, event, job_type = items
            assert event == 1 and jid not in jobs
            t_rollout = profile_time[location][job_type]['generate']
            t_train = profile_time[location][job_type]['train']
            jobs[jid] = (t, job_type, t_rollout, t_train)
        else:
            jid, t, event = items
            assert event == -1 and jid in jobs
            t_start, job_type, t_rollout, t_train = jobs[jid]
            duration = (t - t_start).total_seconds()
            jid_2_num_steps[jid] = math.floor(duration / (t_rollout + t_train))
            cost_per_sec = 1
            if not colo:
                cost_per_sec *= 4 / 3
            if job_type.startswith('32B'):
                cost_per_sec *= 2
            total_cost += duration * cost_per_sec
        curr_num_h800_nodes += (event if not job_type.startswith('32B') else 2 * event)
        if not colo:
            curr_num_h20_nodes += (event if not job_type.startswith('32B') else 2 * event)
        states.append((curr_num_h20_nodes, curr_num_h800_nodes, t))
    assert states[-1][0:2] == (0, 0)
    cost_for_check = sum([(states[i][-1] - states[i - 1][-1]).total_seconds() * \
                          (states[i - 1][0] / 3 + states[i - 1][1]) \
                          for i in range(1, len(states))])
    assert int(total_cost) == int(cost_for_check)
    return total_cost, jid_2_num_steps, states


def sim_baseline(sched: BaselineScheduler, trace_fn: str, mannual_slo: Callable[[], float]):
    profile_time = read_profile()
    trace = read_trace(trace_fn)
    running_jobs: Dict[str, Job] = {}
    total_cost, last_state_cost, last_t = 0, 0, trace[0][1]
    last_state_total_jobs, last_state_invalid_jobs = 0, {}
    time_costs = []  # [(timestamp, cost), ...]
    time_invalid_jobs = []

    for i, job_info in enumerate(tqdm(trace)):
        jid, t, event = job_info[0], job_info[1], job_info[2]
        if i > 0:
            delta_t = (t - last_t).total_seconds()
            total_cost += last_state_cost * delta_t
            time_costs.append((last_t, t, last_state_cost))
            time_invalid_jobs.append((last_t, t, last_state_invalid_jobs, last_state_total_jobs))
        if event == 1:
            assert len(job_info) == 4
            job_type = job_info[3]
            t_rollout = profile_time['disagg'][job_type]['generate']
            t_train = profile_time['disagg'][job_type]['train']
            job = Job(jid, t_rollout, t_train, mannual_slo())
            running_jobs[jid] = job
            # print(f"\n======== Insert Job {job.job_id} [{job.t_rollout=}, {job.t_train=}], {running_jobs=} ========")
            sched.add_job(job, t, job_type.startswith('32B'))
        else:
            del running_jobs[jid]
            # print(f"\n======== Delete Job {jid} [After {running_jobs=}] ========")
            sched.remove_job(jid, t)
        last_state_cost = sum(sched.group_costs.values())
        last_state_invalid_jobs = deepcopy(sched.group_invalid_jobs)
        last_state_total_jobs = sched.total_running_jobs
        last_t = t

    num_invalid_jobs = lambda group_invalid_jobs: sum(len(group_invalid_jobs[grp_id]) for grp_id in group_invalid_jobs)
    slo_violation_counts = [num_invalid_jobs(i[2]) for i in time_invalid_jobs]
    slo_violation_ratio = sum(slo_violation_counts) / sum(i[3] for i in time_invalid_jobs)
    print(f"SLO violation ratio = {slo_violation_ratio}")
    return total_cost, time_costs, time_invalid_jobs


def run_ablation_slo(trace_fn: str, max_group_size: int, slo: Callable[[],float]):
    weave = WeaveScheduler(per_time_cost, max_group_size)
    total_cost, time_costs, time_invalid_jobs = sim_baseline(
        weave,
        trace_fn,
        mannual_slo=slo
    )
    result_str = f"[{slo}] {total_cost=}\n"
    print(result_str)
    for i, (num_rollout_nodes, num_train_nodes, t) in enumerate(weave.num_nodes_trace):
        if i != len(weave.num_nodes_trace) - 1:
            assert weave.num_nodes_trace[i][-1] == time_costs[i][0]
            per_t_cost = weave.num_nodes_trace[i][0] / 3 + weave.num_nodes_trace[i][1]
            assert abs(per_t_cost - time_costs[i][-1]) < 1e-10
        else:
            assert weave.num_nodes_trace[i][-1] == time_costs[i - 1][1]
    assert weave.num_nodes_trace[-1][0:2] == (0, 0)
    return total_cost, weave.average_slowdown(), weave.num_nodes_trace, weave.average_utils()


def plot(colo_states: List[Tuple[int, int, datetime]],
         naived_states: List[Tuple[int, int, datetime]],
         weave_states: List[Tuple[int, int, datetime]],
         colo_cost: float, naived_cost: float, weave_cost: float):
    assert len(colo_states) == len(naived_states) == len(weave_states)
    H800_NODE_USD_PER_HOUR = 8 * 5.28
    def usd_per_h(n_h20_node: int, n_h800_node: int):
        return (n_h20_node / 3 + n_h800_node) * H800_NODE_USD_PER_HOUR
    def modify_attrs(states: List[Tuple[int, int, datetime]]):
        for i in range(len(states)):
            states[i] = (states[i][0] * 8, states[i][1] * 8, usd_per_h(states[i][0], states[i][1]), states[i][-1])
        return states
    for states in [colo_states, naived_states, weave_states]:
        states = modify_attrs(states)
    timings = [(t - colo_states[0][-1]).total_seconds() / 3600 for _, _, _, t in colo_states]
    ylabels = ['# H20', '# H800', 'Per-Hour Cost ($/h)']
    filenames = ['h20', 'h800', 'cost']
    for i in range(3):
        plt.figure(figsize=(6.2, 3))
        if i == 2:
            colo_avg_cost = colo_cost * (H800_NODE_USD_PER_HOUR / 3600) / timings[-1]
            naived_avg_cost = naived_cost * (H800_NODE_USD_PER_HOUR / 3600) / timings[-1]
            weave_avg_cost = weave_cost * (H800_NODE_USD_PER_HOUR / 3600) / timings[-1]
            plt.hlines(colo_avg_cost, timings[0], timings[-1], colors='blue', linestyles='-.')
            plt.hlines(naived_avg_cost, timings[0], timings[-1], colors='red', linestyles='-.')
            plt.hlines(weave_avg_cost, timings[0], timings[-1], colors='black', linestyles='-.')
            print(f'Avg cost (K $/h): colo: {colo_avg_cost / 1000:.2f}, naive-d: {naived_avg_cost / 1000:.2f}, weave: {weave_avg_cost / 1000:.2f}')
        plt.plot(timings, [state[i] for state in colo_states], label='veRL', color='blue', linestyle='-.')
        plt.plot(timings, [state[i] for state in naived_states], label='Naive-D', color='red', linestyle='-.')
        plt.plot(timings, [state[i] for state in weave_states], label='Weave', color='black', linestyle='-')
        # style
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(linestyle='-.', zorder=0)
        plt.ylabel(ylabels[i], fontdict={"fontsize": 14})
        plt.legend(fontsize=14, ncols=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.3))
        plt.xlabel("Time (h)", fontdict={"fontsize": 14})
        plt.tight_layout()
        plt.savefig(f"global_scheduler/wild_time_{filenames[i]}.png")
        plt.savefig(f"global_scheduler/wild_time_{filenames[i]}.pdf")


if __name__ == "__main__":
    random.seed(2345)
    np.random.seed(2345)
    trace_fn = "global_scheduler/trace/wild.trace"
    jid_2_duration = {}
    tr = read_trace(trace_fn)
    for items in tr:
        jid, t = items[:2]
        if jid not in jid_2_duration:
            jid_2_duration[jid] = t
        else:
            jid_2_duration[jid] = (t - jid_2_duration[jid]).total_seconds()

    colo_cost, colo_thpt, colo_states = baseline(trace_fn, True)
    naived_cost, naived_thpt, naived_states = baseline(trace_fn, False)
    weave_cost, weave_slowdown, weave_states, weave_utils = run_ablation_slo(trace_fn, 5, lambda: random.uniform(1.1, 2))
    numbers = list(weave_slowdown.values())
    # Thpt.
    weave_thpt_normed = np.mean(list([1 / sld for sld in weave_slowdown.values()]))
    colo_thpt_normed = np.mean([colo_thpt[jid] / naived_thpt[jid] for jid in colo_thpt])
    print("Mean thpt:")
    print(f"Thpt: "
          f"colo: {colo_thpt_normed/weave_thpt_normed:.4f}, "
          f"naive-d: {1/weave_thpt_normed:.4f}, "
          f"weave: {1:.4f}")
    
    print("Duration-weighted mean thpt:")
    weave_thpt_normed_w = np.sum(list([1 / sld  * jid_2_duration[jid] for jid, sld in weave_slowdown.items()])) / np.sum(list(jid_2_duration.values()))
    colo_thpt_normed_w = np.sum([colo_thpt[jid] / naived_thpt[jid] * jid_2_duration[jid] for jid in colo_thpt]) / np.sum(list(jid_2_duration.values()))
    print(f"Thpt: "
          f"colo: {colo_thpt_normed_w/weave_thpt_normed_w:.4f}, "
          f"naive-d: {1/weave_thpt_normed_w:.4f}, "
          f"weave: {1:.4f}")
    # Cost
    print(f"Cost: colo: {colo_cost:.4f}, naive-d: {naived_cost:.4f}, weave: {weave_cost:.4f}")
    print(f"Cost: colo: {colo_cost / weave_cost:.4f}, naive-d: {naived_cost / weave_cost:.4f}, weave: 1")
    # Util.
    colo_utils = {'rollout': 1, 'train': 1}
    _, _, _, naived_utils = run_ablation_slo(trace_fn, 1, lambda: 1.1)
    for utils in [colo_utils, naived_utils, weave_utils]:
        for k, v in utils.items():
            utils[k] = f"{v * 100:.2f}"
    print(f"Util: colo: {colo_utils}, naive-d: {naived_utils}, weave-d: {weave_utils}")
    plot(colo_states, naived_states, weave_states, colo_cost, naived_cost, weave_cost)
