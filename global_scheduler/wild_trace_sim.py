import time
import os
import sys
import random
import numpy as np
import json
import math
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from typing import Callable, List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor

from global_scheduler.brute_force_solver_new import BruteForceSolver
from global_scheduler.weave_scheduler import WeaveScheduler, per_time_cost
from global_scheduler.baselines import BaselineScheduler, RandomScheduler, MostIdleScheduler
from global_scheduler.structs import Job


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
    for items in trace:
        if len(items) > 3:
            jid, t, event, job_type = items
            assert event == 1 and jid not in jobs
            t_rollout = profile_time[location][job_type]['generate']
            t_train = profile_time[location][job_type]['train']
            jobs[jid] = (t, t_rollout, t_train)
        else:
            jid, t, event = items
            assert event == -1 and jid in jobs
            t_start, t_rollout, t_train = jobs[jid]
            duration = (t - t_start).total_seconds()
            jid_2_num_steps[jid] = math.floor(duration / (t_rollout + t_train))
            cost_per_sec = 1.0 if colo else (1/3 + 1.0)
            if job_type.startswith('32B'):
                cost_per_sec *= 2
            total_cost += duration * cost_per_sec
    return total_cost, jid_2_num_steps


def sim_baseline(sched: BaselineScheduler, trace_fn: str, mannual_slo: float):
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


# Helper function for parallel execution
def compute_opt_cost(jobs_list, max_group_size_):
    solver = BruteForceSolver(jobs_list, max_group_size_, n_iters=20)
    cost, _, _ = solver.solve(max_search_steps=10000)
    return cost


def sim_optimal(trace_fn: str, max_group_size: int, fallback_opt_cost: Dict, mannual_slo: float = -1):
    trace = read_trace(trace_fn)
    try:
        print("Submitting all OPT computation tasks...")
        executor = ProcessPoolExecutor(max_workers=8)
        opt_tasks = []  # [(future, start_time, end_time), ...]
        opt_current_jobs = {}
        opt_last_t = trace[0][1]
        for i, job_info in enumerate(trace):
            jid, t, event = job_info[0], job_info[1], job_info[2]
            if i > 0:
                future = executor.submit(compute_opt_cost, [deepcopy(i) for i in opt_current_jobs.values()], max_group_size)
                opt_tasks.append((future, opt_last_t, t))
            if event == 1:
                assert len(job_info) == 6
                t_roll, t_train, slo = job_info[3], job_info[4], job_info[5]
                opt_current_jobs[jid] = Job(jid, t_roll, t_train, slo if mannual_slo == -1 else mannual_slo)
            else:
                del opt_current_jobs[jid]
            opt_last_t = t

        print("Waiting for all OPT computations to complete...")
        total_opt_cost = 0
        opt_costs = []
        for future, start_time, end_time in opt_tasks:
            opt_cost = future.result()
            real_opt = opt_cost
            if opt_cost > 100000: # inf
                opt_cost = fallback_opt_cost[start_time] # simply reset to the Weave's best value
            delta_t = (end_time - start_time).total_seconds()
            total_opt_cost += opt_cost * delta_t
            opt_costs.append((start_time, end_time, opt_cost))
            print(f"OPT cost for period {start_time} to {end_time}: {real_opt}, duration: {delta_t}")
    finally:
        executor.shutdown(wait=True)
    return total_opt_cost, opt_costs


def run_ablation_slo(trace_fn: str, max_group_size: int, slo: float):
    weave = WeaveScheduler(per_time_cost, max_group_size)
    total_cost, time_costs, time_invalid_jobs = sim_baseline(
        weave,
        trace_fn,
        mannual_slo=slo
    )
    result_str = f"[{slo}] {total_cost=}\n"
    print(result_str)
    return total_cost, weave.average_slowdown()


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

    colo_cost, colo_thpt = baseline(trace_fn, True)
    naived_cost, naived_thpt = baseline(trace_fn, False)
    weave_cost, weave_slowdown = run_ablation_slo(trace_fn, 5, lambda: random.uniform(1.2, 1.5))
    numbers = list(weave_slowdown.values())
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
    
    print(f"Cost: colo: {colo_cost / weave_cost:.4f}, naive-d: {naived_cost / weave_cost:.4f}, weave: 1")
