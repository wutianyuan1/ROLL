import time
import os
import sys
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from typing import Callable, List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor

from global_scheduler.brute_force_solver_new import BruteForceSolver
from global_scheduler.weave_scheduler import WeaveScheduler, per_time_cost
from global_scheduler.baselines import BaselineScheduler, RandomScheduler
from global_scheduler.structs import Job


class JobGenerator:
    def __init__(self, rollout_dist_func: Callable, train_dist_func: Callable, slo_func: Callable):
        self.rollout_dist_func = rollout_dist_func
        self.train_dist_func = train_dist_func
        self.slo_func = slo_func

    def gen(self, job_id):
        return Job(job_id, self.rollout_dist_func(), self.train_dist_func(), self.slo_func())


def read_trace(trace_fn):
    traces = []
    with open(trace_fn, 'r') as f:
        content = f.read().split("\n")
    for line in content:
        if len(line) <= 1:
            continue
        items = line.split(', ')
        if len(items) == 3:
            jid, t, event = items
            t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            traces.append([jid.strip("application_"), t, int(event)])
        else:
            assert len(items) == 6  # + t_roll, t_train, slo
            jid, t, event, t_roll, t_train, slo = items
            t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            traces.append([jid.strip("application_"), t, int(event), float(t_roll), float(t_train), float(slo)])
    return traces


def generate_jobs(trace, slo_func, export_fn_prefix: str):
    ### Similar rollout-train
    small_unif_job_gen  = JobGenerator(lambda: random.uniform(50, 100), lambda: random.uniform(50, 100), slo_func)
    mid_unif_job_gen    = JobGenerator(lambda: random.uniform(100, 200), lambda: random.uniform(100, 200), slo_func)
    large_unif_job_gen  = JobGenerator(lambda: random.uniform(200, 300), lambda: random.uniform(200, 300), slo_func)
    ### Train heavy (TH)
    small_th_job_gen  = JobGenerator(lambda: random.uniform(25, 50), lambda: random.uniform(100, 200), slo_func)
    mid_th_job_gen    = JobGenerator(lambda: random.uniform(50, 100), lambda: random.uniform(200, 400), slo_func)
    large_th_job_gen  = JobGenerator(lambda: random.uniform(100, 200), lambda: random.uniform(400, 600), slo_func)
    ### Rollout heavy (RH)
    small_rh_job_gen  = JobGenerator(lambda: random.uniform(100, 200), lambda: random.uniform(25, 50), slo_func)
    mid_rh_job_gen    = JobGenerator(lambda: random.uniform(200, 400), lambda: random.uniform(50, 100), slo_func)
    large_rh_job_gen  = JobGenerator(lambda: random.uniform(400, 600), lambda: random.uniform(100, 200), slo_func)

    # Try to mix these jobs
    mixed_generators = {
        'uni': [small_unif_job_gen, mid_unif_job_gen, large_unif_job_gen],
        'rh':  [small_rh_job_gen, mid_rh_job_gen, large_rh_job_gen],
        'th':  [small_th_job_gen, mid_th_job_gen, large_th_job_gen],
        'all': [small_unif_job_gen, mid_unif_job_gen, large_unif_job_gen,
                small_rh_job_gen, mid_rh_job_gen, large_rh_job_gen,
                small_th_job_gen, mid_th_job_gen, large_th_job_gen],
    }

    trace_list= read_trace(trace)
    for mix_type in mixed_generators:
        job_generators = mixed_generators[mix_type]
        with open(f"{export_fn_prefix}_{mix_type}.trace", 'w') as f:
            for i, (jid, t, event) in enumerate(trace_list):
                if event == 1:
                    job_gen: JobGenerator = np.random.choice(job_generators)
                    job = job_gen.gen(jid)
                    f.write(f"{job.job_id}, {t}, {event}, {job.t_rollout}, {job.t_train}, {job.slo}\n")
                else:
                    f.write(f"{jid}, {t}, {event}\n")


def sim_baseline(sched: BaselineScheduler, trace_fn: str):
    trace = read_trace(trace_fn)
    running_jobs: Dict[str, Job] = {}
    total_cost, last_state_cost, last_t = 0, 0, trace[0][1]
    last_state_total_jobs, last_state_invalid_jobs = 0, {}
    time_costs = []  # [(timestamp, cost), ...]
    time_invalid_jobs = []

    for i, job_info in tqdm(enumerate(trace[:2000])):
        jid, t, event = job_info[0], job_info[1], job_info[2]
        if i > 0:
            delta_t = (t - last_t).total_seconds()
            total_cost += last_state_cost * delta_t
            time_costs.append((last_t, t, last_state_cost))
            time_invalid_jobs.append((last_t, t, last_state_invalid_jobs, last_state_total_jobs))
        if event == 1:
            assert len(job_info) == 6
            t_roll, t_train, slo = job_info[3], job_info[4], job_info[5]
            job = Job(jid, t_roll, t_train, slo)
            running_jobs[jid] = job
            # print(f"\n======== Insert Job {job.job_id} [{job.t_rollout=}, {job.t_train=}], {running_jobs=} ========")
            sched.add_job(job)
        else:
            del running_jobs[jid]
            # print(f"\n======== Delete Job {jid} [After {running_jobs=}] ========")
            sched.remove_job(jid)
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
    solver = BruteForceSolver(jobs_list, max_group_size_)
    cost, _, _ = solver.solve()
    return cost


def sim_optimal(trace_fn: str, max_group_size: int):
    trace = read_trace(trace_fn)
    try:
        print("Submitting all OPT computation tasks...")
        executor = ProcessPoolExecutor(max_workers=40)
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
                opt_current_jobs[jid] = Job(jid, t_roll, t_train, slo)
            else:
                del opt_current_jobs[jid]
            opt_last_t = t

        print("Waiting for all OPT computations to complete...")
        total_opt_cost = 0
        opt_costs = []
        for future, start_time, end_time in opt_tasks:
            opt_cost = future.result()
            if opt_cost > 100000: # inf
                opt_cost = opt_costs[-1][2] # simply reset to the last value
            delta_t = (end_time - start_time).total_seconds()
            total_opt_cost += opt_cost * delta_t
            opt_costs.append((start_time, end_time, opt_cost))
            print(f"OPT cost for period {start_time} to {end_time}: {opt_cost}, duration: {delta_t}")
    finally:
        executor.shutdown(wait=True)
    return total_opt_cost, opt_costs



if __name__ == "__main__":
    random.seed(2345)
    np.random.seed(2345)
    max_group_size = 3
    generate_jobs("global_scheduler/trace/philly_0_10000_10.trace", lambda: random.uniform(1.1, 2), "global_scheduler/trace/philly_0_10000_10_parsed")
    f = open("global_scheduler/run_results.txt", "w")
    for mix_type in ['all']: #['uni', 'rh', 'th', 'all']:
        total_cost, time_costs, time_invalid_jobs = sim_baseline(
            WeaveScheduler(per_time_cost, max_group_size),
            f"global_scheduler/trace/philly_0_10000_10_parsed_{mix_type}.trace"
        )
        total_rand_cost, time_rand_costs, time_rank_invalid_jobs = sim_baseline(
            RandomScheduler(per_time_cost, max_group_size),
            f"global_scheduler/trace/philly_0_10000_10_parsed_{mix_type}.trace"
        )
        total_opt_cost, opt_costs = sim_optimal(
            f"global_scheduler/trace/philly_0_10000_10_parsed_{mix_type}.trace",
            max_group_size
        )
        result_str = f"[{mix_type}] {total_cost=}, {total_rand_cost=}, {total_opt_cost=}\n"
        f.write(f"{mix_type}--"
                f"Weave|{total_cost}|{time_costs}|{time_invalid_jobs}||"
                f"Random|{total_rand_cost}|{time_rand_costs}|{time_rank_invalid_jobs}||"
                f"Opt|{total_opt_cost}|{opt_costs}|{[]}\n")
        print(result_str)
        # print(f"[{mix_type}] {total_cost=}, {total_opt_cost=}, ratio={total_opt_cost / total_cost if total_cost != 0 else float('inf')}")
    f.close()
