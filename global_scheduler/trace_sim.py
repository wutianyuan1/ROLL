import time
import os
import sys
import random
import numpy as np
from copy import deepcopy
from datetime import datetime
from typing import Callable, List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor

from global_scheduler.brute_force_solver_new import BruteForceSolver
from global_scheduler.weave_scheduler import WeaveScheduler, per_time_cost
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
        jid, t, event = line.split(', ')
        t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        traces.append([jid.strip("application_"), t, int(event)])
    return traces


def compute_opt_cost(jobs_list, max_group_size):
    solver = BruteForceSolver(jobs_list, max_group_size)
    cost, _, _ = solver.solve()
    return cost


def sim_main(max_group_size: int):
    small_unif_job_gen  = JobGenerator(lambda: random.uniform(10, 20), lambda: random.uniform(5, 15), lambda: 1.2)
    mid_unif_job_gen    = JobGenerator(lambda: random.uniform(20, 40), lambda: random.uniform(10, 30), lambda: 1.2)
    large_unif_job_gen  = JobGenerator(lambda: random.uniform(40, 80), lambda: random.uniform(20, 60), lambda: 1.2)
    job_generators = [small_unif_job_gen, mid_unif_job_gen, large_unif_job_gen]
    trace = read_trace("global_scheduler/trace/philly_0_35000_35.trace")
    
    sched = WeaveScheduler(per_time_cost, max_group_size)
    running_jobs: Dict[str, Job] = {}

    total_cost, last_state_cost, last_t = 0, 0, trace[0][1]
    time_costs = []  # [(timestamp, cost), ...]
    job_records = {}

    try:
        for i, (jid, t, event) in enumerate(trace):
            if i > 0:
                delta_t = (t - last_t).total_seconds()
                total_cost += last_state_cost * delta_t
                time_costs.append((last_t, t, last_state_cost))

            if event == 1:
                # if len(running_jobs) >= 8:
                #     break
                job_gen: JobGenerator = np.random.choice(job_generators)
                job = job_gen.gen(jid)
                running_jobs[jid] = job
                job_records[jid] = deepcopy(job)
                print(f"\n======== Insert Job {job.job_id} [{job.t_rollout=}, {job.t_train=}], {len(running_jobs)=} ========")
                sched.add_job(job)
            else:
                del running_jobs[jid]
                print(f"\n======== Delete Job {jid} [After {len(running_jobs)=}] ========")
                sched.remove_job(jid)

            print("!!!", sched.group_costs, sum(sched.group_costs.values()))

            last_state_cost = sum(sched.group_costs.values())
            last_t = t

        # Replay the trace once again, using the same jobs in the record
        print("Submitting all OPT computation tasks...")
        executor = ProcessPoolExecutor(max_workers=8)
        opt_tasks = []  # [(future, start_time, end_time), ...]
        opt_current_jobs = {}
        opt_last_t = trace[0][1]

        for i, (jid, t, event) in enumerate(trace):
            if i > 0:
                future = executor.submit(compute_opt_cost, [deepcopy(i) for i in opt_current_jobs.values()], max_group_size)
                opt_tasks.append((future, opt_last_t, t))
            if event == 1:
                # if len(opt_current_jobs) >= 8:
                #     break
                opt_current_jobs[jid] = job_records[jid]
            else:
                del opt_current_jobs[jid]
            opt_last_t = t

        print("Waiting for all OPT computations to complete...")
        total_opt_cost = 0
        for future, start_time, end_time in opt_tasks:
            opt_cost = future.result()
            delta_t = (end_time - start_time).total_seconds()
            total_opt_cost += opt_cost * delta_t
            print(f"OPT cost for period {start_time} to {end_time}: {opt_cost}, duration: {delta_t}")

    finally:
        executor.shutdown(wait=True)

    print(f"{total_cost=}, {total_opt_cost=}, ratio={total_opt_cost / total_cost if total_cost != 0 else float('inf')}")


if __name__ == "__main__":
    random.seed(2345)
    np.random.seed(2345)
    sim_main(max_group_size=3)
