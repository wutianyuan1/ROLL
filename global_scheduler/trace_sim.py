import time
import os
import sys
import csv
import random
import numpy as np
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


def sim_baseline(sched: BaselineScheduler, trace_fn: str, mannual_slo: float = -1):
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
            assert len(job_info) == 6
            t_roll, t_train, slo = job_info[3], job_info[4], job_info[5]
            job = Job(jid, t_roll, t_train, slo if mannual_slo == -1 else mannual_slo)
            running_jobs[jid] = job
            # print(f"\n======== Insert Job {job.job_id} [{job.t_rollout=}, {job.t_train=}], {running_jobs=} ========")
            sched.add_job(job)
        else:
            del running_jobs[jid]
            # print(f"\n======== Delete Job {jid} [After {running_jobs=}] ========")
            if isinstance(sched, WeaveScheduler):
                sched.remove_job(jid, t)
            else:
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


def run_ablation_types(trace_fn_base:str, max_group_size: int):
    f = open("global_scheduler/run_results_type.txt", "w")
    for mix_type in ['uni', 'rh', 'th', 'all']:
        total_cost, time_costs, time_invalid_jobs = sim_baseline(
            WeaveScheduler(per_time_cost, max_group_size),
            trace_fn_base.format(mix_type),
        )
        fallback_opt_cost = {}
        for (last_t, t, cost_last_t) in time_costs:
            fallback_opt_cost[last_t] = cost_last_t
        total_rand_cost, time_rand_costs, time_rank_invalid_jobs = sim_baseline(
            RandomScheduler(per_time_cost, max_group_size),
            trace_fn_base.format(mix_type),
        )
        total_idle_cost, time_idle_costs, time_idle_invalid_jobs = sim_baseline(
            MostIdleScheduler(per_time_cost, max_group_size),
            trace_fn_base.format(mix_type),
        )
        total_opt_cost, opt_costs = sim_optimal(
            trace_fn_base.format(mix_type),
            max_group_size,
            fallback_opt_cost
        )
        result_str = f"[{mix_type}] {total_cost=}, {total_rand_cost=}, {total_idle_cost=}, {total_opt_cost=}\n"
        f.write(f"{mix_type}--"
                f"Weave|{total_cost}|{time_costs}|{time_invalid_jobs}||"
                f"Random|{total_rand_cost}|{time_rand_costs}|{time_rank_invalid_jobs}||"
                f"MostIdle|{total_idle_cost}|{time_idle_costs}|{time_idle_invalid_jobs}||"
                f"Opt|{total_opt_cost}|{opt_costs}|{[]}\n")
        print(result_str)
    f.close()


def run_ablation_slo(trace_fn: str, max_group_size: int):
    f = open("global_scheduler/run_results_slo.txt", "w")
    for SLO in [1.2, 1.5, 2.0]:
        total_cost, time_costs, time_invalid_jobs = sim_baseline(
            WeaveScheduler(per_time_cost, max_group_size),
            trace_fn
        )
        fallback_opt_cost = {}
        for (last_t, t, cost_last_t) in time_costs:
            fallback_opt_cost[last_t] = cost_last_t
        total_rand_cost, time_rand_costs, time_rank_invalid_jobs = sim_baseline(
            RandomScheduler(per_time_cost, max_group_size),
            trace_fn,
            mannual_slo=SLO
        )
        total_idle_cost, time_idle_costs, time_idle_invalid_jobs = sim_baseline(
            MostIdleScheduler(per_time_cost, max_group_size),
            trace_fn,
            mannual_slo=SLO
        )
        total_opt_cost, opt_costs = 0, [] #sim_optimal(
            # trace_fn,
            # max_group_size,
            # fallback_opt_cost
        # )
        result_str = f"[{SLO}] {total_cost=}, {total_rand_cost=}, {total_idle_cost=}, {total_opt_cost=}\n"
        f.write(f"{SLO}--"
                f"Weave|{total_cost}|{time_costs}|{time_invalid_jobs}||"
                f"Random|{total_rand_cost}|{time_rand_costs}|{time_rank_invalid_jobs}||"
                f"MostIdle|{total_idle_cost}|{time_idle_costs}|{time_idle_invalid_jobs}||"
                f"Opt|{total_opt_cost}|{opt_costs}|{[]}\n")
        print(result_str)
    f.close()


def run_ablation_group_size(trace_fn: str):
    f = open("global_scheduler/run_results_grp_size.txt", "w")
    for group_size in [2, 4, 5]:
        total_cost, time_costs, time_invalid_jobs = sim_baseline(
            WeaveScheduler(per_time_cost, max_group_size=group_size),
            trace_fn,
        )
        fallback_opt_cost = {}
        for (last_t, t, cost_last_t) in time_costs:
            fallback_opt_cost[last_t] = cost_last_t
        total_rand_cost, time_rand_costs, time_rank_invalid_jobs = sim_baseline(
            RandomScheduler(per_time_cost, max_group_size=group_size),
            trace_fn,
        )
        total_idle_cost, time_idle_costs, time_idle_invalid_jobs = sim_baseline(
            MostIdleScheduler(per_time_cost, max_group_size=group_size),
            trace_fn,
        )
        total_opt_cost, opt_costs = sim_optimal(
            trace_fn,
            group_size,
            fallback_opt_cost
        )
        result_str = f"[{group_size}] {total_cost=}, {total_rand_cost=}, {total_idle_cost=}, {total_opt_cost=}\n"
        f.write(f"{group_size}--"
                f"Weave|{total_cost}|{time_costs}|{time_invalid_jobs}||"
                f"Random|{total_rand_cost}|{time_rand_costs}|{time_rank_invalid_jobs}||"
                f"MostIdle|{total_idle_cost}|{time_idle_costs}|{time_idle_invalid_jobs}||"
                f"Opt|{total_opt_cost}|{opt_costs}|{[]}\n")
        print(result_str)
    f.close()

def measure_scheduling_overhead(
    scheduler_class: type, 
    test_points: int, 
    job_generator: JobGenerator, 
    max_group_size: int
) -> List[Tuple[int, float]]:
    """
    Measures the overhead of adding one job to a scheduler that already contains
    a certain number of jobs.

    Args:
        scheduler_class: The scheduler class to test (e.g., WeaveScheduler).
        max_jobs: The maximum number of concurrent jobs to test up to.
        job_generator: An instance of JobGenerator to create jobs.
        max_group_size: The max_group_size parameter for the scheduler.

    Returns:
        A list of tuples, where each tuple is (num_existing_jobs, overhead_in_seconds).
    """
    print(f"--- Measuring overhead for {scheduler_class.__name__} ---")
    overheads = []
    
    # We iterate test_points, and we are measuring the addition of the (i+1)-th job.
    for num_existing_jobs in tqdm(test_points, desc=f"Testing {scheduler_class.__name__}"):
        print(num_existing_jobs)
        # 1. Setup: Create a scheduler and populate it with `num_existing_jobs`
        sched = scheduler_class(per_time_cost, max_group_size)
        for i in range(num_existing_jobs):
            # Use unique job IDs for setup
            background_job = job_generator.gen(f"bg_{i}")
            sched.add_job(background_job)
            
        # 2. Prepare the new job to be timed
        new_job = job_generator.gen(f"new_{num_existing_jobs}")
            
        # 3. Measure the overhead of adding the new job
        start_time = time.perf_counter()
        sched.add_job(new_job)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        overheads.append((num_existing_jobs, duration))
        
    return overheads


def measure_opt_overhead(
    max_jobs: int, 
    job_generator: JobGenerator, 
    max_group_size: int
) -> List[Tuple[int, float]]:
    """
    Measures the overhead of the BruteForceSolver. The overhead is the total time
    to find the optimal placement for a given number of jobs.
    
    Args:
        max_jobs: The maximum number of jobs to solve for.
        job_generator: An instance of JobGenerator to create jobs.
        max_group_size: The max_group_size parameter for the solver.

    Returns:
        A list of tuples, where each tuple is (num_existing_jobs, overhead_in_seconds).
        `num_existing_jobs` is `N-1` to be comparable with other schedulers measuring the N-th job.
    """
    print("--- Measuring overhead for BruteForceSolver (OPT) ---")
    overheads = []

    # Here, `num_total_jobs` is the number of jobs we ask the solver to optimize.
    # This corresponds to adding the `num_total_jobs`-th job.
    for num_total_jobs in tqdm(range(1, max_jobs + 1), desc="Testing OPT"):
        # 1. Generate the full list of jobs for the solver
        jobs_to_solve = [job_generator.gen(f"job_{i}") for i in range(num_total_jobs)]
        
        # 2. Instantiate the solver and measure the `solve` time
        solver = BruteForceSolver(jobs_to_solve, max_group_size, n_iters=20)
        
        start_time = time.perf_counter()
        # The "overhead" for OPT is the entire solving process
        solver.solve(force_enum_all=True)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        
        # `num_total_jobs - 1` is the number of "existing" jobs, to match the x-axis
        # of the other schedulers.
        overheads.append((num_total_jobs - 1, duration))
        
    return overheads

def run_overhead_benchmark(output_csv_filename: str = "scheduler_overheads.csv"):
    """Main function to run the overhead benchmark for all schedulers."""
    
    # --- Configuration ---
    # NOTE: Keep MAX_JOBS_FOR_OPT small, as its complexity is exponential!
    TEST_POINTS = [1, 3, 5, 7, 9, 11, 13, 15, 100, 500, 1000, 2000]  # For fast schedulers
    MAX_JOBS_FOR_OPT = 13      # For BruteForceSolver
    MAX_GROUP_SIZE = 3
    
    # Use the "all" job generator for diverse, realistic jobs
    job_generator = JobGenerator(
        rollout_dist_func=lambda: random.uniform(25, 600),
        train_dist_func=lambda: random.uniform(25, 600),
        slo_func=lambda: random.uniform(1.1, 2.0)
    )
    
    schedulers_to_test = {
        "WEAVE": WeaveScheduler,
    }
    
    all_results = []
    
    # --- Run Benchmarks for Heuristic Schedulers ---
    for name, scheduler_class in schedulers_to_test.items():
        overheads = measure_scheduling_overhead(
            scheduler_class, 
            TEST_POINTS, 
            job_generator, 
            MAX_GROUP_SIZE
        )
        for num_jobs, duration in overheads:
            all_results.append([name, num_jobs, duration])
        
    # --- Run Benchmark for OPT ---
    opt_overheads = measure_opt_overhead(
        MAX_JOBS_FOR_OPT, 
        job_generator, 
        MAX_GROUP_SIZE
    )
    for num_jobs, duration in opt_overheads:
        all_results.append(["OPT (Brute-Force)", num_jobs, duration])
    
    # --- Save to CSV ---
    try:
        with open(output_csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Scheduler', 'NumExistingJobs', 'OverheadSeconds'])
            # Write data rows
            writer.writerows(all_results)
        print(f"Overhead benchmark results saved to {output_csv_filename}")
    except IOError as e:
        print(f"Error writing to CSV file {output_csv_filename}: {e}")


if __name__ == "__main__":
    random.seed(2345)
    np.random.seed(2345)
    max_group_size = 3
    # generate_jobs("global_scheduler/trace/philly_0_30000_20.trace", lambda: random.uniform(1.1, 2), "global_scheduler/trace/philly_0_30000_20_parsed")
    # run_ablation_types("global_scheduler/trace/philly_0_30000_20_parsed_{}.trace", max_group_size)
    run_ablation_slo("global_scheduler/trace/philly_0_30000_20_parsed_all.trace", max_group_size)
    # run_ablation_group_size("global_scheduler/trace/philly_0_30000_20_parsed_all.trace")
    # run_overhead_benchmark("./global_scheduler/scheduler_overheads.csv")
