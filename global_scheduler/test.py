import time
import os
import sys
from global_scheduler.brute_force_solver import IntraGroupSolver
from global_scheduler.simulator import WeaveSimulator
from global_scheduler.weave_scheduler import WeaveScheduler
from global_scheduler.structs import Job


def cost_based_score(utils):
    rollout_waste = [1 - i for i in utils['rollout']]
    train_waste = [1 - i for i in utils['train']]
    return -1 * (sum(rollout_waste) * 0.3 + sum(train_waste))


def test_solver():
    job_A = Job('A', 5, 5)
    job_B = Job('B', 5, 2.5)
    job_C = Job('C', 5, 2.5)
    jobs = [job_A, job_B, job_C]
    t1 = time.time()
    igs = IntraGroupSolver(jobs, sim_steps=1000)
    score, strategy = igs.solve(5, cost_based_score)
    t2 = time.time()
    print(f"Time to solve = {t2 - t1}")
    partition, deploy_jobs, meta_iter = strategy
    sim = WeaveSimulator(deploy_jobs, meta_iter)
    sim.plot(10, "global_scheduler/best.png")


def test_scheduler():
    sched = WeaveScheduler(cost_based_score, 1.0)
    job_seq = [Job("A", 5, 5), Job("B", 5, 5), Job("C", 5, 5)]
    for job in job_seq:
        print(f"\n======== Insert Job {job.job_id} ========")
        print(sched.add_job(job))


if __name__ == "__main__":
    test_scheduler()
