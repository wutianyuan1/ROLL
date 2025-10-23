import time
import os
import sys
import random
from global_scheduler.brute_force_solver_new import BruteForceSolver
from global_scheduler.weave_scheduler import WeaveScheduler, per_time_cost
from global_scheduler.structs import Job



def test_scheduler(n_jobs, max_group_size):
    sched = WeaveScheduler(per_time_cost, max_group_size)
    job_seq = [Job(str(i), random.randint(10, 20), random.randint(10, 20), slo=1.2) for i in range(n_jobs)]
    for job in job_seq:
        print(f"\n======== Insert Job {job.job_id} ========")
        print(sched.add_job(job))
    print("!!!", sched.group_costs, sum(sched.group_costs.values()))
    solver = BruteForceSolver(job_seq, max_group_size)
    ret = solver.solve()
    print(ret)


if __name__ == "__main__":
    random.seed(2345)
    test_scheduler(n_jobs=10, max_group_size=3)
