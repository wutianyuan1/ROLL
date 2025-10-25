from copy import deepcopy
from typing import List, Dict, Callable
from global_scheduler.structs import Job, JobGroup
from global_scheduler.new_simulator import WeaveSimulator
from global_scheduler.baselines import BaselineScheduler


def per_time_cost(jobs: List[Job], num_rollout_nodes: int, train_busy_times: Dict,
                  total_time: float, rollout_cost: float = 1/3, train_cost: float = 1.0,
                  return_invalid: bool = False):
    total_working_time = 0
    valid = True
    invalid_jobs = []
    for job in jobs:
        num_iters = len(train_busy_times['TN'][job.job_id])
        if total_time / num_iters >= job.slo * (job.t_rollout + job.t_train):
            valid = False
            invalid_jobs.append(job.job_id)
        total_working_time += num_iters * (job.t_rollout + job.t_train)
    # cost_per_time = total_time * (1 * train_cost + num_rollout_nodes * rollout_cost) / total_working_time
    cost_per_time = 1 * train_cost + num_rollout_nodes * rollout_cost
    if not return_invalid:
        return cost_per_time if valid else float("inf")
    else:
        return cost_per_time, invalid_jobs


class WeaveScheduler(BaselineScheduler):
    def __init__(self, cost_func: Callable[[Dict], float], max_group_size: int = 5,
                 simulate_steps: int = 100, rollout_cost: float = 1/3, train_cost: float = 1.0):
        super().__init__(cost_func, max_group_size, simulate_steps, rollout_cost, train_cost)

    def add_job(self, job: Job):
        best_rollout_node, best_train_node, best_group, best_cost_delta = None, None, None, float("inf")
        for job_group_name, job_group in self.job_groups.items():
            if len(job_group.jobs) >= self.max_group_size:
                continue
            all_rollout_nodes, all_train_nodes = job_group.all_rollout_nodes, job_group.all_train_nodes

            # Case-1: share train, share rollout (direct insert)
            for train_node in all_train_nodes:
                for rollout_node in all_rollout_nodes:
                    tmp_job = deepcopy(job)
                    tmp_job.rollout_nodes = [rollout_node]
                    tmp_job.train_nodes = [train_node]

                    jobs_in_group = job_group.jobs + [tmp_job]
                    sim = WeaveSimulator(jobs_in_group)
                    rollout_busy_times, train_busy_times, utils, total_time = sim.simulate_run(self.simulate_steps)
                    cost = self.cost_func(jobs_in_group, len(all_rollout_nodes), train_busy_times, total_time, self.rollout_cost, self.train_cost)
                    print(f"Case-1, {rollout_node=}, {train_node=}, {cost=}")
                    if cost - self.group_costs[job_group_name] < best_cost_delta:
                        best_cost_delta = cost - self.group_costs[job_group_name]
                        best_rollout_node = rollout_node
                        best_train_node = train_node
                        best_group = job_group
            # Case-2: share train, but scale-up to an individual rollout
            for train_node in all_train_nodes:
                rollout_node = job_group.next_rollout_node_id()
                tmp_job = deepcopy(job)
                tmp_job.rollout_nodes = [rollout_node]
                tmp_job.train_nodes = [train_node]
                jobs_in_group = job_group.jobs + [tmp_job]
                sim = WeaveSimulator(jobs_in_group)
                rollout_busy_times, train_busy_times, utils, total_time = sim.simulate_run(self.simulate_steps)
                cost = self.cost_func(jobs_in_group, len(all_rollout_nodes) + 1, train_busy_times, total_time, self.rollout_cost, self.train_cost)
                print(f"Case-2, {rollout_node=}, {train_node=}, {cost=}")
                if cost - self.group_costs[job_group_name] < best_cost_delta:
                    best_cost_delta = cost - self.group_costs[job_group_name]
                    best_rollout_node = rollout_node
                    best_train_node = train_node
                    best_group = job_group
        # Case-3: form a new group
        tmp_job = deepcopy(job)
        tmp_job.rollout_nodes = ["0"]
        tmp_job.train_nodes = ["TN"]
        job_group = JobGroup(self.next_group_id(), [tmp_job])
        sim = WeaveSimulator(job_group.jobs)
        rollout_busy_times, train_busy_times, utils, total_time = sim.simulate_run(self.simulate_steps)
        cost = self.cost_func(job_group.jobs, 1, train_busy_times, total_time, self.rollout_cost, self.train_cost)
        print(f"Case-3, {cost=}")
        # If new group is the best
        if cost < best_cost_delta:
            best_cost_delta = cost
            best_rollout_node = job_group.all_rollout_nodes[0]
            best_train_node = job_group.all_train_nodes[0]
            best_group = job_group
            self.job_groups[job_group.group_id] = job_group
            self.group_costs[job_group.group_id] = cost
        # If one existing group is the best, then assign this job to the group
        else:
            tmp_job = deepcopy(job)
            tmp_job.rollout_nodes = [best_rollout_node]
            tmp_job.train_nodes = [best_train_node]
            self.last_group_id -= 1  # new group is not the best, recall the added group ID
            self.job_groups[best_group.group_id].jobs.append(tmp_job)
            self.group_costs[best_group.group_id] += best_cost_delta
        # print(self.group_costs)
        return best_rollout_node, best_train_node, best_group, best_cost_delta


if __name__ == "__main__":
    sched = WeaveScheduler(per_time_cost, 3)
    for i in range(5):
        print("="*80)
        best_rollout_node, best_train_node, best_group, best_cost = sched.add_job(Job(f"{i}", 30, 10, 1.1))
        print(i, best_rollout_node, best_train_node, best_group, best_cost)
