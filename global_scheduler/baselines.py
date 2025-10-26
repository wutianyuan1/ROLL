import numpy as np
from copy import deepcopy
from abc import abstractmethod
from typing import List, Dict, Callable
from global_scheduler.structs import Job, JobGroup
from global_scheduler.new_simulator import WeaveSimulator


class BaselineScheduler:
    def __init__(self, cost_func: Callable[[Dict], float], max_group_size: int = 5,
                 simulate_steps: int = 100, rollout_cost: float = 1/3, train_cost: float = 1.0):
        self.job_groups: Dict[str, JobGroup] = {}
        self.last_group_id = -1
        self.cost_func = cost_func
        # Tolerate T_meta_iter >= overload_ratio * T1 if it is T2-bound
        # self.overload_ratio = overload_ratio
        self.max_group_size = max_group_size
        self.simulate_steps = simulate_steps
        self.rollout_cost = rollout_cost
        self.train_cost = train_cost
        self.group_costs = {}

    def next_group_id(self):
        self.last_group_id += 1
        return f"Group-{self.last_group_id}"

    @abstractmethod
    def add_job(self, job: Job):
        pass

    def remove_job(self, job_id: str, return_invalid: bool = False) -> None:
        removed = False
        for group_id in self.job_groups:
            job_group = self.job_groups[group_id]
            for job in job_group.jobs:
                if job.job_id == job_id:
                    job_group.jobs.remove(job)
                    removed = True
                    if len(job_group.jobs) != 0:
                        sim = WeaveSimulator(job_group.jobs)
                        rollout_busy_times, train_busy_times, utils, total_time = sim.simulate_run(self.simulate_steps)
                        if return_invalid:
                            cost, invalid_jobs = self.cost_func(job_group.jobs, len(job_group.all_rollout_nodes), train_busy_times, total_time, self.rollout_cost, self.train_cost, return_invalid=return_invalid)
                        else:
                            cost = self.cost_func(job_group.jobs, len(job_group.all_rollout_nodes), train_busy_times, total_time, self.rollout_cost, self.train_cost)
                        self.group_costs[group_id] = cost
                    else:
                        self.group_costs[group_id] = 0
                        del self.group_costs[group_id]
                        del self.job_groups[group_id]
                    break
            if removed:
                break
        if not removed:
            print(f"Remove failed: Job {job_id} does not exist.")

class RandomScheduler(BaselineScheduler):
    def __init__(self, cost_func: Callable[[Dict], float], max_group_size: int = 5,
                 simulate_steps: int = 100, rollout_cost: float = 1/3, train_cost: float = 1.0):
        super().__init__(cost_func, max_group_size, simulate_steps, rollout_cost, train_cost)
    
    def add_job(self, job: Job):
        existing_gids = list(self.job_groups.keys())
        new_gid = self.next_group_id()
        possible_gids = [new_gid] + existing_gids
        target_grp_id = np.random.choice(possible_gids)
        if target_grp_id == new_gid:
            # Place the job into a new group
            tmp_job = deepcopy(job)
            tmp_job.rollout_nodes = ["0"]
            tmp_job.train_nodes = ["TN"]
            job_group = JobGroup(target_grp_id, [tmp_job])
            # Assign rollout and train nodes
            best_rollout_node = job_group.all_rollout_nodes[0]
            best_train_node = job_group.all_train_nodes[0]
            # Record the new created group
            self.job_groups[job_group.group_id] = job_group
        else:
            self.last_group_id -= 1  # recall the new group id
            # Assign rollout and train nodes
            job_group = self.job_groups[target_grp_id]
            new_rollout_node_id = str(job_group.last_rollout_node_id + 1)
            possible_rollout_nodes = job_group.all_rollout_nodes + [new_rollout_node_id]
            best_rollout_node = np.random.choice(possible_rollout_nodes)
            best_train_node = job_group.all_train_nodes[0]
            if best_rollout_node == new_rollout_node_id:
                job_group.last_rollout_node_id += 1  # allocate a new rollout ID
            # Place the job into a new group
            tmp_job = deepcopy(job)
            tmp_job.rollout_nodes = [best_rollout_node]
            tmp_job.train_nodes = [best_train_node]
            # Add the job into the existing group
            self.job_groups[job_group.group_id].jobs.append(tmp_job)
        sim = WeaveSimulator(job_group.jobs)
        rollout_busy_times, train_busy_times, utils, total_time = sim.simulate_run(self.simulate_steps)
        cost, invalid_jobs = self.cost_func(
            job_group.jobs, len(job_group.all_rollout_nodes), train_busy_times,
            total_time, self.rollout_cost, self.train_cost, return_invalid=True)
        self.group_costs[job_group.group_id] = cost
        return best_rollout_node, best_train_node, job_group, cost, invalid_jobs

    def remove_job(self, job_id: str) -> None:
        super().remove_job(job_id, return_invalid=True)
