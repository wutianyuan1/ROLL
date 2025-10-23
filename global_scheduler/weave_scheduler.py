from copy import deepcopy
from typing import List, Dict, Callable
from global_scheduler.structs import Job, JobGroup
from global_scheduler.new_simulator import WeaveSimulator


def per_time_cost(jobs: List[Job], num_rollout_nodes: int, train_busy_times: Dict,
                  total_time: float, rollout_cost: float = 1/3, train_cost: float = 1.0):
    total_working_time = 0
    valid = True
    for job in jobs:
        num_iters = len(train_busy_times['TN'][job.job_id])
        if total_time / num_iters >= job.slo * (job.t_rollout + job.t_train):
            valid = False
        total_working_time += num_iters * (job.t_rollout + job.t_train)
    cost_per_time = total_time * (1 * train_cost + num_rollout_nodes * rollout_cost) / total_working_time
    return cost_per_time if valid else float("inf")


class WeaveScheduler:
    def __init__(self, cost_func: Callable[[Dict], float], max_group_size: int = 5,
                 simulate_steps: int = 100, rollout_cost: float = 1/3, train_cost: float = 1.0):
        self.job_groups: Dict[str, JobGroup] = {}
        self.cost_func = cost_func
        # Tolerate T_meta_iter >= overload_ratio * T1 if it is T2-bound
        # self.overload_ratio = overload_ratio
        self.max_group_size = max_group_size
        self.simulate_steps = simulate_steps
        self.rollout_cost = rollout_cost
        self.train_cost = train_cost
        self.group_costs = {}

    def add_job(self, job: Job):
        best_rollout_node, best_train_node, best_group, best_cost_delta = None, None, None, float("inf")
        for job_group_name, job_group in self.job_groups.items():
            if len(job_group.jobs) >= self.max_group_size:
                continue
            all_rollout_nodes, all_train_nodes = job_group.all_rollout_nodes, job_group.all_train_nodes
            # original_phase_times = job_group.get_node_phase_times()
            # T1, T2 = job_group.T1, job_group.T2
            # print(f"{job_group_name=}, {T1=}, {T2=}, {all_rollout_nodes=}, {all_train_nodes=}")
            # assert T1 >= self.overload_ratio * T2, f"{T1=}, {T2=}"

            # Case-1: share train, share rollout (direct insert)
            for train_node in all_train_nodes:
                for rollout_node in all_rollout_nodes:
                    tmp_job = deepcopy(job)
                    tmp_job.rollout_nodes = [rollout_node]
                    tmp_job.train_nodes = [train_node]

                    # T1_after_insert = max(T1, tmp_job.t_rollout + tmp_job.t_train)
                    # T2_after_insert = max(T2,
                    #                       original_phase_times[rollout_node] + tmp_job.t_rollout,
                    #                       original_phase_times[train_node] + tmp_job.t_train)
                    # print(T1_after_insert, T2_after_insert)
                    # if T2_after_insert > self.overload_ratio * T1_after_insert:
                    #     print(f"Case-1 skip: {rollout_node}, {train_node}, {T2_after_insert}>{self.overload_ratio}*{T1_after_insert}")
                    #     continue
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
                rollout_node = f"RN-{len(all_rollout_nodes)}"
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
        tmp_job.rollout_nodes = ["RN-0"]
        tmp_job.train_nodes = ["TN"]
        job_group = JobGroup(f"Group-{len(self.job_groups)}", [tmp_job])
        sim = WeaveSimulator(job_group.jobs)
        rollout_busy_times, train_busy_times, utils, total_time = sim.simulate_run(self.simulate_steps)
        cost = self.cost_func(job_group.jobs, 1, train_busy_times, total_time, self.rollout_cost, self.train_cost)
        print(f"Case-3, {cost=}")
        # If new group is the best
        if cost < best_cost_delta:
            best_cost_delta = cost
            best_rollout_node = "RN-0"
            best_train_node = "TN"
            best_group = job_group
            self.job_groups[job_group.group_id] = job_group
            self.group_costs[job_group.group_id] = cost
        # If one existing group is the best, then assign this job to the group
        else:
            tmp_job = deepcopy(job)
            tmp_job.rollout_nodes = [best_rollout_node]
            tmp_job.train_nodes = [best_train_node]
            self.job_groups[best_group.group_id].jobs.append(tmp_job)
            self.group_costs[best_group.group_id] += best_cost_delta
        # print(self.group_costs)
        return best_rollout_node, best_train_node, best_group, best_cost_delta

    def remove_job(self, job_id: str) -> None:
        removed = False
        for job_group in self.job_groups:
            for job in job_group.jobs:
                if job.job_id == job_id:
                    job_group.jobs.remove(job)
                    removed = True
                    break
            if removed:
                break
        if not removed:
            print(f"Remove failed: Job {job_id} does not exist.")


if __name__ == "__main__":
    sched = WeaveScheduler(per_time_cost, 3)
    for i in range(5):
        print("="*80)
        best_rollout_node, best_train_node, best_group, best_cost = sched.add_job(Job(f"{i}", 30, 10, 1.1))
        print(i, best_rollout_node, best_train_node, best_group, best_cost)
