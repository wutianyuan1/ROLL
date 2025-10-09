from copy import deepcopy
from typing import List, Dict, Callable
from global_scheduler.structs import Job, JobGroup
from global_scheduler.simulator import WeaveSimulator


class WeaveScheduler:
    def __init__(self, score_func: Callable[[Dict], float], overload_ratio: float,
                 max_group_size: int = 5, simulate_steps: int = 100):
        self.job_groups: Dict[str, JobGroup] = {}
        self.score_func = score_func
        # Tolerate T_meta_iter >= overload_ratio * T1 if it is T2-bound
        self.overload_ratio = overload_ratio
        self.max_group_size = max_group_size
        self.simulate_steps = simulate_steps

    def add_job(self, job: Job):
        best_rollout_node, best_train_node, best_group, best_score = None, None, None, -float("inf")
        for job_group_name, job_group in self.job_groups.items():
            if len(job_group.jobs) > self.max_group_size:
                continue
            all_rollout_nodes, all_train_nodes = job_group.all_rollout_nodes, job_group.all_train_nodes
            original_phase_times = job_group.get_node_phase_times()
            T1, T2 = job_group.T1, job_group.T2
            print(f"{job_group_name=}, {T1=}, {T2=}, {all_rollout_nodes=}, {all_train_nodes=}")
            assert T1 >= self.overload_ratio * T2, f"{T1=}, {T2=}"
            # Case-1: share train, share rollout (direct insert)
            for train_node in all_train_nodes:
                for rollout_node in all_rollout_nodes:
                    tmp_job = deepcopy(job)
                    tmp_job.rollout_nodes = [rollout_node]
                    tmp_job.train_nodes = [train_node]

                    T1_after_insert = max(T1, tmp_job.t_rollout + tmp_job.t_train)
                    T2_after_insert = max(T2,
                                          original_phase_times[rollout_node] + tmp_job.t_rollout,
                                          original_phase_times[train_node] + tmp_job.t_train)
                    print(T1_after_insert, T2_after_insert)
                    if T2_after_insert > self.overload_ratio * T1_after_insert:
                        print(f"Case-1 skip: {rollout_node}, {train_node}, {T2_after_insert}>{self.overload_ratio}*{T1_after_insert}")
                        continue

                    sim = WeaveSimulator(job_group.jobs + [tmp_job], job_group.job_ids + [tmp_job.job_id])
                    _, _, utils = sim.simulate_run(self.simulate_steps)
                    score = self.score_func(utils)
                    print(f"Case-1, {rollout_node=}, {train_node=}, {score=}")
                    if score > best_score:
                        best_score = score
                        best_rollout_node = rollout_node
                        best_train_node = train_node
                        best_group = job_group
            # Case-2: share train, but scale-up to an individual rollout
            for train_node in all_train_nodes:
                rollout_node = f"RN-{len(all_rollout_nodes) + 1}"
                tmp_job = deepcopy(job)
                tmp_job.rollout_nodes = [rollout_node]
                tmp_job.train_nodes = [train_node]
                sim = WeaveSimulator(job_group.jobs + [tmp_job], job_group.job_ids + [tmp_job.job_id])
                _, _, utils = sim.simulate_run(self.simulate_steps)
                score = self.score_func(utils)
                print(f"Case-2, {rollout_node=}, {train_node=}, {score=}")
                if score > best_score:
                    best_score = score
                    best_rollout_node = rollout_node
                    best_train_node = train_node
                    best_group = job_group
        # Case-3: form a new group
        tmp_job = deepcopy(job)
        tmp_job.rollout_nodes = ["RN-1"]
        tmp_job.train_nodes = ["TN-1"]
        job_group = JobGroup(f"Group-{len(self.job_groups) + 1}", [tmp_job])
        sim = WeaveSimulator(job_group.jobs, job_group.job_ids)
        _, _, utils = sim.simulate_run(self.simulate_steps)
        score = self.score_func(utils)
        print(f"Case-3, {score=}")
        # If new group is the best
        if score > best_score:
            best_score = score
            best_rollout_node = "RN-1"
            best_train_node = "TN-1"
            best_group = job_group
            self.job_groups[job_group.group_id] = job_group
        # If one existing group is the best, then assign this job to the group
        else:
            tmp_job = deepcopy(job)
            tmp_job.rollout_nodes = [best_rollout_node]
            tmp_job.train_nodes = [best_train_node]
            self.job_groups[best_group.group_id].jobs.append(tmp_job)
        return best_rollout_node, best_train_node, best_group, best_score

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
