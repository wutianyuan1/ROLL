import numpy as np
from typing import List, Tuple, Set
from tqdm import tqdm
from copy import deepcopy
from global_scheduler.structs import Job, JobGroup
from global_scheduler.new_simulator import WeaveSimulator


class BruteForceSolver:
    def __init__(self, jobs: List[Job], max_group_size: int, n_iters: int = 100,
                 rollout_cost: float = 1/3, train_cost: float = 1):
        self.jobs = jobs
        self.max_group_size = max_group_size
        self.n_iters = n_iters
        self.rollout_cost = rollout_cost
        self.train_cost = train_cost

    def build_all_groups(self, job_ids: List[int]) -> List[Tuple[Tuple[int, ...], ...]]:
        job_ids_tuple = tuple(sorted(job_ids))
        return self._build_partitions_recursive(job_ids_tuple)

    def _build_partitions_recursive(self, job_ids: Tuple[int, ...]) -> List[Tuple[Tuple[int, ...], ...]]:
        if not job_ids:
            return [()]
        first_job, rest_jobs = job_ids[0], job_ids[1:]
        sub_partitions = self._build_partitions_recursive(rest_jobs)
        all_new_partitions: Set[Tuple[Tuple[int, ...], ...]] = set()
        for partition in sub_partitions:
            for i, group in enumerate(partition):
                if len(group) < self.max_group_size:
                    new_group = tuple(sorted(group + (first_job,)))
                    new_partition = partition[:i] + (new_group,) + partition[i+1:]
                    all_new_partitions.add(tuple(sorted(new_partition)))
            new_group = (first_job,)
            new_partition = partition + (new_group,)
            all_new_partitions.add(tuple(sorted(new_partition)))
        return sorted(list(all_new_partitions))

    def generate_job_placements(self, gid: int, job_group: List[int]) -> List[Tuple[int, JobGroup]]:
        train_node_name = 'TN'
        all_placements = self.build_all_groups(job_group)
        ret = []
        for pid, placement in enumerate(all_placements):
            p = []
            for rollout_id, shared_jobs in enumerate(placement):
                for jid in shared_jobs:
                    tmp = deepcopy(self.jobs[jid])
                    tmp.rollout_nodes = [str(rollout_id)]
                    tmp.train_nodes = [train_node_name]
                    p.append(tmp)
            ret.append((len(placement), JobGroup(f"{gid}_p{pid}", p)))
        return ret

    def solve_best_placement(self, gid: int, job_group: List[int]) -> None:
        all_placements = self.generate_job_placements(gid, job_group)
        best_cost, best_group = float("inf"), None
        for pid, (num_rollout_nodes, group) in enumerate(all_placements):
            sim = WeaveSimulator(group.jobs)
            rollout_busy_times, train_busy_times, utils, total_time = sim.simulate_run(self.n_iters)
            valid = True
            total_working_time = 0
            for job in group.jobs:
                num_iters = len(train_busy_times['TN'][job.job_id])
                if total_time / num_iters >= job.slo * (job.t_rollout + job.t_train):
                    valid = False
                total_working_time += num_iters * (job.t_rollout + job.t_train)
            # cost_per_time = total_time * (1 * self.train_cost + num_rollout_nodes * self.rollout_cost) / total_working_time
            cost_per_time = 1 * self.train_cost + num_rollout_nodes * self.rollout_cost
            # print(num_rollout_nodes, [(i.job_id, i.rollout_nodes) for i in group.jobs], total_time, cost_per_time, valid, (1 * self.train_cost + num_rollout_nodes * self.rollout_cost))
            if valid and cost_per_time < best_cost:
                best_cost = cost_per_time
                best_group = group
        return best_cost, best_group

    def solve_brute_force(self):
        all_possible_partitions = self.build_all_groups([i for i in range(len(self.jobs))])
        best_partition_cost, best_partition, best_groups = float("inf"), None, None
        for partition in tqdm(all_possible_partitions):
            valid = True
            all_groups_cost = 0
            partition_groups = []
            for gid, group in enumerate(partition):
                best_cost, best_group = self.solve_best_placement(gid, group)
                # If all placements are invalid for this group, then this partitioning is invalid, skip it
                if best_group is None:
                    valid = False
                    break
                all_groups_cost += best_cost
                partition_groups.append(best_group)
            # print(partition, valid, all_groups_cost, best_partition_cost)
            if valid and all_groups_cost < best_partition_cost:
                best_partition_cost = all_groups_cost
                best_partition = partition
                best_groups = partition_groups
        return best_partition_cost, best_partition, best_groups


    def get_random_partitions(self, job_ids: List[int], num: int) -> List[Tuple[Tuple[int, ...], ...]]:
        partitions_set: Set[Tuple[Tuple[int, ...], ...]] = set()
        max_attempts = num * 10 
        attempts = 0
        job_ids_tuple = tuple(job_ids)

        while len(partitions_set) < num and attempts < max_attempts:
            partition = self._generate_one_random_partition(job_ids_tuple)
            partitions_set.add(partition)
            attempts += 1

        if len(partitions_set) < num:
            print(f"Warning: Could only generate {len(partitions_set)} unique partitions after {max_attempts} attempts. "
                  f"The requested number was {num}. This might happen if 'num' is very large "
                  "or constraints are very tight.")

        return list(partitions_set)

    def _generate_one_random_partition(self, job_ids: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
        if not job_ids:
            return ()
        shuffled_jobs = list(job_ids)
        np.random.shuffle(shuffled_jobs)
        groups: List[List[int]] = []
        for job_id in shuffled_jobs:
            possible_placements = []
            for i, group in enumerate(groups):
                if len(group) < self.max_group_size:
                    possible_placements.append(i)
            possible_placements.append(-1)
            chosen_placement = np.random.choice(possible_placements)
            if chosen_placement == -1:
                groups.append([job_id])
            else:
                groups[chosen_placement].append(job_id)
        canonical_partition = tuple(sorted([tuple(sorted(group)) for group in groups]))
        return canonical_partition

    def solve(self, max_search_steps=20000):
        if len(self.jobs) >= 10:
            all_possible_partitions = self.get_random_partitions(
                [i for i in range(len(self.jobs))], num=max_search_steps
            )
            print(f"[N={len(self.jobs)}] Using random sampler to generate {max_search_steps} partitions, generated={len(all_possible_partitions)}")
        else:
            all_possible_partitions = self.build_all_groups(
                [i for i in range(len(self.jobs))]
            )
            print(f"[N={len(self.jobs)}] Using brute-force to enumerate all partitions, all={len(all_possible_partitions)}")
        best_partition_cost, best_partition, best_groups = float("inf"), None, None
        for partition in tqdm(all_possible_partitions):
            valid = True
            all_groups_cost = 0
            partition_groups = []
            for gid, group in enumerate(partition):
                best_cost, best_group = self.solve_best_placement(gid, group)
                # If all placements are invalid for this group, then this partitioning is invalid, skip it
                if best_group is None:
                    valid = False
                    break
                all_groups_cost += best_cost
                partition_groups.append(best_group)
            # print(partition, valid, all_groups_cost, best_partition_cost)
            if valid and all_groups_cost < best_partition_cost:
                best_partition_cost = all_groups_cost
                best_partition = partition
                best_groups = partition_groups
        return best_partition_cost, best_partition, best_groups

if __name__ == '__main__':
    jobs, max_size = list(range(5)), 3
    jobs = [
        Job(str(i), 20, 10, slo=1.01) for i in range(8)
    ]
    solver = BruteForceSolver(jobs, max_size, n_iters=100)
    best_partition_cost, best_partition, best_groups = solver.solve()
    print(best_partition_cost, best_partition)
    for group in best_groups:
        print(group.group_id, [(i.job_id, i.rollout_nodes) for i in group.jobs])
