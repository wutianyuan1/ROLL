import numpy as np
from itertools import combinations
from typing import List, Tuple, Set
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
        self.group_solution_cache = {}
        self.partition_eval_cache = {}
        self.group_catalog_cache = None

    def build_all_groups(self, job_ids: List[int]) -> List[Tuple[Tuple[int, ...], ...]]:
        job_ids_tuple = tuple(sorted(job_ids))
        return self._build_partitions_recursive(job_ids_tuple)

    def canonicalize_partition(self, partition: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
        return tuple(sorted(tuple(sorted(group)) for group in partition if len(group) != 0))

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
        all_placements = self.build_all_groups(job_group)
        ret = []
        for pid, placement in enumerate(all_placements):
            p = []
            pool_train_width = max(self.jobs[jid].train_width for shared_jobs in placement for jid in shared_jobs)
            train_nodes = [f'TN{i}' for i in range(pool_train_width)]
            rollout_node_cursor = 0
            for shared_jobs in placement:
                width2_jobs = [jid for jid in shared_jobs if self.jobs[jid].rollout_width > 1]
                shared_rollout_node = str(rollout_node_cursor)
                rollout_node_cursor += 1
                extra_nodes = {}
                for jid in width2_jobs:
                    extra_nodes[jid] = str(rollout_node_cursor)
                    rollout_node_cursor += 1
                for jid in shared_jobs:
                    tmp = deepcopy(self.jobs[jid])
                    if tmp.rollout_width > 1:
                        tmp.rollout_nodes = [shared_rollout_node, extra_nodes[jid]]
                    else:
                        tmp.rollout_nodes = [shared_rollout_node]
                    tmp.train_nodes = list(train_nodes)
                    p.append(tmp)
            ret.append((rollout_node_cursor, JobGroup(f"{gid}_p{pid}", p)))
        return ret

    def solve_best_placement(self, gid: int, job_group: List[int]) -> None:
        cache_key = tuple(sorted(job_group))
        cached = self.group_solution_cache.get(cache_key)
        if cached is not None:
            best_cost, best_group = cached
            return best_cost, deepcopy(best_group) if best_group is not None else None
        all_placements = self.generate_job_placements(gid, job_group)
        best_cost, best_group = float("inf"), None
        for pid, (num_rollout_nodes, group) in enumerate(all_placements):
            sim = WeaveSimulator(group.jobs)
            rollout_busy_times, train_busy_times, utils, total_time = sim.simulate_run(self.n_iters)
            valid = True
            total_working_time = 0
            for job in group.jobs:
                train_node = job.train_nodes[0]
                num_iters = len(train_busy_times[train_node][job.job_id])
                if total_time / num_iters >= job.slo * (job.t_rollout + job.t_train):
                    valid = False
                total_working_time += num_iters * (job.t_rollout + job.t_train)
            # cost_per_time = total_time * (1 * self.train_cost + num_rollout_nodes * self.rollout_cost) / total_working_time
            cost_per_time = len(group.all_train_nodes) * self.train_cost + num_rollout_nodes * self.rollout_cost
            # print(num_rollout_nodes, [(i.job_id, i.rollout_nodes) for i in group.jobs], total_time, cost_per_time, valid, (1 * self.train_cost + num_rollout_nodes * self.rollout_cost))
            if valid and cost_per_time < best_cost:
                best_cost = cost_per_time
                best_group = group
        self.group_solution_cache[cache_key] = (best_cost, deepcopy(best_group) if best_group is not None else None)
        return best_cost, deepcopy(best_group) if best_group is not None else None

    def _build_group_catalog(self):
        if self.group_catalog_cache is not None:
            return self.group_catalog_cache

        job_ids = list(range(len(self.jobs)))
        singleton_costs = {}
        group_catalog = []
        anchor_to_groups = {jid: [] for jid in job_ids}

        for jid in job_ids:
            cost, best_group = self.solve_best_placement(0, [jid])
            if best_group is None:
                continue
            singleton_costs[jid] = cost

        for size in range(1, self.max_group_size + 1):
            for combo in combinations(job_ids, size):
                if any(jid not in singleton_costs for jid in combo):
                    continue
                cost, best_group = self.solve_best_placement(0, list(combo))
                if best_group is None:
                    continue
                baseline = sum(singleton_costs[jid] for jid in combo)
                savings = baseline - cost
                record = {
                    'members': combo,
                    'cost': cost,
                    'savings': savings,
                    'size': size,
                    'savings_per_job': savings / size,
                }
                group_catalog.append(record)
                for jid in combo:
                    anchor_to_groups[jid].append(record)

        for jid, records in anchor_to_groups.items():
            anchor_to_groups[jid] = sorted(
                records,
                key=lambda rec: (
                    -rec['savings_per_job'],
                    -rec['savings'],
                    -rec['size'],
                    rec['cost'],
                    rec['members'],
                ),
            )

        self.group_catalog_cache = (singleton_costs, group_catalog, anchor_to_groups)
        return self.group_catalog_cache

    def _singleton_partition(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple((jid,) for jid in range(len(self.jobs)))

    def _greedy_complete_cost(self, remaining_jobs: Tuple[int, ...], singleton_costs: dict,
                              anchor_to_groups: dict, max_groups_per_anchor: int) -> float:
        remaining = set(remaining_jobs)
        total_cost = 0.0
        while remaining:
            anchor = min(remaining)
            chosen = None
            for record in anchor_to_groups.get(anchor, [])[:max_groups_per_anchor]:
                members = record['members']
                if set(members).issubset(remaining):
                    chosen = record
                    break
            if chosen is None:
                total_cost += singleton_costs[anchor]
                remaining.remove(anchor)
            else:
                total_cost += chosen['cost']
                for jid in chosen['members']:
                    remaining.remove(jid)
        return total_cost

    def _build_greedy_partition(self, sort_mode: str = 'savings') -> Tuple[Tuple[int, ...], ...]:
        singleton_costs, group_catalog, _ = self._build_group_catalog()
        remaining = set(range(len(self.jobs)))
        partition = []

        if sort_mode == 'savings':
            sorted_groups = sorted(
                group_catalog,
                key=lambda rec: (-rec['savings'], -rec['size'], rec['cost'], rec['members']),
            )
        elif sort_mode == 'density':
            sorted_groups = sorted(
                group_catalog,
                key=lambda rec: (-rec['savings_per_job'], -rec['savings'], -rec['size'], rec['cost'], rec['members']),
            )
        elif sort_mode == 'size':
            sorted_groups = sorted(
                group_catalog,
                key=lambda rec: (-rec['size'], -rec['savings'], rec['cost'], rec['members']),
            )
        else:
            raise ValueError(f'Unsupported sort_mode: {sort_mode}')

        for record in sorted_groups:
            members = set(record['members'])
            if not members.issubset(remaining):
                continue
            if len(record['members']) == 1:
                continue
            if record['savings'] <= 0:
                continue
            partition.append(record['members'])
            remaining.difference_update(members)

        for jid in sorted(remaining):
            if jid in singleton_costs:
                partition.append((jid,))
        return self.canonicalize_partition(tuple(partition))

    def _build_beam_partitions(self, beam_width: int = 64, max_groups_per_anchor: int = 12,
                               max_outputs: int = 64) -> List[Tuple[Tuple[int, ...], ...]]:
        singleton_costs, _, anchor_to_groups = self._build_group_catalog()
        all_jobs = tuple(range(len(self.jobs)))
        if any(jid not in singleton_costs for jid in all_jobs):
            return []

        start_partition = ()
        start_bound = self._greedy_complete_cost(all_jobs, singleton_costs, anchor_to_groups, max_groups_per_anchor)
        beam = [(start_bound, 0.0, start_partition, all_jobs)]
        completed = []
        seen_states = set()

        while beam and len(completed) < max_outputs:
            next_beam = []
            for _, cost_so_far, partition, remaining_jobs in beam:
                if not remaining_jobs:
                    completed.append(self.canonicalize_partition(partition))
                    if len(completed) >= max_outputs:
                        break
                    continue

                anchor = remaining_jobs[0]
                remaining_set = set(remaining_jobs)
                branch_records = []
                for record in anchor_to_groups.get(anchor, [])[:max_groups_per_anchor]:
                    members = record['members']
                    if set(members).issubset(remaining_set):
                        branch_records.append(record)
                if not branch_records:
                    branch_records.append({
                        'members': (anchor,),
                        'cost': singleton_costs[anchor],
                    })

                for record in branch_records:
                    members = record['members']
                    next_remaining = tuple(jid for jid in remaining_jobs if jid not in members)
                    next_partition = self.canonicalize_partition(partition + (members,))
                    state_key = (next_partition, next_remaining)
                    if state_key in seen_states:
                        continue
                    seen_states.add(state_key)
                    next_cost = cost_so_far + record['cost']
                    heuristic_tail = self._greedy_complete_cost(
                        next_remaining, singleton_costs, anchor_to_groups, max_groups_per_anchor
                    )
                    next_score = next_cost + heuristic_tail
                    next_beam.append((next_score, next_cost, next_partition, next_remaining))

            if not next_beam:
                break
            next_beam.sort(key=lambda item: (item[0], item[1], len(item[3]), item[2]))
            beam = next_beam[:beam_width]

        return completed

    def get_heuristic_partitions(self, max_count: int,
                                 seed_partitions: List[Tuple[Tuple[int, ...], ...]] = None) -> List[Tuple[Tuple[int, ...], ...]]:
        seed_partitions = [self.canonicalize_partition(partition) for partition in (seed_partitions or [])]
        partitions_set: Set[Tuple[Tuple[int, ...], ...]] = set(seed_partitions)
        partitions_set.add(self._singleton_partition())

        for sort_mode in ['savings', 'density', 'size']:
            partitions_set.add(self._build_greedy_partition(sort_mode=sort_mode))
            if len(partitions_set) >= max_count:
                return list(partitions_set)

        beam_width = max(16, min(96, max_count // 8 if max_count > 0 else 16))
        max_groups_per_anchor = max(6, min(24, max_count // 16 if max_count > 0 else 6))
        beam_partitions = self._build_beam_partitions(
            beam_width=beam_width,
            max_groups_per_anchor=max_groups_per_anchor,
            max_outputs=max(16, min(max_count, beam_width)),
        )
        for partition in beam_partitions:
            partitions_set.add(partition)
            if len(partitions_set) >= max_count:
                break
        return list(partitions_set)

    def solve_brute_force(self, verbose: bool = True):
        all_possible_partitions = self.build_all_groups([i for i in range(len(self.jobs))])
        best_partition_cost, best_partition, best_groups = float("inf"), None, None
        if verbose:
            print(f"[solver] enumerating all {len(all_possible_partitions)} partitions for N={len(self.jobs)}")
        for partition in all_possible_partitions:
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
            partitions_set.add(self.canonicalize_partition(partition))
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
        canonical_partition = self.canonicalize_partition(tuple(tuple(group) for group in groups))
        return canonical_partition

    def get_neighbor_partitions(self, partition: Tuple[Tuple[int, ...], ...]) -> List[Tuple[Tuple[int, ...], ...]]:
        partition = self.canonicalize_partition(partition)
        neighbors: Set[Tuple[Tuple[int, ...], ...]] = set()

        groups = [list(group) for group in partition]
        for src_idx, src_group in enumerate(groups):
            for job_id in list(src_group):
                for dst_idx in range(len(groups)):
                    if dst_idx == src_idx or len(groups[dst_idx]) >= self.max_group_size:
                        continue
                    new_groups = [list(group) for group in groups]
                    new_groups[src_idx].remove(job_id)
                    new_groups[dst_idx].append(job_id)
                    neighbors.add(self.canonicalize_partition(tuple(tuple(group) for group in new_groups)))
                new_groups = [list(group) for group in groups]
                new_groups[src_idx].remove(job_id)
                new_groups.append([job_id])
                neighbors.add(self.canonicalize_partition(tuple(tuple(group) for group in new_groups)))

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                for left in groups[i]:
                    for right in groups[j]:
                        new_groups = [list(group) for group in groups]
                        new_groups[i].remove(left)
                        new_groups[j].remove(right)
                        new_groups[i].append(right)
                        new_groups[j].append(left)
                        neighbors.add(self.canonicalize_partition(tuple(tuple(group) for group in new_groups)))
        neighbors.discard(partition)
        return sorted(neighbors)

    def evaluate_partition(self, partition: Tuple[Tuple[int, ...], ...]):
        partition = self.canonicalize_partition(partition)
        cached = self.partition_eval_cache.get(partition)
        if cached is not None:
            cost, groups = cached
            return cost, deepcopy(groups) if groups is not None else None

        valid = True
        all_groups_cost = 0
        partition_groups = []
        for gid, group in enumerate(partition):
            best_cost, best_group = self.solve_best_placement(gid, group)
            if best_group is None:
                valid = False
                break
            all_groups_cost += best_cost
            partition_groups.append(best_group)
        if not valid:
            self.partition_eval_cache[partition] = (float("inf"), None)
            return float("inf"), None
        self.partition_eval_cache[partition] = (all_groups_cost, deepcopy(partition_groups))
        return all_groups_cost, deepcopy(partition_groups)

    def greedy_improve_partition(self, start_partition: Tuple[Tuple[int, ...], ...],
                                 max_rounds: int = 8, max_neighbors: int = 256):
        current_partition = self.canonicalize_partition(start_partition)
        current_cost, current_groups = self.evaluate_partition(current_partition)
        if current_groups is None:
            return current_cost, current_partition, current_groups

        for _ in range(max_rounds):
            neighbors = self.get_neighbor_partitions(current_partition)
            if len(neighbors) > max_neighbors:
                np.random.shuffle(neighbors)
                neighbors = neighbors[:max_neighbors]
            best_neighbor_partition = current_partition
            best_neighbor_cost = current_cost
            best_neighbor_groups = current_groups
            for neighbor in neighbors:
                neighbor_cost, neighbor_groups = self.evaluate_partition(neighbor)
                if neighbor_groups is None:
                    continue
                if neighbor_cost + 1e-9 < best_neighbor_cost:
                    best_neighbor_partition = neighbor
                    best_neighbor_cost = neighbor_cost
                    best_neighbor_groups = neighbor_groups
            if best_neighbor_partition == current_partition:
                break
            current_partition = best_neighbor_partition
            current_cost = best_neighbor_cost
            current_groups = best_neighbor_groups
        return current_cost, current_partition, current_groups

    def solve(self, max_search_steps=20000, force_enum_all=False,
              seed_partitions: List[Tuple[Tuple[int, ...], ...]] = None,
              verbose: bool = True):
        seed_partitions = [self.canonicalize_partition(partition) for partition in (seed_partitions or [])]
        if force_enum_all:
            all_possible_partitions = self.build_all_groups(
                [i for i in range(len(self.jobs))]
            )
            if verbose:
                print(f"[solver] N={len(self.jobs)} using brute-force, candidates={len(all_possible_partitions)}")
        else:
            heuristic_budget = max(16, min(max_search_steps, max_search_steps // 2))
            heuristic_partitions = self.get_heuristic_partitions(
                heuristic_budget,
                seed_partitions=seed_partitions,
            )
            if len(self.jobs) >= 10:
                partitions_set: Set[Tuple[Tuple[int, ...], ...]] = set(heuristic_partitions)
                for seed_partition in seed_partitions:
                    for neighbor in self.get_neighbor_partitions(seed_partition):
                        partitions_set.add(neighbor)
                        if len(partitions_set) >= max_search_steps:
                            break
                    if len(partitions_set) >= max_search_steps:
                        break
                if len(partitions_set) < max_search_steps:
                    random_partitions = self.get_random_partitions(
                        [i for i in range(len(self.jobs))], num=max_search_steps - len(partitions_set)
                    )
                    partitions_set.update(random_partitions)
                all_possible_partitions = list(partitions_set)
                if verbose:
                    print(f"[solver] N={len(self.jobs)} using heuristic+sampled search, budget={max_search_steps}, heuristic={len(heuristic_partitions)}, candidates={len(all_possible_partitions)}")
            else:
                all_possible_partitions = self.build_all_groups(
                    [i for i in range(len(self.jobs))]
                )
                if len(all_possible_partitions) <= max_search_steps * 2:
                    if verbose:
                        print(f"[solver] N={len(self.jobs)} using brute-force, candidates={len(all_possible_partitions)}")
                else:
                    partitions_set: Set[Tuple[Tuple[int, ...], ...]] = set(heuristic_partitions)
                    for seed_partition in seed_partitions:
                        for neighbor in self.get_neighbor_partitions(seed_partition):
                            partitions_set.add(neighbor)
                            if len(partitions_set) >= max_search_steps:
                                break
                        if len(partitions_set) >= max_search_steps:
                            break
                    if len(partitions_set) < max_search_steps:
                        partitions_set.update(self.get_random_partitions(
                            [i for i in range(len(self.jobs))], num=max_search_steps - len(partitions_set)
                        ))
                    all_possible_partitions = list(partitions_set)
                    if verbose:
                        print(f"[solver] N={len(self.jobs)} using heuristic+sampled search, budget={max_search_steps}, heuristic={len(heuristic_partitions)}, candidates={len(all_possible_partitions)}")
        best_partition_cost, best_partition, best_groups = float("inf"), None, None
        ranked_candidates = []
        for partition in all_possible_partitions:
            partition_cost, partition_groups = self.evaluate_partition(partition)
            if partition_groups is None:
                continue
            ranked_candidates.append((partition_cost, partition))
            if partition_cost < best_partition_cost:
                best_partition_cost = partition_cost
                best_partition = partition
                best_groups = partition_groups

        if not force_enum_all and ranked_candidates:
            ranked_candidates.sort(key=lambda item: item[0])
            local_search_starts = [partition for _, partition in ranked_candidates[:min(8, len(ranked_candidates))]]
            for seed_partition in seed_partitions:
                if seed_partition not in local_search_starts:
                    local_search_starts.append(seed_partition)
            for partition in local_search_starts:
                improved_cost, improved_partition, improved_groups = self.greedy_improve_partition(partition)
                if improved_groups is not None and improved_cost < best_partition_cost:
                    best_partition_cost = improved_cost
                    best_partition = improved_partition
                    best_groups = improved_groups
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
