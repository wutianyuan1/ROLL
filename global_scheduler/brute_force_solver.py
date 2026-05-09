import sys
import os
import logging
from typing import List, Callable, Dict
from copy import deepcopy
from more_itertools import distinct_permutations
from global_scheduler.structs import Job
from global_scheduler.simulator import WeaveSimulator

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, 'build')
sys.path.insert(0, build_dir)


class IntraGroupSolver:
    def __init__(self, jobs: List[Job], sim_steps: int = 1000):
        self.jobs = jobs
        self.sim_steps = sim_steps
        try:
            import global_scheduler_cpp
            cpp_jobs = []
            for job in jobs:
                cpp_job = global_scheduler_cpp.Job(
                    job.job_id,
                    job.t_rollout,
                    job.t_train,
                    job.rollout_nodes or [],
                    job.train_nodes or []
                )
                cpp_jobs.append(cpp_job)
            self.cpp_solver = global_scheduler_cpp.IntraGroupSolver(cpp_jobs, sim_steps)
            logging.info("[IntraGroupSolver] Using C++ backend...")
        except:
            self.cpp_solver = None
            logging.warning("[IntraGroupSolver] C++ backend is not available, fallback to native python backend.")

    def _generate_deployment(self):
        # We assume t_rollout >= t_train, thus we only consider independent
        # rollout (i.e., N rollout and only 1 train)
        # Also, assume no job will simultaneously use multiple rollout nodes
        def gen_partitions(jobs):
            # Find all possible ways to partition jobs into rollout nodes
            if len(jobs) == 0:
                return [[]]
            first, *rest = jobs
            partitions = gen_partitions(rest)
            result = []
            for part in partitions:
                # Put `first` in a new subset
                result.append([[first]] + part)
                # Put `first` into each existing subset
                for i in range(len(part)):
                    new_part = [subset[:] for subset in part]
                    new_part[i].append(first)
                    result.append(new_part)
            return result

        all_parts = gen_partitions(list(range(len(self.jobs))))
        # Normalize: sort each subset and sort list of subsets
        normalized = []
        seen = set()
        for part in all_parts:
            sorted_boxes = [sorted(box) for box in part]
            sorted_boxes.sort()  # sort boxes lexicographically
            key = tuple(tuple(box) for box in sorted_boxes)
            if key not in seen:
                seen.add(key)
                normalized.append(sorted_boxes)
        
        return normalized

    def _convert_cpp_jobs_to_python(self, cpp_jobs):
        python_jobs = []
        for cpp_job in cpp_jobs:
            python_job = type('Job', (), {})()
            python_job.job_id = cpp_job.job_id
            python_job.t_rollout = cpp_job.t_rollout
            python_job.t_train = cpp_job.t_train
            python_job.rollout_nodes = cpp_job.rollout_nodes
            python_job.train_nodes = cpp_job.train_nodes
            python_jobs.append(python_job)
        return python_jobs

    def solve(self, max_meta_iter_len: int, score_func: Callable[[Dict], float]):
        if self.cpp_solver is not None:
            def cpp_score_func(rollout_utils: List[float], train_utils: List[float]) -> float:
                utils_dict = {'rollout': rollout_utils, 'train': train_utils}
                return score_func(utils_dict)
            result = self.cpp_solver.solve(max_meta_iter_len, cpp_score_func)
            best_strategy = (
                result.partition,
                self._convert_cpp_jobs_to_python(result.job_deployment),
                result.meta_iteration
            )
            return result.score, best_strategy
        all_partitions = self._generate_deployment()
        idx_2_job_id = [job.job_id for job in self.jobs]

        def assemble_jobs(partition: List[List[int]]):
            ret_jobs = deepcopy(self.jobs)
            for rollout_node_id, group in enumerate(partition):
                for job_id in group:
                    assert ret_jobs[job_id].rollout_nodes is None
                    ret_jobs[job_id].rollout_nodes = [f'RN-{rollout_node_id}']
                    ret_jobs[job_id].train_nodes = ['TN']
            return ret_jobs

        def generate_compositions(n: int, k: int, min_val: int):
            # Genetate all ways to a number n into k parts [n_1,...,n_k],
            # subject to sum_{j=1}^{k}n_j = n and \forall j, n_j >= min_val
            if k == 1:
                if n >= min_val:
                    yield [n]
                return
            for i in range(min_val, n - min_val * (k - 1) + 1):
                for rest in generate_compositions(n - i, k - 1, min_val):
                    yield [i] + rest
        
        def generate_meta_iteration_all():
            assert max_meta_iter_len >= len(self.jobs)
            all_combs = []
            for meta_iter_len in range(len(self.jobs), max_meta_iter_len + 1):
                for composition in generate_compositions(meta_iter_len, len(self.jobs), 1):
                    meta_iter_comb = []
                    for i in range(len(composition)):
                        meta_iter_comb += [idx_2_job_id[i]] * composition[i]
                    all_combs.append(meta_iter_comb)
            all_meta_iters = []
            for job_comb in all_combs:
                all_meta_iters += list(distinct_permutations(job_comb))
            return all_meta_iters

        meta_iterations = generate_meta_iteration_all()
        max_score, best_strategy = -float("inf"), None
        print(f"Generated {len(all_partitions)} partitions and {len(meta_iterations)} meta iterations")
        for partition in all_partitions:
            job_deployment = assemble_jobs(partition)
            for meta_iter in meta_iterations:
                sim = WeaveSimulator(job_deployment, meta_iter)
                _, _, utils = sim.simulate_run(self.sim_steps)
                score = score_func(utils)
                if score > max_score:
                    max_score = score
                    best_strategy = (partition, job_deployment, meta_iter)
        return max_score, best_strategy
