import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from typing import List, Optional, Dict, Tuple
from global_scheduler.structs import Job

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, 'build')
sys.path.insert(0, build_dir)


class WeaveSimulator:
    def __init__(self, jobs: List[Job], meta_iter_cycle: Optional[List[str]] = None):
        self.jobs = jobs
        self.jobid_2_jobs = {job.job_id: job for job in self.jobs}
        self.meta_iter_cycle = meta_iter_cycle
        if meta_iter_cycle is None:
            self.meta_iter_cycle = [job.job_id for job in self.jobs]
        self.all_rollout_nodes = list(set([node for job in self.jobs for node in job.rollout_nodes]))
        self.all_train_nodes = list(set([node for job in self.jobs for node in job.train_nodes]))

        # Convert to cpp objects if the cpp backend is available
        try:
            # raise ValueError()
            import global_scheduler_cpp
            cpp_jobs = []
            for job in jobs:
                cpp_job = global_scheduler_cpp.Job(
                    job.job_id,
                    job.t_rollout,
                    job.t_train,
                    job.rollout_nodes,
                    job.train_nodes
                )
                cpp_jobs.append(cpp_job)
            self.cpp_simulator = global_scheduler_cpp.WeaveSimulator(
                cpp_jobs, self.meta_iter_cycle
            )
            logging.info("[Simulator] Using C++ backend...")
        except:
            self.cpp_simulator = None
            logging.warining("[Simulator] C++ backend is not available, fallback to native python backend.")

    def sim_based_util(self, busy_times: Dict[str, List[Tuple[float, float]]]):
        max_t, min_t = 0, float("inf")
        busy_t = 0
        for job_busy_times in busy_times.values():
            for (t_start, t_end) in job_busy_times:
                min_t = min(min_t, t_start)
                max_t = max(max_t, t_end)
                busy_t += t_end - t_start
        return busy_t / (max_t - min_t)

    def simulate_run(self, n_meta_iters: int):
        if self.cpp_simulator is not None:
            result = self.cpp_simulator.simulate_run(n_meta_iters)
            utils = {
                'rollout': [round(util, 4) for util in result.rollout_utils],
                'train': [round(util, 4) for util in result.train_utils]
            }
            return result.rollout_busy_times, result.train_busy_times, utils
        # Else, fallback to python simulator
        # Busy times, key: cluster, value: Dict[job_id -> List[(t_start_i, t_end_i)]]
        rollout_busy_times = {rollout_node: {} for rollout_node in self.all_rollout_nodes}
        train_busy_times = {train_node: {} for train_node in self.all_train_nodes}
        cycle_len = len(self.meta_iter_cycle)
        for meta_iter in range(n_meta_iters):
            for i, job_id in enumerate(self.meta_iter_cycle):
                job = self.jobid_2_jobs[job_id]
                # first, schedule this job's rollout
                # find the previous job that shares the same rollout cluster
                prev_rollout_end = 0
                for j in range(1, cycle_len + 1):
                    cur = (i + cycle_len - j) % cycle_len
                    cur_rollout_nodes = self.jobid_2_jobs[self.meta_iter_cycle[cur]].rollout_nodes
                    # if cur and job share rollout node, job should wait cur to finish
                    shared_nodes = set(cur_rollout_nodes).intersection(job.rollout_nodes)
                    # we take cur_rollout_nodes[0] to find cur's finish
                    # time since all nodes finish simutaneously.
                    if len(shared_nodes) != 0:
                        cur_end = rollout_busy_times[cur_rollout_nodes[0]].get(
                            self.meta_iter_cycle[cur], [(0, 0)]
                        )[-1][1]
                        prev_rollout_end = max(prev_rollout_end, cur_end)
                # t_rollout_begin = max(its_last_train_end, prev_job_rollout_end)
                t_rollout_begin = max(
                    train_busy_times[job.train_nodes[0]].get(job.job_id, [(0, 0)])[-1][1],
                    prev_rollout_end
                )
                for node in job.rollout_nodes:
                    rollout_busy_times[node].setdefault(job.job_id, []).append(
                        (t_rollout_begin, t_rollout_begin + job.t_rollout)
                    )
                # second, schedule this job's train
                # find the previous job that shares the same train cluster
                prev_train_end = 0
                for j in range(1, cycle_len + 1):
                    cur = (i + cycle_len - j) % cycle_len
                    cur_train_nodes = self.jobid_2_jobs[self.meta_iter_cycle[cur]].train_nodes
                    # if cur and job share rollout node, job should wait cur to finish
                    shared_nodes = set(cur_train_nodes).intersection(job.train_nodes)
                    # we take cur_train_nodes[0] to find cur's finish
                    # time since all nodes finish simutaneously.
                    if len(shared_nodes) != 0:
                        cur_end = train_busy_times[cur_train_nodes[0]].get(
                            self.meta_iter_cycle[cur], [(0, 0)]
                        )[-1][1]
                        prev_train_end = max(prev_train_end, cur_end)
                # t_train_begin = max(its_last_rollout_end, prev_job_train_end)
                t_train_begin = max(
                    rollout_busy_times[job.rollout_nodes[0]].get(job.job_id, [(0, 0)])[-1][1],
                    prev_train_end
                )
                for node in job.train_nodes:
                    train_busy_times[node].setdefault(job.job_id, []).append(
                        (t_train_begin, t_train_begin + job.t_train)
                    )

        utils = {'rollout': [], 'train': []}
        for cluster in rollout_busy_times:
            util = self.sim_based_util(rollout_busy_times[cluster])
            # print(f"Rollout[{cluster}].util = {util * 100:.2f}%")
            utils['rollout'].append(round(util, 4))
        for cluster in train_busy_times:
            util = self.sim_based_util(train_busy_times[cluster])
            # print(f"Train[{cluster}].util = {util * 100:.2f}%")
            utils['train'].append(round(util, 4))
        return rollout_busy_times, train_busy_times, utils

    def plot(self, n_meta_iters: int, export_path: Optional[str] = None):
        rollout_busy_times, train_busy_times, _ = self.simulate_run(n_meta_iters)
        colors = sns.color_palette("Set3")
        jobid_2_colors = {job.job_id: color for (job, color) in zip(self.jobs, colors)}
        ax = plt.gca()
        max_x = 0

        # dummy rect for showing the legend
        for job in self.jobs:
            ax.add_patch(patches.Rectangle((0, 0), 0, 0, edgecolor='black', facecolor=jobid_2_colors[job.job_id], label=job.job_id))

        y = 0
        all_clusters = []
        for train_node in train_busy_times:
            cluster_utils = train_busy_times[train_node]
            for job_id in cluster_utils:
                for (train_start, train_end) in cluster_utils[job_id]:
                    rectangle = patches.Rectangle(
                        (train_start, y), train_end - train_start, 1,
                        edgecolor='black',
                        facecolor=jobid_2_colors[job_id]
                    )
                    max_x = max(max_x, train_end)
                    ax.add_patch(rectangle)
            y += 1
            all_clusters.append(train_node)

        for rollout_node in rollout_busy_times:
            cluster_utils = rollout_busy_times[rollout_node]
            for job_id in cluster_utils:
                for (rollout_start, rollout_end) in cluster_utils[job_id]:
                    rectangle = patches.Rectangle(
                        (rollout_start, y), rollout_end - rollout_start, 1,
                        edgecolor='black',
                        facecolor=jobid_2_colors[job_id]
                    )
                    max_x = max(max_x, train_end)
                    ax.add_patch(rectangle)
            y += 1
            all_clusters.append(rollout_node)

        plt.legend()
        plt.xlim(0, max_x + 5)
        plt.ylim(0, y)
        plt.yticks(np.arange(y) + 0.5, all_clusters)
        if export_path is not None:
            plt.savefig(export_path)
