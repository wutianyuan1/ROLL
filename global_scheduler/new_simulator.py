import heapq
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from global_scheduler.structs import Job


class WeaveSimulator:
    """
    Drift-aware discrete-event simulator for disaggregated RL jobs.

    Existing FCFS scheduling semantics are preserved:
    - rollout -> train -> rollout cycles
    - rollout can overlap when rollout nodes do not conflict
    - training is serialized on the shared training pool
    - ready jobs are served FCFS

    The simulator now supports:
    - per-iteration rollout drift via Job.rollout_step_trace
    - proportional train-time drift
    - resumable execution via run_until(...)
    - cancellation/restart hooks for regrouping experiments
    """

    def __init__(self, jobs: List[Job]):
        self.jobs_map: Dict[str, Job] = {job.job_id: job for job in jobs}
        self.all_jobs_map: Dict[str, Job] = {job.job_id: job for job in jobs}
        self.initial_job_order: List[str] = [job.job_id for job in jobs]
        self.current_time: float = 0.0
        self.event_queue: List[Tuple[float, str, str]] = []
        self.rollout_ready_queue: deque = deque()
        self.train_ready_queue: deque = deque()
        self.rollout_node_status: Dict[str, Optional[str]] = {}
        self.train_nodes_busy: bool = False
        self.active_train_job_id: Optional[str] = None
        self.rollout_busy_times: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        self.train_busy_times: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        self._bootstrapped = False
        self._refresh_topology()

    def _refresh_topology(self):
        self.all_rollout_nodes = sorted(list(set(
            node for job in self.jobs_map.values() for node in (job.rollout_nodes or [])
        )))
        self.all_train_nodes = sorted(list(set(
            node for job in self.jobs_map.values() for node in (job.train_nodes or [])
        )))

    def _ensure_busy_time_maps(self):
        self.rollout_busy_times = {
            node: {job_id: [] for job_id in self.all_jobs_map}
            for node in self.all_rollout_nodes
        }
        self.train_busy_times = {
            node: {job_id: [] for job_id in self.all_jobs_map}
            for node in self.all_train_nodes
        }

    def _ensure_job_history_slots(self, job_id: str):
        for node in self.rollout_busy_times:
            self.rollout_busy_times[node].setdefault(job_id, [])
        for node in self.train_busy_times:
            self.train_busy_times[node].setdefault(job_id, [])

    def _ensure_node_history_slots(self):
        for node in self.all_rollout_nodes:
            self.rollout_busy_times.setdefault(node, {})
            for job_id in self.all_jobs_map:
                self.rollout_busy_times[node].setdefault(job_id, [])
        for node in self.all_train_nodes:
            self.train_busy_times.setdefault(node, {})
            for job_id in self.all_jobs_map:
                self.train_busy_times[node].setdefault(job_id, [])

    def _set_idle_resources(self):
        self.rollout_node_status = {node: None for node in self.all_rollout_nodes}
        self.train_nodes_busy = False
        self.active_train_job_id = None

    def reset_state(self, start_time: float = 0.0):
        self.current_time = start_time
        self.event_queue = []
        self.rollout_ready_queue = deque()
        self.train_ready_queue = deque()
        self._set_idle_resources()
        for job in self.jobs_map.values():
            job.reset_runtime_state()
        self._ensure_busy_time_maps()
        self._bootstrapped = False

    def bootstrap(self, start_time: Optional[float] = None, reset_runtime: bool = False,
                  queue_from_state: bool = True):
        if start_time is not None:
            self.current_time = start_time
        self.event_queue = []
        self.rollout_ready_queue = deque()
        self.train_ready_queue = deque()
        self._set_idle_resources()
        if reset_runtime:
            for job in self.jobs_map.values():
                job.reset_runtime_state()
            self._ensure_busy_time_maps()
        elif not self.rollout_busy_times and not self.train_busy_times:
            self._ensure_busy_time_maps()

        if queue_from_state:
            for job_id in self.initial_job_order:
                job = self.jobs_map[job_id]
                if reset_runtime:
                    self.rollout_ready_queue.append(job_id)
                    continue
                if job.next_phase == "rollout":
                    self.rollout_ready_queue.append(job_id)
                elif job.next_phase == "train":
                    self.train_ready_queue.append(job_id)
        self._bootstrapped = True
        self._schedule_pending_jobs()

    def replace_jobs(self, jobs: List[Job], start_time: Optional[float] = None,
                     reset_runtime: bool = False, restart_incomplete: bool = False):
        self.jobs_map = {job.job_id: job for job in jobs}
        self.all_jobs_map = {job.job_id: job for job in jobs}
        self.initial_job_order = [job.job_id for job in jobs]
        self._refresh_topology()
        if reset_runtime:
            for job in jobs:
                job.reset_runtime_state()
        elif restart_incomplete:
            for job in jobs:
                job.restart_incomplete_iteration()
        self.rollout_busy_times = {}
        self.train_busy_times = {}
        self.bootstrap(start_time=self.current_time if start_time is None else start_time,
                       reset_runtime=reset_runtime,
                       queue_from_state=True)

    def _remove_job_from_queues_and_events(self, job_id: str):
        self.rollout_ready_queue = deque(jid for jid in self.rollout_ready_queue if jid != job_id)
        self.train_ready_queue = deque(jid for jid in self.train_ready_queue if jid != job_id)
        self.event_queue = [(t, etype, jid) for (t, etype, jid) in self.event_queue if jid != job_id]
        heapq.heapify(self.event_queue)

    def _truncate_job_inflight(self, job_id: str):
        for node, active_job in list(self.rollout_node_status.items()):
            if active_job != job_id:
                continue
            intervals = self.rollout_busy_times[node][job_id]
            if intervals:
                start, _ = intervals[-1]
                intervals[-1] = (start, self.current_time)
            self.rollout_node_status[node] = None
        if self.active_train_job_id == job_id:
            for node in self.all_train_nodes:
                intervals = self.train_busy_times[node][job_id]
                if intervals:
                    start, _ = intervals[-1]
                    intervals[-1] = (start, self.current_time)
            self.train_nodes_busy = False
            self.active_train_job_id = None

    def fail_job(self, job_id: str, restart_incomplete: bool = True) -> bool:
        if job_id not in self.jobs_map:
            return False
        job = self.jobs_map.pop(job_id)
        self._truncate_job_inflight(job_id)
        self._remove_job_from_queues_and_events(job_id)
        self.initial_job_order = [jid for jid in self.initial_job_order if jid != job_id]
        if restart_incomplete and job.next_phase != "done":
            job.restart_incomplete_iteration()
        self._refresh_topology()
        self._schedule_pending_jobs()
        return True

    def recover_job(self, job_id: str, reset_runtime: bool = False) -> bool:
        if job_id in self.jobs_map or job_id not in self.all_jobs_map:
            return False
        job = self.all_jobs_map[job_id]
        if reset_runtime:
            job.reset_runtime_state()
        self.jobs_map[job_id] = job
        if job_id not in self.initial_job_order:
            self.initial_job_order.append(job_id)
        self._refresh_topology()
        self._ensure_node_history_slots()
        self._ensure_job_history_slots(job_id)
        if job.next_phase == "train":
            self.train_ready_queue.append(job_id)
        elif job.next_phase != "done":
            self.rollout_ready_queue.append(job_id)
        self._schedule_pending_jobs()
        return True

    def _schedule_rollout(self, job: Job):
        rollout_time = job.rollout_time_for_iteration(job.iterations_done)
        train_time = job.train_time_for_rollout(rollout_time)
        end_time = self.current_time + rollout_time
        for node in job.rollout_nodes:
            self.rollout_node_status[node] = job.job_id
            self.rollout_busy_times[node][job.job_id].append((self.current_time, end_time))
        job.current_iter_rollout = rollout_time
        job.current_iter_train = train_time
        job.current_iter_solo = rollout_time + train_time
        job.current_iter_start_time = self.current_time
        job.active_phase = "rollout"
        heapq.heappush(self.event_queue, (end_time, "ROLLOUT_END", job.job_id))

    def _schedule_train(self, job: Job):
        end_time = self.current_time + job.current_iter_train
        self.train_nodes_busy = True
        self.active_train_job_id = job.job_id
        job.active_phase = "train"
        for node in self.all_train_nodes:
            self.train_busy_times[node][job.job_id].append((self.current_time, end_time))
        heapq.heappush(self.event_queue, (end_time, "TRAIN_END", job.job_id))

    def _schedule_pending_jobs(self):
        if not self.train_nodes_busy and self.train_ready_queue:
            job_id_to_train = self.train_ready_queue.popleft()
            self._schedule_train(self.jobs_map[job_id_to_train])

        pending_rollouts = deque()
        while self.rollout_ready_queue:
            job_id_to_rollout = self.rollout_ready_queue.popleft()
            job = self.jobs_map[job_id_to_rollout]
            nodes_required = job.rollout_nodes
            are_nodes_available = all(self.rollout_node_status[node] is None for node in nodes_required)
            if are_nodes_available:
                self._schedule_rollout(job)
            else:
                pending_rollouts.append(job_id_to_rollout)
        self.rollout_ready_queue = pending_rollouts

    def _process_next_event(self, n_iters: Optional[int]):
        event_time, event_type, job_id = heapq.heappop(self.event_queue)
        self.current_time = event_time
        job = self.jobs_map[job_id]

        if event_type == "ROLLOUT_END":
            for node in job.rollout_nodes:
                if self.rollout_node_status.get(node) == job_id:
                    self.rollout_node_status[node] = None
            job.active_phase = None
            job.next_phase = "train"
            job.completed_rollout_durations.append(job.current_iter_rollout)
            self.train_ready_queue.append(job_id)

        elif event_type == "TRAIN_END":
            self.train_nodes_busy = False
            self.active_train_job_id = None
            job.active_phase = None
            job.completed_train_durations.append(job.current_iter_train)
            job.completed_solo_durations.append(job.current_iter_solo)
            job.completed_cycle_durations.append(self.current_time - job.current_iter_start_time)
            job.completed_iteration_end_times.append(self.current_time)
            job.iterations_done += 1
            job.current_iter_rollout = None
            job.current_iter_train = None
            job.current_iter_solo = None
            job.current_iter_start_time = None
            if n_iters is None or job.iterations_done < n_iters:
                job.next_phase = "rollout"
                self.rollout_ready_queue.append(job_id)
            else:
                job.next_phase = "done"

        self._schedule_pending_jobs()

    def run_until(self, target_time: Optional[float] = None, n_iters: Optional[int] = None):
        if not self._bootstrapped:
            self.bootstrap(start_time=self.current_time, reset_runtime=False, queue_from_state=True)

        while True:
            if n_iters is not None and all(job.iterations_done >= n_iters for job in self.jobs_map.values()):
                break
            if target_time is not None and self.current_time >= target_time:
                break
            if not self.event_queue:
                if target_time is not None:
                    self.current_time = target_time
                break
            next_time = self.event_queue[0][0]
            if target_time is not None and next_time > target_time:
                self.current_time = target_time
                break
            self._process_next_event(n_iters=n_iters)

    def apply_pause(self, duration: float):
        if duration <= 0:
            return
        self.current_time += duration

    def _truncate_running_intervals(self):
        for node, job_id in self.rollout_node_status.items():
            if job_id is None:
                continue
            intervals = self.rollout_busy_times[node][job_id]
            if intervals:
                start, _ = intervals[-1]
                intervals[-1] = (start, self.current_time)
        if self.active_train_job_id is not None:
            for node in self.all_train_nodes:
                intervals = self.train_busy_times[node][self.active_train_job_id]
                if intervals:
                    start, _ = intervals[-1]
                    intervals[-1] = (start, self.current_time)

    def cancel_inflight_work(self, restart_incomplete: bool = True):
        self._truncate_running_intervals()
        self.event_queue = []
        self.rollout_ready_queue = deque()
        self.train_ready_queue = deque()
        self._set_idle_resources()
        if restart_incomplete:
            for job in self.jobs_map.values():
                if job.next_phase != "done":
                    job.restart_incomplete_iteration()
        self.bootstrap(start_time=self.current_time, reset_runtime=False, queue_from_state=True)

    def get_observed_rollout_time(self, job_id: str) -> float:
        job = self.jobs_map[job_id]
        if job.completed_rollout_durations:
            return job.completed_rollout_durations[-1]
        return job.t_rollout

    def get_job_slowdowns(self) -> Dict[str, List[float]]:
        slowdowns = {}
        for job_id, job in self.jobs_map.items():
            job_slowdowns = []
            for cycle_time, solo_time in zip(job.completed_cycle_durations, job.completed_solo_durations):
                if solo_time > 0:
                    job_slowdowns.append(cycle_time / solo_time)
            slowdowns[job_id] = job_slowdowns
        return slowdowns

    def get_job_average_slowdowns(self) -> Dict[str, float]:
        job_slowdowns = self.get_job_slowdowns()
        return {
            job_id: sum(values) / len(values)
            for job_id, values in job_slowdowns.items() if values
        }

    def _clean_busy_times(self) -> Tuple[Dict, Dict]:
        cleaned_rollout_busy_times = {node: {} for node in self.rollout_busy_times}
        for node, jobs_data in self.rollout_busy_times.items():
            for j_id, intervals in jobs_data.items():
                valid_intervals = [interval for interval in intervals if interval[1] > interval[0]]
                if valid_intervals:
                    cleaned_rollout_busy_times[node][j_id] = valid_intervals

        cleaned_train_busy_times = {node: {} for node in self.train_busy_times}
        for node, jobs_data in self.train_busy_times.items():
            for j_id, intervals in jobs_data.items():
                valid_intervals = [interval for interval in intervals if interval[1] > interval[0]]
                if valid_intervals:
                    cleaned_train_busy_times[node][j_id] = valid_intervals
        return cleaned_rollout_busy_times, cleaned_train_busy_times

    def simulate_run(self, n_iters: int) -> Tuple[Dict, Dict, Dict, float]:
        if not self.jobs_map:
            return {}, {}, {"rollout": [], "train": []}, 0.0

        self.reset_state(start_time=0.0)
        self.bootstrap(start_time=0.0, reset_runtime=True, queue_from_state=True)
        self.run_until(n_iters=n_iters)
        total_time = self.current_time

        cleaned_rollout_busy_times, cleaned_train_busy_times = self._clean_busy_times()

        utils = {"rollout": {}, "train": []}
        if total_time > 0:
            for node in self.all_rollout_nodes:
                node_busy_time = sum(
                    end - start
                    for j_data in self.rollout_busy_times[node].values()
                    for start, end in j_data
                    if end > start
                )
                utils["rollout"][node] = node_busy_time / total_time
            if self.all_train_nodes:
                first_train_node = self.all_train_nodes[0]
                total_train_busy_time = sum(
                    end - start
                    for j_data in self.train_busy_times[first_train_node].values()
                    for start, end in j_data
                    if end > start
                )
                train_util = total_train_busy_time / total_time
                utils["train"] = [train_util] * len(self.all_train_nodes)
        else:
            utils["rollout"] = {node: 0.0 for node in self.all_rollout_nodes}
            utils["train"] = [0.0] * len(self.all_train_nodes)

        return cleaned_rollout_busy_times, cleaned_train_busy_times, utils, total_time

    def get_recorded_state(self) -> Tuple[Dict, Dict, float]:
        rollout_busy_times, train_busy_times = self._clean_busy_times()
        return rollout_busy_times, train_busy_times, self.current_time

    def _plot_busy_times(self, rollout_busy_times: Dict, train_busy_times: Dict, color_settings: Dict,
                         show_lgd: bool = False, export_path: Optional[str] = None,
                         xlim_job_id: Optional[str] = None, xlim_pad: float = 5.0):
        jobid_2_rcolors = {job_id: color_settings[job_id][0] for job_id in self.all_jobs_map}
        jobid_2_tcolors = {job_id: color_settings[job_id][1] for job_id in self.all_jobs_map}
        jobid_2_hatches = {job_id: color_settings[job_id][2] for job_id in self.all_jobs_map}
        ax = plt.gca()
        max_x = 0

        if show_lgd:
            for job_id in self.all_jobs_map:
                ax.add_patch(patches.Rectangle((0, 0), 0, 0, edgecolor='black', facecolor=jobid_2_rcolors[job_id], label=job_id + "-Rollout"))
                ax.add_patch(patches.Rectangle((0, 0), 0, 0, edgecolor='black', facecolor=jobid_2_tcolors[job_id], label=job_id + "-Train"))

        y = 0
        all_clusters = []
        for train_node in train_busy_times:
            cluster_utils = train_busy_times[train_node]
            for job_id in cluster_utils:
                for (train_start, train_end) in cluster_utils[job_id]:
                    rectangle = patches.Rectangle(
                        (train_start, y), train_end - train_start, 1,
                        edgecolor='white',
                        facecolor=jobid_2_tcolors[job_id],
                        hatch=jobid_2_hatches[job_id],
                        zorder=0,
                    )
                    ax.add_patch(rectangle)
                    rectangle = patches.Rectangle(
                        (train_start, y), train_end - train_start, 1,
                        edgecolor='black',
                        facecolor='none',
                        zorder=100,
                    )
                    ax.add_patch(rectangle)
                    max_x = max(max_x, train_end)
            y += 1
            all_clusters.append(train_node)

        for rollout_node in rollout_busy_times:
            cluster_utils = rollout_busy_times[rollout_node]
            for job_id in cluster_utils:
                for (rollout_start, rollout_end) in cluster_utils[job_id]:
                    rectangle = patches.Rectangle(
                        (rollout_start, y), rollout_end - rollout_start, 1,
                        edgecolor='white',
                        facecolor=jobid_2_rcolors[job_id],
                        hatch=jobid_2_hatches[job_id],
                        zorder=0,
                    )
                    ax.add_patch(rectangle)
                    rectangle = patches.Rectangle(
                        (rollout_start, y), rollout_end - rollout_start, 1,
                        edgecolor='black',
                        facecolor='none',
                        zorder=100,
                    )
                    ax.add_patch(rectangle)
                    max_x = max(max_x, rollout_end)
            y += 1
            all_clusters.append(rollout_node)

        if show_lgd:
            plt.legend()
        xlim_end = max_x
        if xlim_job_id is not None:
            candidate_ends = []
            for train_node, cluster_utils in train_busy_times.items():
                if xlim_job_id in cluster_utils and cluster_utils[xlim_job_id]:
                    candidate_ends.append(cluster_utils[xlim_job_id][-1][1])
            if candidate_ends:
                xlim_end = max(candidate_ends)
        plt.xlim(0, xlim_end + xlim_pad)
        plt.ylim(0, y)
        plt.yticks(np.arange(y) + 0.5, all_clusters)
        if export_path is not None:
            plt.savefig(export_path)

    def plot(self, n_meta_iters: int, color_settings: Dict, show_lgd: bool = False,
             export_path: Optional[str] = None, xlim_job_id: Optional[str] = None,
             xlim_pad: float = 5.0):
        rollout_busy_times, train_busy_times, _, _ = self.simulate_run(n_meta_iters)
        self._plot_busy_times(
            rollout_busy_times,
            train_busy_times,
            color_settings,
            show_lgd=show_lgd,
            export_path=export_path,
            xlim_job_id=xlim_job_id,
            xlim_pad=xlim_pad,
        )

    def plot_recorded_history(self, color_settings: Dict, show_lgd: bool = False,
                              export_path: Optional[str] = None, xlim_job_id: Optional[str] = None,
                              xlim_pad: float = 5.0):
        rollout_busy_times, train_busy_times, _ = self.get_recorded_state()
        self._plot_busy_times(
            rollout_busy_times,
            train_busy_times,
            color_settings,
            show_lgd=show_lgd,
            export_path=export_path,
            xlim_job_id=xlim_job_id,
            xlim_pad=xlim_pad,
        )
