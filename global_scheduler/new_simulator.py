import heapq
import numpy as np
from collections import deque
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from global_scheduler.structs import Job


class WeaveSimulator:
    """
    A discrete-event simulator for multiple disaggregated RLHF jobs.

    This simulator models a First-Come, First-Served (FCFS) scheduling policy
    for both rollout and training phases, based on the provided scheduler logic.

    Key principles emulated from the provided scheduler code:
    - A job cycle consists of Rollout -> Train -> Rollout -> ...
    - Rollout can happen in parallel for jobs with non-overlapping node requirements.
    - Training is sequential; only one job can train at a time using a shared
      set of training nodes.
    - Jobs waiting for a phase (rollout or train) are placed in a FCFS queue.
    - A job at the front of a queue is scheduled as soon as its required
      resources become available.
    """
    def __init__(self, jobs: List[Job]):
        self.jobs_map: Dict[str, Job] = {job.job_id: job for job in jobs}
        # Preserve the initial order for FCFS initialization
        self.initial_job_order: List[str] = [job.job_id for job in jobs]

        # Discover all unique nodes from the job definitions
        self.all_rollout_nodes: List[str] = sorted(list(set(
            node for job in jobs for node in job.rollout_nodes
        )))
        self.all_train_nodes: List[str] = []
        if jobs and jobs[0].train_nodes:
            self.all_train_nodes = sorted(list(set(jobs[0].train_nodes)))

        # Simulation state variables (will be reset for each run)
        self.current_time: float = 0.0
        # Event queue is a min-heap: (time, event_type, job_id)
        self.event_queue: List[Tuple[float, str, str]] = []
        # FCFS ready queues for jobs waiting for resources
        self.rollout_ready_queue: deque = deque()
        self.train_ready_queue: deque = deque()

        # Resource availability status
        self.rollout_node_status: Dict[str, Optional[str]] = {}
        self.train_nodes_busy: bool = False

        # Data collection for results
        self.rollout_busy_times: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        self.train_busy_times: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    def _reset_state(self):
        """Resets the simulator to its initial state for a new run."""
        self.current_time = 0.0
        self.event_queue = []
        self.rollout_ready_queue = deque()
        self.train_ready_queue = deque()

        for job in self.jobs_map.values():
            job.iterations_done = 0

        self.rollout_node_status = {node: None for node in self.all_rollout_nodes}
        self.train_nodes_busy = False

        self.rollout_busy_times = {node: {job_id: [] for job_id in self.jobs_map} for node in self.all_rollout_nodes}
        self.train_busy_times = {node: {job_id: [] for job_id in self.jobs_map} for node in self.all_train_nodes}

    def _schedule_pending_jobs(self):
        """
        Emulates the FCFS scheduler policy. It checks the ready queues and
        schedules jobs if their required resources are free.
        """
        # --- Train Scheduling (Single shared resource) ---
        if not self.train_nodes_busy and self.train_ready_queue:
            job_id_to_train = self.train_ready_queue.popleft()
            job = self.jobs_map[job_id_to_train]

            self.train_nodes_busy = True
            end_time = self.current_time + job.t_train
            heapq.heappush(self.event_queue, (end_time, "TRAIN_END", job_id_to_train))

            for node in self.all_train_nodes:
                self.train_busy_times[node][job_id_to_train].append((self.current_time, end_time))

        # --- Rollout Scheduling (Potentially parallel) ---
        # Iterate through the ready queue; if a job can be scheduled, start it.
        # If not, it remains in the queue for the next check.
        pending_rollouts = deque()
        while self.rollout_ready_queue:
            job_id_to_rollout = self.rollout_ready_queue.popleft()
            job = self.jobs_map[job_id_to_rollout]

            nodes_required = job.rollout_nodes
            are_nodes_available = all(self.rollout_node_status[node] is None for node in nodes_required)

            if are_nodes_available:
                # Allocate resources and schedule completion event
                for node in nodes_required:
                    self.rollout_node_status[node] = job_id_to_rollout
                end_time = self.current_time + job.t_rollout
                heapq.heappush(self.event_queue, (end_time, "ROLLOUT_END", job_id_to_rollout))
                for node in nodes_required:
                    self.rollout_busy_times[node][job_id_to_rollout].append((self.current_time, end_time))
            else:
                # Resources not available, job must wait
                pending_rollouts.append(job_id_to_rollout)
        
        self.rollout_ready_queue = pending_rollouts

    def simulate_run(self, n_iters: int) -> Tuple[Dict, Dict, Dict]:
        """
        Runs the simulation for all jobs until each has completed n_iters iterations.

        Args:
            n_iters: The minimum number of iterations each job must complete.

        Returns:
            A tuple containing:
            - rollout_busy_times: Dict mapping node name to a dict of
              job_id -> List of (start_time, end_time) tuples.
            - train_busy_times: Same structure as above, for training nodes.
            - utils: Dict with "rollout" and "train" keys, each mapping to a
              list of utilization floats (busy_time / total_time) for each node.
        """
        if not self.jobs_map:
            return {}, {}, {"rollout": [], "train": []}

        self._reset_state()

        # At t=0, all jobs are ready for their first rollout, in the specified FCFS order.
        for job_id in self.initial_job_order:
            self.rollout_ready_queue.append(job_id)

        # Initial scheduling attempt at t=0
        self._schedule_pending_jobs()

        # Main discrete-event simulation loop
        while True:
            # Termination condition: all jobs have completed at least n_iters
            if any(job.iterations_done >= n_iters for job in self.jobs_map.values()):
                break

            if not self.event_queue:
                # This may happen if n_iters is 0, or in a deadlock scenario.
                if any(job.iterations_done < n_iters for job in self.jobs_map.values()):
                    print("Warning: Event queue empty, but simulation not finished. Possible deadlock.")
                break

            # Advance time to the next event
            event_time, event_type, job_id = heapq.heappop(self.event_queue)
            self.current_time = event_time
            job = self.jobs_map[job_id]

            # Process the event
            if event_type == "ROLLOUT_END":
                # Release rollout nodes
                for node in job.rollout_nodes:
                    if self.rollout_node_status.get(node) == job_id:
                        self.rollout_node_status[node] = None
                # Job is now ready to train
                self.train_ready_queue.append(job_id)

            elif event_type == "TRAIN_END":
                # Release training nodes
                self.train_nodes_busy = False
                job.iterations_done += 1
                # If more iterations are needed, job is ready for the next rollout
                # This models the zero-cost "UPDATE" phase from the scheduler logic.
                if job.iterations_done < n_iters:
                    self.rollout_ready_queue.append(job_id)

            # After resources are freed, check if any waiting jobs can be scheduled
            self._schedule_pending_jobs()

        total_time = self.current_time

        # --- Finalize and format results ---
        
        # 1. Clean busy time dicts to remove entries for jobs that never used a node
        cleaned_rollout_busy_times = {node: {} for node in self.all_rollout_nodes}
        for node, jobs_data in self.rollout_busy_times.items():
            for j_id, intervals in jobs_data.items():
                if intervals:
                    cleaned_rollout_busy_times[node][j_id] = intervals
        
        cleaned_train_busy_times = {node: {} for node in self.all_train_nodes}
        for node, jobs_data in self.train_busy_times.items():
            for j_id, intervals in jobs_data.items():
                if intervals:
                    cleaned_train_busy_times[node][j_id] = intervals

        # 2. Calculate utilization statistics
        utils = {"rollout": [], "train": []}
        if total_time > 0:
            for node in self.all_rollout_nodes:
                node_busy_time = sum(end - start for j_data in self.rollout_busy_times[node].values() for start, end in j_data)
                utils["rollout"].append(node_busy_time / total_time)

            if self.all_train_nodes:
                first_train_node = self.all_train_nodes[0]
                total_train_busy_time = sum(end - start for j_data in self.train_busy_times[first_train_node].values() for start, end in j_data)
                train_util = total_train_busy_time / total_time
                utils["train"] = [train_util] * len(self.all_train_nodes)
        else: # Handle zero total_time case (e.g., n_iters=0)
            utils["rollout"] = [0.0] * len(self.all_rollout_nodes)
            utils["train"] = [0.0] * len(self.all_train_nodes)

        return cleaned_rollout_busy_times, cleaned_train_busy_times, utils, total_time

    def plot(self, n_meta_iters: int, export_path: Optional[str] = None):
        rollout_busy_times, train_busy_times, _, _ = self.simulate_run(n_meta_iters)
        colors = sns.color_palette("Set3")
        jobid_2_colors = {job.job_id: color for (job, color) in zip(self.jobs_map.values(), colors)}
        ax = plt.gca()
        max_x = 0

        # dummy rect for showing the legend
        for job in self.jobs_map.values():
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

# Example Usage:
if __name__ == '__main__':
    # Define a common set of training nodes
    training_cluster = ["TN0"]

    # Scenario 1: Two jobs with conflicting rollout nodes
    print("--- Scenario 1: Rollout Conflict ---")
    jobs1 = [
        Job(job_id="job_A", t_rollout=40, t_train=40, rollout_nodes=["RN0", "RN1"], train_nodes=training_cluster),
        Job(job_id="job_B", t_rollout=40, t_train=20, rollout_nodes=["RN0"], train_nodes=training_cluster),
        Job(job_id="job_C", t_rollout=40, t_train=20, rollout_nodes=["RN1"], train_nodes=training_cluster),
    ]
    simulator1 = WeaveSimulator(jobs1)
    rollout_times, train_times, utils = simulator1.simulate_run(n_iters=1000)

    simulator1.plot(n_meta_iters=10, export_path="global_scheduler/new.png")
    # Expected: job_A starts rollout at t=0, finishes at t=10. job_B waits, starts at t=10, finishes at t=15.
    # job_A starts train at t=10, finishes at t=30. job_B starts train at t=30, finishes at t=45.
    print("Rollout Busy Times:")
    import json
    print(json.dumps(rollout_times, indent=2))
    print("\nTrain Busy Times:")
    print(json.dumps(train_times, indent=2))
    print(f"\nUtils: {utils}")
    print(f"Total simulation time: {simulator1.current_time:.2f}\n")


    # # Scenario 2: Two jobs with independent rollout nodes
    # print("\n--- Scenario 2: Parallel Rollout ---")
    # jobs2 = [
    #     Job(job_id="job_C", t_rollout=10, t_train=20, rollout_nodes=["rollout_node_1"], train_nodes=training_cluster),
    #     Job(job_id="job_D", t_rollout=5, t_train=15, rollout_nodes=["rollout_node_2"], train_nodes=training_cluster),
    # ]
    # simulator2 = WeaveSimulator(jobs2)
    # rollout_times, train_times, utils = simulator2.simulate_run(n_iters=2)

    # # Expected: job_C and job_D start rollout in parallel at t=0.
    # # job_D finishes rollout at t=5, gets in train queue first.
    # # job_D starts train at t=5, finishes at t=20.
    # # job_C finishes rollout at t=10, waits for training.
    # # job_C starts train at t=20, finishes at t=40.
    # # Then second iteration begins.
    # print("Rollout Busy Times:")
    # print(json.dumps(rollout_times, indent=2))
    # print("\nTrain Busy Times:")
    # print(json.dumps(train_times, indent=2))
    # print(f"\nUtils: {utils}")
    # print(f"Total simulation time: {simulator2.current_time:.2f}")

