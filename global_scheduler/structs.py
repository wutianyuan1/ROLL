from typing import List, Optional


class Job:
    def __init__(self, job_id: str, t_rollout: float, t_train: float, slo: float = None,
                 rollout_nodes: Optional[List[str]] = None,
                 train_nodes: Optional[List[str]] = None,
                 rollout_step_trace: Optional[List[float]] = None,
                 rollout_width: int = 1,
                 train_width: int = 1):
        self.job_id = job_id
        self.t_rollout = t_rollout
        self.t_train = t_train
        self.slo = slo
        # We simplify the condition here (avoid early-starting)
        # A job's rollout/train can start iff all its required 
        # rollout/train nodes are available.
        self.rollout_nodes = rollout_nodes
        self.train_nodes = train_nodes
        self.rollout_step_trace = rollout_step_trace
        self.rollout_width = rollout_width
        self.train_width = train_width
        self.reset_runtime_state()

    def __repr__(self):
        return f"<Job[{self.job_id}] on {self.rollout_nodes}>"

    def rollout_time_for_iteration(self, iteration_idx: int) -> float:
        if not self.rollout_step_trace:
            return self.t_rollout
        trace_idx = min(iteration_idx, len(self.rollout_step_trace) - 1)
        return self.rollout_step_trace[trace_idx]

    def train_time_for_rollout(self, rollout_time: float) -> float:
        if self.t_rollout == 0:
            return self.t_train
        return self.t_train * rollout_time / self.t_rollout

    def train_time_for_iteration(self, iteration_idx: int) -> float:
        return self.train_time_for_rollout(self.rollout_time_for_iteration(iteration_idx))

    def solo_time_for_iteration(self, iteration_idx: int) -> float:
        rollout_time = self.rollout_time_for_iteration(iteration_idx)
        return rollout_time + self.train_time_for_rollout(rollout_time)

    def reset_runtime_state(self):
        self.iterations_done = 0
        self.next_phase = "rollout"
        self.active_phase = None
        self.current_iter_rollout = None
        self.current_iter_train = None
        self.current_iter_solo = None
        self.current_iter_start_time = None
        self.completed_rollout_durations: List[float] = []
        self.completed_train_durations: List[float] = []
        self.completed_solo_durations: List[float] = []
        self.completed_cycle_durations: List[float] = []
        self.completed_iteration_end_times: List[float] = []

    def restart_incomplete_iteration(self):
        self.next_phase = "rollout"
        self.active_phase = None
        self.current_iter_rollout = None
        self.current_iter_train = None
        self.current_iter_solo = None
        self.current_iter_start_time = None


class JobGroup:
    def __init__(self, group_id: str, jobs: List[Job]):
        self.group_id = group_id
        self.jobs = jobs
        self.last_rollout_node_id = 0

    def next_rollout_node_id(self) -> str:
        self.last_rollout_node_id += 1
        return str(self.last_rollout_node_id)

    def __repr__(self):
        return self.group_id

    @property
    def all_rollout_nodes(self) -> List[str]:
        all_nodes = []
        for job in self.jobs:
            all_nodes += (job.rollout_nodes or [])
        return list(set(all_nodes))

    @property
    def all_train_nodes(self) -> List[str]:
        all_nodes = []
        for job in self.jobs:
            all_nodes += (job.train_nodes or [])
        return list(set(all_nodes))

    def get_node_phase_times(self):
        node_2_phase_times = {}
        for job in self.jobs:
            for rn in job.rollout_nodes:
                node_2_phase_times.setdefault(rn, 0)
                node_2_phase_times[rn] += job.t_rollout
            for tn in job.train_nodes:
                node_2_phase_times.setdefault(tn, 0)
                node_2_phase_times[tn] += job.t_train
        return node_2_phase_times

    @property
    def T1(self) -> float:
        '''Max single-job iteration time'''
        return max(i.t_rollout + i.t_train for i in self.jobs) if len(self.jobs) != 0 else 0

    @property
    def T2(self) -> float:
        '''Max phase time'''
        return max(i for i in self.get_node_phase_times().values()) if len(self.jobs) != 0 else 0

    @property
    def job_ids(self) -> List[str]:
        return [i.job_id for i in self.jobs]
