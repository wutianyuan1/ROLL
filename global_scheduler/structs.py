from typing import List, Optional


class Job:
    def __init__(self, job_id: str, t_rollout: float, t_train: float,
                 rollout_nodes: Optional[List[str]] = None,
                 train_nodes: Optional[List[str]] = None):
        self.job_id = job_id
        self.t_rollout = t_rollout
        self.t_train = t_train
        # We simplify the condition here (avoid early-starting)
        # A job's rollout/train can start iff all its required 
        # rollout/train nodes are available.
        self.rollout_nodes = rollout_nodes
        self.train_nodes = train_nodes

    def __repr__(self):
        return f"<Job[{self.job_id}] on {self.rollout_nodes}>"


class JobGroup:
    def __init__(self, group_id: str, jobs: List[Job]):
        self.group_id = group_id
        self.jobs = jobs

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
