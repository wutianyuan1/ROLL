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
    def __init__(self, jobs: List[Job]):
        self.jobs = jobs
