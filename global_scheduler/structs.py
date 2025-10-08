from typing import List


class Job:
    def __init__(self, job_id: str, t_rollout: float, t_train: float,
                 rollout_nodes: str, train_nodes: str):
        self.job_id = job_id
        self.t_rollout = t_rollout
        self.t_train = t_train
        # We simplify the condition here (avoid early-starting)
        # A job's rollout/train can start iff all its required 
        # rollout/train nodes are available.
        self.rollout_nodes = rollout_nodes
        self.train_nodes = train_nodes


class JobGroup:
    def __init__(self, jobs: List[Job]):
        self.jobs = jobs
