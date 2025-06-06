import copy
from typing import Any, List

import pytest
import ray
import time
import torch
import os
from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.worker import Worker, RankInfo
from roll.distributed.scheduler.decorator import register, Dispatch
from roll.distributed.scheduler.resource_manager import ResourceManager


class TestWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.value = self.rank
        x = torch.zeros(1000*1000*1000, dtype=torch.float32, device="cuda:0")
        y = torch.zeros(1000*1000*1000, dtype=torch.float32, device="cuda:1")
        print(f"Worker {self.rank} initialized with value {self.value} and GPUs {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def add(self, x):
        print("Adding value:", x, "to worker rank:", self.rank)
        return

def test_gpu_id():
    ray.init(log_to_driver=True)

    print(f"Available: {ray.available_resources()}")

    resource_manager = ResourceManager(num_nodes=1, num_gpus_per_node=2)

    test_worker_config = WorkerConfig(name="test_worker", world_size=1, num_gpus_per_worker=2, device_mapping="[0,1]")
    test_cluster: Any = Cluster(
        name=test_worker_config.name,
        resource_manager=resource_manager,
        worker_cls=TestWorker,
        worker_config=test_worker_config,
    )
    time.sleep(60)
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    res = test_cluster.add(x=x)
    print(res)
    



if __name__ == "__main__":
    test_gpu_id()
