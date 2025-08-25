import os

import ray
import torch


@ray.remote(num_gpus=1)
def get_visible_gpus():
    return ray.get_gpu_ids()


@ray.remote(num_gpus=1)
def get_node_rank():
    return int(os.environ.get("NODE_RANK", "0"))


@ray.remote(num_gpus=1)
def get_device_type():
    return torch.cuda.get_device_properties(0).name