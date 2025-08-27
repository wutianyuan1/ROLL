import dataclasses
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import ray
from ray.util.placement_group import PlacementGroup

from roll.utils.ray_utils import get_visible_gpus, get_node_rank, get_device_type


class ResourceManager:
    def __init__(self, da_2_num_gpus_per_node: Dict[str, int], da_2_num_nodes: Dict[str, int]):
        """
            The ResourceManager centrally manages the required GPU/CPU resources,
            facilitating Ray to deploy Actors on specified GPU devices.
            `da_2_num_gpus_per_node`: {device_affinity: num_gpus_per_node},
            `da_2_num_nodes`: {device_affinity: num_nodes}.
        """
        # TODO: Lunxi: We can not match the order of device affinities of following dicts with the actual order
        # of device types when calling ray.util.placement_group, so we temporarily adjust the order mannually.
        da_2_num_gpus_per_node = {k: da_2_num_gpus_per_node[k] for k in sorted(da_2_num_gpus_per_node, reverse=True)}
        da_2_num_nodes = {k: da_2_num_nodes[k] for k in sorted(da_2_num_nodes, reverse=True)}

        assert da_2_num_gpus_per_node.keys() == da_2_num_nodes.keys()
        available_resources = ray.available_resources()
        available_gpu = available_resources.get("GPU", 0)

        da_2_nodes_maybe_used = {da: [] for da in da_2_num_gpus_per_node}
        ray_nodes = ray.nodes()
        DEVICE_TYPE_STR = "accelerator_type:"
        gpu_enough = True
        for da in da_2_num_gpus_per_node:
            remain_num_nodes = da_2_num_nodes[da]
            for node in ray_nodes:
                resource = node["Resources"]
                node_gpu_num = int(resource.get("GPU", 0))
                for key in resource.keys():
                    if key.startswith(DEVICE_TYPE_STR) and da == f"NVIDIA {key[len(DEVICE_TYPE_STR):]}" and node_gpu_num >= da_2_num_gpus_per_node[da]:
                        da_2_nodes_maybe_used[da].append(node)
                        remain_num_nodes -= 1
                        break
            if remain_num_nodes != 0:
                gpu_enough = False
            assert gpu_enough, (f"The Ray clusters(ray_gpus_per_node: {[node['Resources'].get('GPU', 0) for node in ray_nodes]}) cannot meet the "
                                f"required number of nodes (`num_nodes`{da_2_num_nodes}).")

        self.da_2_num_nodes = da_2_num_nodes
        self.da_2_gpu_per_node = da_2_num_gpus_per_node
        self.da_2_num_gpus = {da: self.da_2_gpu_per_node[da] * self.da_2_num_nodes[da] for da in self.da_2_num_nodes}
        print(f"***** [roll.distributed.xxx.ResourceManager] da_2_num_nodes: {self.da_2_num_nodes}, da_2_gpu_per_node: {self.da_2_gpu_per_node}, da_2_num_gpus: {self.da_2_num_gpus} *****")
        print(f"***** nodes_maybe_used = {[(da, [node['Resources'] for node in nodes]) for da, nodes in da_2_nodes_maybe_used.items()]} *****")

        self.da_2_placement_groups: Dict[str, List[PlacementGroup]] = {}
        self.da_2_node_ranks: Dict[str, List[int]] = {}
        self.da_2_gpu_ranks: Dict[str, List[int]] = {}
        self.da_2_node2pg: Dict[str, Dict[int, PlacementGroup]] = {}
        for da in self.da_2_num_nodes:
            if self.da_2_gpu_per_node[da] > 0:
                assert self.da_2_num_gpus[da] <= available_gpu, f"[{da}] num_gpus {self.da_2_num_gpus[da]} > available_gpu {available_gpu}"
                available_gpu -= self.da_2_num_gpus[da]

                bundles = []
                for i in range(self.da_2_num_nodes[da]):
                    node = da_2_nodes_maybe_used[da][i]
                    node_cpu = int(node["Resources"]["CPU"])
                    bundles.append({"GPU": self.da_2_gpu_per_node[da], "CPU": max(32, 1)})

                self.da_2_placement_groups[da] = [ray.util.placement_group([bundle]) for bundle in bundles]
                ray.get([pg.ready() for pg in self.da_2_placement_groups[da]])
                gpu_ranks = ray.get(
                    [
                        get_visible_gpus.options(placement_group=pg, num_gpus=self.da_2_gpu_per_node[da]).remote()
                        for pg in self.da_2_placement_groups[da]
                    ]
                )

                gpu_types = ray.get(
                    [
                        get_device_type.options(placement_group=pg, num_gpus=self.da_2_gpu_per_node[da]).remote()
                        for pg in self.da_2_placement_groups[da]
                    ]
                )
                print(f"***** [{da}] gpu ranks: {gpu_ranks} *****")
                self.da_2_node_ranks[da] = list(range(len(self.da_2_placement_groups[da])))
                for node_rank in self.da_2_node_ranks[da]:
                    assert gpu_types[node_rank] == da, f"***** node_rank: {node_rank}, gpu_type: {gpu_types[node_rank]}, da: {da}, node_ranks: {self.da_2_node_ranks[da]}, gpu_types: {gpu_types} *****"

                self.da_2_gpu_ranks[da] = [int(gpu_rank[0]) for gpu_rank in gpu_ranks]
                self.da_2_node2pg[da] = {}
                for node_rank, placement_group in zip(self.da_2_node_ranks[da], self.da_2_placement_groups[da]):
                    self.da_2_node2pg[da][node_rank] = placement_group
                print(f"***** da_2_node2pg[{da}]: {self.da_2_node2pg[da]} *****")
            else:
                print("***** YOU SHOULD NOT REACH HERE *****")
                assert self.da_2_num_nodes[da] == 1
                node = da_2_nodes_maybe_used[da][0]
                node_cpu = int(node["Resources"]["CPU"])
                bundles = [{"CPU": node_cpu}] * self.da_2_num_nodes[da]
                self.da_2_placement_groups[da] = [ray.util.placement_group([bundle]) for bundle in bundles]
                ray.get([pg.ready() for pg in self.da_2_placement_groups[da]])
                self.da_2_node_ranks[da] = [0]
                self.da_2_node2pg[da] = {}
                for node_rank, placement_group in zip(self.da_2_node_ranks[da], self.da_2_placement_groups[da]):
                    self.da_2_node2pg[da][node_rank] = placement_group
                print(f"***** da_2_node2pg[{da}]: {self.da_2_node2pg[da]} *****")

    def nodes_placement_group(self, node_rank, device_affinity: str) -> PlacementGroup:
        """
        mesh table是 m×n，获取第node_rank nodel上gpu_rank的PlacementGroup，用于把ray.Actor部署到指定的GPU上
        """
        return self.da_2_node2pg[device_affinity][node_rank]

    def destroy_placement_group(self, device_affinity: str):
        [ray.util.remove_placement_group(pg) for pg in self.da_2_placement_groups[device_affinity]]

    def allocate_placement_group(self, world_size, device_mapping: List[int] = None, device_affinity: str = None) -> List[List[Dict]]:
        """
            Allocate resources according to device_mapping (numbered by GPU RANK)
            - GPUs: Specify required GPU indices via device_mapping
            - CPUs: Specify via world_size

            Return Type: List[List[Dict]]
              Dict Keys:
                - node_rank
                - gpu_rank
                - placement_group
              List[Dict]: Represents GPUs allocated to a worker and access to placement groups
              Example: If num_gpus_per_worker=8, then len(List[Dict])=8

            A Worker is defined as a group of resource owners (can span multiple machines) that can independently use allocated resources to execute computation operations.
        """
        if device_affinity is None or device_affinity not in self.da_2_placement_groups:
            if device_affinity is not None:
                print(f"***** Warning: cannot find device type {device_affinity}, fallback to the global pool. *****")
            device_affinity = 'Default'
        allocated_pg = []
        ray_address = f"{ray.get_runtime_context().gcs_address}"
        if device_mapping:
            num_gpus_per_worker = len(device_mapping) // world_size
            grouped_ranks = [
                list(device_mapping[i : i + num_gpus_per_worker])
                for i in range(0, len(device_mapping), num_gpus_per_worker)
            ]
            for group in grouped_ranks:
                pg_list = []
                for rank in group:
                    node_rank = rank // self.da_2_gpu_per_node[device_affinity]
                    gpu_rank = rank % self.da_2_gpu_per_node[device_affinity]

                    assert node_rank < self.da_2_num_nodes[device_affinity], \
                        (f"device_mapping used gpus are more than "
                         f"num_nodes×num_gpus_per_node={self.da_2_num_nodes[device_affinity]}×{self.da_2_gpu_per_node[device_affinity]}")

                    pg = self.nodes_placement_group(node_rank, device_affinity)
                    pg_list.append(
                        dict(node_rank=node_rank, gpu_rank=gpu_rank, placement_group=pg, ray_address=ray_address)
                    )
                allocated_pg.append(pg_list)
        else:
            # Try to spread the CPU workers across various nodes to avoid the out-of-memory (OOM) situation caused
            # by the concentration of CPU workers in one place and the resulting peak memory usage.
            for rank in range(world_size):
                # TODO: Lunxi: priority, intra-cluster or inter-cluster?
                # Temporarily apply inter-cluster priority.
                da = list(self.da_2_num_nodes.keys())[rank % len(self.da_2_num_nodes)]
                node_rank = rank % self.da_2_num_nodes[da]
                print(f"***** CPU worker rank: {rank}, (da, node_rank): ({da}, {node_rank}) *****")
                allocated_pg.append(
                    [
                        dict(
                            node_rank=node_rank,
                            gpu_rank=None,
                            placement_group=self.nodes_placement_group(node_rank, da),
                            ray_address=ray_address,
                        )
                    ]
                )

        assert len(allocated_pg) == world_size

        return allocated_pg
