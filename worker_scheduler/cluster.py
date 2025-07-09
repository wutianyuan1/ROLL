import copy
import redis
from typing import List, Dict
from monitor import WorkerStatus, TenantJob


class Cluster:
    def __init__(self, shared_storage: redis.StrictRedis, train_all_devices: List[int], gen_all_devices: List[int]) -> None:
        self.shared_storage = shared_storage
        self.train_all_devices = train_all_devices
        self.gen_all_devices = gen_all_devices
        self.avail_train_devices = copy.deepcopy(self.train_all_devices)
        self.avail_gen_devices = copy.deepcopy(self.gen_all_devices)
        self.gen_cluster_size = len(gen_all_devices)
        self.train_cluster_size = len(train_all_devices)
        self.tenants: Dict[str, TenantJob] = {}

    def _try_allocate(self, num_workers_to_run: int, gpus_per_worker: int,
                      num_total_workers: int, avail_list: List[int]):
        '''Allocate num_workers_to_run generate workers from the current available devices.'''
        worker_ids_to_start = []
        alloc_list = []
        for worker_id in range(num_total_workers):
            can_alloc_i = True
            for j in range(worker_id * gpus_per_worker, (worker_id + 1) * gpus_per_worker):
                if j not in avail_list:
                    can_alloc_i = False
            if can_alloc_i:
                alloc_list += list(range(worker_id * gpus_per_worker, (worker_id + 1) * gpus_per_worker))
                worker_ids_to_start.append(worker_id)
        if len(worker_ids_to_start) < num_workers_to_run:
            return None, avail_list
        ret_avail_list = sorted(list(set(avail_list) - set(alloc_list)))
        return worker_ids_to_start, ret_avail_list

    def schedule_gen(self, tenant_name: str, num_workers_to_run: int):
        tenant_handle = self.tenants[tenant_name]
        gen_gpus_per_worker = tenant_handle.config.gen_gpus_per_worker
        worker_ids_to_run, ret_avail_gen_devices = self._try_allocate(
            num_workers_to_run, gen_gpus_per_worker,
            self.gen_cluster_size // gen_gpus_per_worker, self.avail_gen_devices
        )
        if worker_ids_to_run is not None:
            self.avail_gen_devices = ret_avail_gen_devices
            for worker_id in worker_ids_to_run:
                monitor = self.tenants[tenant_name].get_gen(worker_id)
                status = monitor.get_status()
                assert status != WorkerStatus.GENERATING, f"{tenant_name}-{worker_id} is already in {status} status"
                monitor.set_status(WorkerStatus.GENERATING)
        else:
            print(f"*** Schedule generation for {tenant_name} failed, no enough resources!")
