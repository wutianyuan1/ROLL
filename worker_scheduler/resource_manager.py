from typing import List, Optional, Dict, Tuple
from event import Phase
from dataclasses import dataclass

@dataclass
class JobStatus:
    job_name: str
    phase: Phase
    gpus_per_gen_worker: int
    max_gen_gpus: int
    max_train_gpus: int
    # For a running generation phase, 1 <= |allocated_gpus| <= max_gen_gpus
    # For a running train phase, |allocated_gpus| = max_train_gpus
    allocated_gpus: List[int]


class ResourceManager:
    def __init__(self, gen_device_affinity: str, gen_device_ids: List[int], train_device_affinity: str,  train_device_ids: List[int]) -> None:
        self.gen_device_affinity = gen_device_affinity
        self.train_device_affinity = train_device_affinity
        self.gen_available_devices = sorted(gen_device_ids.copy())
        self.train_available_devices = sorted(train_device_ids.copy())
        self.gen_resource_mapping: Dict[int, str] = {i: None for i in gen_device_ids}
        self.train_resource_mapping: Dict[int, str] = {i: None for i in train_device_ids}
        # job_mapping: job_name -> JobStatus
        self.job_mapping: Dict[str, JobStatus] = {}
        self.gen_cluster_size = len(self.gen_available_devices)
        self.train_cluster_size = len(self.train_available_devices)

    @property
    def num_available_devices(self):
        return len(self.gen_available_devices) + len(self.train_available_devices)

    @property
    def cluster_size(self):
        return self.gen_cluster_size + self.train_cluster_size

    def release(self, job_name: str, device_affinity: str, gpu_list: List[int]):
        '''Release the given GPU list'''
        assert device_affinity in [self.gen_device_affinity, self.train_device_affinity], \
            f"Try to release GPUs in {device_affinity}, which doesn't match device type of train/gen devices"
        if device_affinity == self.gen_device_affinity:
            assert len(set(gpu_list).intersection(self.gen_available_devices)) == 0
            for i in gpu_list:
                assert i in self.gen_resource_mapping, f"Try to release GPU: {i}, which is not in gen pool"
                self.gen_resource_mapping[i] = None
                self.gen_available_devices.append(i)
                self.job_mapping[job_name].allocated_gpus.remove(i)
            self.gen_available_devices.sort()
        else:
            assert len(set(gpu_list).intersection(self.train_available_devices)) == 0
            for i in gpu_list:
                assert i in self.train_resource_mapping, f"Try to release GPU: {i}, which is not in train pool"
                self.train_resource_mapping[i] = None
                self.train_available_devices.append(i)
                self.job_mapping[job_name].allocated_gpus.remove(i)
            self.train_available_devices.sort()

    def register_job(self, init_job_status: JobStatus):
        assert init_job_status.max_gen_gpus <= self.gen_cluster_size
        assert init_job_status.max_train_gpus <= self.train_cluster_size
        self.job_mapping[init_job_status.job_name] = init_job_status

    def update_phase(self, job_name: str, new_phase: Phase):
        self.job_mapping[job_name].phase = new_phase

    def allocate_worker(self, job_name: str, pool: str) -> Optional[List]:
        '''Allocate `gpus_per_worker` GPUs to job_name'''
        assert pool in ['train', 'gen'], f"Resource pool can only be train or gen, got {pool}"
        if pool == 'gen':
            gpus_per_worker = self.job_mapping[job_name].gpus_per_gen_worker
            assert self.gen_cluster_size % gpus_per_worker == 0
            start_gen_gid = min(self.gen_resource_mapping.keys())
            for i in range(start_gen_gid, start_gen_gid + self.gen_cluster_size, gpus_per_worker):
                can_allocate = True
                for j in range(i, i + gpus_per_worker):
                    if j not in self.gen_available_devices:
                        can_allocate = False
                if can_allocate:
                    for j in range(i, i + gpus_per_worker):
                        self.gen_available_devices.remove(j)
                        assert self.gen_resource_mapping[j] is None
                        self.gen_resource_mapping[j] = job_name
                    self.job_mapping[job_name].allocated_gpus += list(range(i, i + gpus_per_worker))
                    return list(range(i, i + gpus_per_worker))
        else:
            # For train allocation request, we allocate max_train_gpus for it at once
            num_train_gpus = self.job_mapping[job_name].max_train_gpus
            free_gpus = [i for i in self.train_resource_mapping.keys() if self.train_resource_mapping[i] is None]
            assert len(free_gpus) == len(self.train_available_devices)
            if len(free_gpus) < num_train_gpus:
                return None
            gpus_to_allocate = free_gpus[:num_train_gpus].copy()
            for j in gpus_to_allocate:
                self.train_available_devices.remove(j)
                assert self.train_resource_mapping[j] is None
                self.train_resource_mapping[j] = job_name
            self.train_available_devices.sort()
            assert self.job_mapping[job_name].allocated_gpus == [], f"Non-empty allocated_gpus before train: {self.job_mapping[job_name].allocated_gpus}"
            self.job_mapping[job_name].allocated_gpus = gpus_to_allocate.copy()
            return gpus_to_allocate
        return None

    def cleanup_by_name(self, job_name: str) -> int:
        '''Returns the released GPU count'''
        cleanup_count = 0
        for i in self.gen_resource_mapping:
            if self.gen_resource_mapping[i] == job_name:
                assert i not in self.gen_available_devices
                self.gen_resource_mapping[i] = None
                self.gen_available_devices.append(i)
                cleanup_count += 1
        for i in self.train_resource_mapping:
            if self.train_resource_mapping[i] == job_name:
                assert i not in self.train_available_devices
                self.train_resource_mapping[i] = None
                self.train_available_devices.append(i)
                cleanup_count += 1
        self.job_mapping[job_name].allocated_gpus = []
        self.gen_available_devices.sort()
        self.train_available_devices.sort()
        return cleanup_count

    def allocate_all(self, job_name: str) -> None:
        '''Allocate all GPUs to job_name'''
        for i in self.gen_resource_mapping:
            self.gen_resource_mapping[i] = job_name
        for i in self.train_resource_mapping:
            self.train_resource_mapping[i] = job_name
        self.job_mapping[job_name].allocated_gpus = self.gen_available_devices.copy() + self.train_available_devices.copy()
        self.gen_available_devices = []
        self.train_available_devices = []

    def get_running_jobs(self):
        '''Get all running job names'''
        running_jobs = []
        for job in self.job_mapping:
            if len(self.job_mapping[job].allocated_gpus) != 0:
                running_jobs.append(job)
        return running_jobs
