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
    # For a running generation phase, |allocated_gpus| = max_train_gpus
    allocated_gpus: List[int]


class ResourceManager:
    def __init__(self, gen_device_mapping: List[int]) -> None:
        self.available_devices = sorted(gen_device_mapping.copy())
        self.resource_mapping: Dict[int, str] = {i: None for i in gen_device_mapping}
        # job_mapping: job_name -> JobStatus
        self.job_mapping: Dict[str, JobStatus] = {}
        self.cluster_size = len(self.available_devices)

    def release(self, job_name: str, gpu_list: List[int]):
        '''Release the given GPU list'''
        assert len(set(gpu_list).intersection(self.available_devices)) == 0
        self.available_devices += gpu_list
        self.available_devices.sort()
        for i in gpu_list:
            self.resource_mapping[i] = None
            self.job_mapping[job_name].allocated_gpus.remove(i)

    def register_job(self, init_job_status: JobStatus):
        assert init_job_status.max_gen_gpus <= self.cluster_size
        assert init_job_status.max_train_gpus <= self.cluster_size
        self.job_mapping[init_job_status.job_name] = init_job_status

    def update_phase(self, job_name: str, new_phase: Phase):
        self.job_mapping[job_name].phase = new_phase

    def allocate_worker(self, job_name: str) -> Optional[List]:
        '''Allocate `gpus_per_worker` GPUs to job_name'''
        gpus_per_worker = self.job_mapping[job_name].gpus_per_gen_worker
        assert self.cluster_size % gpus_per_worker == 0
        for i in range(0, self.cluster_size, gpus_per_worker):
            can_allocate = True
            for j in range(i, i + gpus_per_worker):
                if j not in self.available_devices:
                    can_allocate = False
            if can_allocate:
                for j in range(i, i + gpus_per_worker):
                    self.available_devices.remove(j)
                    assert self.resource_mapping[j] is None
                    self.resource_mapping[j] = job_name
                self.job_mapping[job_name].allocated_gpus += list(range(i, i + gpus_per_worker))
                return list(range(i, i + gpus_per_worker))
        return None

    def cleanup_by_name(self, job_name: str) -> int:
        '''Returns the released GPU count'''
        cleanup_count = 0
        for i in self.resource_mapping:
            if self.resource_mapping[i] == job_name:
                assert i not in self.available_devices
                self.resource_mapping[i] = None
                self.available_devices.append(i)
                cleanup_count += 1
        self.job_mapping[job_name].allocated_gpus = []
        self.available_devices.sort()
        return cleanup_count

    def allocate_all(self, job_name: str) -> None:
        '''Allocate all GPUs to job_name'''
        for i in self.resource_mapping:
            self.resource_mapping[i] = job_name
        self.job_mapping[job_name].allocated_gpus = self.available_devices.copy()
        self.available_devices = []

    def get_running_jobs(self):
        '''Get all running job names'''
        running_jobs = []
        for job in self.job_mapping:
            if len(self.job_mapping[job].allocated_gpus) != 0:
                running_jobs.append(job)
        return running_jobs
