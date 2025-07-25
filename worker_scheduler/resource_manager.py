from typing import List, Optional, Dict, Tuple
from event import Phase


class ResourceManager:
    def __init__(self, gen_device_mapping: List[int]) -> None:
        self.available_devices = sorted(gen_device_mapping.copy())
        self.resource_mapping: Dict[int, str] = {i: None for i in gen_device_mapping}
        # job_mapping: job_name -> (phase, allocated_gpus)
        self.job_mapping: Dict[str, Tuple[Phase, List[int]]] = {}
        self.cluster_size = len(self.available_devices)

    def release(self, job_name: str, gpu_list: List[int]):
        '''Release the given GPU list'''
        assert len(set(gpu_list).intersection(self.available_devices)) == 0
        self.available_devices += gpu_list
        self.available_devices.sort()
        for i in gpu_list:
            self.resource_mapping[i] = None
            self.job_mapping[job_name][1].remove(i)

    def register_job(self, job_name: str):
        self.job_mapping[job_name] = [Phase.INIT, []]

    def update_phase(self, job_name: str, new_phase: Phase):
        self.job_mapping[job_name][0] = new_phase

    def allocate_worker(self, job_name: str, gpu_per_worker: int) -> Optional[List]:
        '''Allocate `gpu_per_worker` GPUs to job_name'''
        assert self.cluster_size % gpu_per_worker == 0
        for i in range(0, self.cluster_size, gpu_per_worker):
            can_allocate = True
            for j in range(i, i + gpu_per_worker):
                if j not in self.available_devices:
                    can_allocate = False
            if can_allocate:
                for j in range(i, i + gpu_per_worker):
                    self.available_devices.remove(j)
                    assert self.resource_mapping[j] is None
                    self.resource_mapping[j] = job_name
                self.job_mapping[job_name][1] += list(range(i, i + gpu_per_worker))
                return list(range(i, i + gpu_per_worker))
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
        self.job_mapping[job_name][1] = []
        self.available_devices.sort()
        return cleanup_count

    def allocate_all(self, job_name: str) -> None:
        '''Allocate all GPUs to job_name'''
        for i in self.resource_mapping:
            self.resource_mapping[i] = job_name
        self.job_mapping[job_name][1] = self.available_devices.copy()
        self.available_devices = []

    def get_running_jobs(self):
        '''Get all running job names'''
        running_jobs = []
        for job in self.job_mapping:
            if len(self.job_mapping[job][1]) != 0:
                running_jobs.append(job)
        return running_jobs
