import os
from typing import List, Optional, Dict, Tuple
from event import Phase
from dataclasses import dataclass

@dataclass
class JobStatus:
    job_name: str
    phase: Phase
    gpus_per_gen_worker: int
    gen_gpu_list: List[int]
    train_gpu_list: List[int]
    # For a running generation phase, 1 <= |allocated_gpus| <= |gen_gpu_list|
    # For a running train phase, |allocated_gpus| = |train_gpu_list|
    allocated_gpus: List[int]


def parse_job_offset(fn: str):
    with open(fn, 'r') as f:
        content = f.read().split("\n")
    offset_mapping = {}
    for line in content:
        if len(line) <= 1 or line[0] == '#':
            continue
        job, gen_offset, train_offset = line.split(" ")
        offset_mapping[job] = (int(gen_offset), int(train_offset))
    return offset_mapping


class ResourceManager:
    def __init__(self, gen_device_affinity: str, gen_device_ids: List[int], train_device_affinity: str,  train_device_ids: List[int], offset_mapping_fn: str) -> None:
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
        # offset_mapping: maps job's local GPU ID to global GPU ID
        self.offset_mapping: Dict[str, Tuple[int, int]] = parse_job_offset(offset_mapping_fn)
        self.is_colo = (os.environ.get("COLO", "0") == "1")

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
        if self.gen_device_affinity == self.train_device_affinity:
            # unified device types, then train and gen resources should have different device IDs
            '''
            assert len(
                set(self.train_resource_mapping.keys())\
                .intersection(self.gen_resource_mapping.keys())
            ) == 0
            '''
            for i in gpu_list:
                if i in self.gen_resource_mapping:
                    self.gen_resource_mapping[i] = None
                    self.gen_available_devices.append(i)
                if i in self.train_resource_mapping:
                    self.train_resource_mapping[i] = None
                    self.train_available_devices.append(i)
                else:
                    raise ValueError(f"Try to release GPU {i}, but neither in train {self.train_resource_mapping} nor gen {self.gen_resource_mapping}.")
                self.job_mapping[job_name].allocated_gpus.remove(i)
            if self.is_colo:
                self.train_available_devices = list(set(self.train_available_devices))
                self.gen_available_devices = list(set(self.gen_available_devices))
            self.train_available_devices.sort()
            self.gen_available_devices.sort()
        else:
            if device_affinity == self.gen_device_affinity:
                # assert len(set(gpu_list).intersection(self.gen_available_devices)) == 0
                for i in gpu_list:
                    assert i in self.gen_resource_mapping, f"Try to release GPU: {i}, which is not in gen pool"
                    gid = i + self.offset_mapping[job_name][0]
                    self.gen_resource_mapping[gid] = None
                    self.gen_available_devices.append(gid)
                    self.job_mapping[job_name].allocated_gpus.remove(gid)
                self.gen_available_devices.sort()
            else:
                # assert len(set(gpu_list).intersection(self.train_available_devices)) == 0
                for i in gpu_list:
                    assert i in self.train_resource_mapping, f"Try to release GPU: {i}, which is not in train pool"
                    gid = i + self.offset_mapping[job_name][1]
                    self.train_resource_mapping[gid] = None
                    self.train_available_devices.append(gid)
                    self.job_mapping[job_name].allocated_gpus.remove(gid)
                self.train_available_devices.sort()

    def register_job(self, init_job_status: JobStatus):
        assert len(init_job_status.gen_gpu_list) <= self.gen_cluster_size
        assert len(init_job_status.train_gpu_list) <= self.train_cluster_size
        self.job_mapping[init_job_status.job_name] = init_job_status

    def update_phase(self, job_name: str, new_phase: Phase):
        self.job_mapping[job_name].phase = new_phase

    def allocate_worker(self, job_name: str, pool: str) -> Optional[List]:
        '''Allocate `gpus_per_worker` GPUs to job_name'''
        assert pool in ['train', 'gen'], f"Resource pool can only be train or gen, got {pool}"
        if pool == 'gen':
            gpus_per_worker = self.job_mapping[job_name].gpus_per_gen_worker
            assert self.gen_cluster_size % gpus_per_worker == 0
            offset = self.offset_mapping[job_name][0]
            start_gen_gid = min(self.job_mapping[job_name].gen_gpu_list) + offset
            stop_gen_gid = max(self.job_mapping[job_name].gen_gpu_list) + offset + 1
            for i in range(start_gen_gid, stop_gen_gid, gpus_per_worker):
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
                    return list(range(i - offset, i + gpus_per_worker - offset))
        else:
            # For train allocation request, we allocate |train_gpu_list| for it at once
            offset = self.offset_mapping[job_name][1]
            num_train_gpus = len(self.job_mapping[job_name].train_gpu_list)
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
            return [i - offset for i in gpus_to_allocate]
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
        offset = self.offset_mapping[job_name][0]
        for i in self.job_mapping[job_name].gen_gpu_list:
            self.gen_resource_mapping[i + offset] = job_name
            try:
                self.gen_available_devices.remove(i + offset)
            except:
                assert False, f"i: {i}, offset: {offset}, gen_available_devices: {self.gen_available_devices}, offset_mapping: {self.offset_mapping}, "
        gen_global_ids = [i + offset for i in self.job_mapping[job_name].gen_gpu_list]
        offset = self.offset_mapping[job_name][1]
        for i in self.job_mapping[job_name].train_gpu_list:
            self.train_resource_mapping[i + offset] = job_name
            self.train_available_devices.remove(i + offset)
        train_global_ids = [i + offset for i in self.job_mapping[job_name].train_gpu_list]
        # self.job_mapping[job_name].allocated_gpus = list(set(self.gen_available_devices.copy() + self.train_available_devices.copy()))
        self.job_mapping[job_name].allocated_gpus = list(set(gen_global_ids + train_global_ids)) if self.is_colo else gen_global_ids + train_global_ids
        # self.gen_available_devices = []
        # self.train_available_devices = []

    def get_running_jobs(self):
        '''Get all running job names'''
        running_jobs = []
        for job in self.job_mapping:
            if len(self.job_mapping[job].allocated_gpus) != 0:
                running_jobs.append(job)
        return running_jobs

    def get_free_gpus(self, job_name: str, pool: str):
        assert pool in ['gen', 'train']
        if pool == 'gen':
            offset = self.offset_mapping[job_name][0]
            return [i for i in self.job_mapping[job_name].gen_gpu_list if self.gen_resource_mapping[i + offset] is None]
        else:
            offset = self.offset_mapping[job_name][1]
            return [i for i in self.job_mapping[job_name].train_gpu_list if self.train_resource_mapping[i + offset] is None]


def test_colo():
    rm = ResourceManager('NVIDIA H800', list(range(8)), 'NVIDIA H800', list(range(8)), 'examples/case3_mapping.txt')
    rm.register_job(JobStatus('job1', Phase.INIT, 1, list(range(8)), list(range(8)), []))
    rm.allocate_all('job1')
    print(rm.gen_available_devices, rm.train_available_devices, rm.gen_resource_mapping, rm.train_resource_mapping, rm.job_mapping['job1'].allocated_gpus)
    rm.release('job1', 'NVIDIA H800', list(range(8)))
    print(rm.gen_available_devices, rm.train_available_devices, rm.gen_resource_mapping, rm.train_resource_mapping, rm.job_mapping['job1'].allocated_gpus)
    for iter in range(5):
        print(f"==== iter {iter} ====")
        print(rm.gen_available_devices, rm.train_available_devices, rm.gen_resource_mapping, rm.train_resource_mapping, rm.job_mapping['job1'].allocated_gpus)
        for i in range(8):
            rm.allocate_worker('job1', 'gen')
        print(rm.gen_available_devices, rm.train_available_devices, rm.gen_resource_mapping, rm.train_resource_mapping, rm.job_mapping['job1'].allocated_gpus)
        rm.release('job1', 'NVIDIA H800', list(range(8)))
        print(rm.gen_available_devices, rm.train_available_devices, rm.gen_resource_mapping, rm.train_resource_mapping, rm.job_mapping['job1'].allocated_gpus)
        rm.allocate_worker("job1", 'train')
        print(rm.gen_available_devices, rm.train_available_devices, rm.gen_resource_mapping, rm.train_resource_mapping, rm.job_mapping['job1'].allocated_gpus)
        rm.cleanup_by_name("job1")
        print(rm.gen_available_devices, rm.train_available_devices, rm.gen_resource_mapping, rm.train_resource_mapping, rm.job_mapping['job1'].allocated_gpus)


def test_disagg():
    rm = ResourceManager('NVIDIA H20', list(range(24)), 'NVIDIA H800', list(range(8)), 'examples/case3_mapping.txt')
    rm.register_job(JobStatus('job1', Phase.INIT, 1, list(range(8)), list(range(8)), []))
    rm.register_job(JobStatus('job2', Phase.INIT, 1, list(range(8)), list(range(8)), []))
    rm.register_job(JobStatus('job3', Phase.INIT, 1, list(range(8)), list(range(8)), []))

    rm.allocate_all('job1')
    rm.release('job1', 'NVIDIA H20', list(range(8)))
    rm.release('job1', 'NVIDIA H800', list(range(8)))

    rm.allocate_all('job2')
    rm.release('job2', 'NVIDIA H20', list(range(8)))
    rm.release('job2', 'NVIDIA H800', list(range(8)))

    rm.allocate_all('job3')
    rm.release('job3', 'NVIDIA H20', list(range(8)))
    rm.release('job3', 'NVIDIA H800', list(range(8)))

    print("==="*10)
    for i in range(8):
        rm.allocate_worker('job1', 'gen')
        rm.allocate_worker('job2', 'gen')
        rm.allocate_worker('job3', 'gen')
        print("==="*5)
        for i in range(1, 4):
            print(rm.get_free_gpus(f'job{i}', 'gen'))

    print("==="*10)
    for i in range(8):
        rm.release('job1', 'NVIDIA H20', [i])
        rm.release('job2', 'NVIDIA H20', [i])
        rm.release('job3', 'NVIDIA H20', [i])
        print("==="*5)
        for i in range(1, 4):
            print(rm.get_free_gpus(f'job{i}', 'gen'))

    print("==="*10)
    rm.allocate_worker('job1', 'train')
    print(rm.get_free_gpus('job1', 'train'), rm.get_free_gpus('job2', 'train'), rm.get_free_gpus('job3', 'train'))
    rm.cleanup_by_name('job1')
    print(rm.get_free_gpus('job1', 'train'), rm.get_free_gpus('job2', 'train'), rm.get_free_gpus('job3', 'train'))

    rm.allocate_worker('job2', 'train')
    print(rm.get_free_gpus('job1', 'train'), rm.get_free_gpus('job2', 'train'), rm.get_free_gpus('job3', 'train'))
    rm.cleanup_by_name('job2')
    print(rm.get_free_gpus('job1', 'train'), rm.get_free_gpus('job2', 'train'), rm.get_free_gpus('job3', 'train'))

    rm.allocate_worker('job3', 'train')
    print(rm.get_free_gpus('job1', 'train'), rm.get_free_gpus('job2', 'train'), rm.get_free_gpus('job3', 'train'))
    rm.cleanup_by_name('job3')
    print(rm.get_free_gpus('job1', 'train'), rm.get_free_gpus('job2', 'train'), rm.get_free_gpus('job3', 'train'))


if __name__ == "__main__":
    if os.environ.get("COLO", "0") == "1":
        test_colo()
    else:
        test_disagg()
    