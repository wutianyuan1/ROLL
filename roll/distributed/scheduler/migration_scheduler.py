import time
import ray

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Callable
from roll.third_party.vllm.vllm_utils import EngineStats


class MigrationSchedulerBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def migrate(self, get_worker_stat_fns: List[Callable[[], EngineStats]], request_mapping: Dict[str, int]
                )-> Dict[Tuple[int, int], List[str]]:
        '''
        Input:
            - get_worker_stat_fns: A list of functions. When calling an item in it,
            it returns each worker's engine stats, including running/waiting/swapped
            /unfinished request count and max batch size of each worker.
            - request_mapping: a dictionary maps from a request_id to its worker_id.
        Output:
            - a dictionary, which key is a tuple (from_worker, to_worker), and the value
            is corresponding request_id's to migrate from `from_worker` to `to_worker`, note
            that vLLM uses strings as req_ids.
        '''
        raise NotImplementedError("Not implemented in base class")


class NaiveMigrationScheduler(MigrationSchedulerBase):
    CHECK_INTERVAL = 1

    def __init__(self):
        super().__init__()
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.migrated = False

    def _aggregate_reqs(self, request_mapping: Dict[str, int]) -> Dict[int, List[str]]:
        per_worker_reqs = {}
        for req_id, worker_id in request_mapping.items():
            if worker_id not in per_worker_reqs:
                per_worker_reqs[worker_id] = [req_id]
            else:
                per_worker_reqs[worker_id].append(req_id)
        return per_worker_reqs

    def migrate(self, get_worker_stat_fns: List[Callable[[], EngineStats]], request_mapping: Dict[str, int]
                )-> Dict[Tuple[int, int], List[str]]:
        '''
        Naive policy, assume we only have several (e.g., 128) requests on 4 separate workers,
        request_id%4 is worker_id. We migrate all requests to worker_1 after 5 seconds since
        start, and if the requests are already migrated, we do not migrate them again.
        '''
        current_time = time.time()
        if current_time - self.last_check_time <= NaiveMigrationScheduler.CHECK_INTERVAL:
            print(f"==== scheduler: skip ({current_time - self.last_check_time})")
            return {}
        self.last_check_time = current_time
        if current_time - self.start_time >= 5 and (not self.migrated):
            self.migrated = True
            worker_stats: List[EngineStats] = [ray.get(fn.remote()) for fn in get_worker_stat_fns]
            engine_unfinished_reqs = [i.num_unfinished_reqs for i in worker_stats]
            print(f"==== scheduler check: migrated={self.migrated}, since_start={current_time - self.start_time}, engine_unfinished_reqs={engine_unfinished_reqs}, request_mapping_len={len(request_mapping)}")
            # TODO: Lunxi: This equation does not hold when num_return_sequences > 1,
            # as EngineStats.num_unfinished_reqs counts the number of unfinished responses in fact.
            # assert sum(engine_unfinished_reqs) == len(request_mapping), f"engine_unfinished_reqs({sum(engine_unfinished_reqs)}) != request_mapping_len({len(request_mapping)})"
            per_worker_reqs = self._aggregate_reqs(request_mapping)
            print(f"==== per-worker {per_worker_reqs}")
            return {(0, 1): per_worker_reqs[0], (2, 1): per_worker_reqs[2], (3, 1): per_worker_reqs[3]}
        else:
            return {}


class StaticAggMigrationScheduler(MigrationSchedulerBase):
    CHECK_INTERVAL = 1

    def __init__(self, dest_workers: List[int], max_batch_size: int):
        super().__init__()
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.migrated = False
        self.started = True
        self.dest_workers = dest_workers
        self.batch_size = max_batch_size

    def _aggregate_reqs(self, request_mapping: Dict[str, int], ready_workers: List[int]) -> Dict[int, List[str]]:
        per_worker_reqs = {worker_id: [] for worker_id in ready_workers}
        for req_id, worker_id in request_mapping.items():
            assert worker_id in per_worker_reqs
            per_worker_reqs[worker_id].append(req_id)
        return per_worker_reqs

    def migrate(self, get_worker_stat_fns: List[Callable[[], EngineStats]], request_mapping: Dict[str, int], ready_workers: List[int]
                )-> Dict[Tuple[int, int], List[str]]:
        '''
        static policy, assume we have several (e.g., 128) requests on N separate workers,
        We want to migrate them to dest_workers once possible. Migration only happens once.
        '''
        current_time = time.time()
        if current_time - self.last_check_time <= StaticAggMigrationScheduler.CHECK_INTERVAL:
            print(f"==== scheduler: skip ({current_time - self.last_check_time})")
            return {}
        self.last_check_time = current_time

        worker_stats: List[EngineStats] = [ray.get(fn.remote()) for fn in get_worker_stat_fns]
        engine_unfinished_reqs = [i.num_unfinished_reqs for i in worker_stats]
        if len(engine_unfinished_reqs) > 0 and (not self.started):
            self.started = True
            self.start_time = current_time
            self.last_check_time = current_time
        if self.started and current_time - self.start_time >= 5 and (not self.migrated):
            print(f"==== scheduler check: migrated={self.migrated}, since_start={current_time - self.start_time}, engine_unfinished_reqs={engine_unfinished_reqs}, request_mapping_len={len(request_mapping)}")
            if len(request_mapping) > 0 and sum(engine_unfinished_reqs) <= len(self.dest_workers) * self.batch_size:
                # key: ready worker id, value: # active reqs on this worker (may be 0)
                per_worker_reqs = self._aggregate_reqs(request_mapping, ready_workers)
                # src worker: a worker that has at least 1 req and do not belong to dst workers
                src_list = [i for i in range(len(get_worker_stat_fns)) if \
                            i in per_worker_reqs and len(per_worker_reqs[i]) > 0 and i not in self.dest_workers]
                for worker in self.dest_workers:
                    if worker not in ready_workers:
                        # Madoka: if the dest worker is not available, just do nothing...
                        self.migrated = True
                        return {}
                migration_plan = {}
                for dest in self.dest_workers:
                    avail = self.batch_size - len(per_worker_reqs[dest])
                    while avail > 0 and len(src_list) != 0:
                        src = src_list.pop()
                        if len(per_worker_reqs[src]) <= avail:
                            migration_plan[(src, dest)] = per_worker_reqs[src]
                            avail -= len(per_worker_reqs[src])
                        else:
                            migration_plan[(src, dest)] = per_worker_reqs[src][:avail]
                            per_worker_reqs[src] = per_worker_reqs[src][avail:]
                            assert len(per_worker_reqs[src]) > 0
                            avail = 0
                            src_list.append(src)
                self.migrated = True
                return migration_plan
            else:
                return {}
        else:
            return {}
