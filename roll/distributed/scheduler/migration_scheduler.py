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
