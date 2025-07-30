import time
import ray
from typing import List, Dict, Tuple, Callable
from roll.third_party.vllm.vllm_utils import EngineStats
from roll.distributed.scheduler.migration_scheduler import MigrationSchedulerBase


class StaticAggMigrationSchedulerTest(MigrationSchedulerBase):
    CHECK_INTERVAL = 0.5

    def __init__(self, dest_workers: List[int], max_bsz: int):
        super().__init__()
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.N_dest = len(dest_workers)
        self.dest_workers = dest_workers
        self.max_bsz = max_bsz
        self.migrated = False

    def _aggregate_reqs(self, total_workers: int, request_mapping: Dict[str, int]) -> Dict[int, List[str]]:
        per_worker_reqs = {i: [] for i in range(total_workers)}
        for req_id, worker_id in request_mapping.items():
            if worker_id not in per_worker_reqs:
                per_worker_reqs[worker_id] = [req_id]
            else:
                per_worker_reqs[worker_id].append(req_id)
        return per_worker_reqs

    def migrate(self, get_worker_stat_fns: List[Callable[[], EngineStats]], request_mapping: Dict[str, int]
                )-> Dict[Tuple[int, int], List[str]]:
        current_time = time.time()
        if current_time - self.last_check_time <= StaticAggMigrationSchedulerTest.CHECK_INTERVAL:
            return {}
        self.last_check_time = current_time
        print(f"==== scheduler: check at {current_time - self.start_time}")
        if current_time - self.start_time >= 5 and (not self.migrated):
            worker_stats: List[EngineStats] = [ray.get(fn.remote()) for fn in get_worker_stat_fns]
            engine_unfinished_reqs = [i.num_unfinished_reqs for i in worker_stats]
            # If remaining reqs more than destination capacity, not migrate
            if sum(engine_unfinished_reqs) > self.max_bsz * self.N_dest:
                return {}
            # Else (remaining can be aggregated), then perform migration
            self.migrated = True
            per_worker_reqs = self._aggregate_reqs(len(get_worker_stat_fns), request_mapping)
            remaining_slots = {i: self.max_bsz - len(per_worker_reqs[i]) for i in self.dest_workers}
            all_requests_to_migrate = []
            for worker_id, reqs in per_worker_reqs.items():
                if worker_id in self.dest_workers:
                    continue
                all_requests_to_migrate += [(worker_id, r) for r in reqs]
            migration_plan = {}
            req_count = 0
            all_finish = False
            for dest_wid in remaining_slots:
                for _ in range(remaining_slots[dest_wid]):
                    src_wid, req_id = all_requests_to_migrate[req_count]
                    key = (src_wid, dest_wid)
                    if key not in migration_plan:
                        migration_plan[key] = [req_id]
                    else:
                        migration_plan[key].append(req_id)
                    req_count += 1
                    if req_count == len(all_requests_to_migrate):
                        all_finish = True
                        break
                if all_finish:
                    break
            return migration_plan
        else:
            return {}
