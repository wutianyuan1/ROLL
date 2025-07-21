import redis
import array
import enum
import dataclasses
from typing import Dict, List, Optional


class WorkerStatus(enum.Enum):
    PENDING = enum.auto()
    INITIALIZING = enum.auto()
    RUNNING = enum.auto()
    COMPLETED = enum.auto()
    CRASHED = enum.auto()
    UNKNOWN = enum.auto()

status_map = {name.lower(): status for (name, status) in WorkerStatus._member_map_.items()}


class ActorWorkerMonitor:
    def __init__(self, actor_name: str, shared_storage: redis.StrictRedis) -> None:
        self.actor_name = actor_name
        self.shared_storage = shared_storage

    def _try_get(self, key, return_bytes=False) -> Optional[str]:
        try:
            resp = self.shared_storage.get(key)
            if return_bytes:
                return resp
            else:
                assert hasattr(resp, 'decode'), f'Response has type {type(resp)}, which cannot decode'.capitalize
                return resp.decode()
        except Exception as e:
            print(f"**** Error in redis get: {e}")
            return None

    def get_device_mapping(self) -> Optional[List[int]]:
        device_bytes = self._try_get(f"{self.actor_name}_device_mapping")
        if device_bytes is None:
            return None
        devices = array.array('I')
        devices.frombytes(device_bytes)
        return devices.tolist()

    def get_status(self) -> WorkerStatus:
        status_str = self._try_get(f'{self.actor_name}_status', return_bytes=False)
        if status_str in status_map:
            return status_map[status_str]
        return WorkerStatus.UNKNOWN

    def set_status(self, status: WorkerStatus) -> None:
        self.shared_storage.set(f"{self.actor_name}_status", status.name.lower())


'''
Assumption: we can schedule generation job in worker granularity,
but reference and training should be view as a whole.
HACK: Suppose there are N GPUs in total, and a tenant requires
T GPUs per worker. Although it may only need K workers (K < N/T),
its device mapping still contain all N GPUs. But the scheduler may
only give it K*T GPUs at the same time. This is to allow we can
place the K workers on any subset of these N GPUs. The world_size is
the actual value of K.
'''
@dataclasses.dataclass
class TenantConfig:
    gen_world_size: int
    gen_gpus_per_worker: int
    gen_all_devices: List[int]
    train_all_devices: List[int]


class TenantJob:
    def __init__(self, name: str, tenant_config: TenantConfig, shared_storage: redis.StrictRedis) -> None:
        self.name = name
        self.config = tenant_config
        self.shared_storage = shared_storage
        self.monitors: Dict[str, ActorWorkerMonitor] = {}
        for i in range(self.config.gen_world_size):
            self.monitors[f"{self.name}_actor_infer-{i}"] =\
                ActorWorkerMonitor(f"{self.name}_actor_infer-{i}", shared_storage)
        self.monitors[f"{self.name}_train"] =\
            ActorWorkerMonitor(f"{self.name}_train", shared_storage)

    def get_ref_train(self) -> ActorWorkerMonitor:
        return self.monitors[f"{self.name}_train"]

    def get_gen(self, worker_id) -> ActorWorkerMonitor:
        return self.monitors[f"{self.name}_actor_infer-{worker_id}"]
