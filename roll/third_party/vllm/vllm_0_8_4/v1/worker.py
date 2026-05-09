import gc
from typing import Optional

import torch
from vllm.v1.worker.gpu_worker import Worker
from vllm.device_allocator.cumem import CuMemAllocator

from roll.third_party.vllm.worker_helper import WorkerHelper
from roll.utils.logging import get_logger
from roll.utils.send_recv_utils import RecvBucketManager

logger = get_logger()


class Worker084(Worker, WorkerHelper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_loaded : bool = True
        self.kv_cache_loaded : bool = True

    def reload_model(self):
        if not self.weight_loaded:
            self.wake_up(["weights"])
            self.weight_loaded = True

    def load_states(self):
        self.reload_model()
        if not self.kv_cache_loaded:
            self.wake_up(["kv_cache"])
            self.kv_cache_loaded = True

    def offload_states(self):
        assert (self.weight_loaded and self.kv_cache_loaded) or (not self.weight_loaded and not self.kv_cache_loaded)
        if not self.weight_loaded:
            return
        self.sleep()
        self.weight_loaded = False
        self.kv_cache_loaded = False
        if hasattr(self, 'recv_manager'):
            self.recv_manager.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def broadcast_bucket(self, src_pp_rank, meta_infos, bucket_size):
        RecvBucketManager.dict_to_meta(meta_infos)
        super().broadcast_bucket(src_pp_rank, meta_infos, bucket_size)

    def update_parameter_in_bucket(self, meta_infos, buffer, ranks_in_worker):
        RecvBucketManager.dict_to_meta(meta_infos)
        buffer = torch.tensor(buffer, dtype=torch.int8, device='cuda')
        super().update_parameter_in_bucket(meta_infos, buffer, ranks_in_worker)
