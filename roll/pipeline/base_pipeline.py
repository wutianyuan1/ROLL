import os
import time
from contextlib import contextmanager
from concurrent import futures
from typing import List, Optional, Tuple
from collections import Counter

from codetiming import Timer
from transformers import set_seed
from redis import StrictRedis

from roll.distributed.executor.model_update_group import ModelUpdateGroup
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.utils.checkpoint_manager import CheckpointManager
from roll.utils.functionals import reduce_metrics
from roll.utils.logging import get_logger
from roll.utils.tracking import create_tracker
from roll.utils.worker_state import WorkerState
from roll.configs.base_config import BaseConfig


logger = get_logger()


def get_job_name(shared_storage: Optional[StrictRedis]) -> Tuple[str, int]:
    '''Returns the job_name (str) and job_id (int)'''
    if shared_storage is not None:
        shared_storage.setnx("job_id", 0)
        job_id = shared_storage.incr("job_id", 1)
        job_name = "job" + str(job_id)
    else:
        job_name = os.environ.get("JOB_NAME", "defaultjob")
        job_id = 0
    return job_name, job_id


class BasePipelineMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls)
        # Search through the passed-in arguments to find the config object
        config_obj = None
        for arg in args:
            if isinstance(arg, BaseConfig):
                config_obj = arg
        for arg in kwargs.values():
            if isinstance(arg, BaseConfig):
                config_obj = arg
        if hasattr(obj, '__pre_init__'):
            # Executed before __init__, right after object creation
            if config_obj is not None:
                num_gpus_per_gen_worker = config_obj.actor_infer.num_gpus_per_worker
                max_gen_gpus = len(config_obj.actor_infer.device_mapping)
                max_train_gpus = len(config_obj.actor_train.device_mapping)
            else:
                print("[BasePipelineMeta] cannot find config object")
                num_gpus_per_gen_worker = 1
                max_gen_gpus = 1
                max_train_gpus = 1
            print(f"[BasePipelineMeta] num_gpus_per_gen_worker={num_gpus_per_gen_worker}, max_gen_gpus={max_gen_gpus}, max_train_gpus={max_train_gpus}")
            obj.__pre_init__(num_gpus_per_gen_worker, max_gen_gpus, max_train_gpus)
        obj.__init__(*args, **kwargs)
        if hasattr(obj, '__post_init__'):
            # Executed after __init__
            obj.__post_init__()
        return obj


class BasePipeline(metaclass=BasePipelineMeta):
    model_update_groups: List[ModelUpdateGroup] = []
    checkpoint_clusters: List = []

    def __pre_init__(self,
                     num_gpus_per_gen_worker: int = 1,
                     max_gen_gpus: int = 1,
                     max_train_gpus: int = 1):
        self.master_addr = os.environ.get("MASTER_ADDR", "localhost")
        self.scheduler_port = int(os.environ.get("SCHEDULER_PORT", 9969))
        self.check_interval = 0.2
        self.step_counter = Counter()
        try:
            self.shared_storage = StrictRedis(
                host=self.master_addr,
                port=self.scheduler_port,
                db=0,
                decode_responses=True
            )
            self.shared_storage.set("test", "1")
        except:
            self.shared_storage = None
        self.job_name, self.job_id = get_job_name(self.shared_storage)
        print(f"=== Job created: <Name={self.job_name}, JobID={self.job_id}>")
        if self.shared_storage is not None:
            # (1) set job status to 'created' and notify the scheduler via redis pubsub
            self.set_key(f"{self.job_name}:status", "created", nx=True)
            self.shared_storage.publish("tenant_events", f"{self.job_name}:created")
            # (2) set gpus_per_gen_worker, max_gen_gpus, and max_train_gpus
            self.set_key(f"{self.job_name}:gpus_per_gen_worker", num_gpus_per_gen_worker)
            self.set_key(f"{self.job_name}:max_gen_gpus", max_gen_gpus)
            self.set_key(f"{self.job_name}:max_train_gpus", max_train_gpus)
            # (3) wait until the scheduler changes the status and wakes it up.
            self.wait_key(f"{self.job_name}:status", "initializing")

    def __init__(self, pipeline_config):
        set_seed(seed=pipeline_config.seed)
        self.pipeline_config = pipeline_config
        self.resource_manager = ResourceManager(da_2_num_nodes=self.pipeline_config.da_2_num_nodes,
                                                da_2_num_gpus_per_node=self.pipeline_config.da_2_num_gpus_per_node)
        self.state = WorkerState()
        self.checkpoint_manager = CheckpointManager(checkpoint_config=self.pipeline_config.checkpoint_config)
        self.tracker = create_tracker(
            tracker_name=self.pipeline_config.track_with,
            config=self.pipeline_config.to_dict(),
            **self.pipeline_config.tracker_kwargs,
        )
        self.resume_from_checkpoint = False
        self.executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=5)
        self.resume_futures = []

        if self.pipeline_config.resume_from_checkpoint:
            self.resume_from_checkpoint = self.pipeline_config.resume_from_checkpoint

            logger.info(f"resume_from_checkpoint: {self.resume_from_checkpoint}")
            load_dir = os.path.join(self.resume_from_checkpoint, "pipeline")
            self.state = WorkerState.load_from_json(load_dir=load_dir, tag="pipeline")

            def resume_metrics():
                for metrics in self.state.log_history:
                    self.tracker.log(values=metrics, step=metrics["system/step"])

            self.resume_futures.append(self.executor.submit(resume_metrics))

    def __post_init__(self):
        if self.shared_storage is not None:
            # Notify the scheduler that the resources used for initialization can be released
            for da, num_gpus in self.resource_manager.da_2_num_gpus.items():
                release_content = da + ',' + ",".join([str(i) for i in range(num_gpus)])
                self.shared_storage.publish("tenant_events", f"{self.job_name}:init:release_gpu[{release_content}]")
            # Publish that the current job's initialization is finished
            self.shared_storage.publish("tenant_events", f"{self.job_name}:init:done")
            # After initialization, change itself's status to 'running'
            self.set_key(f"{self.job_name}:status", "running")

    def wait_key(self, key: str, expected: str):
        """Wait until a the value corresponds to key in redis contains `expected`"""
        while True:
            resp = self.shared_storage.get(key)
            if resp is not None and expected in resp:
                break
            time.sleep(self.check_interval)
        return resp

    def set_key(self, key: str, value: str, nx=False):
        self.shared_storage.set(key, value, nx=nx)

    @contextmanager
    def scheduler_section(self, section_name: str):
        """
        A section for scheduler. When enters it, we should wait
        for a specific scheduler_key until its value contains
        expected_value, then give the control to the subsequent block.
        """
        try:
            if self.shared_storage is not None:
                # Safety: If status does not exist, set status to pending before taking actions
                self.set_key(f"{self.job_name}:{section_name}:status", "pending", nx=True)
                # Wait until the expected value occurs
                self.wait_key(f"{self.job_name}:{section_name}:status", "running")
                print(f"Entered context: {section_name} is ready to run now!")
            # Give the control to `with` block
            yield self.shared_storage
        finally:
            # Reset status to pending after the section finished
            self.set_key(f"{self.job_name}:{section_name}:status", "pending")
            # Notify the scheduler current section is finished.
            self.shared_storage.publish("tenant_events", f"{self.job_name}:{section_name}:done[{self.step_counter[section_name]}]")
            self.step_counter[section_name] += 1
            print(f"Exited context: {section_name}, finished steps = {self.step_counter[section_name]}")


    def run(self):
        pass

    def set_model_update_pair(self, src_cluster, tgt_cluster, frequency=1):
        self.model_update_groups.append(
            ModelUpdateGroup(src_cluster=src_cluster, tgt_cluster=tgt_cluster, frequency=frequency)
        )

    def set_checkpoint_clusters(self, *clusters):
        self.checkpoint_clusters.extend(clusters)

    def model_update(self, global_step):
        metrics = {}
        for model_update_group in self.model_update_groups:
            metrics.update(model_update_group.model_update(global_step))
        return metrics

    def do_checkpoint(self, global_step):
        metrics = self.state.log_history[-1]
        metrics["system/step"] = global_step
        if global_step > 0 and global_step % self.pipeline_config.save_steps == 0:
            ckpt_metrics_refss = []
            for cluster in self.checkpoint_clusters:
                ckpt_metrics_refss.append(cluster.do_checkpoint(global_step=global_step, blocking=False))

            for ckpt_metrics_refs in ckpt_metrics_refss:
                ckpt_metrics = DataProto.materialize_concat(data_refs=ckpt_metrics_refs)
                metrics.update(reduce_metrics(ckpt_metrics.meta_info.pop("metrics", {})))

            ckpt_id = f"checkpoint-{global_step}"
            pipeline_save_dir = os.path.join(self.pipeline_config.output_dir, "pipeline", ckpt_id)
            save_dir = os.path.join(self.pipeline_config.output_dir, "pipeline", ckpt_id, "pipeline")
            self.state.save_to_json(save_dir=save_dir, tag="pipeline")
            self.state.save_rng_state(save_dir=save_dir, tag="pipeline")
            self.checkpoint_manager.upload(ckpt_id=ckpt_id, local_state_path=pipeline_save_dir)

        futures.wait(self.resume_futures)
        self.resume_futures.clear()
