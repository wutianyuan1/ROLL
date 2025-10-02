import asyncio
import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

from transformers import PreTrainedTokenizer, ProcessorMixin

from roll.agentic.rollout.base_env_manager import BaseEnvManager
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider, default_processor_provider
from roll.pipeline.agentic.agentic_config import EnvManagerConfig
from roll.pipeline.agentic.env_manager.step_env_manager import StepEnvManager
from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager
from roll.utils.import_utils import safe_import_class
import os

class VLTrajEnvManager: pass

class EnvironmentWorker(Worker):
    """
      Within a group, all environments share identical states by using the same seed.
      To reduce the overhead of dedicating one process per environment, parallelism is redesigned as **process + threads** :
      - One `EnvironmentWorker` holds multiple `EnvStateManager`s.
      - Each `EnvStateManager` manages the rollout loop for a single environment.
      - `EnvStateManager.run_rollout_loop` runs inside dedicated threads.
        TODO: GiGPO: https://arxiv.org/abs/2505.10978
    """

    def __init__(self, worker_config: EnvManagerConfig):
        super().__init__(worker_config)
        self.worker_config: EnvManagerConfig = worker_config
        self.env_managers: Dict[int, BaseEnvManager] = {}
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.processor: Optional[ProcessorMixin] = None
        self.env_configs: Dict[int, Dict] = worker_config.env_configs[self.rank]
        self.thread_lock = threading.Lock()
        self.output_queue = None

        # 添加日志目录配置
        self.log_dir = getattr(worker_config, 'log_dir', './output/logs/env_worker')
        os.makedirs(self.log_dir, exist_ok=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def initialize(self,
                   pipeline_config,
                   generate_scheduler,
                   output_queue,
                   collator: Optional[callable] = None,
                   mode: str = "train"):
        super().initialize(pipeline_config)
        self.output_queue = output_queue
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.processor = default_processor_provider(model_args=self.worker_config.model_args)
        def create_env_manager(env_id, env_config):
            self.logger.info(f"use env_manager_cls: {env_config['env_manager_cls']}")
            env_manager_cls = safe_import_class(env_config["env_manager_cls"])

            assert env_manager_cls is not None

            # 为每个环境创建独立的日志文件
            env_log_file = os.path.join(self.log_dir, f"worker_{self.rank}_env_{env_id}_{env_config.get('env_type', 'unknown')}.log")
            
            # 创建环境特定的日志配置
            env_log_config = copy.deepcopy(env_config)
            env_log_config['log_file'] = env_log_file
            env_log_config['env_id'] = env_id
            env_log_config['worker_id'] = self.rank


            # if env_manager_cls in [TrajEnvManager, StepEnvManager]:
            if issubclass(env_manager_cls, (TrajEnvManager, StepEnvManager)):
                return env_id, env_manager_cls(
                    worker_config=self.worker_config,
                    pipeline_config=pipeline_config,
                    env_config=env_log_config,  # 使用包含日志配置的配置
                    tokenizer=copy.deepcopy(self.tokenizer),
                    generate_scheduler=generate_scheduler,
                    output_queue=output_queue,
                    thread_lock=self.thread_lock,
                    mode=mode
                )
            # elif env_manager_cls == VLTrajEnvManager:
            elif issubclass(env_manager_cls, VLTrajEnvManager):
                tokenizer = copy.deepcopy(self.tokenizer)
                processor = copy.deepcopy(self.processor)
                return env_id, env_manager_cls(
                    worker_config=self.worker_config,
                    pipeline_config=pipeline_config,
                    env_config=env_log_config,  # 使用包含日志配置的配置
                    tokenizer=tokenizer,
                    processor=processor,
                    generate_scheduler=generate_scheduler,
                    output_queue=output_queue,
                    thread_lock=self.thread_lock,
                    mode=mode
                )
            else: raise ValueError(f"Unsupported env_manager_cls: {env_manager_cls}")
        with ThreadPoolExecutor(max_workers=min(len(self.env_configs), 64)) as executor:
            futures = [
                executor.submit(create_env_manager, env_id, env_config)
                for env_id, env_config in self.env_configs.items()
            ]
            for future in as_completed(futures):
                try:
                    env_id, env_manager = future.result()
                    self.env_managers[env_id] = env_manager
                except Exception as e:
                    self.logger.error(f"Failed to initialize env_manager: {e}", exc_info=True)
                    raise e

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def run_rollout_loop(self, current_step, seed):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=len(self.env_managers)) as pool:
            try:
                await asyncio.gather(
                    *[
                        loop.run_in_executor(pool, env_manager.run_rollout_loop, DataProto(meta_info={"current_step": current_step, "seed": seed}))
                        for env_manager in self.env_managers.values()
                    ]
                )
            except Exception as e:
                self.logger.error(f"EnvManager run with except: {e}", exc_info=True)
                ref = self.output_queue.put_exception.remote(e)
                await asyncio.wrap_future(ref.future())
                raise e

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def update_step(self, global_step):
        for env_manager in self.env_managers.values():
            env_manager.update_step(global_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def stop(self):
        for env_manager in self.env_managers.values():
            env_manager.stop()
