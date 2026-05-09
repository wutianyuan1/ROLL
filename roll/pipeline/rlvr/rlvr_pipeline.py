import copy
import json
import math
import os
import argparse
import dataclasses
from functools import partial
from typing import Any, Dict, List, Optional

import time
import psutil
import redis
import random
from codetiming import Timer
from pathlib import Path

try:
    import datasets
    import ray
    import torch
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from ray.util.timer import _Timer
    from ray.runtime_env import RuntimeEnv

    from roll.multi_tenant.lock_utils import redis_lock
    from roll.datasets.chat_template import get_chat_template
    from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys
    from roll.distributed.executor.cluster import Cluster
    from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
    from roll.distributed.scheduler.protocol import DataProto
    from roll.models.model_providers import default_tokenizer_provider
    from roll.pipeline.rlvr.rlvr_config import RLVRConfig
    from roll.utils.functionals import (
        compute_advantage,
        reduce_metrics,
        RunningMoments,
        get_sample_level_mask,
        reward_postprocess,
        compute_token_reward,
        agg_loss,
    )
    from roll.utils.kl_controller import get_kl_controller
    from roll.utils.metrics.metrics_manager import MetricsManager
    from roll.utils.gpu_mem_monitor import GPUMemoryMonitor
    HAS_RLVR_DEPS = True
    RLVR_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - emulated path fallback
    datasets = None
    ray = None
    class _TorchStub:
        @staticmethod
        def no_grad():
            def decorator(fn):
                return fn
            return decorator
    torch = _TorchStub()
    NodeAffinitySchedulingStrategy = None
    _Timer = None
    RuntimeEnv = None
    Cluster = None
    DynamicSamplingScheduler = None
    DataProto = None
    default_tokenizer_provider = None
    RLVRConfig = Any
    compute_advantage = None
    reduce_metrics = None
    RunningMoments = None
    get_sample_level_mask = None
    reward_postprocess = None
    compute_token_reward = None
    agg_loss = None
    get_kl_controller = None
    MetricsManager = None
    GPUMemoryMonitor = None
    redis_lock = None
    get_chat_template = None
    DataCollatorWithPaddingForPaddedKeys = None
    HAS_RLVR_DEPS = False
    RLVR_IMPORT_ERROR = exc

from roll.pipeline.base_pipeline import BasePipeline, PipelineSectionCrash
try:
    from roll.configs.base_config import BaseConfig
    from roll.configs.worker_config import WorkerConfig
except Exception:  # pragma: no cover - lightweight fallback for emulated demos
    @dataclasses.dataclass
    class WorkerConfig:
        name: Optional[str] = None
        worker_cls: Optional[str] = None
        device_mapping: str = "[0]"
        device_affinity: Optional[str] = None
        num_gpus_per_worker: int = 1
        training_args: Any = dataclasses.field(default_factory=lambda: type("TrainArgs", (), {"seed": 42, "max_steps": 1})())

        def __post_init__(self):
            if isinstance(self.device_mapping, str):
                self.device_mapping = eval(self.device_mapping)

    @dataclasses.dataclass
    class BaseConfig:
        exp_name: str = "emu"
        seed: int = 42
        rpc_timeout: int = 3600
        output_dir: str = "./output"
        logging_dir: str = "./output/logs"
        track_with: str = "stdout"
        tracker_kwargs: dict = dataclasses.field(default_factory=dict)
        max_steps: int = 1
        save_steps: int = 1000000
        logging_steps: int = 1
        eval_steps: int = 1000000
        rollout_batch_size: int = 1
        val_batch_size: int = 1
        local_rank: int = -1
        resume_from_checkpoint: bool = False
        checkpoint_config: dict = dataclasses.field(default_factory=dict)
        prompt_length: int = 16
        response_length: Optional[int] = None
        sequence_length: Optional[int] = 32
        alive_check_interval: int = 10
        profiler_timeline: bool = False
        profiler_memory: bool = False
        profiler_output_dir: str = "./output/profiler"
        system_envs: dict = dataclasses.field(default_factory=dict)
        num_nodes: int = 1
        num_gpus_per_node: int = 1

        def to_dict(self):
            return dataclasses.asdict(self)

        def __post_init__(self):
            if self.response_length is None and self.sequence_length is None:
                self.sequence_length = self.prompt_length
            da_2_total_devices = {}
            self.da_2_num_gpus_per_node = {}
            for attribute_name in dir(self):
                attribute = getattr(self, attribute_name)
                if isinstance(attribute, WorkerConfig):
                    da = getattr(attribute, "device_affinity", "Default")
                    da_2_total_devices.setdefault(da, [])
                    self.da_2_num_gpus_per_node.setdefault(da, self.num_gpus_per_node)
                    da_2_total_devices[da].extend(attribute.device_mapping)
            max_gpu_nums = {
                device_affinity: (max(devices) + 1 if devices else 1)
                for device_affinity, devices in da_2_total_devices.items()
            }
            self.da_2_num_nodes = {
                device_affinity: 1
                for device_affinity in da_2_total_devices
            }
from roll.utils.logging import get_logger

logger = get_logger()


def preprocess_dataset(dataset, prompt_len, encode_function, num_proc):
    # 处理数据
    print(f"Begin : {dataset}")
    dataset = dataset.map(
        encode_function,
        batched=True,
        num_proc=num_proc,
        desc="Encoding dataset",
        load_from_cache_file=False,
    )
    # 过滤cutoff
    dataset = dataset.filter(
        lambda data_i: 5 < len(data_i["input_ids"]) <= prompt_len,
        num_proc=num_proc,
        desc="Filtering dataset",
    )
    print(f"Filtering prompt len: {dataset}")
    print(f"Encoding: {dataset}")
    return dataset


def get_encode_function(template_name, tokenizer):
    chat_template_func = get_chat_template(template_name, tokenizer)

    def encode_function(data_i):
        text_list = []
        if "messages" in data_i:
            for messages in data_i["messages"]:
                if isinstance(messages, str):
                    messages = json.loads(messages)
                text_list.append(chat_template_func(messages))
        elif "prompt" in data_i:
            for prompt in data_i["prompt"]:
                text_list.append(prompt)
        encodings = tokenizer(text_list)
        return encodings

    return encode_function

def update_dataset_domain(tag_2_domain: Dict[str, set[str]], row):
    if 'domain' in row and row['domain'] is not None:
        return row
    row["domain"] = tag_2_domain.get(row["tag"], "math_rule")
    return row

def query_filter_fn(data_list: List[DataProto], config: RLVRConfig) -> bool:
    """
    各domain的过滤规则可以自定义
    """
    response_level_rewards = [data.batch["response_level_rewards"] for data in data_list]
    if len(response_level_rewards) == 1:
        return True
    rewards = torch.cat(response_level_rewards, dim=0)

    domain = data_list[0].non_tensor_batch["domain"][0]
    query_filter_config = config.rewards[domain].query_filter_config

    if query_filter_config.type == "no_filter":
        return True
    elif query_filter_config.type == "mean_filter":
        threshold_up = query_filter_config.filter_args.get("threshold_up", math.inf)
        threshold_down = query_filter_config.filter_args.get("threshold_down", -1)
        if torch.mean(rewards) <= threshold_down or torch.mean(rewards) >= threshold_up:
            return False
    elif query_filter_config.type == "std_filter":
        std_threshold = query_filter_config.filter_args.get("std_threshold", -1)
        if torch.std(rewards) <= std_threshold:
            return False
    return True


def get_mem_usage_str():
    mem = psutil.virtual_memory()
    return f"total={mem.total/1024**3} available={mem.available/1024**3} free={mem.free/1024**3} used={mem.used/1024**3}"


class RLVRPipeline(BasePipeline):

    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)
        init_stats_file = open("init_stats.txt", "w")
        init_stats_file.write(f"Mem_init: {get_mem_usage_str()}\n")

        dataset_paths = []
        if self.pipeline_config.actor_train.data_args.file_name:
            dataset_paths.extend(self.pipeline_config.actor_train.data_args.file_name)

        print(f'load_dataset_paths: {chr(10)} {chr(10).join(dataset_paths)}')
        dataset = datasets.load_dataset('json', data_files=dataset_paths)['train']

        self.val_dataset = None
        if self.pipeline_config.validation:
            val_dataset_paths = self.pipeline_config.validation.data_args.file_name
            self.val_dataset = datasets.load_dataset("json", data_files=val_dataset_paths)["train"]

        # 加上format，然后转ids的func
        template_name = (
            self.pipeline_config.global_template
            if self.pipeline_config.global_template
            else self.pipeline_config.actor_train.data_args.template
        )
        encode_function = get_encode_function(template_name, self.tokenizer)

        dataset = preprocess_dataset(
            dataset,
            self.pipeline_config.prompt_length,
            encode_function,
            num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
        )
        # update domain field
        dataset = dataset.map(
            partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
            num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
            desc="update_dataset_domain",
            load_from_cache_file=False
        )
        self.domain_datasets: Dict[str, datasets.Dataset] = {}
        for domain in self.pipeline_config.actor_train.data_args.domain_interleave_probs.keys():
            self.domain_datasets[domain] = dataset.filter(
                lambda example, dom: example["domain"] == dom,
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                fn_kwargs={"dom": domain},
            )
            assert len(self.domain_datasets[domain]) > 0, f"domain dataset {domain} has no data"

        if self.val_dataset:
            self.val_dataset = preprocess_dataset(
                self.val_dataset,
                self.pipeline_config.prompt_length,
                encode_function,
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
            )
            self.val_dataset = self.val_dataset.map(
                partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                desc="update_val_dataset_domain",
                load_from_cache_file=False
            )

        assert 'domain' in dataset.column_names, "domain field should set in dataset"
        assert 'domain' in self.val_dataset.column_names, "domain field should set in val dataset"
        print(dataset)

        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )

        assert self.pipeline_config.max_steps > 0, "max_steps must be greater than 0"
        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)

        self.actor_train: Any = Cluster(
            job_name=self.job_name,
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.actor_infer: Any = Cluster(
            job_name=self.job_name,
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        self.reference: Any = Cluster(
            job_name=self.job_name,
            name=self.pipeline_config.reference.name,
            worker_cls=self.pipeline_config.reference.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )
        if self.pipeline_config.adv_estimator == "gae":
            self.critic: Any = Cluster(
                job_name=self.job_name,
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )
        self.rewards: Dict[str, Any] = {
            key: Cluster(
                job_name=self.job_name,
                name=f"reward-{key}",
                worker_cls=worker_config.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=worker_config,
            )
            for key, worker_config in self.pipeline_config.rewards.items()
        }

        domain_ratios = self.pipeline_config.actor_train.data_args.domain_interleave_probs
        self.generate_schedulers: Dict[str, DynamicSamplingScheduler] = {}
        self.domain_batch_size = {}
        domain_list = list(domain_ratios.keys())
        accumulated = 0
        for i, domain in enumerate(domain_list):
            if i == len(domain_list) - 1:
                domain_batch_size = self.pipeline_config.rollout_batch_size - accumulated
            else:
                domain_batch_size = int(domain_ratios[domain] * self.pipeline_config.rollout_batch_size)
            accumulated += domain_batch_size
            generate_scheduler = DynamicSamplingScheduler.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
                runtime_env=RuntimeEnv(env_vars=dict(os.environ))
            ).remote(pipeline_config=self.pipeline_config, job_name=self.job_name)
            ray.get(
                generate_scheduler.set_scheduler.remote(
                    actor_cluster=self.actor_infer,
                    reward_clusters={domain: self.rewards[domain]},
                    dataset=self.domain_datasets[domain],
                    collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                    collect_fn_kwargs=dict(max_length=self.pipeline_config.prompt_length, padding="max_length"),
                    response_filter_fn=lambda data_item, config: True,
                    query_filter_fn=query_filter_fn,
                    response_callback_fn=generate_scheduler.report_response.remote,
                    state=self.state.kv.get(f"scheduler_state_{domain}", None),
                )
            )
            self.generate_schedulers[domain] = generate_scheduler
            self.domain_batch_size[domain] = domain_batch_size

            assert domain_batch_size < len(self.domain_datasets[domain]), (f"domain_batch_size {domain_batch_size} must be "
                                                                           f"less than the number of domain datasets {len(self.domain_datasets[domain])}")

        if self.val_dataset:
            val_pipeline_config = copy.deepcopy(self.pipeline_config)
            val_pipeline_config.use_additional_prompts = False
            val_pipeline_config.enable_migration = False
            self.val_generate_scheduler = DynamicSamplingScheduler.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
                runtime_env=RuntimeEnv(env_vars=dict(os.environ))
            ).remote(pipeline_config=val_pipeline_config, job_name=self.job_name)
        if self.val_dataset:
            ray.get(
                self.val_generate_scheduler.set_scheduler.remote(
                    actor_cluster=self.actor_infer,
                    reward_clusters=self.rewards,
                    dataset=self.val_dataset,
                    collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                    collect_fn_kwargs=dict(max_length=self.pipeline_config.prompt_length, padding="max_length"),
                    response_filter_fn=lambda data_item, config: True,
                    query_filter_fn=lambda data_list, config: True,
                    response_callback_fn=self.val_generate_scheduler.report_response.remote,
                )
            )

        init_stats_file.write(f"Mem_before_load: {get_mem_usage_str()}\n")
        t0 = time.time()
        refs = []
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)
        t1 = time.time()
        init_stats_file.write(f"Time_infer {t1-t0}\n")
        init_stats_file.write(f"Mem_after_infer: {get_mem_usage_str()}\n")

        refs = []
        refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=False))
        for key, cluster in self.rewards.items():
            refs.extend(cluster.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)
        t2 = time.time()
        init_stats_file.write(f"Time_reference_reward {t2-t1}\n")
        init_stats_file.write(f"Mem_after_reference_reward: {get_mem_usage_str()}\n")

        refs = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)
        refs = []

        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)
        t3 = time.time()
        init_stats_file.write(f"Time_train {t3-t2}\n")
        init_stats_file.write(f"Mem_after_train: {get_mem_usage_str()}\n")

        self.actor_infer.load_states()
        t4 = time.time()
        init_stats_file.write(f"Time_warm_infer {t4-t3}\n")
        self.actor_infer.offload_states(blocking=True)

        t5 = time.time()
        self.reference.load_states()
        self.actor_train.load_states()
        t6 = time.time()
        init_stats_file.write(f"Time_warm_train {t6-t5}\n")

        self.reference.offload_states(blocking=True)
        self.actor_train.offload_states(blocking=True)
        init_stats_file.close()
        del init_stats_file

        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

        if self.pipeline_config.adv_estimator == "gae":
            self.set_checkpoint_clusters(self.actor_train, self.critic)
        else:
            self.set_checkpoint_clusters(self.actor_train)

        self.running = {}
        for domain in self.rewards.keys():
            self.running[domain] = RunningMoments()

    @torch.no_grad()
    def run(self):
        # 计算tokens per second 系统吞吐

        # 创建一个专门管理监控指标的类
        metrics_mgr = MetricsManager()

        tps_timer = _Timer(window_size=5)
        actor_infer_timer = _Timer(window_size=5)
        actor_infer_response_timer = _Timer(window_size=5)
        actor_train_timer = _Timer(window_size=5)
        self.actor_train.offload_states(blocking=True)

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            logger.info(f"pipeline step {global_step} start...")

            metrics_mgr.clear_metrics()
            with tps_timer, Timer(name="step_total", logger=None) as step_total_timer:
                # with self.scheduler_section("update"), GPUMemoryMonitor(interval_sec=0.02) as gmonitor:
                ### Update Begin
                # 先model update，resume时不需要保存infer cluster的状态
                if self.pipeline_config.adv_estimator == "gae":
                    self.critic.offload_states(blocking=True)
                # self.actor_train.offload_states(blocking=True)

                with Timer(name="step_model_update", logger=None) as step_model_update_timer:
                    model_update_metrics: Dict = self.model_update(global_step)
                    # time.sleep(10)
                    metrics_mgr.add_metrics(model_update_metrics)
                    metrics_mgr.add_metric("time/step_model_update", step_model_update_timer.last)
                ### Update End

                batch: DataProto = DataProto()
                batch.meta_info = {"global_step": global_step}

                # 要按domain group by生成对应的batch
                with self.scheduler_section("generate"):
                    with actor_infer_timer, actor_infer_response_timer, Timer(
                        name="step_generate", logger=None
                    ) as step_generate_timer:
                        domain_batches = {}
                        batch.meta_info["generation_config"] = self.actor_infer.worker_config.generating_args.to_dict()
                        self.actor_infer.start_server(data=DataProto(meta_info=batch.meta_info))
                        for reward_cluster in self.rewards.values():
                            reward_cluster.load_states()

                        if self.val_dataset and global_step % self.pipeline_config.eval_steps == 0:
                            with Timer(name="val_step", logger=None) as val_step_timer:
                                val_metrics = self.val(init=False)
                                metrics_mgr.add_metrics(val_metrics)
                                metrics_mgr.add_metric("time/val_step", val_step_timer.last)

                        batch.meta_info["is_offload_states"] = False
                        scheduler_refs = {}
                        for domain, scheduler in self.generate_schedulers.items():
                            scheduler_refs[domain] = scheduler.get_batch.remote(data=batch, batch_size=self.domain_batch_size[domain])
                        for domain, scheduler_ref in scheduler_refs.items():
                            domain_batch: DataProto = ray.get(scheduler_ref, timeout=self.pipeline_config.rpc_timeout)
                            metrics_mgr.add_domain_metrics(
                                domain, reduce_metrics(domain_batch.meta_info.pop("metrics", {}))
                            )
                            domain_batches[domain] = domain_batch
                        generate_output = DataProto.concat([domain_batch for domain_batch in domain_batches.values()])
                        generate_output.meta_info.pop("is_offload_states", None)

                        for reward_cluster in self.rewards.values():
                            reward_cluster.offload_states()
                        gen_metrics = self.actor_infer.stop_server()
                    metrics_mgr.add_domain_metrics(domain, reduce_metrics(gen_metrics.meta_info.pop("metrics", {})))
                    metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

                batch = generate_output

                with self.scheduler_section("train"):
                    with Timer(name="cal_ref_log_probs", logger=None) as cal_ref_log_probs_timer:
                        ref_log_probs = self.reference.compute_log_probs(batch, blocking=True)
                        metrics_mgr.add_reduced_metrics(ref_log_probs.meta_info.pop("metrics", {}))
                        ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                        batch = batch.union(ref_log_probs)
                        metrics_mgr.add_metric("time/ref_log_probs_values", cal_ref_log_probs_timer.last)

                    with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                        batch.meta_info["is_offload_states"] = False
                        if self.pipeline_config.adv_estimator == "gae":
                            values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)
                        old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                        old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                        agg_entropy = agg_loss(
                            loss_mat=old_log_probs.batch["entropy"],
                            loss_mask=batch.batch["response_mask"][:, 1:],
                            loss_agg_mode="token-mean",
                        )
                        batch.meta_info["agg_entropy"] = agg_entropy

                        if self.pipeline_config.adv_estimator == "gae":
                            values = DataProto.materialize_concat(data_refs=values_refs)
                            batch = batch.union(values)
                            metrics_mgr.add_reduced_metrics(values.meta_info.pop("metrics", {}))

                        batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                        metrics_mgr.add_reduced_metrics(old_log_probs.meta_info.pop("metrics", {}))
                        metrics_mgr.add_metric("time/old_log_probs", cal_old_logpb_timer.last)

                    # 要按domain group by处理reward
                    batch.batch["prompt_id"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
                    batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                    batch_list = []
                    for domain, domain_batch in batch_grouped.items():
                        # 1. 处理mask相关策略， 获取sample level mask
                        with Timer(name="get_sample_level_mask", logger=None) as get_sample_level_mask_timer:
                            domain_batch, mask_metrics = get_sample_level_mask(domain_batch, self.pipeline_config)
                            metrics_mgr.add_metrics(mask_metrics)
                            metrics_mgr.add_metric("time/get_sample_level_mask", get_sample_level_mask_timer.last)

                        # 2. 处理reward相关策略
                        with Timer(name="reward_postprocess", logger=None) as reward_postprocess_timer:
                            domain_batch, response_level_metrics = reward_postprocess(
                                domain_batch, self.pipeline_config, self.running
                            )
                            metrics_mgr.add_metrics(response_level_metrics)
                            metrics_mgr.add_metric("time/reward_postprocess", reward_postprocess_timer.last)

                        # 3. 计算token level rewards
                        with Timer(name="get_token_reward", logger=None) as get_token_reward_timer:
                            domain_batch, token_level_metrics = compute_token_reward(
                                domain_batch, self.pipeline_config, self.kl_ctrl
                            )
                            metrics_mgr.add_metrics(token_level_metrics)
                            metrics_mgr.add_metric("time/get_token_reward", get_token_reward_timer.last)

                        # 4. 计算advantage
                        final_response_mask = domain_batch.batch["final_response_mask"].clone()
                        with Timer(name="compute_advantage", logger=None) as compute_advantage_timer:
                            domain_batch = compute_advantage(
                                data=domain_batch,
                                gamma=self.pipeline_config.gamma,
                                lambd=self.pipeline_config.lambd,
                                adv_estimator=self.pipeline_config.adv_estimator,
                                advantage_clip=self.pipeline_config.advantage_clip,
                                whiten_advantages=self.pipeline_config.whiten_advantages,
                                whiten_rewards=self.pipeline_config.whiten_rewards,
                                response_mask=final_response_mask,
                            )
                            domain_metrics = reduce_metrics(domain_batch.meta_info.pop("metrics", {}))
                            metrics_mgr.add_domain_metrics(domain, domain_metrics)
                            metrics_mgr.add_metric("time/compute_advantage", compute_advantage_timer.last)
                            batch_list.append(domain_batch)

                    batch = DataProto.concat(batch_list)
                    batch.reorder(indices=torch.argsort(batch.batch["prompt_id"]))
                    batch.pop("prompt_id")

                    metrics_mgr.add_all_metrics(
                        global_step,
                        batch,
                        resource_manager=self.resource_manager,
                        actor_infer=self.actor_infer,
                        actor_train=self.actor_train,
                    )
                    batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                    metrics_mgr.add_domain_all_metrics(global_step, batch_grouped)

                    with Timer(name="step_train", logger=None) as step_train_timer:
                        if self.pipeline_config.adv_estimator == "gae":
                            critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                        with actor_train_timer:
                            # implement critic warmup
                            if self.pipeline_config.critic_warmup <= global_step:
                                # update actor
                                actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                                actor_train_metrics: DataProto = DataProto.materialize_concat(
                                    data_refs=actor_train_metrics_refs
                                )
                                metrics_mgr.add_reduced_metrics(actor_train_metrics.meta_info.pop("metrics", {}))

                        if self.pipeline_config.adv_estimator == "gae":
                            critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                            metrics_mgr.add_reduced_metrics(critic_train_metrics.meta_info.pop("metrics", {}))

                        metrics_mgr.add_metric("time/step_train", step_train_timer.last)
                        self.actor_train.offload_states(blocking=True)
                    tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                    actor_infer_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                    actor_infer_response_timer.push_units_processed(
                        n=torch.sum(batch.batch["response_mask"]).detach().item()
                    )
                    actor_train_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

                    metrics = metrics_mgr.get_metrics()
                    # do ckpt
                    self.state.step = global_step
                    self.state.log_history.append(metrics)
                    for domain, scheduler in self.generate_schedulers.items():
                        self.state.kv[f"scheduler_state_{domain}"] = ray.get(scheduler.get_scheduler_state.remote())

                    self.do_checkpoint(global_step=global_step)

                    self.tracker.log(values=metrics, step=global_step)

                    if global_step % self.pipeline_config.logging_steps == 0:
                        if int(os.environ.get("RAY_PROFILING", "0")):
                            timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                            os.makedirs(timeline_dir, exist_ok=True)
                            ray.timeline(
                                filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                            )

                        prompts = self.tokenizer.batch_decode(generate_output.batch["prompts"], skip_special_tokens=True)
                        responses = self.tokenizer.batch_decode(
                            generate_output.batch["responses"], skip_special_tokens=True
                        )
                        generate_examples = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)][:10]
                        logger.info(json.dumps(generate_examples, ensure_ascii=False))
                        logger.info(json.dumps(metrics, ensure_ascii=False))

                logger.info(f"pipeline step {global_step} finished")
                global_step += 1
                if global_step >= int(os.environ.get("MAX_STEPS", self.pipeline_config.max_steps)):
                    break
        logger.info("pipeline complete!")

    @torch.no_grad()
    def val(self, init: bool = False):
        val_metrics_mgr = MetricsManager()
        batch = DataProto()

        with Timer(name="step_generate", logger=None) as step_generate_timer:
            batch.meta_info["is_offload_states"] = False
            batch.meta_info["generation_config"] = self.pipeline_config.validation.generating_args.to_dict()
            # As we invoke val() inside the generate phase, initialization should be skipped
            if init:
                self.actor_infer.start_server(data=DataProto(meta_info=batch.meta_info))
                for reward_cluster in self.rewards.values():
                    reward_cluster.load_states()
            generate_output: DataProto = ray.get(
                self.val_generate_scheduler.get_batch.remote(data=batch, batch_size=len(self.val_dataset)),
                timeout=self.pipeline_config.rpc_timeout
            )
            generate_output.meta_info.pop("is_offload_states", None)
            if init:
                self.actor_infer.stop_server()
                for reward_cluster in self.rewards.values():
                    reward_cluster.offload_states()
            val_metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

        batch = generate_output
        val_correct_mean = (batch.batch["scores"] == 1).detach().float().mean().item()
        val_metrics_mgr.add_metric("val_correct/all/mean", val_correct_mean)
        logger.info(json.dumps({"val_correct/all/mean": val_correct_mean}, ensure_ascii=False))

        epoch_batch = batch.pop(batch_keys=["scores"], non_tensor_batch_keys=["tag"])

        grouped_batch = epoch_batch.group_by("tag")
        for group_key, group_batch in grouped_batch.items():
            score_mean = group_batch.batch["scores"].mean().item()
            print(f"{group_key}:  {score_mean}")
            val_metrics_mgr.add_domain_metrics(
                "val_correct", {f"{group_key}/mean": (group_batch.batch["scores"] == 1).detach().float().mean().item()}
            )

        return val_metrics_mgr.get_metrics()


@dataclasses.dataclass
class EmulatedRLVRConfig(BaseConfig):
    actor_infer: WorkerConfig = dataclasses.field(default_factory=WorkerConfig)
    actor_train: WorkerConfig = dataclasses.field(default_factory=WorkerConfig)


class EmulatedRLVRPipeline(BasePipeline):
    def __init__(
        self,
        pipeline_config: EmulatedRLVRConfig,
        generate_real_sec: float,
        train_real_sec: float,
        time_scale: float,
        max_iters: int,
        event_log_dir: str,
        failure_iter: Optional[int] = None,
        failure_phase: Optional[str] = None,
        failure_after_ratio: float = 0.0,
    ):
        super().__init__(pipeline_config)
        self.generate_real_sec = generate_real_sec
        self.train_real_sec = train_real_sec
        self.time_scale = time_scale
        self.max_iters = max_iters
        self.failure_iter = failure_iter
        self.failure_phase = failure_phase
        self.failure_after_ratio = failure_after_ratio
        self.event_log_dir = Path(event_log_dir)
        self.event_log_dir.mkdir(parents=True, exist_ok=True)
        self.event_log_path = self.event_log_dir / f"{self.job_name}.jsonl"
        self.current_iter = self._load_current_iter()

    def _state_key(self, name: str) -> str:
        return f"{self.job_name}:emu:{name}"

    def _load_current_iter(self) -> int:
        if self.shared_storage is None:
            return 0
        value = self.shared_storage.get(self._state_key("current_iter"))
        return int(value) if value is not None else 0

    def _log_event(self, event_type: str, phase: Optional[str] = None, **payload) -> None:
        record = {
            "time": time.time(),
            "job": self.job_name,
            "iter": self.current_iter,
            "event": event_type,
        }
        if phase is not None:
            record["phase"] = phase
        record.update(payload)
        with self.event_log_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _persist_state(self, phase: str, completed_iter: Optional[int] = None) -> None:
        if self.shared_storage is None:
            return
        self.shared_storage.set(self._state_key("current_iter"), self.current_iter)
        self.shared_storage.set(self._state_key("phase"), phase)
        if completed_iter is not None:
            self.shared_storage.set(self._state_key("last_completed_iter"), completed_iter)

    def _sleep_or_crash(self, phase: str, scaled_duration: float) -> None:
        should_fail = (
            self.failure_iter is not None
            and self.current_iter == self.failure_iter
            and phase == self.failure_phase
            and os.environ.get("RECOVER_JOB", "0") != "1"
        )
        if should_fail:
            prefix = scaled_duration * self.failure_after_ratio
            suffix = max(0.0, scaled_duration - prefix)
            if prefix > 0:
                time.sleep(prefix)
            self._log_event("section_crash", phase=phase, elapsed=prefix)
            self.fail_current_section(
                reason=f"Injected crash at iter={self.current_iter} phase={phase} after_ratio={self.failure_after_ratio}"
            )
            if suffix > 0:
                time.sleep(suffix)
        else:
            time.sleep(scaled_duration)

    def run(self):
        if self.is_recovery:
            self._log_event("job_recovered")
        while self.current_iter < self.max_iters:
            self._persist_state("generate")
            try:
                with self.scheduler_section("generate"):
                    self._log_event("section_enter", phase="generate")
                    self._sleep_or_crash("generate", self.generate_real_sec / self.time_scale)
                self._log_event("section_exit", phase="generate")
            except PipelineSectionCrash:
                self._persist_state("generate")
                self._log_event("job_exit_after_crash", phase="generate")
                return 1

            self._persist_state("train")
            try:
                with self.scheduler_section("train"):
                    self._log_event("section_enter", phase="train")
                    self._sleep_or_crash("train", self.train_real_sec / self.time_scale)
                self._log_event("section_exit", phase="train")
            except PipelineSectionCrash:
                self._persist_state("generate")
                self._log_event("job_exit_after_crash", phase="train")
                return 1

            self.current_iter += 1
            self._persist_state("generate", completed_iter=self.current_iter - 1)
            self._log_event("iteration_complete", completed_iter=self.current_iter - 1)

        if self.shared_storage is not None:
            self.shared_storage.publish("tenant_events", f"{self.job_name}:completed")
            self.set_key(f"{self.job_name}:status", "completed")
        self._log_event("job_completed", completed_iters=self.current_iter)
        return 0


def _build_emulated_config(job_name: str, gen_devices: str, train_devices: str) -> EmulatedRLVRConfig:
    return EmulatedRLVRConfig(
        exp_name=f"emu-{job_name}",
        output_dir=f"/tmp/emu-{job_name}",
        logging_dir="/tmp/emu-logs",
        track_with="stdout",
        max_steps=1,
        save_steps=1000000,
        logging_steps=1,
        eval_steps=1000000,
        prompt_length=16,
        sequence_length=32,
        actor_infer=WorkerConfig(
            name="actor_infer",
            device_affinity=os.environ.get("GDA", "EMU-GEN"),
            device_mapping=gen_devices,
            num_gpus_per_worker=1,
        ),
        actor_train=WorkerConfig(
            name="actor_train",
            device_affinity=os.environ.get("TDA", "EMU-TRAIN"),
            device_mapping=train_devices,
            num_gpus_per_worker=1,
        ),
    )


def _run_emulated_cli(args: argparse.Namespace) -> int:
    config = _build_emulated_config(
        job_name=args.job_name,
        gen_devices=args.gen_devices,
        train_devices=args.train_devices,
    )
    pipeline = EmulatedRLVRPipeline(
        pipeline_config=config,
        generate_real_sec=args.generate_real_sec,
        train_real_sec=args.train_real_sec,
        time_scale=args.time_scale,
        max_iters=args.max_iters,
        event_log_dir=args.event_log_dir,
        failure_iter=args.failure_iter,
        failure_phase=args.failure_phase,
        failure_after_ratio=args.failure_after_ratio,
    )
    return pipeline.run()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emulated", action="store_true")
    parser.add_argument("--job-name", required=False, default=os.environ.get("JOB_NAME", "emu-job"))
    parser.add_argument("--gen-devices", default="[0]")
    parser.add_argument("--train-devices", default="[0]")
    parser.add_argument("--generate-real-sec", type=float, default=600.0)
    parser.add_argument("--train-real-sec", type=float, default=100.0)
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--event-log-dir", default="/tmp/emu_events")
    parser.add_argument("--failure-iter", type=int, default=None)
    parser.add_argument("--failure-phase", default=None, choices=[None, "generate", "train"])
    parser.add_argument("--failure-after-ratio", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    if cli_args.emulated:
        raise SystemExit(_run_emulated_cli(cli_args))
    raise SystemExit(
        "This module now supports --emulated for scheduler demos. "
        "Use the project entrypoints for full RLVR training."
    )
