import copy
import gc
import os
import itertools
import redis
import time
import array
import queue
from concurrent import futures
from typing import List, Optional, Union, Dict
import asyncio

import ray
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed
from mcore_adapter.models.converter.convert_utils import RecvBucketManager
from vllm import SamplingParams, RequestOutput
from vllm.utils import random_uuid

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.third_party.vllm import LLM
from roll.third_party.vllm import AsyncLLM
from roll.third_party.vllm.vllm_utils import EngineStats
from roll.utils.collective import collective
from roll.utils.functionals import concatenate_input_and_output, GenerateRequestType
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType

logger = get_logger()


class VllmStrategy(InferenceStrategy):
    strategy_name = "vllm"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.model: Union[LLM, AsyncLLM]
        self.executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=1)
        self.pending_size = 1
        self.recv_manager = RecvBucketManager()
        self.command_queue: Optional[queue.Queue] = None

        self.request_metas = {}
        self.req_id_2_resp_start_idx = {}
        self.group_name = "vllm_worker_default"
        self.running = False
        self.ready = False

    def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)
        vllm_config = copy.deepcopy(self.worker_config.strategy_args.strategy_config)
        engine_mode = vllm_config.pop("engine_mode", "sync")  # sync/async
        self.pending_size = vllm_config.pop("pending_size", 1)
        self.command_queue = queue.Queue()

        if self.worker_config.model_args.dtype == "fp32":
            dtype = "float32"
        elif self.worker_config.model_args.dtype == "fp16":
            dtype = "float16"
        elif self.worker_config.model_args.dtype == "bf16":
            dtype = "bfloat16"
        else:
            dtype = "auto"
        vllm_config.update(
            {
                "model": self.worker_config.model_args.model_name_or_path,
                "dtype": dtype,
                "enforce_eager": vllm_config.get("enforce_eager", True),
                "trust_remote_code": True,
                "seed": self.worker.pipeline_config.seed,
                "disable_custom_all_reduce": vllm_config.get(
                    "disable_custom_all_reduce", True
                ),  # potentially hangs in tp>1
                "enable_prefix_caching": vllm_config.get("enable_prefix_caching", False),
                "load_format": vllm_config.get("load_format", "dummy"),  # use model update passed value
            }
        )
        logger.info(f"vllm_config: {vllm_config}")
        assert not dist.is_initialized()
        if engine_mode == "sync":
            self.model = LLM(resource_placement_groups=self.worker_config.resource_placement_groups, **vllm_config)
            self.tokenizer = self.model.get_tokenizer()
        else:
            self.model = AsyncLLM(
                resource_placement_groups=self.worker_config.resource_placement_groups, **vllm_config
            )
            loop = asyncio.get_event_loop()
            self.tokenizer = loop.run_until_complete(self.model.get_tokenizer())
        additional_special_tokens = self.tokenizer.additional_special_tokens
        special_tokens = [
            add_token
            for add_token in self.tokenizer.added_tokens_decoder.values()
            if add_token.special and add_token.content not in additional_special_tokens
        ]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}, replace_additional_special_tokens=False
        )
        logger.info(f"add {special_tokens} to additional_special_tokens: {self.tokenizer.additional_special_tokens}")

        self.worker.rank_info.my_rank = self.worker.rank
        self.worker.rank_info.dp_size = self.worker.world_size
        collective.init_collective_group(
            world_size=self.worker.world_size,
            rank=self.worker.rank,
            group_name=self.group_name,
            master_addr=self.worker.master_addr,
            master_port=self.worker.master_port,
        )
        try:
            self.shared_storage = redis.StrictRedis(
                host=os.environ.get("MASTER_ADDR", "localhost"),
                port=int(os.environ.get("SCHEDULER_PORT", "9969")),
                db=0
            )
            self.shared_storage.set("test", "1")
        except:
            print("*** Scheduler not found, migration is not functional!!!")
            self.shared_storage = None

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        vllm实现compute log probs在这里实现即可
        """
        pass

    def generate(self, batch: DataProto, generation_config) -> torch.Tensor:
        sampling_params = create_sampling_params_for_vllm(gen_kwargs=generation_config)

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        vllm_input_args = {}
        if "multi_modal_data" in batch.non_tensor_batch:
            vllm_input_args["prompts"] = batch.non_tensor_batch["multi_modal_data"]
        else:
            vllm_input_args["prompt_token_ids"] = gather_unpadded_input_ids(
                input_ids=input_ids, attention_mask=attention_mask
            )

        vllm_outputs = self.model.generate(sampling_params=sampling_params, use_tqdm=False, **vllm_input_args)

        # (bs * num_return_sequences, max_response_len)
        output_ids = gather_outputs_to_pad_tensor(
            request_outputs=vllm_outputs,
            pad_token_id=self.tokenizer.pad_token_id,
            device=input_ids.device,
        )

        # (bs * num_return_sequences, input_len + max_response_len)
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids, num_return_sequences=sampling_params.n
        )

        return output

    def process_vllm_output(self, vllm_outputs: List[RequestOutput], request_complete_callback):
        # We use the request id format to distinguish migrated responses and normal requests.
        # Request id of a migrated response is like f"{req_id}_{resp_id}_{num_seqs}".
        shard_storage_pipeline = self.shared_storage.pipeline()
        migrated_resps: List[RequestOutput] = []
        normal_reqs: List[RequestOutput] = []
        for output_request in vllm_outputs:
            if "_" in output_request.request_id:
                assert len(output_request.request_id.split("_")) == 3
                assert len(output_request.outputs) == 1, "Migrated response should have only one output."
                migrated_resps.append(output_request)
            else:
                normal_reqs.append(output_request)
        # print(f"=== normal reqs: {[req.request_id for req in normal_reqs]} migrated resps: {[resp.request_id for resp in migrated_resps]}")

        # Deal with migrated responses.
        # For a migrated response, do not process it, check if its original request has been finished instead.
        # If so, collect responses and then process the original request, else cache it in shared storage.
        origin_req_id_2_num_seqs = {}
        origin_req_id_2_resps = {}
        for migrated_resp in migrated_resps:
            # Request id of a migrated response is like f"{req_id}_{resp_id}_{num_seqs}".
            origin_req_id, resp_id, num_seqs, = migrated_resp.request_id.split("_")

            if origin_req_id not in origin_req_id_2_num_seqs:
                origin_req_id_2_num_seqs[origin_req_id] = int(num_seqs)
            else:
                assert origin_req_id_2_num_seqs[origin_req_id] == int(num_seqs)

            if origin_req_id not in origin_req_id_2_resps:
                origin_req_id_2_resps[origin_req_id] = []
            origin_req_id_2_resps[origin_req_id].append((resp_id, migrated_resp))

        for origin_req_id, resps in origin_req_id_2_resps.items():
            assert origin_req_id in self.req_id_2_resp_start_idx
            resp_start_idx = self.req_id_2_resp_start_idx[origin_req_id]
            keys = [key.decode() for key in self.shared_storage.scan_iter(match=f"finished_token_ids_{origin_req_id}_*")]
            # print(f"=== origin_req_id={origin_req_id}, cached responses: {len(keys)}, finished responses at this step: {len(resps)}, num_seqs: {origin_req_id_2_num_seqs[origin_req_id]}")
            assert len(keys) + len(resps) <= origin_req_id_2_num_seqs[origin_req_id], f"Cached responses ({len(keys)}) + finished responses ({len(resps)}) exceed num_seqs ({origin_req_id_2_num_seqs[origin_req_id]}) for request {origin_req_id}."
            if len(resps) + len(keys) == origin_req_id_2_num_seqs[origin_req_id]:
                # Process the original request with the callback function.
                output_token_ids = []
                assert origin_req_id in self.request_metas
                # Collect cached responses.
                for key in keys:
                    token_ids_bytes = self.shared_storage.get(key)
                    token_ids_arr = array.array("I")
                    token_ids_arr.frombytes(token_ids_bytes)
                    token_ids_list = token_ids_arr.tolist()
                    assert token_ids_list[0] == resp_start_idx, f"Separation index {token_ids_list[0]} does not match expected {resp_start_idx} for request {origin_req_id}."
                    output_token_ids.append(tuple(token_ids_list[resp_start_idx + 1:]))
                # Collect responses just finished at this step.
                for resp_id, resp in resps:
                    previous_output_token_ids = list(resp.prompt_token_ids[resp_start_idx:])
                    output_token_ids.append(tuple(previous_output_token_ids + list(resp.outputs[0].token_ids)))
                output_data = DataProto(meta_info=self.request_metas[origin_req_id])
                output_data.meta_info["output_token_ids"] = output_token_ids
                print("=== request_complete_callback for migrated request:", origin_req_id)
                request_complete_callback(data=output_data)
            else:
                # Cache finished migrated responses.
                # Set separation index according to exising finished responses.
                for resp_id, resp in resps:
                    all_token_ids = [resp_start_idx] + list(resp.prompt_token_ids) + list(resp.outputs[0].token_ids)
                    serialized_tokens = array.array('I', all_token_ids)
                    shard_storage_pipeline.set(f"finished_token_ids_{origin_req_id}_{resp_id}", serialized_tokens.tobytes())
        shard_storage_pipeline.execute()

        # 转成response id, request_complete_callback
        # Deal with normal requests.
        normal_req_ids = [req.request_id for req in normal_reqs]
        normal_req_ids_set = set(normal_req_ids)
        # TODO: Lunxi: For duplicate requests returned by engine, just ignore them temporarily.
        if len(normal_req_ids_set) != len(normal_req_ids):
            duplicate_req_ids = []
            for req_id in normal_req_ids_set:
                if normal_req_ids.count(req_id) > 1:
                    duplicate_req_ids.append(req_id)
            print(f"=== normal reqs: {[req.request_id for req in normal_reqs]} migrated resps: {[resp.request_id for resp in migrated_resps]} duplicate req ids: {duplicate_req_ids}")
        for request_output in normal_reqs:
            if request_output.request_id not in normal_req_ids_set:
                continue
            normal_req_ids_set.remove(request_output.request_id)
            output_token_ids = []
            request_id = request_output.request_id
            if request_id not in self.request_metas:
                continue
            for completion_output in request_output.outputs:
                output_token_ids.append(completion_output.token_ids)
            output_data = DataProto(meta_info=self.request_metas[request_id])
            output_data.meta_info["output_token_ids"] = output_token_ids
            print("=== request_complete_callback for unmigrated request:", request_id)
            request_complete_callback(data=output_data)
        assert len(normal_req_ids_set) == 0

    def start_server(self, data: DataProto, request_complete_callback, offload_manager):
        # collective.barrier(group_name=self.group_name)
        self.running = True
        self.ready = False

        # Block start_server execution here, wait for a running signal from the shared storage
        if self.shared_storage is not None:
            while True:
                status_bytes = self.shared_storage.get(self.worker.worker_name + "_status")
                if status_bytes is not None:
                    status_str = status_bytes.decode()
                    if status_str == 'running':
                        print("=== Got running signal!")
                        break
                time.sleep(0.1)
        # Enter the offload manager, GPU memory will be allocated after here.
        offload_manager.__enter__()
        self.ready = True

        while True:
            while not self.command_queue.empty():
                command, batch = self.command_queue.get_nowait()
                if command == GenerateRequestType.ADD:
                    input_ids = batch.batch["input_ids"]
                    attention_mask = batch.batch["attention_mask"]
                    request_id = batch.meta_info["request_id"]
                    self.request_metas[request_id] = batch.meta_info
                    # Add the meta info of the original request if adding a migrated response.
                    # Meta info of the original request is similar to that of every migrated response of it,
                    # while the latter has additional fields 'origin_request_id' and 'resp_start_idx',
                    # modified 'request_id' and modified 'num_return_sequences'.
                    if "origin_request_id" in batch.meta_info:
                        origin_req_id, _, num_seqs = batch.meta_info["request_id"].split("_")
                        assert origin_req_id == batch.meta_info["origin_request_id"]
                        if origin_req_id not in self.request_metas:
                            origin_req_meta_info = copy.deepcopy(batch.meta_info)
                            origin_req_meta_info["request_id"] = origin_req_meta_info.pop("origin_request_id")
                            origin_req_meta_info.pop("resp_start_idx")
                            origin_req_meta_info["generation_config"]["num_return_sequences"] = num_seqs
                            self.request_metas[origin_req_id] = origin_req_meta_info
                        else:
                            origin_req_meta_info = self.request_metas[origin_req_id]
                            assert origin_req_meta_info["request_id"] == origin_req_id
                            assert origin_req_meta_info["generation_config"]["num_return_sequences"] == num_seqs

                    generation_config = batch.meta_info.get("generation_config")
                    max_new_tokens = batch.meta_info.get("max_new_tokens", generation_config["max_new_tokens"])
                    max_new_tokens = min(max_new_tokens, generation_config["max_new_tokens"])
                    sampling_params = create_sampling_params_for_vllm(
                        gen_kwargs={**generation_config, "max_new_tokens": max_new_tokens}
                    )
                    prompt_token_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
                    self.model.add_requests(
                        request_ids=[request_id], prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
                    )
                    if 'resp_start_idx' in batch.meta_info:
                        origin_req_id = batch.meta_info['origin_request_id']
                        if origin_req_id in self.req_id_2_resp_start_idx:
                            assert self.req_id_2_resp_start_idx[origin_req_id] == batch.meta_info['resp_start_idx']
                        else:
                            self.req_id_2_resp_start_idx[origin_req_id] = batch.meta_info['resp_start_idx']
                elif command == GenerateRequestType.MIGRATE:
                    # print(f"[Worker {os.getpid()}] Migrate {batch}")
                    assert self.shared_storage is not None, "Cannot migrate requests due to no redis found."
                    migrate_pipeline = self.shared_storage.pipeline()
                    if batch.meta_info["role"] == "src":
                        my_rank = batch.meta_info["my_rank"]
                        dst_rank = batch.meta_info["dst_rank"]
                        desired_req_ids = batch.meta_info['req_ids']
                        reqs_to_migrate, finished_reqs = self.model.fetch_responses_to_migrate(desired_req_ids)
                        # If desired requests has been finished, we can directly process them.
                        self.process_vllm_output(vllm_outputs=finished_reqs, request_complete_callback=request_complete_callback)
                        num_resps_to_migrate = {}
                        for req_output in reqs_to_migrate:
                            # For each request, cache its finished responses in shared storage first,
                            # when migrated responses get finished, we can merge them and process this request with all responses finished.
                            for resp_id, output_response in enumerate(req_output.outputs):
                                print(f"=== token_ids_{req_output.request_id}_{resp_id}: prompt_length={len(req_output.prompt_token_ids)} output_length={len(output_response.token_ids)}")
                                # all_token_ids: [0] is the seperation index of prompt and response, [1:sep_idx+1] are prompt tokens, [sep_idx+1:] are response tokens
                                all_token_ids = [len(req_output.prompt_token_ids)] + list(req_output.prompt_token_ids) + list(output_response.token_ids)
                                serialized_tokens = array.array('I', all_token_ids)
                                if output_response.finish_reason is None:
                                    migrate_pipeline.set(f"token_ids_{my_rank}_{dst_rank}_{req_output.request_id}_{resp_id}", serialized_tokens.tobytes())
                                else:
                                    # Cache finished responses to merge with migrated responses in the future to finish this request.
                                    migrate_pipeline.set(f"finished_token_ids_{req_output.request_id}_{resp_id}", serialized_tokens.tobytes())
                            num_finished_resps = len([resp for resp in req_output.outputs if resp.finish_reason is not None])
                            num_resps_to_migrate[req_output.request_id] = len(req_output.outputs) - num_finished_resps
                            print(f"=== migrate request: {req_output.request_id}, finished responses: {num_finished_resps}, responses to migrate: {num_resps_to_migrate[req_output.request_id]}")
                            print(f"*** abort request: {req_output.request_id}")
                            self.model.abort_request(request_id=req_output.request_id)
                        print(f"=== migrate {sum(num_resps_to_migrate.values())} responses: {num_resps_to_migrate}")
                    migrate_pipeline.execute()
                    self.shared_storage.publish("migrate_progress", f"done_{my_rank}_{dst_rank}")

                elif command == GenerateRequestType.ABORT:
                    request_id = batch.meta_info["request_id"]
                    self.model.abort_request(request_id=request_id)
                elif command == GenerateRequestType.STOP:
                    self.model.abort_request(request_id=list(self.request_metas.keys()))
                    self.request_metas.clear()
                    self.req_id_2_resp_start_idx.clear()
                    while not self.command_queue.empty():
                        self.command_queue.get_nowait()
                    # Run llm_engine again to consume all out standing requests and
                    # stop model execute loop, otherwise collective_rpc will stuck by
                    # model execute loop or there will be garbage output at next step.
                    self.model.clear_unfinished_requests()
                    self.running = False
                    return

            vllm_outputs: List[RequestOutput] = self.model.fetch_output()
            self.process_vllm_output(vllm_outputs=vllm_outputs, request_complete_callback=request_complete_callback)

    def add_request(self, command, data: DataProto):
        self.command_queue.put((command, data))

    def get_stats(self) -> EngineStats:
        return self.model.get_stats()

    async def async_generate(self, batch: DataProto, generation_config: Dict) -> torch.Tensor:
        # TODO: refactor async_generate interface. not supported now!
        raise NotImplementedError()
        from vllm.inputs.data import TokensPrompt

        sampling_params = create_sampling_params_for_vllm(gen_kwargs=generation_config)

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask
        assert input_ids.size(0) == 1, f"async_generate: batch['input_ids'] must have exactly one batch dimension"

        prompt_token_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)

        # TODO meaningful request id?
        #   async_generate如何实现abort_request
        request_id = random_uuid()
        result_generator = self.model.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_token_ids[0]),
            sampling_params=sampling_params,
            request_id=request_id,
        )
        vllm_output: Optional[RequestOutput] = None
        async for request_output in result_generator:
            vllm_output = request_output
        assert vllm_output is not None

        # (bs * num_return_sequences, max_response_len)
        output_ids = gather_outputs_to_pad_tensor(
            request_outputs=[vllm_output], pad_token_id=self.tokenizer.pad_token_id, device=input_ids.device
        )
        # (bs * num_return_sequences, input_len + max_response_len)
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids, num_return_sequences=sampling_params.n
        )
        return output

    # offload/reload 接口
    def load_states(self, *args, **kwargs):
        self.model.load_states()

    def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            self.model.offload_states()
        self.recv_manager.clear()
        gc.collect()
        torch.cuda.empty_cache()

    # 参数同步相关接口
    def setup_collective_group(self, comm_plan, backend="nccl"):
        self.model.setup_collective_group(comm_plan=comm_plan, backend=backend, rank_in_cluster=self.worker.rank)

    def broadcast_parameter(self, src_pp_rank, dtype, shape, parameter_name):
        self.model.broadcast_parameter(src_pp_rank, dtype, shape, parameter_name)

    def broadcast_bucket(self, src_pp_rank, meta_infos, bucket_size):
        self.model.broadcast_bucket(src_pp_rank, meta_infos, bucket_size)

    def update_parameter(self, parameter_name, weight, ranks_in_worker):
        self.model.update_parameter(parameter_name, weight, ranks_in_worker)

    def update_parameter_in_bucket(self, meta_infos, buffer, ranks_in_worker):
        self.model.update_parameter_in_bucket(meta_infos, buffer, ranks_in_worker)


def gather_unpadded_input_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    gathered_input_ids = [ids[mask.bool()].tolist() for ids, mask in zip(input_ids, attention_mask)]
    return gathered_input_ids


def gather_outputs_to_pad_tensor(request_outputs: List["RequestOutput"], pad_token_id, device="cuda") -> torch.Tensor:
    token_ids_list_of_lists = [
        torch.tensor(completion_output.token_ids, device=device)
        for request_output in request_outputs
        for completion_output in request_output.outputs
    ]
    output_tensor = pad_sequence(token_ids_list_of_lists, batch_first=True, padding_value=pad_token_id)
    return output_tensor


def create_sampling_params_for_vllm(gen_kwargs):
    if gen_kwargs["num_beams"] > 1:
        return SamplingParams(
            max_tokens=gen_kwargs["max_new_tokens"],
            stop_token_ids=gen_kwargs["eos_token_id"],
            repetition_penalty=gen_kwargs["repetition_penalty"],
            n=gen_kwargs["num_return_sequences"],
            best_of=gen_kwargs["num_beams"],
            use_beam_search=True,
            logprobs=0,
        )
    return SamplingParams(
        max_tokens=gen_kwargs["max_new_tokens"],
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        stop_token_ids=gen_kwargs["eos_token_id"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        n=gen_kwargs["num_return_sequences"],
        logprobs=0,
    )


def compare_sampling_params(params1: SamplingParams, params2: SamplingParams) -> bool:
    # 只比较采样参数的配置
    param_attrs = [
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "n",
        "stop_token_ids" "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "min_p",
        "best_of",
        "stop",
        "ignore_eos",
        "use_beam_search",
        "best_of",
        "use_beam_search",
    ]

    # 比较每个采样参数
    for attr in param_attrs:
        if hasattr(params1, attr) and hasattr(params2, attr):
            val1 = getattr(params1, attr)
            val2 = getattr(params2, attr)
            if val1 != val2:
                print(f"采样参数 {attr} 不同: {val1} != {val2}")
                return False
    return True
