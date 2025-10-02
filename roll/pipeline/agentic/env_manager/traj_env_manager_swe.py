import copy
from contextlib import nullcontext
from omegaconf import DictConfig
from threading import Lock
from typing import Dict, List, Optional

import roll.agentic.env # add

import gem
import numpy as np
import ray
import torch
from codetiming import Timer
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from roll.agentic.llm_proxy import create_llm_proxy, BaseLLMProxy
from roll.agentic.rollout.base_env_manager import RolloutCache, BaseEnvManager
from roll.agentic.rollout.env_action_limiter import get_global_limiter
from roll.agentic.rollout.rollout_scheduler import GroupQueueManager
from roll.agentic.rollout.token_mask_utils import split_by_token, \
    token_ids_to_assistant_mask, custom_apply_chat_template
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig
from roll.utils.constants import GenerateStopReason
from roll.utils.functionals import pad_to_length
from roll.utils.logging import get_logger
from roll.utils.str_utils import contains_renderable_field
import json
import time
import os
import re

DEBUG = True
# for long traj debug
class Colors:
    """ANSI颜色代码"""
    HEADER = '\033[95m'      # 紫色
    BLUE = '\033[94m'        # 蓝色
    PINK = '\033[91m'        # 粉色
    GREEN = '\033[92m'       # 绿色
    YELLOW = '\033[93m'      # 黄色
    RED = '\033[91m'         # 红色
    BOLD = '\033[1m'         # 粗体
    UNDERLINE = '\033[4m'    # 下划线
    END = '\033[0m'          # 结束颜色

def pretty_print(color=Colors.END,text_color:str='',text:str=''):
    """打印彩色文本"""
    print(f"{Colors.BOLD}{color}{text_color}{Colors.END}{text}")
def write_data_json(data,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,'w',encoding="utf-8") as f:
        info_str = json.dumps(data, ensure_ascii=False)
        f.write(info_str)

class TrajEnvManager(BaseEnvManager):
    def __init__(self,
                 worker_config: EnvManagerConfig,
                 pipeline_config: AgenticConfig,
                 env_config: DictConfig,
                 tokenizer: PreTrainedTokenizer,
                 generate_scheduler,
                 output_queue: GroupQueueManager,
                 thread_lock: Lock,
                 mode='train',
                 *args, **kwargs):
        """
        """
        super().__init__()
        self.logger = get_logger()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: DictConfig = env_config

        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler: RequestScheduler = generate_scheduler

        # # 配置环境特定的日志
        # if 'log_file' in self.env_config:
        #     self.setup_env_logging(
        #         self.env_config['log_file'], 
        #         self.env_config.get('env_id', 'unknown'), 
        #         self.env_config.get('worker_id', 'unknown')
        #     )
            
        # EnvManager states
        self.rollout_cache: Optional[RolloutCache] = None
        self.group_seed = None
        self.episode_id = 0
        self.current_step = -1
        self.running = False
        self.use_thread_lock = self.env_config.get("use_thread_lock", False) # 避免同时执行大量cpu操作, 可以通过env_config配置
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()
        with self.thread_lock:
            if "seed" in self.env_config['config']:
                self.env_config['config']["seed"] = self.env_config['group_seed']
            self.env = gem.make(env_id=self.env_config["env_type"], **self.env_config['config'])

        # Set environment step concurrency limit
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = None
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(tag=env_tag, max_concurrent_calls=self.max_env_step_concurrent)

        print('====== traj_env_manager_swe init ======')
        cfg_template = self.pipeline_config.custom_envs[self.env_config["tag"]]
        self.agent_system_template = cfg_template["agent_system_template"]
        self.agent_instance_template = cfg_template["agent_instance_template"] # added for swe_env
        self.agent_obs_template = cfg_template["agent_obs_template"] # added for swe_env
        self.agent_last_step_template = cfg_template["agent_last_step_template"] # added for swe_env

        # TODO: add rewards_scheduler for local ray reward workers
        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env
        )

    def run_rollout_loop(self, data: DataProto):
        """
        1. Each time run_rollout_loop is called,
           it will continuously play episodes until it receives a command that data collection is complete.
           The seed needs to be reset to ensure consistency across all groups.
           episode_id is reset to 0.

        Seed update logic:
           group_seed = base_seed + group_id
           episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """
        assert not self.running
        assert "seed" in data.meta_info
        current_step = data.meta_info.get("current_step", None)
        self.running = True
        is_sync_training: bool = current_step is not None
        if is_sync_training:
            self.current_step = current_step
        assert self.current_step >= 0
        self.episode_id = 0
        self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']
        rollout_cache: RolloutCache = self.reset()
        start_step = self.current_step

        log_stats = {"generate_time": [], "step_time": [], "current_step": []}

        while self.running:

            with Timer(name="generate", logger=None) as generate_timer:
                lm_output: DataProto = self.make_decision(rollout_cache)
                stop_reason = lm_output.meta_info.pop("stop_reason")
                self.logger.info(f'[STOP_REASON] {stop_reason}')
            log_stats["current_step"].append(self.current_step)
            log_stats["generate_time"].append(generate_timer.last)

            with Timer(name="step", logger=None) as step_timer:
                if stop_reason == GenerateStopReason.FINISH:
                    rollout_cache: RolloutCache = self.step(lm_output)
            log_stats["step_time"].append(step_timer.last)

            if self.running and (rollout_cache.terminated or stop_reason == GenerateStopReason.MAX_LENGTH):
                print('[DEBUG]STOP')
                self.logger.debug(f"group_id: {self.env_config['group_id']} env_id: {self.env_config['env_id']} episode_id: {self.episode_id} start_step {start_step} gen_stats: {log_stats}")
                log_stats = {"generate_time": [], "step_time": [], "current_step": []}

                rollout: DataProto = self.formulate_rollouts(rollout_cache)
                traj_group_id = f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
                traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                rollout.non_tensor_batch["traj_group_id"] = np.array([traj_group_id] * rollout.batch.batch_size[0], dtype=object)
                rollout.non_tensor_batch["traj_id"] = np.array([traj_id] * rollout.batch.batch_size[0], dtype=object)
                ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout))

                if not self.running or (is_sync_training and self.episode_id >= self.worker_config.max_traj_per_env):
                    self.rollout_cache: Optional[RolloutCache] = None
                    self.logger.debug(
                        f"env_id: {self.env_config['env_id']} max_traj_per_env {self.worker_config.max_traj_per_env} reached, stopping rollout loop")
                    break

                rollout_cache = self.reset()

    def reset(self) -> RolloutCache:
        self.rollout_cache = RolloutCache(env_id=self.env_config['env_id'],
                                          group_id=self.env_config['group_id'],
                                          tag=self.env_config['tag'])

        seed = self.group_seed + self.episode_id

        # with self.thread_lock:
            # `observation` describes the current game-state prompt;
            # `info["suffix"]` carries the current environment-specific state string.
        observation, info = self.env.reset(seed=seed)
        self.rollout_cache.history.append({
            "observation": observation,
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            **info,
        })
        self.episode_id += 1
        return self.rollout_cache

    def step(self, llm_output: DataProto):
        responses = self.tokenizer.batch_decode(
            llm_output.batch['responses'],
            skip_special_tokens=True
        )

        observation, reward, terminated, truncated, info = self.env.step(action=responses[0])
        suffix = info.pop("suffix", None)

        self.rollout_cache.step += 1
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= self.env_config.max_steps:
            self.rollout_cache.terminated = True
            if not terminated:
                self.rollout_cache.truncated = True
        self.rollout_cache.history[-1]['reward'] = reward
        self.rollout_cache.history[-1]['penalty'] = 0
        metrics = info.get("metrics", {})
        if not metrics.get("action_is_valid", True):
            self.rollout_cache.history[-1]['penalty'] = self.worker_config.format_penalty
        self.rollout_cache.history[-1]['llm_response'] = responses[0]
        if info is not None:
            self.rollout_cache.history[-1].update(info)

        self.rollout_cache.history.append({
            "observation": observation,
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
        })
        if suffix is not None:
            self.rollout_cache.history[-1]["suffix"] = suffix

        if self.mode == "val" and self.pipeline_config.render_save_dir and hasattr(self.env, "render"):
            frame = self.env.render(mode='rgb_array')
            if isinstance(frame, np.ndarray):
                self.rollout_cache.frames.append(frame)

        # debug
        # if DEBUG and self.rollout_cache.step < 2:
            # print('\n\n***** self.rollout_cache.step == 2 *****')
            # print('[DEBUG] self.rollout_cache: \n', self.rollout_cache)
        # elif DEBUG and self.rollout_cache.terminated:
            # print('\n\n***** terminated *****')
            # print('[DEBUG] self.rollout_cache: \n', self.rollout_cache)
        """
        @input:
            llm_output: DataProto
        @output: (需要check一下)
            self.rollout_cache: 包含env_id=0, group_id=0, tag='SWEEnvTrain', history=[{},{}...], frames=[], truncated=False, terminated=True, step=7
            其中.history[-1] 包含reward, penalty, llm_response, suffix, metrics, info, observation, actions_left
        """
        return self.rollout_cache

    def make_decision(self, rollout_cache: RolloutCache):
        content = self.rollout_cache.history[-1]
        render_dict = {"observation": content["observation"]}
        if contains_renderable_field(self.agent_obs_template, "turn_idx"):
            render_dict["turn_idx"] = self.rollout_cache.step + 1
        if contains_renderable_field(self.agent_obs_template, "suffix"):
            render_dict["suffix"] = content.get("suffix", "")
        if contains_renderable_field(self.agent_obs_template, "actions_left"):
            render_dict["actions_left"] = content["actions_left"]
        if contains_renderable_field(self.agent_obs_template, "max_response_length"):
            render_dict["max_response_length"] = self.env_config["max_tokens_per_step"]
        # current messages
        messages = []
        if self.rollout_cache.step == 0:
            messages = [{"role": "system", "content": self.agent_system_template}]
            messages.append({"role": "user", "content": self.agent_instance_template.format(problem_statement=content.get("observation", ""))})
        else:
            messages.append({"role": "user", "content": self.agent_obs_template.format(**render_dict)})
            messages[-1]['content'] = self.agent_last_step_template.format(observation=messages[-1]['content'],actions_left=content["actions_left"])
        content["messages"] = messages
        prompt_ids = custom_apply_chat_template(messages=messages, tokenizer=self.tokenizer, add_generation_prompt=True)

        # write_data_json(messages,f'./output/{time.strftime("%Y%m%d_%H%M%S", time.localtime())}-messages.json')
        # self.logger.info(f'************* make_decision *************')
        # self.logger.info(f'[DEBUG][STEP: {self.rollout_cache.step}][messages]{[item["role"] for item in messages]}')
        # pretty_print(Colors.BLUE,f'************* make_decision *************')
        # pretty_print(Colors.BLUE,f'[DEBUG][STEP: {self.rollout_cache.step}][messages]{[item["role"] for item in messages]}')
        # for item in messages:
        #     # self.logger.info(f'[ROLE: {item["role"]}] {item["content"]}')
        #     pretty_print(Colors.BLUE,f'\n**********[ROLE: {item["role"]}]**********\n',f'{item["content"]}')
        # pretty_print(Colors.BLUE,f'**********************************************')

        # format_messages需要加入env.py中的parse_action逻辑，和obs对齐。所以需要重新reformat_messages。
        # TODO: input_messages会重构之前的histroy，所以直接拼接histroy_token_ids会对不齐。改成基于input_messages来计算input_ids。
        
        # reformat messages
        def parse_action(response_text):
            # Regex to match (non-greedily) from `<function=` up to the first `</function>`
            pattern = re.compile(r"(?s)(<function=.*?</function>)")
            match = pattern.search(response_text)

            if match:
                action = match.group(1)  # The entire <function=...></function> block
                thought = response_text[: match.start()]  # Everything before the block
            else:
                # If no match, treat entire text as "thought"
                thought = response_text
                action = ""

            # Strip leading/trailing whitespace
            thought = thought.strip()
            action = action.strip()

            return thought, action

        def reformat_messages(messages_ori):
            # 需要把模型输出的content按照parse_action重新处理
            messages = []
            last = len(messages_ori) - 1
            for i in range(len(messages_ori)):
                item = messages_ori[i]
                if item['role'] == 'system' or (item['role'] == 'user' and i < 2):
                    content_new = item['content']
                elif item["role"] == "assistant":
                    # 与env.py中的parse_action对齐
                    thought, action = parse_action(item["content"])
                    if action.strip() == '':  content_new = item['content'] # 存在完全解析为空的情况，影响训练
                    else: content_new = f"{thought}\n{action}\n"
                elif item['role'] == 'user' and i >= 2 and i != last:
                    # rm unuseless tips in middle turns
                    content_new = item['content'].split("ATTENTION: ")[0] # attention
                elif item['role'] == 'user' and i == last:
                    content_new = item['content']
                else:
                    raise ValueError(f'Unknown role: {item["role"]}')
                messages.append({"role": item['role'], "content": content_new})
            return messages

        # input messages
        if self.rollout_cache.step < 2: # 0,1
            history_message = []
        else:
            history_message = [item for items in self.rollout_cache.history[:-1] for item in items["messages"]]
        input_messages = reformat_messages(history_message + messages)
        # pretty_print(Colors.BLUE,f'**********[input_messages]**********')
        # for item in input_messages:
        #     pretty_print(Colors.BLUE,f'\n**********[input_messages][ROLE: {item["role"]}]**********\n',f'{item["content"]}')

        # input_ids = history_token_ids + prompt_ids
        input_ids = custom_apply_chat_template(messages=input_messages, tokenizer=self.tokenizer, add_generation_prompt=True)

        # sequence length warining
        if len(input_ids) >= self.pipeline_config.sequence_length:
            self.logger.warning(f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {len(input_ids)},"
                                f"maybe you should increase the response_length")
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})
        
        # convert to tensor for lm_input
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        position_ids = attention_mask.cumsum(dim=-1)
        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])
        
        # compute max_new_tokens
        max_new_tokens = min(self.env_config["max_tokens_per_step"],
                             self.worker_config.generating_args.max_new_tokens,
                             self.pipeline_config.sequence_length-input_ids.shape[1])
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]

        # pretty_print(Colors.PINK,f'\n**********[input_messages]**********')
        # pretty_print(text = input_messages)
        # pretty_print(Colors.PINK,f'**********[rollout_cache.history]**********')
        # print(self.rollout_cache)
        # if self.rollout_cache.step == 2:
        # for item in self.rollout_cache.history:
            # print(item)

        lm_output: DataProto = self.llm_proxy.generate(messages=input_messages,
                                                       lm_input=lm_input,
                                                       generation_config=generation_config)

        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})

        response_ids = lm_output.batch['responses'][0]
        response_ids = response_ids.tolist()
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        content["prompt_ids"] = prompt_ids # 这里为什么不能注释，copy.deepcopy?
        content["response_ids"] = response_ids
        content["messages"].append({"role": "assistant", "content": self.tokenizer.decode(response_ids, skip_special_tokens=True)})

        return lm_output

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        """

        """
        if 'observation' in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)
        history = rollout_cache.history[:-1]
        last_cache = copy.deepcopy(rollout_cache.history[-1])

        traj_messages = [item for items in self.rollout_cache.history for item in items["messages"]]
        # pretty_print(Colors.GREEN,f'********************* traj_messages *********************')
        # for item in traj_messages:
        #     pretty_print(Colors.GREEN,f'\n**********[traj_messages][ROLE: {item["role"]}]**********\n',f'{item["content"]}')
        # pretty_print(Colors.GREEN,f'**********************************************************')
        
        pretty_print(Colors.PINK,f'\n**********[METRIC]**********\n',f'{last_cache["metrics"]}')
        last_cache.pop("reward", None)
        history.append(last_cache)
        #  last_cache keys: dict_keys(['observation', 'actions_left', 'suffix', 'messages', 'prompt_ids', 'response_ids', 'penalty', 'llm_response', 'metrics'])

        # write_data_json(traj_messages,f'./output/{time.strftime("%Y%m%d_%H%M%S", time.localtime())}-traj_messages.json')
        # write_data_json(last_cache["metrics"],f'./output/{time.strftime("%Y%m%d_%H%M%S", time.localtime())}-metric.json')

        scores = [i['reward'] for i in self.rollout_cache.history]
        episode_score = sum(scores)

        token_ids = []
        prompt_masks = []
        response_masks = []
        step_response_length_list = []
        step_prompt_length_list = []
        for items in self.rollout_cache.history:
            token_ids.extend(items["prompt_ids"])
            token_ids.extend(items["response_ids"])
            prompt_masks.extend([1] * len(items["prompt_ids"]) + [0] * len(items["response_ids"]))
            response_masks.extend([0] * len(items["prompt_ids"]) + [1] * len(items["response_ids"]))
            step_response_length = len(items["response_ids"])
            step_response_length_list.append(step_response_length)
            step_prompt_length = len(items["prompt_ids"])
            step_prompt_length_list.append(step_prompt_length)
        

        input_ids =torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * len(token_ids), dtype=torch.long).unsqueeze(0)
        response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)
        step_response_length_tensor = torch.tensor(step_response_length_list, dtype=torch.float)
        step_prompt_length_tensor = torch.tensor(step_prompt_length_list, dtype=torch.float)

        first_response_idx = response_masks.index(1)
        prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
        prompt_mask =torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
        score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)
        score_tensor[0][-1] = episode_score
        position_ids = attention_mask.cumsum(dim=-1)

        lm_input = DataProto()
        lm_input.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0])

        response_length = response_mask.sum(dim=-1).float().mean().item()

        # TODO: move pad to pipeline
        input_ids = pad_to_length(input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id)
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)

        metrics = self.rollout_cache.history[-1].get('metrics', {})
        if metrics.get('env_timeout'):
            response_mask = torch.zeros_like(response_mask)
            prompt_mask = torch.zeros_like(prompt_mask)
            score_tensor = torch.zeros_like(score_tensor)

        lm_input.batch.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "prompt_mask": prompt_mask,
            "scores": score_tensor,
        })
        lm_input.non_tensor_batch.update({
            "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
            "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
            "tags": np.array([self.rollout_cache.tag], dtype=object),
            "frames": np.array([self.rollout_cache.frames], dtype=object),
            "step_scores": np.array([scores], dtype=object),
            "episode_scores": np.array([episode_score], dtype=object),
            "traj_rollout_time": np.array([float(metrics.get('traj_rollout_time', 0))], dtype=object),
        })

        # length
        avg_step_response_length = round(step_response_length_tensor.mean().item(), 2)
        avg_step_prompt_length = round(step_prompt_length_tensor.mean().item(), 2)
        max_step_response_length = round(step_response_length_tensor.max().item(), 2)
        max_step_prompt_length = round(step_prompt_length_tensor.max().item(), 2)
        min_step_response_length = round(step_response_length_tensor.min().item(), 2)
        min_step_prompt_length = round(step_prompt_length_tensor.min().item(), 2)

        # traj-level metric
        env_metric = {
            'success': float(metrics.get('success', episode_score > 0)),
            "reward": float(metrics.get('reward', episode_score)),
            "truncated": float(metrics.get("truncated", 0)),
            "env_timeout": float(metrics.get("env_timeout", 0)),
            'num_actions': rollout_cache.step,
            "step_count": float(metrics.get('step_count', 0)),
            "retry_times": float(metrics.get('retry_times', 0)),
            "action_is_valid": float(metrics.get('action_is_valid', 0)),
            "action_is_effective": float(metrics.get('action_is_effective', 0)),
            "traj_reset_time": float(metrics.get('traj_reset_time', 0)),
            "traj_reward_time": float(metrics.get('traj_reward_time', 0)),
            "traj_total_time": float(metrics.get('traj_total_time', 0)),
            "traj_rollout_time": float(metrics.get('traj_rollout_time', 0)),
            "avg_step_response_length": avg_step_response_length,
            "avg_step_prompt_length": avg_step_prompt_length,
            "max_step_response_length": max_step_response_length,
            "max_step_prompt_length": max_step_prompt_length,
            "min_step_response_length": min_step_response_length,
            "min_step_prompt_length": min_step_prompt_length,
        }
        traj_keys = list(env_metric.keys())
        # step-level metric
        custom_metric = {}
        for turn in self.rollout_cache.history:
            for k, v in turn.get('metrics', {}).items():
                if k in traj_keys: continue
                if k == 'task_idx': continue
                if k not in custom_metric:
                    custom_metric[k] = []
                custom_metric[k].append(float(v))
        for k, v in custom_metric.items():
            env_metric[k] = np.sum(v) / len(self.rollout_cache.history)
        # add tag
        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        # response_length
        env_metric["env/response_length"] = response_length
        lm_input.meta_info = {"metrics": env_metric}
        print(f'\n[formulate_rollouts][env_metric]{env_metric}')
        
        base_dir = self.pipeline_config.base_dir
        prompt_length = torch.tensor(prompt_masks).sum(dim=-1).float().mean().item()
        length = prompt_length + response_length
        max_seq_length = self.pipeline_config.sequence_length
        step_count = (len(traj_messages)-1) // 2
        print(f'[DEBUG] length: {length}, max_seq_length: {max_seq_length}, prompt_length: {prompt_length}, response_length: {response_length}')
        if length > max_seq_length:
            stop_reason = "max_length"
        elif metrics.get('success', True): 
            stop_reason = "finish"
        elif metrics.get('env_timeout'):
            stop_reason = "env_timeout"
        else: 
            stop_reason = "truncated"
        
        task_idx = metrics.get('task_idx', 0)
        tag = self.env_config["tag"]
        # 在这里写文件
        save = {
            "task_idx": task_idx,
            "env_id": self.env_config["env_id"],
            "group_id": self.env_config["group_id"],
            "tag": self.env_config["tag"],
            "length": length,
            "step_count": step_count,
            "stop_reason": stop_reason,
            "episode_score": episode_score,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "max_seq_length": max_seq_length,
            "traj_messages": traj_messages,
            "metrics": metrics,
            "env_metric": env_metric,
        }
        if metrics.get('env_timeout'):
            log_path = os.path.join(base_dir,'rollouts',f'env_timeout-{tag}_{task_idx}_{time.strftime("%m%d%H%M%S", time.localtime())}-{self.env_config["env_id"]}_re{episode_score}_step{step_count}_rlgh{response_length}_plgh{prompt_length}_srlgh{avg_step_response_length}_splgh{avg_step_prompt_length}.json')
        else:
            log_path = os.path.join(base_dir,'rollouts',f'{tag}_{task_idx}_{time.strftime("%m%d%H%M%S", time.localtime())}-{self.env_config["env_id"]}_re{episode_score}_step{step_count}_rlgh{response_length}_plgh{prompt_length}_srlgh{avg_step_response_length}_splgh{avg_step_prompt_length}.json')
        write_data_json(save,log_path)
        return lm_input


    def setup_env_logging(self, log_file, env_id, worker_id):
        """为特定环境设置独立的日志文件"""
        import logging
        import os
        
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 创建环境特定的logger
        env_logger = logging.getLogger(f"worker_{worker_id}_env_{env_id}")
        env_logger.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if not env_logger.handlers:
            # 创建文件处理器
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # 创建格式化器
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # 添加处理器到logger
            env_logger.addHandler(file_handler)
        
        # 保存logger引用
        self.env_logger = env_logger
        
        # 重定向stdout和stderr到日志文件
        self.redirect_output_to_log(log_file)
        print(f'重定向输出到日志文件: {log_file}')

    def redirect_output_to_log(self, log_file):
        """重定向标准输出和错误输出到日志文件"""
        import sys
        import os
        
        # 创建日志文件
        self.log_file = open(log_file, 'a', encoding='utf-8', buffering=1)
        
        # 保存原始的stdout和stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # 创建自定义的输出流
        class LogFileWriter:
            def __init__(self, log_file, original_stream):
                self.log_file = log_file
                self.original_stream = original_stream
            
            def write(self, text):
                self.log_file.write(text)
                self.log_file.flush()
                self.original_stream.write(text)
            
            def flush(self):
                self.log_file.flush()
                self.original_stream.flush()
        
        # 重定向输出
        sys.stdout = LogFileWriter(self.log_file, self.original_stdout)
        sys.stderr = LogFileWriter(self.log_file, self.original_stderr)

    def close(self):
        """关闭环境管理器时恢复标准输出"""
        if hasattr(self, 'log_file'):
            # 恢复标准输出
            import sys
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            # 关闭日志文件
            self.log_file.close()
        # 关闭环境
        if hasattr(self, 'env'):
            self.env.close()

    def stop(self):
        """停止环境管理器"""
        self.running = False
        self.close()  # 调用close方法恢复输出