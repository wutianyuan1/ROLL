from types import NoneType
import numpy as np
import random
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
from typing import Optional, Any,Tuple
import time
import logging
import uuid
import gem
from gem import Env
from roll.agentic.utils import all_seed

from roll.agentic.env.swe_env.src.agenthub.agent.agent import AgentArgs, Agent
from roll.agentic.env.swe_env.src.agenthub.action import Action
from roll.agentic.env.swe_env.src.agenthub.utils.log import get_logger
from roll.agentic.env.swe_env.src.agenthub.runtime.docker import DockerRuntime
from roll.agentic.env.swe_env.src.agenthub.trajectory import TrajectoryStep, Trajectory
from roll.agentic.env.swe_env.src.agenthub.observation import Observation
from roll.agentic.env.swe_env.src.agenthub.agent.commands import ParseCommandBash
from roll.agentic.env.swe_env.src.agenthub.tools import (
    search_tool,
    file_editor,
    bash_execute_tool,
    finish_tool,
)
from roll.agentic.env.swe_env.utils import _lazy_load_jsonl_lines_spec_idx, RepoEnv
import re
from pathlib import Path
import json
import os
import copy

def write_data_json(data,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,'w',encoding="utf-8") as f:
        info_str = json.dumps(data, ensure_ascii=False)
        f.write(info_str)

class SWEEnv(Env, gym.Env):
    def __init__(self,
                 render_mode: str = "text",
                 max_steps: int = 50,
                 max_reset_retry_times: int= 20,
                 format_penalty=0.0,
                 mode: str = "train", # train, val, spec-xx
                 data_path: str = "data/part_0.jsonl",
                 train_idx_range: Tuple[int, int] = (0, 4577), # 训练集任务ID范围
                 val_idx_range: Tuple[int, int] = (0,128), # 验证集任务ID范围
                 tools: list[str] = ["swe_env/src/agenthub/tools/search.py",
                                    "swe_env/src/agenthub/tools/file_editor.py",
                                    "swe_env/src/agenthub/tools/execute_bash.py",
                                    "swe_env/src/agenthub/tools/finish.py"],
                 action_pattern="^<answer>(.*?)</answer>$",
                 special_token_list=("<think>", "</think>", "<answer>","</answer>", "<|im_start|>", "<|im_end|>"),
                 swe_rex_host="https://xrl-aliyun.alibaba-inc.com/swe-rex/docker",
                 traj_dir: str = "./traj/trainset/",
                 swe_requirment_dir: str = "data/swe/250820_valset_v1_swe_bench_verified_requirment",
                 **kwargs
    ):
        
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list
        self.format_penalty = format_penalty

        self.logger = get_logger()
        
        # 环境信息(不变)
        self.mode = mode
        self.train_idx_range = train_idx_range
        self.val_idx_range = val_idx_range
        self.repo_env = RepoEnv(logger=self.logger)
        self.data_path = data_path
        self.swe_rex_host = swe_rex_host
        # 基本参数(不变)
        self.max_reset_retry_times = max_reset_retry_times
        self.max_steps = max_steps
        # self.tools = tools
        current_file_path = Path(__file__).resolve()
        self.tools = [f'{current_file_path.parent.parent}/{data_path}' for data_path in tools]
        self.tool_names = ['file_editor', 'execute_bash', 'search', 'finish'] # TODO
        self.traj_dir = traj_dir

        # 当前参数(会更新)
        self.retry_time = 0
        self.step_count = 0
        self.task_idx = None
        self.history = []
        self.data_line = []
        self.container_name = None
        self.route_key = None
        self.problem_statement = None
        self.issue = None
        self.metrics = None
        self.terminate = False
        self.truncated = False
        self.env_timeout = False
        self.reward = 0
        self.action_is_valid_lst = [] # 用于metric
        self.action_is_effective_lst = [] # 用于metric


        # 时间参数
        self.traj_reset_time = 0
        self.traj_step_time = 0
        self.traj_reward_time = 0
        self.traj_total_time = 0
        self.traj_rollout_time = 0
        self.time_start = 0

        self.max_env_time = 60 * 40 # 单环境最长rollout40min, 超时则return mask。

        # 环境参数
        os.environ['SWE_REQUIRMENT_DIR'] = swe_requirment_dir
        
    def get_task_suffix(self) -> Any:
        problem_statement,issue = self.get_instruction()
        return problem_statement
        # if self.render_mode == "text":
        #     return (
        #         f"Here is the current state of the SWE:\n{self.render(mode='text')}\n"
        #     )
        # else:
        #     return self.render(mode=self.render_mode)
    
    def get_task_idx_and_data(self,seed):
        # get task_idx
        print(f'self.train_idx_range: {self.train_idx_range}')
        with all_seed(seed):
            if self.mode == "train":
                idx = random.randint(self.train_idx_range[0], self.train_idx_range[1])
            elif self.mode == "val":
                idx = random.randint(self.val_idx_range[0], self.val_idx_range[1])
            elif self.mode.startswith("spec-"):
                idx = int(self.mode.split("-")[1])
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
            
        # load data from oss
        cur_data_path = self.data_path.replace("part_0.jsonl", f"part_{idx//100}.jsonl")
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]load_data from {cur_data_path}. (cur_task_idx: {idx})')
        data_line = _lazy_load_jsonl_lines_spec_idx(cur_data_path, idx)
        return idx, data_line
    
    def gen_route_key(self,data_line):
        if "docker_image" in data_line:
            route_key = f"{self.task_idx}_{data_line['docker_image']}"
        elif "docker_image" not in data_line:
            route_key = f"{self.task_idx}"
        return route_key
    
    def get_instruction(self):
        problem_statement = self.data_line['problem_statement']
        # print(f'[DEBUG][problem_statement]\n{problem_statement}')
        try: 
            issue = re.search(r"\[ISSUE\](.*)\[/ISSUE\]", problem_statement, re.DOTALL).group(1) # r2e-gym trainset
        except: 
            issue = problem_statement # swe-bench-verified
        # print(f'[DEBUG][issue]\n{issue}')
        # exit()
        return problem_statement,issue
        
    def reset(self, seed=None):
        st = time.time()
        self.time_start = st
        self.clean_record()

        while self.retry_times < self.max_reset_retry_times: # Attention: 这里必须成功，否则会failed
            # gen task_idx and load data_line
            task_idx, data_line = self.get_task_idx_and_data(seed)
            self.task_idx = task_idx
            self.data_line = data_line
            if self.retry_times == 0: print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]start reset, task_idx: {self.task_idx}')
            # gen route_key
            # data_line["route_key"] = self.gen_route_key(data_line)
            data_line["route_key"] = None
            data_line["swe_rex_host"] = self.swe_rex_host

            # init docker_runtime 
            # TODO：这里后续可以在环境reset里写多进程reset，哪个先成功就用哪个
            reset_info = self.repo_env.reset(data_line) # {'container_name':ERROR/SUCCESS,'setup_env_result':ERROR/SUCCESS,"reset_retry_times": int, "route_key": self.route_key}
            self.container_name = reset_info.get("container_name", None)
            self.route_key = reset_info.get("route_key", None)
            self.retry_times += reset_info.get("retry_times", 1)
            self.setup_env_result = reset_info.get("setup_env_result", None)

            if 'ERROR' in self.container_name or 'ERROR' in self.setup_env_result:
                continue
            else:
                # self.logger.info(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]reset success (retry_times: {self.retry_times})')
                break
        
        if "ERROR" in self.container_name:
            print(f'[ERROR IN ENV][{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]reset failed (retry_times: {self.retry_times})')
            self.max_env_time = 0
            return '', {"suffix": ''}

        self.data_line['container_name'] = self.container_name

        # add tools to repo_env
        self.repo_env.add_commands(self.tools)

        # get_instruction
        self.problem_statement, self.issue = self.get_instruction()

        # record time
        self.traj_reset_time = round(time.time() - st, 4)

        self.history.append({"role": "user", "content": self.problem_statement})

        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]finish reset, task_idx: {self.task_idx}, retry_times: {self.retry_times}, retry_time: {self.traj_reset_time}s')
        return self.problem_statement, {"suffix": self.issue} # TODO suffix和obs的区别
    
    def step(self, action: str):
        """
        @input:
            action: <answer>Right</answer>
        @output:
            [obs] At turn 1, you moved Down, which is effective.
            [reward] 0.0
            [terminate] False
            [truncated] False
            [info] {'suffix': 'Here is the current state of the FrozenLake:\n____\n_OP_\n___O\nGO__\n', 'metrics': {'action_is_effective': True, 'action_is_valid': True, 'success': False, 'format_penalty': 0.0}, 'action': 1, 'action_content': 'Down', 'think_content': ''}
        """
        # print(f'\n\n-------------- SWEEnv.step(start) --------------')
        st = time.time()

        self.step_count += 1
        obs = ''
        info = {"suffix": "","metrics": ""}
        bash_output, error_code, execute_time = '', '', 0
        action_is_valid,action_is_effective = False,False
        action_original = copy.deepcopy(action)
        
        # update history with response
        self.history.append({"role": "assistant_original", "content": f'{action}'})

        print(f'[ENV交互时间消耗]{(time.time() - self.time_start)/60} min.')
        if time.time() - self.time_start > self.max_env_time:
            self.truncated = True
            self.env_timeout = True
            self.metrics = {
                "reward": self.reward,
                "success": self.terminate,
                "truncated": self.truncated,
                "format_penalty": self.format_penalty,
                "action_is_valid": round(sum(self.action_is_valid_lst)/len(self.action_is_valid_lst), 4) if self.action_is_valid_lst else 0.0,
                "action_is_effective": round(sum(self.action_is_effective_lst)/len(self.action_is_effective_lst), 4) if self.action_is_effective_lst else 0.0,
                "step_count": self.step_count,
                "retry_times": self.retry_times,
                "traj_reset_time": self.traj_reset_time,
                "traj_step_time": round(self.traj_step_time, 4),
                "traj_reward_time": self.traj_reward_time,
                "traj_total_time": self.traj_total_time,
                "traj_rollout_time": self.traj_rollout_time,
                "task_idx": self.task_idx # for rollout_log
            }  # TODO： 
            info = {
                "suffix": obs,
                "metrics": self.metrics,
            }
            print(f'[ERROR IN ENV][ENV TIMEOUT]task_idx: {self.task_idx}, route_key: {self.route_key}, docker_image: {self.data_line.get("docker_image", "")}')
            info['metrics']['env_timeout'] = True # 增加了一个这个
            return f"ERROR: The command took too long to execute (>{self.max_env_time}s)",self.reward,self.terminate,self.truncated,info

        # split think content
        if "</think>" in action: action = action.split("</think>")[-1].strip()

        # parse action from response & format exam
        action_info = self.parse_action(action)
        format_info = self.format_exam(action_info) # error, error_msg
        # print(f'\n[DEBUG][模型输入original]{[action_original]}[DEBUG][parse_action]{action_info["action_content"]}\n[DEBUG][format_info]{format_info}')


        # run action and get obs
        if not format_info['error']:
            # run action
            bash_output, error_code, execute_time = self.repo_env.run_action(action_info["action"],timeout=180) # Action object
            obs = str(Observation(bash_output, error_code, action_info["action"]))
        else:
            obs = format_info.get("error_msg", "")

        # invalid action
        if "finish" in action_info["action"].function_name.lower() or "submit" in action_info["action_content"].lower(): # 任务完成
            self.terminate = True
            action_is_valid,action_is_effective = True,True
        
        try:
            if not ("Invalid Action" in str(error_code)) and not format_info['error']: # 无效动作
                action_is_valid = True # cd等不被允许的动作也视为成功
            if error_code == "0":  # 动作执行成功： 
                action_is_effective = True
        except Exception as e:
            print(f'[DEBUG][error in action valid]{e}')
            action_is_valid = False
            action_is_effective = False

        self.history.append({"role": "assistant", "content": f'{action_info["action_content"]}','action_is_valid':action_is_valid,'action_is_effective':action_is_effective})
        self.action_is_valid_lst.append(1 if action_is_valid else 0)
        self.action_is_effective_lst.append(1 if action_is_effective else 0)
        

        # 超过最大轮数强制结束
        if not self.terminate and self.step_count >= self.max_steps:
            self.truncated = True
        # 计算reward
        if self.terminate or self.truncated:
            st_reward = time.time()
            self.reward = self.repo_env.calculate_reward()
            self.traj_reward_time = round(time.time() - st_reward, 4)
            self.traj_total_time = round(self.traj_reset_time + self.traj_step_time + self.traj_reward_time, 4)
            self.traj_rollout_time = round(time.time() - self.time_start, 4)

        self.traj_step_time += round(time.time() - st, 4) - self.traj_reward_time

        self.metrics = {
            "reward": self.reward,
            "success": self.terminate,
            "truncated": self.truncated,
            "format_penalty": self.format_penalty,
            "action_is_valid": round(sum(self.action_is_valid_lst)/len(self.action_is_valid_lst), 4) if self.action_is_valid_lst else 0.0,
            "action_is_effective": round(sum(self.action_is_effective_lst)/len(self.action_is_effective_lst), 4) if self.action_is_effective_lst else 0.0,
            "step_count": self.step_count,
            "retry_times": self.retry_times,
            "traj_reset_time": self.traj_reset_time,
            "traj_step_time": round(self.traj_step_time, 4),
            "traj_reward_time": self.traj_reward_time,
            "traj_total_time": self.traj_total_time,
            "traj_rollout_time": self.traj_rollout_time,
            "task_idx": self.task_idx # for rollout_log
        }  # TODO： 
        info = {
            "suffix": obs,
            "metrics": self.metrics,
        }
        self.history.append({"role": "user", "content": f'{obs}'})
        # print('-------------------------------------------------')
        print(f'[DEBUG][self.step_count]{self.step_count}(max_steps: {self.max_steps})')
        # print(f'[DEBUG][obs]{obs}')
        # print(f'[DEBUG][reward]{self.reward}')
        # print(f'[DEBUG][terminate]{self.terminate}')
        # print(f'[DEBUG][truncated]{self.truncated}')
        print(f'[DEBUG][metrics]{self.metrics}')
        # print(f'[DEBUG][action_is_valid]{action_is_valid}')
        # print(f'[DEBUG][action_is_effective]{action_is_effective}')
        # print(f'-------------- SWEEnv.step(end) -----------------')
        if self.terminate or self.truncated:
            self.close()
        return obs,self.reward,self.terminate,self.truncated,info

    def parse_action(self, response_text):
        """
        Extracts:
        - thought: everything before the first <function=...> block
        - action: the entire first <function=...></function> block
        Returns (thought, action).
        """
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


        # convert action to Action object
        action_obj = Action.from_string(action)
        # print(f'[ATTENTION][response_text]{response_text}\n[before] action: {action}\n[after]action_obj: {action_obj}')

        action_info = {
            "response": response_text,
            "action": action_obj, # Action object
            "action_content": action, # string
            "think_content": thought, # string
        }
        return action_info

    def format_exam(self,action_info:dict):
        """        
        """
        thought, action, response = action_info['think_content'], action_info['action'], action_info['action_content']

        # format error type1
        action_dict = action.to_dict()
        if action_dict['function'] == '':
            return {'error':True,"error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<function=func1><parameter=param1>xxx</parameter><parameter=param2>xxx</parameter></function>'."}
        # format error type2
        elif "<parameter>" in response:
            return {'error':True,"error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<parameter=xxxx>', not '<parameter>xxxx>.'"}
        elif "<function>" in response:
            return {'error':True,"error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<function=func1>', not '<function>func1>.'"}
        elif "<parameter" in response and "</parameter>" not in response:
            return {'error':True,"error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<parameter=param1>xxx</parameter>'. Do not forget to close the parameter tag."}
        elif "<function" in response and "</function>" not in response:
            return {'error':True,"error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<function=func1>xxxx</function>'. Do not forget to close the function tag."}
        elif not action:
            return {'error':True,"error_msg": "ERROR: The model output is illegal, please check it carefully."}
        # format error type3: function name
        function_name = action_dict['function']
        if function_name not in self.tool_names:
            return {'error':True,"error_msg": f"Invalid Action: input action must be one of allowed actions. Allowed actions: ['file_editor', 'execute_bash', 'search', 'finish']. Current input action: {function_name}. "}
         


        return {"error": False,"error_msg": ""}

    def render(self, mode=None):
        # 源frozen_lake的render函数，将环境的内部数值状态转换为直观的文本或图像表示，用在get_instruction中
        # 这里应该不需要写
        pass

    def sample_random_action(self):
        actions = []
        # 执行正确动作
        actions.append("Now let's look for the @return_future decorator. Based on the issue description, it's likely in one of the tornado modules. Let's search for it:\n\n<function=search>\n  <parameter=search_term>return_future</parameter>\n  <parameter=path>/testbed</parameter>\n</function>")
        actions.append("Now let's look for the @return_future decorator. Based on the issue description, it's likely in one of the tornado modules. Let's search for it:\n\n<function=search>\n  <parameter=search_term>return_future</parameter>\n  <parameter=path>/testbed</parameter>\n</function>")
        actions.append("Now let's look for the @return_future decorator. Based on the issue description, it's likely in one of the tornado modules. Let's search for it:\n\n<function=search>\n  <parameter=search_term>return_future</parameter>\n  <parameter=path>/testbed</parameter>\n</function>")
        # 执行不允许的动作（exit code: 1）
        actions.append("the issue:\n\n<function=execute_bash>\n  <parameter=cmd>cd /testbed && python reproduce_issue.py</parameter>\n</function>")
        # 执行错误动作（exit code: 2）
        actions.append('<function=execute_bash>\n  <parameter=cmd>python reproduce_issue.py</parameter>\n</function>')
        # 提交
        actions.append("tness.\n\n<function=finish>\n  <parameter=command>submit</parameter>\n  <parameter=result>I have successfully implemented a fix for the GitHub issue \"Callback Receives Unexpected Argument When No Result is Provided\" in the `@return_future` decorator.\n\n**Changes Made:**\n1. Added `import inspect` to `/testbed/tornado/concurrent.py`\n2. Modified the `run_callback` function to intelligently handle callbacks when no result is provided:\n   - When `result is None`, inspect the callback signature\n   - If callback accepts no parameters, call it without arguments\n   - Otherwise, call it with the `None` result as before\n   - Includes robust fallback mechanism for cases where signature inspection fails\n\n**Key Benefits:**\n- \u2705 Fixes the original issue: callbacks expecting no arguments work correctly\n- \u2705 Maintains backward compatibility: all existing tests pass\n- \u2705 Handles edge cases: works with lambdas, default parameters, *args, **kwargs\n- \u2705 Robust: graceful fallback when signature inspection fails\n- \u2705 Minimal impact: only changes behavior when result is `None`\n\nThe fix ensures that when a `@return_future` decorated function calls its callback without arguments, the client callback is invoked appropriately based on its signature, resolving the TypeError while preserving all existing functionality.</parameter>\n</function>")
        return random.choice(actions)

    def close(self):
        if self.history and self.data_line:
            if self.terminate:stop_reason = "terminate"
            elif self.env_timeout:stop_reason = "env_timeout"
            elif self.truncated:stop_reason = "truncated"
            else:stop_reason = "unknown"
            save = {
                "task_idx": self.task_idx,
                "reward": self.reward,
                "terminate": self.terminate,
                "truncated": self.truncated,
                "stop_reason": stop_reason,
                "container_name": self.container_name,
                "route_key": self.route_key,
                # "problem_statement": self.problem_statement,
                "retry_times": self.retry_times,
                "docker_image": self.data_line.get('docker_image', ''),
                "history": self.history,
                "metrics": self.metrics,
                "retry_times": self.retry_times,
            }
            os.makedirs(self.traj_dir,exist_ok=True)
            valid_score = round(sum(self.action_is_valid_lst)/len(self.action_is_valid_lst), 2) if self.action_is_valid_lst else 0.0
            effective_score = round(sum(self.action_is_effective_lst)/len(self.action_is_effective_lst), 2) if self.action_is_effective_lst else 0.0
            log_path = os.path.join(self.traj_dir,f'{self.task_idx}-{time.strftime("%m%d_%H%M%S", time.localtime())}-re{self.reward}_v{valid_score}_e{effective_score}_{stop_reason}_st{self.step_count}_time{self.traj_total_time}.json')
            write_data_json(save,log_path)
        # close
        self.repo_env.close()
        # 这里不能clean_record，否则返回就不对了。

    def clean_record(self):
        print(f'[DEBUG][clean record finish]')
        self.history = []
        self.data_line = None
        self.container_name = None
        self.route_key = None
        self.task_idx = None
        self.step_count = 0
        self.retry_times = 0
        self.traj_reset_time = 0
        self.traj_step_time = 0
        self.traj_reward_time = 0
        self.traj_total_time = 0
        self.problem_statement, self.issue = '', ''
        self.reward,self.terminate,self.truncated = 0,False,False
        self.env_timeout = False

    def get_history(self):
        return self.history

    def get_key_params(self):
        return {
            "task_idx": self.task_idx,
            "container_name": self.container_name,
            "route_key": self.route_key,
            "problem_statement": self.problem_statement,
            "retry_times": self.retry_times,
            "docker_image": self.data_line.get('docker_image', ''),
        }
    
if __name__ == "__main__":
    
    """
    env.step 返回的info中必须有suffix字段
    """
    try:
        env = SWEEnv(
                mode="train", 
                data_path="data/swe/250814_trainset_v1_r2e_lite_vpc_4578_split100/part_0.jsonl", 
                train_idx_range=(0, 4577), val_idx_range=(0,128),
                max_steps=2,
                max_reset_retry_times=20,
                format_penalty=0.0,
                tools=["swe_env/src/agenthub/tools/file_editor.py",
                    "swe_env/src/agenthub/tools/execute_bash.py",
                    "swe_env/src/agenthub/tools/search.py",
                    "swe_env/src/agenthub/tools/finish.py"],
                action_pattern="^<answer>(.*?)</answer>$",
                special_token_list=("<think>", "</think>", "<answer>","</answer>", "<|im_start|>", "<|im_end|>"),
                swe_rex_host="https://xrl-aliyun.alibaba-inc.com/swe-rex/docker",
                )
        obs, info = env.reset(seed=42)
        print(f'[obs]\n{obs}')
        print(f'[info]\n{info}')
        while True:
            # keyboard = input("Enter action: ")
            # if keyboard == "q":
                # break
            keyboard = """ Let me check the exact content of the file to find the correct location:

<function=file_editor>
<parameter=command>view</parameter>
<parameter=path>/testbed/sphinx/directives/other.py</parameter>
<parameter=view_range>[120, 160]</parameter>
</function>

<function=file_editor>
<parameter=command>str_replace</parameter>
<parameter=path>/testbed/sphinx/directives/other.py</parameter>
<parameter=old_str>        rawtext = ''.join(lines)

        # start-line and end-line options
        startline = self.options.get('start-line', None)</parameter>
<parameter=new_str>        rawtext = ''.join(lines)

        # Emit source-read event for included files to allow extensions to modify content
        # Use relative filename for docname to be consistent with main document processing
        if hasattr(env.app, 'emit'):
            source = [rawtext]
            env.app.emit('source-read', rel_filename, source)
            rawtext = source[0]

        # start-line and end-line options
        startline = self.options.get('start-line', None)</parameter>
</function>

Now let's test the fix:

<function=execute_bash>
<parameter=cmd>cd /testbed && python test_include_issue.py</parameter>
</function>

Perfect! The fix is working. The issue has been resolved by adding the `source-read` event emission in the `Include` directive, allowing extensions to modify included file content before it's processed.

<function=finish>
<parameter=command>submit</parameter>
<parameter>result>Fixed the issue where the `source-read` event was not being emitted for included files when using the `.. include::` directive.

**Problem**: The `Include` directive used `env.read_file()` which doesn't emit the `source-read` event, while main documents use `env.read_doc()` which does emit this event.

**Solution**: Modified the `Include` directive in `/testbed/sphinx/directives/other.py` to emit the `source-read` event after reading included file content, allowing extensions to modify the content before it's processed further.

**Key changes**:
- Added code in the `Include.run()` method to emit the `source-read` event with the included file's relative path as docname
- Content is modified after reading but before applying directive options like `start-line`, `end-line`, etc.
- Works for nested includes and maintains backward compatibility

The fix has been verified to work correctly with the reproduction case provided in the GitHub issue.</parameter>
</function>"""
            obs, reward, terminate, truncated, info = env.step(keyboard)
            print(f'[obs]\n{obs}')
            print(f'[reward]{reward}')
            print(f'[terminate]{terminate}')
            print(f'[truncated]{truncated}')
            print(f'[info]\n{info}')
            history = env.get_history()
            print(f'[history]\n{history}')
            if terminate or truncated:
                break
    finally:
        env.close()
