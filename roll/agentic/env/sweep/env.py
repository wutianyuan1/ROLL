import numpy as np
import random
from typing import Optional, Any, Tuple, Dict

import gem
from gem import Env
from roll.agentic.env.parse_action_utils import default_parser_action_func
from roll.agentic.utils import all_seed


class SweepEnv(Env):
    """
    自定义的清扫任务环境
    不依赖 Gymnasium，完全自定义实现
    """
    
    def __init__(self,
                 grid_size: int = 5,
                 max_steps: int = 50,
                 render_mode: str = "text",
                 grid_lookup=None,
                 grid_vocab=None,
                 action_lookup=None,
                 env_instruction=None,
                 format_penalty=0.0,
                 action_pattern="^<answer>(.*?)</answer>$",
                 special_token_list=("<think>", "</think>", "<answer>","</answer>", "<|im_start|>", "<|im_end|>"),
                 **kwargs
                 ):
        
        # 默认配置
        self.GRID_LOOKUP = {0: "P", 1: "_", 2: "D", 3: "C", 4: "X"}
        self.GRID_VOCAB = {
            "P": "player", 
            "_": "empty", 
            "D": "dirt", 
            "C": "clean", 
            "X": "player on dirt"
        }
        self.ACTION_LOOKUP = {0: "Left", 1: "Down", 2: "Right", 3: "Up", 4: "Clean"}
        
        # 环境指令
        self.env_instruction = (
            "You are solving the Sweep puzzle. "
            "Move around the grid and clean all the dirt. "
            "You can move in four directions or clean the current position. "
            f"The answer must be one of action in a turn, format is <answer>Right</answer>"
        )
        
        # 覆盖默认配置
        if grid_lookup is not None:
            self.GRID_LOOKUP = grid_lookup
        if grid_vocab is not None:
            self.GRID_VOCAB = grid_vocab
        if action_lookup is not None:
            self.ACTION_LOOKUP = action_lookup
        if env_instruction is not None:
            self.env_instruction = env_instruction
            
        # 环境参数
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.format_penalty = format_penalty
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list
        
        # 初始化环境状态
        self.grid = None
        self.player_pos = None
        self.dirt_count = 0
        self.step_count = 0
        self.cleaned_count = 0
        
    def _generate_random_grid(self, seed=None):
        """生成随机网格"""
        if seed is not None:
            random.seed(seed)
            
        # 创建空网格
        self.grid = np.full((self.grid_size, self.grid_size), 1)  # 1 = empty
        
        # 随机放置玩家起始位置
        self.player_pos = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
        self.grid[self.player_pos[0], self.player_pos[1]] = 0  # 0 = player
        
        # 随机放置污垢
        self.dirt_count = random.randint(self.grid_size, self.grid_size * 2)
        dirt_positions = []
        for _ in range(self.dirt_count):
            while True:
                pos = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
                if pos != self.player_pos and self.grid[pos[0], pos[1]] == 1:
                    self.grid[pos[0], pos[1]] = 2  # 2 = dirt
                    dirt_positions.append(pos)
                    break
                    
        return self.grid
    
    def _is_valid_move(self, new_pos):
        """检查移动是否有效"""
        return (0 <= new_pos[0] < self.grid_size and 
                0 <= new_pos[1] < self.grid_size)
    
    def _move_player(self, action):
        """移动玩家"""
        if action == 0:  # Left
            new_pos = [self.player_pos[0], self.player_pos[1] - 1]
        elif action == 1:  # Down
            new_pos = [self.player_pos[0] + 1, self.player_pos[1]]
        elif action == 2:  # Right
            new_pos = [self.player_pos[0], self.player_pos[1] + 1]
        elif action == 3:  # Up
            new_pos = [self.player_pos[0] - 1, self.player_pos[1]]
        else:
            return False
            
        if self._is_valid_move(new_pos):
            # 更新网格
            self.grid[self.player_pos[0], self.player_pos[1]] = 1  # 原位置变空
            self.player_pos = new_pos
            self.grid[self.player_pos[0], self.player_pos[1]] = 0  # 新位置放玩家
            return True
        return False
    
    def _clean_position(self):
        """清理当前位置的污垢"""
        if self.grid[self.player_pos[0], self.player_pos[1]] == 2:  # 如果是污垢
            self.grid[self.player_pos[0], self.player_pos[1]] = 3  # 变成清洁
            self.cleaned_count += 1
            return True
        return False
    
    def get_instructions(self) -> str:
        """获取环境指令"""
        grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join(
            [f"{k}: {v}" for k, v in self.GRID_VOCAB.items()])
        action_lookup_str = "\nYour available actions are:\n" + ", ".join(
            [f"{v}" for k, v in self.ACTION_LOOKUP.items()])
        return self.env_instruction + grid_vocab_str + action_lookup_str
    
    def get_task_suffix(self) -> Any:
        """获取任务后缀信息"""
        if self.render_mode == "text":
            return f"Here is the current state of the Sweep grid:\n{self.render(mode='text')}\n"
        else:
            return self.render(mode=self.render_mode)
    
    def reset(self, seed=None):
        """重置环境"""
        Env.reset(self, seed)
        self.step_count = 0
        self.cleaned_count = 0
        
        with all_seed(seed):
            self._generate_random_grid(seed)
            
        return self.get_instructions(), {"suffix": self.get_task_suffix()}
    
    def step(self, action: str):
        """执行一步动作"""
        self.step_count += 1
        action_info = self.parse_action(action)
        
        if action_info["action"] is None:
            terminate_obs = f"At turn {self.step_count}, You did not provide a valid action."
            reward = self.format_penalty
            metrics = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": self.cleaned_count >= self.dirt_count,
                "format_penalty": self.format_penalty
            }
            info = {
                "suffix": self.get_task_suffix(),
                "metrics": metrics,
            }
            info.update(action_info)
            return terminate_obs, reward, False, False, info
        
        action_id = action_info["action"]
        action_effective = False
        
        if action_id == 4:  # Clean action
            action_effective = self._clean_position()
            if action_effective:
                next_obs = f"At turn {self.step_count}, you cleaned the dirt, which is effective."
            else:
                next_obs = f"At turn {self.step_count}, you tried to clean, but there was no dirt here."
        else:  # Move action
            action_effective = self._move_player(action_id)
            if action_effective:
                next_obs = f"At turn {self.step_count}, you moved {action_info['action_content']}, which is effective."
            else:
                next_obs = f"At turn {self.step_count}, you tried to move {action_info['action_content']}, which is not effective."
        
        # 计算奖励
        reward = 1.0 if action_id == 4 and action_effective else 0.0
        
        # 检查是否完成
        terminated = self.cleaned_count >= self.dirt_count
        truncated = self.step_count >= self.max_steps
        
        metrics = {
            "action_is_effective": action_effective,
            "action_is_valid": True,
            "success": terminated,
            "format_penalty": self.format_penalty,
            "cleaned_count": self.cleaned_count,
            "dirt_count": self.dirt_count
        }
        
        info = {
            "suffix": self.get_task_suffix(),
            "metrics": metrics,
        }
        info.update(action_info)
        
        return next_obs, reward, terminated, truncated, info
    
    def parse_action(self, text):
        """解析动作文本"""
        return default_parser_action_func(text, self.action_pattern, self.ACTION_LOOKUP, self.special_token_list)
    
    def render(self, mode=None):
        """渲染环境"""
        if not mode:
            mode = self.render_mode
            
        if mode == "text":
            # 创建显示网格
            display_grid = self.grid.copy()
            
            # 如果玩家在污垢上，标记为特殊状态
            if self.grid[self.player_pos[0], self.player_pos[1]] == 2:
                display_grid[self.player_pos[0], self.player_pos[1]] = 4  # X = player on dirt
            
            # 转换为文本表示
            return "\n".join("".join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in display_grid)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def sample_random_action(self):
        """随机采样动作"""
        return random.choice(list(self.ACTION_LOOKUP.values()))
    
    def close(self):
        """关闭环境"""
        pass


if __name__ == "__main__":
    # 测试环境
    env = SweepEnv(grid_size=5, max_steps=20)
    obs, info = env.reset(seed=42)
    print('***** reset *****')
    print('[obs]\n', obs)
    print('[info]', info)
    print(f'[info["suffix"]]\n{info["suffix"]}')
    
    while True:
        keyboard = input("Enter action (0-4, q to quit): ")
        if keyboard == "q":
            break
        try:
            action = int(keyboard)
            if action not in range(5):
                print("Invalid action, must be 0-4")
                continue
        except ValueError:
            print("Invalid input, must be a number")
            continue
            
        action_text = f"<answer>{env.ACTION_LOOKUP[action]}</answer>"
        obs, reward, terminate, truncated, info = env.step(action_text)
        print('***** step *****')
        print('[obs]\n', obs)
        print('[reward]', reward)
        print('[terminate]', terminate)
        print('[truncated]', truncated)
        print('[info]', info)
        print(f'[info["suffix"]]\n{info["suffix"]}')
        
        if terminate or truncated:
            break 