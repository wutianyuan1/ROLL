
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from roll.agentic.env.base import BaseEnvConfig

@dataclass
class SokobanMCPEnvConfig(BaseEnvConfig):
    base_env_instruction: str = (
        "You are an expert Sokoban solver. Your goal is to push all boxes (X) onto the targets (O). "
        "On each turn, you MUST output a single word representing your move. "
        "Your response must be wrapped in <answer> tags. For example: <answer>Up</answer>"
    )    
    server_url: Optional[str] = None
    action_pattern: str = r"<answer>\s*(.*?)\s*</answer>"
    action_lookup: Optional[Dict[int, str]] = field(
        default_factory=lambda: {1: "Up", 2: "Down", 3: "Left", 4: "Right"}
    )
    special_token_list: Optional[List[str]] = field(default_factory=lambda: ["<think>", "</think>", "<answer>",                                                                             
                                                                            "</answer>", "<|im_start|>", "<|im_end|>"])
    
    def __post_init__(self):
        if self.server_url is None:
            raise ValueError(
                "A 'server_url' must be provided when creating an instance of MCPEnvConfig or its subclasses."
            )
        action_lookup_str = "\nYour available actions are:\n" + ", ".join(
            [f"{v}" for k, v in self.action_lookup.items()])
        self.env_instruction = self.env_instruction + action_lookup_str
    