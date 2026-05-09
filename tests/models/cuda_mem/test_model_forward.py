import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.models.cuda_mem.utils import log_gpu_memory_usage

device = torch.device("cuda:0")

path = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16)
model.eval()
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(
    path, trust_remote_code=True, use_fast=True, split_special_tokens=True, padding_side="left", padding=True
)
tokenizer.pad_token = tokenizer.eos_token

prompts = [
    "Compared with Google, Microsoft",
    "据悉，美国总统",
]
batch = tokenizer(prompts, return_tensors="pt", padding=True)

model = model.to("cpu")
torch.cuda.empty_cache()
log_gpu_memory_usage(head="initialize offload")
breakpoint()

with torch.no_grad():
    # model prepare done
    model = model.to(device)
    input_ids = batch["input_ids"].to(device)
    outputs = model.forward(input_ids)

del outputs, input_ids, model
gc.collect()
torch.cuda.empty_cache()
log_gpu_memory_usage(head="forward offload")

breakpoint()
