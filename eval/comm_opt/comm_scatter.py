import os
import time
import argparse
import numpy as np

import torch
import torch.distributed as dist
from modelscope import AutoConfig, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-7B"

def setup_distributed():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(
        f"Initialized process rank {rank} of {world_size} on device cuda:{local_rank}."
    )
    return rank, world_size, local_rank

def create_dummy_model(model_id, device):
    """Creates a model with dummy weights."""
    print(f"[{dist.get_rank()}] Creating dummy model from config: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.to(device)
    total_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return model, total_size_mb

def get_sharded_state_dict(model, num_shards):
    """
    Deterministically splits the model's state_dict into N shards (lists of key-tensor pairs).
    This version pads shards to ensure they all have the same number of tensors,
    which simplifies the all-gather logic.
    """
    state_dict_items = list(model.state_dict().items())
    num_tensors = len(state_dict_items)
    
    # Use ceiling division to determine max shard size
    max_chunk_size = (num_tensors + num_shards - 1) // num_shards
    
    shards = []
    all_keys_in_order = []
    for i in range(num_shards):
        shard = []
        shard_keys = []
        for j in range(max_chunk_size):
            idx = i + j * num_shards
            if idx < num_tensors:
                key, tensor = state_dict_items[idx]
                shard.append(tensor)
                shard_keys.append(key)
        shards.append(shard)
        all_keys_in_order.append(shard_keys)
        
    return shards, all_keys_in_order

def run_transfer_test(model, total_size_mb, rank, world_size, local_rank, node1_group, shards, keys_in_shards):
    num_gpus_per_node = world_size // 2
    node_rank = rank // num_gpus_per_node
    device = torch.device(f"cuda:{local_rank}")

    # --- Timers ---
    shard_send_time = 0.0
    gather_time = 0.0
    
    # Global barrier to synchronize before starting the test
    dist.barrier()
    if rank == 0:
        global_start_time = time.time()

    # --- Stage 1: Sharded Point-to-Point Send (Scatter) ---
    if node_rank == 0: # Senders on Node 0
        my_shard = shards[local_rank]
        peer_rank = rank + num_gpus_per_node
        print(f"[Rank {rank}] -> [Rank {peer_rank}] Starting Stage 1: Sending shard {local_rank} ({len(my_shard)} tensors).")
        
        torch.cuda.synchronize(device=device)
        start_time = time.time()

        for tensor in my_shard:
            dist.send(tensor=tensor, dst=peer_rank)
            
        torch.cuda.synchronize(device=device)
        shard_send_time = time.time() - start_time
        print(f"[Rank {rank}] -> [Rank {peer_rank}] Stage 1 finished in {shard_send_time:.4f}s.")

    else: # Receivers on Node 1
        peer_rank = rank - num_gpus_per_node
        my_shard_idx = local_rank
        tensors_to_receive = shards[my_shard_idx]
        print(f"[Rank {rank}] <- [Rank {peer_rank}] Starting Stage 1: Receiving shard {my_shard_idx} ({len(tensors_to_receive)} tensors).")

        for tensor in tensors_to_receive:
            dist.recv(tensor=tensor, src=peer_rank)
        print(f"[Rank {rank}] <- [Rank {peer_rank}] Stage 1 finished.")


    # --- Stage 2: Intra-Node All-Gather with Key Mapping ---
    if node_rank == 1:
        dist.barrier(group=node1_group)
        print(f"[Rank {rank}] Starting Stage 2: Intra-Node All-Gather.")

        if rank == num_gpus_per_node:  # Let rank 8 time this stage
            torch.cuda.synchronize(device=device)
            start_gather_time = time.time()

        # 构建 key -> shard_idx 的映射（谁拥有哪个 key）
        key_to_shard = {}
        for shard_idx, key_list in enumerate(keys_in_shards):
            for key in key_list:
                key_to_shard[key] = shard_idx

        # 遍历所有 keys，依次 gather 每个 tensor
        all_keys = sum(keys_in_shards, [])  # Flatten all keys across shards

        for key in all_keys:
            # 找到这个 key 属于哪个 shard（即发送方 local_rank）
            owner_shard_idx = key_to_shard[key]

            # 获取当前 rank 自己 shard 中的 tensor
            local_key_list = keys_in_shards[local_rank]
            if key in local_key_list:
                tensor_idx = local_key_list.index(key)
                input_tensor = shards[local_rank][tensor_idx]
            else:
                # 如果当前 rank 没有这个 key，构造一个 dummy tensor（用于 all_gather 对齐）
                # 但我们只真正使用拥有该 key 的 rank 的数据
                dummy_shape = model.state_dict()[key].shape
                input_tensor = torch.empty(dummy_shape, device=device, dtype=model.state_dict()[key].dtype)

            gathered_tensors = [torch.empty_like(input_tensor) for _ in range(num_gpus_per_node)]

            dist.all_gather(gathered_tensors, input_tensor, group=node1_group)

            # 写入到自己的 state_dict
            model.state_dict()[key].copy_(gathered_tensors[owner_shard_idx])

        torch.cuda.synchronize(device=device)
        if rank == num_gpus_per_node:
            gather_time = time.time() - start_gather_time
        print(f"[Rank {rank}] Stage 2 finished.")



    # --- Collect and Report Timings ---
    dist.barrier()
    if rank == 0:
        total_time = time.time() - global_start_time
    
    # Collect all shard_send_times from Node 0 senders.
    local_time_tensor = torch.tensor([shard_send_time], dtype=torch.float32, device=device)
    all_times_tensor = torch.zeros(world_size, dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(all_times_tensor, local_time_tensor)
    
    # Rank 0 does the final reporting
    if rank == 0:
        
        # Get send times from ranks 0-7
        sender_times = all_times_tensor.cpu().numpy()[:num_gpus_per_node]
        
        # Receive gather_time from rank 8
        gather_time_tensor = torch.zeros(1, device=device)
        dist.recv(tensor=gather_time_tensor, src=num_gpus_per_node)
        final_gather_time = gather_time_tensor.cpu().item()

        # Final Report
        print("\n" + "="*60)
        print("      SCATTER-GATHER TRANSFER SPEED RESULTS")
        print("="*60)
        print(f"Model Size:              {total_size_mb:.2f} MB")
        print("-" * 60)
        print(f"Stage 1 (Max Shard Send): {np.max(sender_times):.4f} s")
        print(f"Stage 2 (Gather Time):    {final_gather_time:.4f} s")
        print("-" * 60)
        print(f"Total End-to-End Time:   {total_time:.4f} s")
        print("="*60 + "\n")
        
    elif rank == num_gpus_per_node: # Rank 8 sends its measured time
        gather_time_tensor = torch.tensor([gather_time], device=device)
        dist.send(tensor=gather_time_tensor, dst=0)

def main():
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) != 16:
        print("Error: This script is designed for WORLD_SIZE=16 (2 nodes x 8 GPUs).")
        return
    rank, world_size, local_rank = setup_distributed()
    num_gpus_per_node = world_size // 2
    device = torch.device(f"cuda:{local_rank}")
    model, total_size_mb = create_dummy_model(MODEL_ID, device)
    if rank == 0:
        print("\n" + "="*60)
        print(f"Transfer Scheme: Sharded P2P Send, followed by Intra-Node All-Gather")
        print(f"Model Size per Transfer: {total_size_mb:.2f} MB")
        print("="*60 + "\n")
        # Create a process group for Node 1 (ranks 8 to 15)
    node1_ranks = list(range(num_gpus_per_node, world_size))
    node1_group = dist.new_group(ranks=node1_ranks)
    # Get the sharding scheme. Everyone calculates this so they know their roles.
    shards, keys_in_shards = get_sharded_state_dict(model, num_gpus_per_node)
    for i in range(3):
        run_transfer_test(model, total_size_mb, rank, world_size, local_rank, node1_group, shards, keys_in_shards)
    dist.destroy_process_group()
    print(f"[{rank}] Cleaned up and finished.")

if __name__ == "__main__":
    main()
