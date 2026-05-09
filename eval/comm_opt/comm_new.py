import os
import time
import argparse
import numpy as np

import torch
import torch.distributed as dist

def setup_distributed():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"Initialized process rank {rank} of {world_size} on device cuda:{local_rank}.")
    return rank, world_size, local_rank

def create_dummy_model(model_size_billion: float, device):
    """Creates a dummy model with 256 equally-sized fp16 tensors."""
    total_params = int(model_size_billion * 1e9)
    num_tensors = 256
    params_per_tensor = total_params // num_tensors

    model_state_dict = {}
    for i in range(num_tensors):
        shape = (params_per_tensor,)  # 一维张量更方便处理
        model_state_dict[f"param_{i}"] = torch.randn(shape, dtype=torch.float16, device=device)

    class DummyModel(torch.nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            self._state_dict = state_dict
        def state_dict(self):
            return self._state_dict

    model = DummyModel(model_state_dict)
    total_size_mb = total_params * 2 / (1024 ** 2)  # fp16 = 2 bytes per param
    return model, total_size_mb

def get_sharded_state_dict(model, num_shards):
    """
    Splits the model's state_dict into N shards deterministically.
    Each shard is a list of (key, tensor) pairs.
    """
    state_dict_items = list(model.state_dict().items())
    num_tensors = len(state_dict_items)
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

def run_transfer_test(model, total_size_mb, rank, world_size, local_rank, receiver_group, shards, keys_in_shards):
    num_senders = world_size // 2
    node_rank = rank // (world_size // 2) if world_size > 1 else 0
    device = torch.device(f"cuda:{local_rank}")

    shard_send_time = 0.0
    gather_time = 0.0

    dist.barrier()

    if rank == 0:
        global_start_time = time.time()

    # --- Stage 1: Sharded Point-to-Point Send (Scatter) ---
    if rank < num_senders:
        sender_idx = rank
        receiver_rank = sender_idx + num_senders
        my_shard = shards[sender_idx]

        print(f"[Rank {rank}] -> [Rank {receiver_rank}] Starting Stage 1: Sending shard {sender_idx} ({len(my_shard)} tensors).")

        torch.cuda.synchronize(device=device)
        start_time = time.time()

        for tensor in my_shard:
            dist.send(tensor=tensor, dst=receiver_rank)

        torch.cuda.synchronize(device=device)
        shard_send_time = time.time() - start_time
        print(f"[Rank {rank}] -> [Rank {receiver_rank}] Stage 1 finished in {shard_send_time:.4f}s.")

    else:
        receiver_idx = rank - num_senders
        sender_rank = receiver_idx
        tensors_to_receive = shards[receiver_idx]

        print(f"[Rank {rank}] <- [Rank {sender_rank}] Starting Stage 1: Receiving shard {receiver_idx} ({len(tensors_to_receive)} tensors).")

        for tensor in tensors_to_receive:
            dist.recv(tensor=tensor, src=sender_rank)
        print(f"[Rank {rank}] <- [Rank {sender_rank}] Stage 1 finished.")

    # --- Stage 2: Intra-Node All-Gather with Key Mapping ---
    if rank >= num_senders:
        dist.barrier(group=receiver_group)
        print(f"[Rank {rank}] Starting Stage 2: Intra-Group All-Gather.")

        if rank == num_senders:
            torch.cuda.synchronize(device=device)
            start_gather_time = time.time()
        
        for holder_rank, shard in enumerate(shards):
            for tensor in shard:
                dist.broadcast(tensor, holder_rank + num_senders, group=receiver_group)
                # dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=receiver_group)

        torch.cuda.synchronize(device=device)
        if rank == num_senders:
            gather_time = time.time() - start_gather_time
        print(f"[Rank {rank}] Stage 2 finished.")

    dist.barrier()

    if rank == 0:
        total_time = time.time() - global_start_time

    local_time_tensor = torch.tensor([shard_send_time], dtype=torch.float32, device=device)
    all_times_tensor = torch.zeros(world_size, dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(all_times_tensor, local_time_tensor)

    if rank == 0:
        sender_times = all_times_tensor.cpu().numpy()[:num_senders]
        gather_time_tensor = torch.zeros(1, device=device)
        dist.recv(tensor=gather_time_tensor, src=num_senders)
        final_gather_time = gather_time_tensor.cpu().item()

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

    elif rank == num_senders:
        gather_time_tensor = torch.tensor([gather_time], device=device)
        dist.send(tensor=gather_time_tensor, dst=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=float, default=7.0, help="Model size in billion parameters.")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    if world_size % 2 != 0:
        if rank == 0:
            print("Error: This script requires even number of ranks.")
        return

    device = torch.device(f"cuda:{local_rank}")
    model, total_size_mb = create_dummy_model(args.model_size, device)

    if rank == 0:
        print("\n" + "="*60)
        print(f"Transfer Scheme: Sharded P2P Send, followed by Intra-Group All-Gather")
        print(f"Model Size: {total_size_mb:.2f} MB")
        print("="*60 + "\n")

    receiver_ranks = list(range(world_size // 2, world_size))
    receiver_group = dist.new_group(ranks=receiver_ranks)
    shards, keys_in_shards = get_sharded_state_dict(model, world_size // 2)

    for i in range(3):
        run_transfer_test(model, total_size_mb, rank, world_size, local_rank, receiver_group, shards, keys_in_shards)

    dist.destroy_process_group()
    print(f"[{rank}] Cleaned up and finished.")

if __name__ == "__main__":
    main()
