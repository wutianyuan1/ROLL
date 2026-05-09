import os
import time
import argparse
import numpy as np

import torch
import torch.distributed as dist
from modelscope import AutoConfig, AutoModelForCausalLM

# Using the same dummy model setup
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

def run_transfer_test(rank, world_size, local_rank):
    """
    Main worker function for the two-stage transfer.
    - Stage 1: Rank 0 sends to Rank 8.
    - Stage 2: Rank 8 broadcasts to Ranks 9-15.
    """
    if world_size % 2 != 0:
        if rank == 0:
            print("Error: This script requires an even number of processes for two nodes.")
        return
        
    num_gpus_per_node = world_size // 2
    device = torch.device(f"cuda:{local_rank}")
    
    # --- Setup Model and Process Groups ---
    model, total_size_mb = create_dummy_model(MODEL_ID, device)
    if rank == 0:
        print("\n" + "="*60)
        print(f"Transfer Scheme: P2P (0->8) followed by Intra-Node Broadcast (8->9..15)")
        print(f"Model Size per Transfer: {total_size_mb:.2f} MB")
        print("="*60 + "\n")

    # Create a process group for Node 1 (ranks 8 to 15)
    node1_ranks = list(range(num_gpus_per_node, world_size))
    # All processes must call new_group, even if they are not in it
    node1_group = dist.new_group(ranks=node1_ranks)

    # --- Timers and State Variables ---
    send_time = 0.0
    broadcast_time = 0.0
    total_time = 0.0
    
    # Synchronize all processes before starting the entire operation
    dist.barrier()
    
    if rank == 0:
        global_start_time = time.time()

    # --- Stage 1: Point-to-Point Transfer (Rank 0 to Rank 8) ---
    state_dict = model.state_dict()
    gateway_rank = num_gpus_per_node # This is rank 8

    if rank == 0:
        print(f"[Rank 0] Starting Stage 1: Sending model to gateway [Rank {gateway_rank}].")
        torch.cuda.synchronize(device=device)
        start_send_time = time.time()
        
        for tensor in state_dict.values():
            dist.send(tensor=tensor, dst=gateway_rank)
            
        torch.cuda.synchronize(device=device)
        send_time = time.time() - start_send_time
        print(f"[Rank 0] Stage 1 finished. Send time: {send_time:.4f} s.")

    elif rank == gateway_rank:
        print(f"[Rank {gateway_rank}] Starting Stage 1: Receiving model from [Rank 0].")
        for key, tensor in state_dict.items():
            # Receive into the existing tensor buffers
            dist.recv(tensor=tensor, src=0)
        print(f"[Rank {gateway_rank}] Stage 1 finished. Received all tensors.")

    # --- Stage 2: Intra-Node Broadcast on Node 1 ---

    # All processes in the group must participate in the broadcast call.
    # We use a barrier on the subgroup to ensure the receiver (rank 8) is ready
    # before the others (9-15) start listening for the broadcast.
    if rank in node1_ranks:
        # This barrier is crucial. It ensures rank 8 has finished receiving
        # before it starts broadcasting.
        dist.barrier(group=node1_group)

        if rank == gateway_rank:
            print(f"[Rank {gateway_rank}] Starting Stage 2: Broadcasting model to Node 1 peers.")
            torch.cuda.synchronize(device=device)
            start_broadcast_time = time.time()

        # The src of a broadcast within a group is its rank *within the global group*.
        broadcast_src_in_group = 8
        
        for tensor in state_dict.values():
            dist.broadcast(tensor=tensor, src=broadcast_src_in_group, group=node1_group)

        # Synchronize GPU to ensure broadcast operation is complete
        torch.cuda.synchronize(device=device)

        if rank == gateway_rank:
            broadcast_time = time.time() - start_broadcast_time
            print(f"[Rank {gateway_rank}] Stage 2 finished. Broadcast time: {broadcast_time:.4f} s.")
        else:
            print(f"[Rank {rank}] Stage 2 finished. Received model from broadcast.")

    # --- Collect and Report Timings ---

    # Global barrier to wait for everyone to finish before stopping the total timer
    dist.barrier()
    
    if rank == 0:
        total_time = time.time() - global_start_time
        
        # Rank 0 now needs to receive the broadcast_time from Rank 8
        broadcast_time_tensor = torch.zeros(1, device=device)
        dist.recv(tensor=broadcast_time_tensor, src=gateway_rank)
        
        # Final Report
        print("\n" + "="*60)
        print("          TWO-STAGE TRANSFER SPEED RESULTS")
        print("="*60)
        print(f"Model Size:              {total_size_mb:.2f} MB")
        print("-" * 60)
        print(f"Stage 1 (P2P Send Time): {send_time:.4f} s")
        print(f"Stage 2 (Bcast Time):    {broadcast_time_tensor.item():.4f} s")
        print("-" * 60)
        print(f"Total End-to-End Time:   {total_time:.4f} s")
        print("="*60 + "\n")
        
    elif rank == gateway_rank:
        # Send the measured broadcast time to rank 0 for reporting
        broadcast_time_tensor = torch.tensor([broadcast_time], device=device)
        dist.send(tensor=broadcast_time_tensor, dst=0)

def main():
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) != 16:
        print("Error: This script is designed for WORLD_SIZE=16 (2 nodes x 8 GPUs).")
        return
        
    rank, world_size, local_rank = setup_distributed()
    run_transfer_test(rank, world_size, local_rank)
    dist.destroy_process_group()
    print(f"[{rank}] Cleaned up and finished.")

if __name__ == "__main__":
    main()

#######
# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=33.98.65.84 --master_port=13456 comm_bcast.py
#######
