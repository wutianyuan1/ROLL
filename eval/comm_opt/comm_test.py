import os
import time
import argparse
import numpy as np

import torch
import torch.distributed as dist
from modelscope import AutoConfig, AutoModelForCausalLM

# Set the model name from ModelScope
# Using a smaller, well-known Qwen model for configuration purposes.
# The logic works for any model size.
MODEL_ID = "Qwen/Qwen2.5-7B"

def setup_distributed():
    """
    Initializes the distributed process group.
    torchrun will set the necessary environment variables.
    """
    # NCCL is the recommended backend for GPU-to-GPU communication
    dist.init_process_group(backend="nccl")
    
    # Get rank and world size from the environment
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Set the device for the current process
    torch.cuda.set_device(local_rank)
    
    print(
        f"Initialized process rank {rank} of {world_size} on device cuda:{local_rank}."
    )
    return rank, world_size, local_rank


def create_dummy_model(model_id, device):
    """
    Creates a model with the same architecture as the target model but with
    randomly initialized "dummy" weights to avoid slow downloads.
    
    Args:
        model_id (str): The model identifier from ModelScope/HuggingFace.
        device (torch.device): The device to place the model on.

    Returns:
        torch.nn.Module: The instantiated model with dummy weights.
    """
    print(f"[{dist.get_rank()}] Creating dummy model from config: {model_id}")
    # trust_remote_code is often necessary for custom model architectures like Qwen
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # from_config creates the model structure without loading pretrained weights
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # Move model to the assigned GPU
    model.to(device)
    
    # Get model size for reporting
    total_params = sum(p.numel() for p in model.parameters())
    total_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    return model, total_params, total_size_mb

def run_transfer_test(rank, world_size, local_rank):
    """
    The main worker function for each process.
    - Node 0 processes (rank 0-7) send the model.
    - Node 1 processes (rank 8-15) receive the model.
    """
    # Determine the role of this node
    num_gpus_per_node = world_size // 2
    node_rank = rank // num_gpus_per_node

    # Create the dummy model and place it on the correct GPU
    device = torch.device(f"cuda:{local_rank}")
    model, total_params, total_size_mb = create_dummy_model(MODEL_ID, device)
    
    # Only rank 0 prints model details to avoid clutter
    if rank == 0:
        print("\n" + "="*50)
        print(f"Model: {MODEL_ID}")
        print(f"Total Parameters: {total_params / 1e9:.2f} Billion")
        print(f"Total Size: {total_size_mb:.2f} MB")
        print("="*50 + "\n")

    # Determine the communication peer
    if node_rank == 0:
        # Sender on Node 0
        peer_rank = rank + num_gpus_per_node
        print(f"[Rank {rank}] Role: SENDER. Peer: Rank {peer_rank}")
    else:
        # Receiver on Node 1
        peer_rank = rank - num_gpus_per_node
        print(f"[Rank {rank}] Role: RECEIVER. Peer: Rank {peer_rank}")

    # Synchronize all processes before starting the transfer
    dist.barrier()

    elapsed_time = 0.0
    if node_rank == 0:  # Senders
        state_dict = model.state_dict()
        
        # Ensure all model setup operations on GPU are complete before timing
        torch.cuda.synchronize(device=device)
        
        start_time = time.time()

        # Send each tensor in the state_dict
        for key, tensor in state_dict.items():
            dist.send(tensor=tensor, dst=peer_rank)
        
        # Wait for all send operations to complete on the GPU
        torch.cuda.synchronize(device=device)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"[Rank {rank}] -> [Rank {peer_rank}] Transfer completed in {elapsed_time:.4f} seconds.")

    else:  # Receivers
        # Create a new state_dict to receive into. Tensors must have correct shape/dtype.
        state_dict = model.state_dict()
        
        # Receive each tensor
        for key, tensor in state_dict.items():
            dist.recv(tensor=tensor, src=peer_rank)
        
        print(f"[Rank {rank}] <- [Rank {peer_rank}] Successfully received all model tensors.")

    # --- Collect and Report Timings ---
    
    # All processes must participate in the collective operation
    local_time_tensor = torch.tensor([elapsed_time], dtype=torch.float32, device=device)
    
    # Create a tensor on each GPU to hold all results
    all_times_tensor = torch.zeros(world_size, dtype=torch.float32, device=device)

    # Gather times from all processes. all_gather is a blocking operation.
    dist.all_gather_into_tensor(all_times_tensor, local_time_tensor)
    
    # Synchronize to make sure the gathering is complete before printing
    dist.barrier()
    
    # Let rank 0 process and print the final results
    if rank == 0:
        # Move results to CPU for analysis
        all_times_cpu = all_times_tensor.cpu().numpy()
        
        # Times are from the senders (ranks 0 to num_gpus_per_node-1)
        sender_times = all_times_cpu[:num_gpus_per_node]
        
        # Calculate throughput
        # Effective throughput is size / time
        throughputs_gbps = [(total_size_mb / 1024 * 8) / t for t in sender_times if t > 0]

        print("\n" + "="*50)
        print("          TRANSFER SPEED RESULTS")
        print("="*50)
        print(f"Model Size per Transfer: {total_size_mb:.2f} MB")
        print("-" * 50)
        print(f"Max Time (Slowest Transfer): {np.max(sender_times):.4f} s")
        print(f"Min Time (Fastest Transfer): {np.min(sender_times):.4f} s")
        print(f"Mean Time:                   {np.mean(sender_times):.4f} s")
        print(f"Median Time:                 {np.median(sender_times):.4f} s")
        print("-" * 50)
        print(f"Mean Throughput:             {np.mean(throughputs_gbps):.2f} Gbps")
        print("="*50 + "\n")


def main():
    # This script assumes it's launched by `torchrun`
    # torchrun sets MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK
    
    # Verify we have the right setup
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) != 16:
        print("Error: This script is designed for WORLD_SIZE=16 (2 nodes x 8 GPUs).")
        print("Please launch with --nnodes=2 and --nproc_per_node=8.")
        return

    rank, world_size, local_rank = setup_distributed()
    run_transfer_test(rank, world_size, local_rank)
    
    # Clean up the distributed environment
    dist.destroy_process_group()
    print(f"[{rank}] Cleaned up and finished.")


if __name__ == "__main__":
    main()

#######
# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=33.98.65.84 --master_port=13456 comm_test.py
#######
