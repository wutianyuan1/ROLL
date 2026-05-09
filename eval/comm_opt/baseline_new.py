import os
import time
import argparse
import numpy as np

import torch
import torch.distributed as dist


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


def create_dummy_model(model_size_billion: float, device):
    """
    Creates a dummy model with 256 equally-sized fp16 tensors.

    Args:
        model_size_billion (float): Model size in billions of parameters.
        device (torch.device): The device to place the model on.

    Returns:
        torch.nn.Module, int, float: Model, total params, total size in MB
    """
    total_params = int(model_size_billion * 1e9)
    num_tensors = 256
    params_per_tensor = total_params // num_tensors

    model_state_dict = {}
    for i in range(num_tensors):
        shape = (params_per_tensor,)  # 一维张量更易处理
        model_state_dict[f"param_{i}"] = torch.randn(shape, dtype=torch.float16, device=device)

    class DummyModel(torch.nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            self._state_dict = state_dict
        def state_dict(self):
            return self._state_dict

    model = DummyModel(model_state_dict)

    total_size_mb = total_params * 2 / (1024 ** 2)  # fp16 = 2 bytes per param
    return model, total_params, total_size_mb


def run_transfer_test(model, total_params, total_size_mb, rank, world_size, local_rank):
    """
    The main worker function for each process.
    - First half of ranks send the model.
    - Second half of ranks receive the model.
    """
    # Determine the role of this node
    num_senders = world_size // 2
    if rank < num_senders:
        # Sender
        peer_rank = rank + num_senders
        print(f"[Rank {rank}] Role: SENDER. Peer: Rank {peer_rank}")
    else:
        # Receiver
        peer_rank = rank - num_senders
        print(f"[Rank {rank}] Role: RECEIVER. Peer: Rank {peer_rank}")

    # Synchronize all processes before starting the transfer
    dist.barrier()

    elapsed_time = 0.0
    device = torch.device(f"cuda:{local_rank}")

    if rank < num_senders:  # Senders
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

        # Times are from the senders (ranks 0 to num_senders-1)
        sender_times = all_times_cpu[:num_senders]

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=float, default=7.0, help="Model size in billion parameters.")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    # Verify we have an even number of ranks
    if world_size % 2 != 0:
        if rank == 0:
            print("Error: This script requires an even number of ranks.")
        return

    device = torch.device(f"cuda:{local_rank}")
    model, total_params, total_size_mb = create_dummy_model(args.model_size, device)

    # Only rank 0 prints model details to avoid clutter
    if rank == 0:
        print("\n" + "="*50)
        print(f"Model Size: {args.model_size:.2f} Billion Parameters")
        print(f"Total Size: {total_size_mb:.2f} MB")
        print("="*50 + "\n")

    run_transfer_test(model, total_params, total_size_mb, rank, world_size, local_rank)

    # Clean up the distributed environment
    dist.destroy_process_group()
    print(f"[{rank}] Cleaned up and finished.")


if __name__ == "__main__":
    main()
