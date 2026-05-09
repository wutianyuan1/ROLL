# multi_load_mp.py
import time
import sys
from multiprocessing import Process, Barrier, current_process
from transformers import AutoModelForCausalLM

NUM_PROCS = 8  # 8 GPUs

def worker(barrier: Barrier, rank: int):
    # 等待所有进程就绪
    barrier.wait()
    # 开始加载模型
    device = f"cuda:{rank}"
    _ = AutoModelForCausalLM.from_pretrained(
        f"Qwen/Qwen2.5-{sys.argv[1]}B",
        device_map=device,
    )

def main():
    barrier = Barrier(NUM_PROCS)
    procs = []

    t_start = time.time()

    for rank in range(NUM_PROCS):
        p = Process(target=worker, args=(barrier, rank))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    t_end = time.time()
    print(f"{sys.argv[1]}B: Total load time for {NUM_PROCS} processes: {t_end - t_start:.2f} s")

if __name__ == "__main__":
    main()

