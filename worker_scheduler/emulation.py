import threading
import time
import pprint
import matplotlib.pyplot as plt
import redis
from collections import defaultdict


class GPUResourceManager:
    """
    Global GPU resource manager, we set up a lock for each GPU.
    acquire_many is non-blocking, but release all on failures
    """
    def __init__(self, all_gpu_ids):
        self._locks = {gid: threading.Lock() for gid in all_gpu_ids}

    def acquire_many(self, gpu_ids):
        acquired = []
        for gid in sorted(gpu_ids):
            lock = self._locks.get(gid)
            if lock is None:
                raise ValueError(f"GPU {gid} does not exist")
            if not lock.acquire(blocking=False):
                # failure, release all GPUs we have acquired.
                for l in acquired:
                    l.release()
                return False
            acquired.append(lock)
        return True

    def release_many(self, gpu_ids):
        for gid in sorted(gpu_ids):
            lock = self._locks.get(gid)
            if lock and lock.locked():
                lock.release()


class EmulatedJob:
    """
    A Job has many Generation worker with different t_gens, and a single
    Training worker.
    Each iteration within a job:
      1) Start all Generation workers in parallel, and wait them all done.
      2) Run Training worker
    We use a shared GPUResourceManager to make sure GPUs are exclusively
    acquired.
    Each running time slot (start, end, label) is recorded to self.gpu_usage.
    """
    def __init__(self,
                 job_name: str,
                 gen_worker_configs: list, # Each item in list is a dict containing: {name: str, gen_gpu_ids: List[int], t_gen: float (s)}
                 train_gpu_ids: list,
                 t_init: float,
                 t_train: float,
                 iterations: int,
                 gpu_manager: GPUResourceManager,
                 shared_storage: redis.StrictRedis):
        self.job_name = job_name
        self.gen_cfgs = gen_worker_configs
        self.train_gpu_ids = train_gpu_ids
        self.t_init = t_init
        self.t_train = t_train
        self.iterations = iterations
        self.gpu_mgr = gpu_manager
        self.shared_storage = shared_storage

        # usage: gpu_id -> List[(start, end, label)]
        self.gpu_usage = defaultdict(list)

        self._thread = None
        self._t0 = None  # record the start time

    def wait_key(self, key: str, expected: str):
        """Wait until a the value corresponds to key in redis contains `expected`"""
        while True:
            resp = self.shared_storage.get(key)
            if resp is not None:
                resp = resp.decode()
                if expected in resp:
                    break
        return resp

    def set_key(self, key: str, value: str):
        self.shared_storage.set(key, value)

    def start(self):
        """Run this job using a background thread"""
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, name=f"Job-{self.job_name}")
        self._thread.start()
        return self

    def join(self):
        """Wait for job complete and record gpu_usage"""
        if self._thread:
            self._thread.join()
        return dict(self.gpu_usage)

    def _record_usage(self, gpu_ids, t_start, t_end, label):
        """Record the relative time slot to self._t0"""
        rel_s = t_start - self._t0
        rel_e = t_end   - self._t0
        for gid in gpu_ids:
            self.gpu_usage[gid].append((rel_s, rel_e, label))

    def _run_initialization(self):
        """
        Job initialization phase
         - try acquire all GPUs of its train and gen
         - sleep self.t_init
         - release resoure and record usage
        """
        label = f"{self.job_name}:init:Init"
        all_gpus = [i for i in self.train_gpu_ids]
        for gen_config in self.gen_cfgs:
            all_gpus += gen_config['gen_gpu_ids']
        if not self.gpu_mgr.acquire_many(all_gpus):
            raise RuntimeError(f"{label} cannot acquire GPUs {all_gpus}")
        t0 = time.time()
        time.sleep(self.t_init)
        t1 = time.time()
        self._record_usage(all_gpus, t0, t1, label)
        self.gpu_mgr.release_many(all_gpus)

    def _run_generation(self, cfg, iteration):
        """
        An iteration of a Generation worker:
         - try acquire cfg['gen_gpu_ids']
         - sleep cfg['t_gen']
         - release resoure and record usage
        """
        self.set_key(f"{self.job_name}_{cfg['name']}_status", "pending")
        self.wait_key(f"{self.job_name}_{cfg['name']}_status", "running")
        label = f"{self.job_name}:{cfg['name']}:G{iteration}"
        gpus = cfg['gen_gpu_ids']
        if not self.gpu_mgr.acquire_many(gpus):
            raise RuntimeError(f"{label} cannot acquire GPUs {gpus}")
        t0 = time.time()
        time.sleep(cfg['t_gen'])
        t1 = time.time()
        self._record_usage(gpus, t0, t1, label)
        self.gpu_mgr.release_many(gpus)
        self.set_key(f"{self.job_name}_{cfg['name']}_status", "pending")

    def _run_training(self, iteration):
        """
        An iteration of a Training worker:
         - try acquire self.train_gpu_ids
         - sleep self.t_train
         - release resoure and record usage
        """
        label = f"{self.job_name}:train:T{iteration}"
        gpus = self.train_gpu_ids
        if not self.gpu_mgr.acquire_many(gpus):
            raise RuntimeError(f"{label} cannot acquire GPUs {gpus}")
        t0 = time.time()
        time.sleep(self.t_train)
        t1 = time.time()
        self._record_usage(gpus, t0, t1, label)
        self.gpu_mgr.release_many(gpus)

    def _run(self):
        try:
            # Initialization phase
            print(f"> Job {self.job_name} start!")
            self.set_key(f"{self.job_name}_status", "pending")
            self.shared_storage.publish("tenant_events", f"{self.job_name}:created")
            self.wait_key(f"{self.job_name}_status", "initializing")
            print(f"> Job {self.job_name} can init now!")
            self._run_initialization()
            self.set_key(f"{self.job_name}_status", "running")
            self.shared_storage.publish("tenant_events", f"{self.job_name}:init_done")
            print(f"> Job {self.job_name} init done!")
            # RL Loop
            for it in range(1, self.iterations + 1):
                # 1) Generation phase, run all gen workers in parallel
                threads = []
                for cfg in self.gen_cfgs:
                    t = threading.Thread(
                        target=self._run_generation,
                        args=(cfg, it),
                        name=f"{self.job_name}-{cfg['name']}-G{it}"
                    )
                    threads.append(t)
                    t.start()
                # Wait for all gen finish
                for t in threads:
                    t.join()

                # 2) Training phase, run training function
                self.shared_storage.publish("tenant_events", f"{self.job_name}:gen_done:{it}")
                print(f"> Job {self.job_name} generation done, wait for training!")
                self.set_key(f"{self.job_name}_train_status", "pending")
                self.wait_key(f"{self.job_name}_train_status", "running")
                print(f"> Job {self.job_name} can train now!")
                self._run_training(it)
                self.set_key(f"{self.job_name}_train_status", "pending")
                self.shared_storage.publish("tenant_events", f"{self.job_name}:train_done:{it}")
            self.set_key(f"{self.job_name}_status", "completed")
            self.shared_storage.publish("tenant_events", f"{self.job_name}:completed")
            print(f"Job {self.job_name} completes {self.iterations} iterations")
        except Exception as e:
            self.set_key(f"{self.job_name}_status", "crashed")
            self.shared_storage.publish("tenant_events", f"{self.job_name}:crashed")
            print(f"Job {self.job_name} crashed: {e}")


def plot_gantt(gpu_usage, ax):
    yticks, ylabels = [], []
    # Each line's height is 9, the distance between lines is 1 => 10 in total
    height = 9
    cmap = plt.get_cmap("tab20").colors
    color_map = {}
    ci = 0

    for idx, gpu in enumerate(sorted(gpu_usage)):
        intervals = gpu_usage[gpu]
        y = idx * (height + 1)
        yticks.append(y + height/2)
        ylabels.append(f"GPU {gpu}")

        for (s, e, label) in intervals:
            job, worker, bar_text = label.split(":")
            label = job
            if label not in color_map:
                color_map[label] = cmap[ci % len(cmap)]
                ci += 1
            ax.text(s + (e - s) / 3, y + height / 2, f"{worker}-{bar_text}", fontdict={"fontsize": 12})
            ax.broken_barh(
                [(s, e - s)],
                (y, height),
                facecolors=color_map[label],
                edgecolors="black",
                label=label if label not in ax.get_legend_handles_labels()[1] else ""
            )

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time")
    ax.set_ylabel("GPU ID")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
    plt.tight_layout()
    plt.savefig("gantt.png")


if __name__ == "__main__":
    # Global GPU resource pool
    ALL_GPUS = list(range(8))
    gpu_mgr = GPUResourceManager(ALL_GPUS)
    fig, ax = plt.subplots(figsize=(15, 6))
    shared_storage = redis.StrictRedis(host='localhost', port=9969, db=0)

    job_configs = {
        "job1": dict(job_name="job1",
                    gen_worker_configs=[
                        dict(name="actor_infer-0", gen_gpu_ids=[0], t_gen=0.5),
                        dict(name="actor_infer-1", gen_gpu_ids=[1], t_gen=0.5),
                        dict(name="actor_infer-2", gen_gpu_ids=[2], t_gen=3.0),
                        dict(name="actor_infer-3", gen_gpu_ids=[3], t_gen=0.5),
                    ],
                    train_gpu_ids=[4, 5, 6, 7],
                    t_init=1.0,
                    t_train=2.0,
                    iterations=3,
                    gpu_manager=gpu_mgr,
                    shared_storage=shared_storage
                ),
        # "job2": dict(job_name="job2",
        #             gen_worker_configs=[
        #                 dict(name="w0", gen_gpu_ids=[0], t_gen=1.0),
        #                 dict(name="w1", gen_gpu_ids=[1], t_gen=1.0),
        #                 dict(name="w2", gen_gpu_ids=[2], t_gen=1.0),
        #                 dict(name="w3", gen_gpu_ids=[3], t_gen=1.5),
        #             ],
        #             train_gpu_ids=[4, 5, 6, 7],
        #             t_init=1.0,
        #             t_train=1.5,
        #             iterations=3,
        #             gpu_manager=gpu_mgr,
        #             shared_storage=shared_storage
        #         ),
    }

    jobs = {job_name: EmulatedJob(**job_kwargs) for job_name, job_kwargs in job_configs.items()}

    for job_name in jobs:
        jobs[job_name].start()

    all_usages = {}
    for job_name in jobs:
        usage = jobs[job_name].join()
        pprint.pprint(usage)
        for key in usage:
            if key in all_usages:
                all_usages[key] += usage[key]
            else:
                all_usages[key] = usage[key]
    pprint.pprint(all_usages)
    plot_gantt(all_usages, ax)
