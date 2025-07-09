import threading
import time
import pprint
import matplotlib.pyplot as plt
import redis
import itertools
from collections import defaultdict
from typing import List

HYPER_PERIOD_SCHEDULE = [f'job{i}' for i in [1, 2, 3, 4, 2]]

class GPUResourceManager:
    """
    Global GPU resource manager, we set up a lock for each GPU.
    acquire_many is non-blocking, but release all on failures
    """
    def __init__(self, all_gpu_ids: List[int], hyper_period_schedule: List[str]):
        job_names = set(hyper_period_schedule)
        self._semaphores = {gid: {job_name: threading.Semaphore(0) for job_name in job_names} for gid in all_gpu_ids}
        # Let the first job in the hyper-period schedule acquires GPUs at the beginning.
        for gid in all_gpu_ids:
            self._semaphores[gid][hyper_period_schedule[0]].release()

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

    def acquire_many_blocked(self, job_name: str, gpu_ids: List[int]):
        """
        Acquire desired GPUs in a blocked manner.
        Each GPU is acquired in a blocked manner as well.
        """
        acquired: List[threading.Semaphore] = []
        for gid in sorted(gpu_ids):
            semaphores = self._semaphores.get(gid)
            if semaphores is None:
                raise ValueError(f"GPU {gid} does not exist")
            semaphore = semaphores.get(job_name)
            if semaphore is None:
                raise ValueError(f"GPU {gid} does not exist")
            semaphore.acquire(blocking=True)
            acquired.append(semaphore)
        return True

    def release_many(self, next_job: str, gpu_ids: List[int]):
        """
        Release GPUs to `next_job` in the hyper-period schedule.
        """
        for gid in sorted(gpu_ids):
            semaphores = self._semaphores.get(gid)
            if semaphores:
                semaphore = semaphores.get(next_job)
                if semaphore:
                    # This GPU has been acquired by the job invoking this method.
                    assert not semaphore.acquire(blocking=False)
                    semaphore.release()

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
                 shared_storage: redis.StrictRedis,
                 hyper_period_schedule: List[str]):
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

        # Record the next job candidates according to the hyper-period schedule.
        self.hyper_period_schedule = hyper_period_schedule
        self.next_job_candidates = []
        for i, j_name in enumerate(self.hyper_period_schedule):
            if j_name == self.job_name:
                self.next_job_candidates.append(self.hyper_period_schedule[(i + 1) % len(self.hyper_period_schedule)])
        # Different workers may be released to the next job at different times. So each worker has its own counter.
        self.gen_next_job_counters = {cfg['name']: itertools.count(0, 1) for cfg in self.gen_cfgs}
        # All training workers are released to the next job at the same time.
        self.train_next_job_counter = itertools.count(0, 1)

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

    def start(self, global_t0: float):
        """Run this job using a background thread"""
        # self._t0 = time.time()
        self._t0 = global_t0
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
        if not self.gpu_mgr.acquire_many_blocked(self.job_name, all_gpus):
            raise RuntimeError(f"{label} cannot acquire GPUs {all_gpus}")
        t0 = time.time()
        time.sleep(self.t_init)
        t1 = time.time()
        self._record_usage(all_gpus, t0, t1, label)
        # Find the next job in initialization schedule and release all GPUs to it.
        job_name_set = []
        for job_name in self.hyper_period_schedule:
            if job_name not in job_name_set:
                job_name_set.append(job_name)
        next_job = job_name_set[(job_name_set.index(self.job_name) + 1) % len(job_name_set)]
        self.gpu_mgr.release_many(next_job, all_gpus)

    def _run_generation(self, cfg, iteration):
        """
        An iteration of a Generation worker:
         - try acquire cfg['gen_gpu_ids']
         - sleep cfg['t_gen']
         - release resoure and record usage
        """
        # assert self.shared_storage.get(f"{self.job_name}_{cfg['name']}_status") != b"running", "Generation already running?"
        # self.set_key(f"{self.job_name}_{cfg['name']}_status", "pending")
        self.wait_key(f"{self.job_name}_{cfg['name']}_status", "running")
        label = f"{self.job_name}:{cfg['name']}:G{iteration}"
        gpus = cfg['gen_gpu_ids']
        if not self.gpu_mgr.acquire_many_blocked(self.job_name, gpus):
            raise RuntimeError(f"{label} cannot acquire GPUs {gpus}")
        t0 = time.time()
        time.sleep(cfg['t_gen'])
        t1 = time.time()
        self._record_usage(gpus, t0, t1, label)
        # Find the next job in hyper-period schedule and release this worker to it.
        next_job = self.next_job_candidates[next(self.gen_next_job_counters[cfg['name']]) % len(self.next_job_candidates)]
        self.gpu_mgr.release_many(next_job, gpus)
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
        if not self.gpu_mgr.acquire_many_blocked(self.job_name, gpus):
            raise RuntimeError(f"{label} cannot acquire GPUs {gpus}")
        t0 = time.time()
        time.sleep(self.t_train)
        t1 = time.time()
        self._record_usage(gpus, t0, t1, label)
        # Find the next job in hyper-period schedule and release this worker to it.
        next_job = self.next_job_candidates[next(self.train_next_job_counter) % len(self.next_job_candidates)]
        self.gpu_mgr.release_many(next_job, gpus)

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
                # assert self.shared_storage.get(f"{self.job_name}_train_status") != b"running", "Training already running?"
                # self.set_key(f"{self.job_name}_train_status", "pending")
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
            # ax.text(s + (e - s) / 3, y + height / 2, f"{worker}-{bar_text}", fontdict={"fontsize": 12})
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
    gpu_mgr = GPUResourceManager(ALL_GPUS, HYPER_PERIOD_SCHEDULE)
    fig, ax = plt.subplots(figsize=(15, 6))
    shared_storage = redis.StrictRedis(host='localhost', port=9969, db=0)
    shared_storage.flushdb()

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
                    shared_storage=shared_storage,
                    hyper_period_schedule=HYPER_PERIOD_SCHEDULE
                ),
        "job2": dict(job_name="job2",
                    gen_worker_configs=[
                        dict(name="actor_infer-0", gen_gpu_ids=[0], t_gen=1.0),
                        dict(name="actor_infer-1", gen_gpu_ids=[1], t_gen=1.0),
                        dict(name="actor_infer-2", gen_gpu_ids=[2], t_gen=1.0),
                        dict(name="actor_infer-3", gen_gpu_ids=[3], t_gen=1.5),
                    ],
                    train_gpu_ids=[4, 5, 6, 7],
                    t_init=1.0,
                    t_train=1.5,
                    iterations=6,
                    gpu_manager=gpu_mgr,
                    shared_storage=shared_storage,
                    hyper_period_schedule=HYPER_PERIOD_SCHEDULE
                ),
        "job3": dict(job_name="job3",
                    gen_worker_configs=[
                        dict(name="actor_infer-0", gen_gpu_ids=[0], t_gen=1.0),
                        dict(name="actor_infer-1", gen_gpu_ids=[1], t_gen=3.0),
                        dict(name="actor_infer-2", gen_gpu_ids=[2], t_gen=1.0),
                        dict(name="actor_infer-3", gen_gpu_ids=[3], t_gen=1.5),
                    ],
                    train_gpu_ids=[4, 5, 6, 7],
                    t_init=1.0,
                    t_train=1.5,
                    iterations=3,
                    gpu_manager=gpu_mgr,
                    shared_storage=shared_storage,
                    hyper_period_schedule=HYPER_PERIOD_SCHEDULE
                ),
        "job4": dict(job_name="job4",
                    gen_worker_configs=[
                        dict(name="actor_infer-0", gen_gpu_ids=[0], t_gen=3.0),
                        dict(name="actor_infer-1", gen_gpu_ids=[1], t_gen=1.0),
                        dict(name="actor_infer-2", gen_gpu_ids=[2], t_gen=1.5),
                        dict(name="actor_infer-3", gen_gpu_ids=[3], t_gen=1.0),
                    ],
                    train_gpu_ids=[4, 5, 6, 7],
                    t_init=1.0,
                    t_train=1.5,
                    iterations=3,
                    gpu_manager=gpu_mgr,
                    shared_storage=shared_storage,
                    hyper_period_schedule=HYPER_PERIOD_SCHEDULE
                ),
    }
    # TODO: Assumption: all jobs have the same hyper-period count.
    assert all([job_kwargs['iterations'] % HYPER_PERIOD_SCHEDULE.count(job_name) == 0 for job_name, job_kwargs in job_configs.items()])
    hyper_period_counts = [job_kwargs['iterations'] // HYPER_PERIOD_SCHEDULE.count(job_name) for job_name, job_kwargs in job_configs.items()]
    assert all([count == hyper_period_counts[0] for count in hyper_period_counts])
    jobs = {job_name: EmulatedJob(**job_kwargs) for job_name, job_kwargs in job_configs.items()}

    t0 = time.time()
    for job_name in jobs:
        jobs[job_name].start(t0)

    all_usages = {}
    for job_name in jobs:
        usage = jobs[job_name].join()
        # pprint.pprint(usage)
        for key in usage:
            if key in all_usages:
                all_usages[key] += usage[key]
            else:
                all_usages[key] = usage[key]
    # pprint.pprint(all_usages)
    for key in all_usages:
        all_usages[key].sort(key=lambda x: x[0])
    with open('usages.log', 'w') as f:
        f.write(pprint.pformat(all_usages))
    plot_gantt(all_usages, ax)
