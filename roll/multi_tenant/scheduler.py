import os
import time
import redis
import subprocess
from collections import deque
from roll.multi_tenant.lock_utils import redis_lock

class Scheduler:
    def __init__(self):
        self.master_addr = os.environ.get("MASTER_ADDR", "localhost")
        self.scheduler_port = int(os.environ.get("SCHEDULER_PORT", 9969))
        cmd = f"redis-server --port {self.scheduler_port} --save \"\""
        self.redis_process = subprocess.Popen(cmd, shell=True)
        time.sleep(5)
        while True:
            try:
                self.shared_storage = redis.StrictRedis(
                    host=self.master_addr,
                    port=self.scheduler_port,
                    db=0,
                    decode_responses=True
                )
                break
            except:
                continue
        self.get_tenant_interval = 1
        self.check_job_interval = 0.1
        self.tenant_count, self.tenants = self.get_tenants()
        self.shared_storage.set("running_init_job", "empty")
        self.shared_storage.set("running_gen_job", "empty")
        self.shared_storage.set("running_train_job", "empty")

    def __del__(self):
        if self.redis_process:
            self.redis_process.terminate()

    def get_tenants(self):
        names = self.shared_storage.lrange("tenant_list", 0, -1)
        return len(names), names


    def schedule(self):
        '''
        Prioritize generation scheduling.
        '''
        step = 0
        ready_to_gen = deque()
        ready_to_train = deque()
        ready_to_init = deque()
        current_gen = None
        current_train = None
        current_init = None
        while True:
            # Check and accept new tenants
            if step % self.get_tenant_interval == 0:
                tenant_count, tenants = self.get_tenants()
                new_tenants = list(set(tenants) - set(self.tenants))
                new_count = tenant_count - self.tenant_count
                self.tenant_count, self.tenants = tenant_count, tenants
                ready_to_init.extend(new_tenants)
                # ready_to_gen.extend(new_tenants)
                if len(new_tenants) != 0:
                    print(f"Step {step}: new_count={new_count}, new_tenants={new_tenants}, ready_to_gen={ready_to_gen}, ready_to_train={ready_to_train}")
            if len(self.tenants) == 0:
                print("No tenants found, continues")
                time.sleep(1)
                continue
            # print(f"Step {step}: tenant_count={self.tenant_count}, tenants={self.tenants}")

            running_init_job = self.shared_storage.get("running_init_job")
            if running_init_job == 'empty':
                if current_init is not None:
                    print(f"Add {current_init} to gen queue")
                    ready_to_gen.append(current_init)
                    current_init = None
                if len(ready_to_init) != 0:
                    print(f"Step {step}: has {len(ready_to_init)} jobs to initialize, next init: {ready_to_init[0]}")
                    # Update the current initialize job
                    current_init = ready_to_init.popleft()
                    self.shared_storage.set("running_init_job", current_init)
                    print(f"Schedule new init: {current_init}")

            running_gen_job = self.shared_storage.get("running_gen_job")
            if running_gen_job == 'empty':
                # print(f"To schedule new gen, previous gen is {current_gen}")
                # The previous generation job is done, move it to the train queue
                if current_gen is not None:
                    print(f"Add {current_gen} to train queue")
                    ready_to_train.append(current_gen)
                    current_gen = None
                if len(ready_to_gen) != 0:
                    print(f"Step {step}: has {len(ready_to_gen)} jobs to generate, next gen: {ready_to_gen[0]}")
                    # Update the current generation job
                    current_gen = ready_to_gen.popleft()
                    self.shared_storage.set("running_gen_job", current_gen)
                    print(f"Schedule new generation: {current_gen}")

            running_train_job = self.shared_storage.get("running_train_job")
            if running_train_job == 'empty':
                # print(f"To schedule new train, previous train is {current_train}")
                # The previous training job is done, move it to the generation queue
                if current_train is not None:
                    print(f"Add {current_train} to generation queue")
                    ready_to_gen.append(current_train)
                    current_train = None
                if len(ready_to_train) != 0:
                    print(f"Step {step}: has {len(ready_to_train)} jobs to train, next train: {ready_to_train[0]}")
                    # Update the current generation job
                    current_train = ready_to_train.popleft()
                    self.shared_storage.set("running_train_job", current_train)
                    print(f"Schedule new training: {current_train}")
            time.sleep(self.check_job_interval)


if __name__ == '__main__':
    s = Scheduler()
    s.schedule()
