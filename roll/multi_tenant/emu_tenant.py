import time
import redis
import os
from sys import argv
from roll.multi_tenant.lock_utils import redis_lock


class SimTenant:
    def __init__(self, name, global_steps, gen_time, train_time):
        self.name = name
        self.global_steps = global_steps
        self.gen_time = gen_time
        self.train_time = train_time
        self.master_addr = os.environ.get("MASTER_ADDR", "localhost")
        self.scheduler_port = int(os.environ.get("SCHEDULER_PORT", 9969))
        self.report_interval = 0.2
        self.check_interval = 0.2
        self.shared_storage = redis.StrictRedis(
            host=self.master_addr,
            port=self.scheduler_port,
            db=0,
            decode_responses=True
        )
        self.register()

    def register(self):
        with redis_lock(self.shared_storage, "tenant_list"):
            self.shared_storage.lpush("tenant_list", self.name)

    def gen(self, step):
        current_running = self.shared_storage.get("running_gen_job")
        while current_running != self.name:
            print(f"### Cur is {current_running} != {self.name}")
            time.sleep(self.check_interval)
            current_running = self.shared_storage.get("running_gen_job")
        n = int(self.gen_time // self.report_interval)
        for i in range(n):
            print(f">>> gen step {step}: {i}")
            time.sleep(self.report_interval)
        self.shared_storage.set("running_gen_job", "empty")

    def train(self, step):
        current_running = self.shared_storage.get("running_train_job")
        while current_running != self.name:
            print(f"@@@ Cur is {current_running} != {self.name}")
            time.sleep(self.check_interval)
            current_running = self.shared_storage.get("running_train_job")
        n = int(self.train_time // self.report_interval)
        for i in range(n):
            print(f"$$$ train step {step}: {i}")
            time.sleep(self.report_interval)
        self.shared_storage.set("running_train_job", "empty")

    def run(self):
        for i in range(self.global_steps):
            self.gen(i)
            self.train(i)


if __name__ == '__main__':
    t = SimTenant(argv[1], 1000000, int(argv[2]), int(argv[3]))
    t.run()
