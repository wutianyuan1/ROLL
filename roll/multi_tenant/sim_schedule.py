import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Callable
from sys import argv


class Tenant:
    def __init__(self, t_train, t_gen):
        self.name = "".join(random.choice("qazwsxedcrfvtgbyhnujmikolp") for _ in range(5))
        self.t_train = t_train
        self.t_gen = t_gen
        self.gen_start = 0
        self.train_start = 0

    def __hash__(self):
        return self.name


def sim_schedule(tenants: List[Tenant], n_iters: int, select_train: Callable, select_gen: Callable):
    ready_to_gen = []
    ready_to_train = []
    running_gen_job = [None, 0]
    running_train_job = [None, 0]
    current_gen: Tenant = None
    current_train: Tenant = None
    done_gen = {}
    done_train = {}
    ready_to_gen.extend(tenants)
    now = 0
    while True:
        if running_gen_job[0] is None:
            # The previous generation job is done, move it to the train queue
            if current_gen is not None:
                if (current_gen.name not in done_train) or (len(done_train[current_gen.name]) < n_iters):
                    ready_to_train.append(current_gen)
                current_gen = None
            if len(ready_to_gen) != 0:
                # Update the current generation job
                current_gen = select_gen(ready_to_gen)
                current_gen.gen_start = now
                running_gen_job = [current_gen, current_gen.t_gen]
        else:
            running_gen_job[1] -= 1
            if running_gen_job[1] == 0:
                if running_gen_job[0].name in done_gen:
                    done_gen[running_gen_job[0].name].append((running_gen_job[0].gen_start, now))
                else:
                    done_gen[running_gen_job[0].name] = [(running_gen_job[0].gen_start, now)]
                running_gen_job[0] = None

        if running_train_job[0] is None:
            # The previous training job is done, move it to the generation queue
            if current_train is not None:
                if (current_train.name not in done_gen) or (len(done_gen[current_train.name]) < n_iters):
                    ready_to_gen.append(current_train)
                current_train = None
            if len(ready_to_train) != 0:
                # Update the current generation job
                current_train = select_train(ready_to_train)
                current_train.train_start = now
                running_train_job = [current_train, current_train.t_train]
        else:
            running_train_job[1] -= 1
            if running_train_job[1] == 0:
                if running_train_job[0].name in done_train:
                    done_train[running_train_job[0].name].append((running_train_job[0].train_start, now))
                else:
                    done_train[running_train_job[0].name] = [(running_train_job[0].train_start, now)]
                running_train_job[0] = None
        now += 1
        min_train_count = min([len(v) if isinstance(v, list) else 0 for v in done_train.values()]) if len(done_train) != 0 else 0
        min_gen_count = min([len(v) if isinstance(v, list) else 0 for v in done_gen.values()]) if len(done_gen) != 0 else 0
        if min_train_count >= n_iters and min_gen_count >= n_iters:
            break
    return done_gen, done_train

def plot(tenants, done_gen, done_train):
    fig = plt.figure(figsize=(6, 2))
    ax = plt.subplot(111)
    clist = ['red', 'blue', 'green', 'orange']
    colors = {}
    maxt = 0
    for i in range(len(tenants)):
        colors[tenants[i].name] = clist[i]
    for k, v in done_gen.items():
        for item in v:
            rect = patches.Rectangle((item[0], 1), item[1] - item[0], 1, edgecolor='black', facecolor=colors[k])
            maxt = max(item[1], maxt)
            ax.add_patch(rect)
    for k, v in done_train.items():
        for item in v:
            rect = patches.Rectangle((item[0], 0), item[1] - item[0], 1, edgecolor='black', facecolor=colors[k])
            maxt = max(item[1], maxt)
            ax.add_patch(rect)
    ax.plot([0], [0])
    ax.set_yticks([0.5, 1.5], ['Train', 'Gen'])
    ax.set_title(f"total_time={maxt}")
    print(maxt)


def rand_tenant(gl, gh, tl, th):
    return Tenant(random.randint(tl, th), random.randint(gl, gh))


def rand_select(l):
    idx = random.randint(0, len(l) - 1)
    return l.pop(idx)


if __name__ == '__main__':
    random.seed(42)
    # ts = [Tenant(4, 7), Tenant(15, 6), Tenant(5, 17)]
    ts = [Tenant(60, 60) for _ in range(3)]
    left_select = lambda x: x.pop(0)
    random.seed(int(argv[1]))
    done_gen, done_train = sim_schedule(
        ts,
        10,
        left_select,
        left_select
        # rand_select,
        # rand_select
    )
    plot(ts, done_gen, done_train)
    plt.tight_layout()
    plt.savefig("sim_tenant.png")
