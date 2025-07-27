import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from math import ceil


class LengthDistDataset(object):
    def __init__(self, path: str, num_steps: int):
        with open(path, 'r') as f:
            content  = f.read().split("\n")
        self.lengths = []
        for line in content:
            if len(line) <= 1:
                continue
            items = [int(i) for i in line.split(" ")[1:]]
            self.lengths += items
        assert len(self.lengths) % num_steps == 0
        per_step_samples = len(self.lengths) // num_steps
        self.lengths = np.array(self.lengths).reshape((num_steps, per_step_samples))

    def __getitem__(self, i) -> np.ndarray:
        assert i < len(self.lengths)
        np.random.shuffle(self.lengths[i])
        return self.lengths[i]

    def flatten(self) -> np.ndarray:
        return self.lengths.reshape(-1)


def sim_migrate(req_lens: np.ndarray,
                L: int,  N: int, N_dest: int, N_src: int, B: int,
                plot: bool=True):
    '''
    L: migration timing,
    N: total #GPUs, N_src/dest: #src/dest GPUs,
    B: max batchsize per GPU.
    '''
    R = len(req_lens)  # total number of requests
    L_max = np.max(req_lens)  # max output length
    assert N_dest + N_src == N
    assert R == N * B
    assert 1 <= L <= L_max
    finished_reqs = req_lens[np.where(req_lens <= L)]
    unfinished_reqs = req_lens[np.where(req_lens > L)]
    print(L, len(finished_reqs), len(unfinished_reqs), N_dest, B, B*(L_max-L)*(N - N_dest))
    if len(unfinished_reqs) > N_dest * B:
        print("Too many requests, cannot fit to destination GPUs")
        return -1
    if plot:
        plt.barh(np.arange(R), np.clip(req_lens, 0, L), label='initial')
        plt.barh(np.arange(len(unfinished_reqs)), unfinished_reqs - L, left=L, label='migrated')
        plt.hlines(np.arange(N) * B, 0, L_max, linestyles='-.', color='black')
        plt.yticks(np.arange(N) * B + B//2, [f"GPU-{i}" for i in np.arange(N)])
        plt.legend()
        plt.savefig("figures/offline_req.png")

def offline_solver(req_lens: np.ndarray,
                L_max: int, N: int, B: int):
    '''
    N: total #GPUs, N_src/dest: #src/dest GPUs,
    B: max batchsize per GPU.
    '''
    req_lens.sort()
    # L_max = req_lens[-1]
    req_lens = req_lens.reshape((N,B))
    migration = [0] * 3 # [L(migration_length), N_m(migrated_GPU), (L_max - L)*N_m]
    migration[0] = L_max
    for i in range(N):
        if (i + 1) * B * (L_max - req_lens[i][-1]) > migration[2]:
            migration[0] = req_lens[i][-1]
            migration[1] = i + 1
            migration[2] = (i + 1) * B * (L_max - req_lens[i][-1])
    return migration


def online_solver(cur_req_lens: np.ndarray, pre_req_lens: np.ndarray,
                  optimal_N_out_prev: int, optimal_len_prev: int, L_max: int,
                  N: int, B: int, N_interval: Optional[int] = None,
                  L_interval: Optional[int] = None) -> Tuple[bool, int, float]:
    '''Returns (bool): whether to migrate at current timestamp'''
    # create confidence interval of optimal number of migrated_out gpus. -> (N_lo, N_hi)
    N_lo = optimal_N_out_prev - N_interval if N_interval is not None else 0
    N_hi = optimal_N_out_prev + N_interval if N_interval is not None else N
    # create confidence interval(???) of optimal len of migrated_out gpus. -> (L_lo, L_hi)
    L_lo = optimal_len_prev - L_interval if L_interval is not None else 0
    L_hi = optimal_len_prev + L_interval if L_interval is not None else L_max

    # If current time (i.e., max length) not in [L_lo, L_hi], ignore check
    cur_max_length = np.max(cur_req_lens)
    if cur_max_length < L_lo or cur_max_length > L_hi:
        return False, -1, -1


    # Count the number of unfinished requests in current step (which have max length)    
    num_cur_unfinished = len(cur_req_lens[np.where(cur_req_lens == cur_max_length)])
    # The unfinished requests will occupy at least ceil(num_cur_unfinished / B) GPUs
    # Therefore, the number of GPUs that can be released currently is as follows.
    cur_max_N_out = N - ceil(num_cur_unfinished / B)

    # If current migrated out #GPUs not in [N_lo, N_hi], ignore check
    if cur_max_N_out < N_lo or cur_max_N_out > N_hi:
        return False, -1, -1
    

    # print("=== check:", cur_max_length, optimal_len_prev, cur_max_N_out, optimal_N_out_prev)
    pre_req_lens.sort()
    L_prev_at_cur_max_N_out = pre_req_lens[cur_max_N_out * B]
    L_prev_at_cur_max_N_out_plus_one = pre_req_lens[(cur_max_N_out + 1) * B]
    # print(L_prev_at_cur_max_N_out, L_prev_at_cur_max_N_out_plus_one)
    # Calculate the expected timestamp of releasing one more GPU
    L_expected_release = (L_prev_at_cur_max_N_out_plus_one - L_prev_at_cur_max_N_out)\
        * (L_max - cur_max_length) / (L_max - L_prev_at_cur_max_N_out) + cur_max_length
    # print("expect release time:", L_expected_release)
    cur_profit = cur_max_N_out * (L_max - cur_max_length)
    expected_next_profit = (cur_max_N_out + 1) * (L_max - L_expected_release)
    # print(f"cur profit: {cur_profit}, next profit: {expected_next_profit}")
    if cur_max_N_out >= N_hi or cur_max_length >= L_hi:
        return True, cur_max_N_out, cur_profit * B
    return cur_profit >= expected_next_profit, cur_max_N_out, cur_profit * B


if __name__ == '__main__':
    ds = LengthDistDataset("/home/twubt/workspace/ROLL/experiments/lunxi_logs/16k/step5_bs64_n4/logs_without_state/unmig_r1/output_lens.log", 5)
    # ds = LengthDistDataset("output_lens_10_unmig.log", 10)

    N_total = 16
    B_per_gpu = 16
    L_max = 16500
    # L_max = 4200 
    step_num = 5

    offline_params, online_params = [], []
    mig_para = offline_solver(ds[0], L_max, N_total, B_per_gpu)
    offline_params.append(mig_para)
    # We just skip step 0 of online solver, so use some dummy value to fill it
    online_params.append([-1, -1, -1])
    for i in range(1, step_num):
        print("--> current situation of step", i)
        print(f"  optimal migration parameter of prev step ({i -1}) is {mig_para}")
        opt_N_out_prev = mig_para[1]
        opt_L_prev = mig_para[0]
        cur_req_lens = ds[i]
        dt = 1
        max_profit_obtained = -114514
        for t in range(1, np.max(cur_req_lens), dt):
            cur_req_lens_at_t = cur_req_lens.copy()
            cur_req_lens_at_t = cur_req_lens_at_t.clip(0, t)
            ret, cur_max_N_out, cur_profit = online_solver(
                cur_req_lens_at_t, ds[i - 1], opt_N_out_prev,
                opt_L_prev, L_max, N_total, B_per_gpu, 2, 500)
            if ret:
                break
            max_profit_obtained = max(max_profit_obtained, cur_profit)
            #TODO: check this condition, find a better metric to measure break timing
            if (max_profit_obtained - cur_profit) / max_profit_obtained > 0.1:
                break
        print(f"  online optimal t={t}, cur_max_N_out: {cur_max_N_out}, cur profit: {cur_profit}")
        online_params.append([t, cur_max_N_out, cur_profit])
        # Forward: calculate offline profit
        mig_para = offline_solver(ds[i], L_max, N_total, B_per_gpu)
        offline_params.append(mig_para)

    for i in range(1, step_num):
        profit_error = (offline_params[i][2] - online_params[i][2]) / offline_params[i][2]
        N_error = offline_params[i][1] - online_params[i][1]
        t_error = offline_params[i][0] - online_params[i][0]
        print(f"Step {i} error: profit_error={profit_error*100:.3f}%, N_error={N_error}, t_error={t_error}")
