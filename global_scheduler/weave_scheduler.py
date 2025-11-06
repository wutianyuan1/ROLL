from copy import deepcopy
from typing import List, Dict, Callable, Tuple, Optional
from datetime import datetime
from global_scheduler.structs import Job, JobGroup
from global_scheduler.new_simulator import WeaveSimulator
from global_scheduler.baselines import BaselineScheduler


def per_time_cost(jobs: List[Job], num_rollout_nodes: int, train_busy_times: Dict,
                  total_time: float, rollout_cost: float = 1/3, train_cost: float = 1.0,
                  return_invalid: bool = False):
    total_working_time = 0
    valid = True
    invalid_jobs = {}
    for job in jobs:
        num_iters = len(train_busy_times['TN'][job.job_id])
        if total_time / num_iters >= job.slo * (job.t_rollout + job.t_train):
            valid = False
            invalid_jobs[job.job_id] = (total_time / num_iters) / (job.t_rollout + job.t_train)
        total_working_time += num_iters * (job.t_rollout + job.t_train)
    # cost_per_time = total_time * (1 * train_cost + num_rollout_nodes * rollout_cost) / total_working_time
    cost_per_time = 1 * train_cost + num_rollout_nodes * rollout_cost
    if not return_invalid:
        return cost_per_time if valid else float("inf")
    else:
        if len(invalid_jobs) != 0:
            return cost_per_time * max(invalid_jobs.values()), invalid_jobs
        return cost_per_time * 1, invalid_jobs


def get_slowdowns(jobs: List[Job], total_time: float, train_busy_times: Dict):
    '''
    Calculate the slowdown of each involved job.
    '''
    jid_2_sld: Dict[str, float] = {}
    for job in jobs:
        num_iters = len(train_busy_times['TN'][job.job_id])
        jid_2_sld[job.job_id] = (total_time / num_iters) / (job.t_rollout + job.t_train)
    return jid_2_sld


class WeaveScheduler(BaselineScheduler):
    def __init__(self, cost_func: Callable[[Dict], float], max_group_size: int = 5,
                 simulate_steps: int = 100, rollout_cost: float = 1/3, train_cost: float = 1.0):
        super().__init__(cost_func, max_group_size, simulate_steps, rollout_cost, train_cost)
        # key: job_id, value: List[(slowdown, timing)]
        self.jid_2_sld_trace: Dict[str, List[Tuple[float, datetime]]] = {}
        # List[(# rollout (H20) nodes, # train (H800) nodes, timing)]
        self.num_nodes_trace: List[Tuple[int, int, datetime]] = []
        # List[('r': rollout util info., 't': train util. info, 'time': timing)]
        # r / t util. info = List[{'gid': str, 'u': float, 'n': int}]
        # 'gid' for job group id, 'u' for r / t util. of job group, 'n' for # r/t nodes of job group)]
        self.util_trace: List[Dict] = []

    def record_num_nodes(self, timing: datetime):
        '''
        Update self.num_nodes_trace.
        '''
        num_rollout_nodes = sum([len(jg.all_rollout_nodes) for jg in self.job_groups.values()])
        num_train_nodes = sum([len(jg.all_train_nodes) for jg in self.job_groups.values()])
        self.num_nodes_trace.append((num_rollout_nodes, num_train_nodes, timing))

    def record_utils_after_add(self, timing: datetime, jg_id: str, utils: Dict):
        '''
        Update self.util_trace.
        '''
        if len(self.util_trace) > 0:
            self.util_trace.append(deepcopy(self.util_trace[-1]))
        else:
            self.util_trace = [{'r': [], 't': [], 'time': None}]
        self.util_trace[-1]['time'] = timing
        # average util. across nodes
        rollout_util = sum(utils['rollout'].values()) / len(utils['rollout'].values())
        train_util = sum(utils['train']) / len(utils['train'])
        if jg_id not in [r_info['gid'] for r_info in self.util_trace[-1]['r']]:
            self.util_trace[-1]['r'].append({'gid': jg_id, 'u': rollout_util, 'n': len(utils['rollout'])})
            self.util_trace[-1]['t'].append({'gid': jg_id, 'u': train_util, 'n': len(utils['train'])})
        else:
            for i in range(len(self.util_trace[-1]['r'])):
                if jg_id == self.util_trace[-1]['r'][i]['gid']:
                    self.util_trace[-1]['r'][i] = {'gid': jg_id, 'u': rollout_util, 'n': len(utils['rollout'])}
                    self.util_trace[-1]['t'][i] = {'gid': jg_id, 'u': train_util, 'n': len(utils['train'])}
                    break

    def record_utils_after_remove(self, timing: datetime, jg_id: Optional[str], utils: Optional[Dict]):
        self.util_trace.append(deepcopy(self.util_trace[-1]))
        self.util_trace[-1]['time'] = timing
        if jg_id is not None:
            assert jg_id in [r_info['gid'] for r_info in self.util_trace[-1]['r']]
            if utils is not None:
                # average util. across nodes
                rollout_util = sum(utils['rollout'].values()) / len(utils['rollout'].values())
                train_util = sum(utils['train']) / len(utils['train'])
                for i in range(len(self.util_trace[-1]['r'])):
                    if jg_id == self.util_trace[-1]['r'][i]['gid']:
                        self.util_trace[-1]['r'][i] = {'gid': jg_id, 'u': rollout_util, 'n': len(utils['rollout'])}
                        self.util_trace[-1]['t'][i] = {'gid': jg_id, 'u': train_util, 'n': len(utils['train'])}
                        break
            else:
                for i in range(len(self.util_trace[-1]['r'])):
                    if jg_id == self.util_trace[-1]['r'][i]['gid']:
                        self.util_trace[-1]['r'].pop(i)
                        self.util_trace[-1]['t'].pop(i)
                        break

    def add_job(self, job: Job, timing: datetime, double_rollout_resource: bool):
        best_rollout_nodes, best_train_node, best_group, best_cost_delta, best_slowdowns, best_utils = None, None, None, float("inf"), None, None
        for job_group_name, job_group in self.job_groups.items():
            if len(job_group.jobs) >= self.max_group_size:
                continue
            all_rollout_nodes, all_train_nodes = job_group.all_rollout_nodes, job_group.all_train_nodes

            # Case-1: share train, share rollout (direct insert)
            for train_node in all_train_nodes:
                if double_rollout_resource and len(all_rollout_nodes) < 2:
                    break
                if not double_rollout_resource:
                    candidate_rollout_nodes = [[node] for node in all_rollout_nodes]
                else:
                    candidate_rollout_nodes = [[all_rollout_nodes[i], all_rollout_nodes[j]] \
                                               for i in range(len(all_rollout_nodes)) for j in range(i, len(all_rollout_nodes))]
                for rollout_nodes in candidate_rollout_nodes:
                    tmp_job = deepcopy(job)
                    tmp_job.rollout_nodes = rollout_nodes
                    tmp_job.train_nodes = [train_node]

                    jobs_in_group = job_group.jobs + [tmp_job]
                    sim = WeaveSimulator(jobs_in_group)
                    rollout_busy_times, train_busy_times, utils, total_time = sim.simulate_run(self.simulate_steps)
                    cost = self.cost_func(jobs_in_group, len(all_rollout_nodes), train_busy_times, total_time, self.rollout_cost, self.train_cost)
                    slowdowns = get_slowdowns(jobs_in_group, total_time, train_busy_times)
                    # print(f"Case-1, {rollout_node=}, {train_node=}, {cost=}")
                    if cost - self.group_costs[job_group_name] < best_cost_delta:
                        best_cost_delta = cost - self.group_costs[job_group_name]
                        best_rollout_nodes = rollout_nodes
                        best_train_node = train_node
                        best_group = job_group
                        best_slowdowns = slowdowns
                        best_utils = utils
            # Case-2: share train, but scale-up to an individual rollout
            for train_node in all_train_nodes:
                if not double_rollout_resource:
                    rollout_nodes = [job_group.next_rollout_node_id()]
                else:
                    rollout_nodes = [job_group.next_rollout_node_id(), job_group.next_rollout_node_id()]
                tmp_job = deepcopy(job)
                tmp_job.rollout_nodes = rollout_nodes
                tmp_job.train_nodes = [train_node]
                jobs_in_group = job_group.jobs + [tmp_job]
                sim = WeaveSimulator(jobs_in_group)
                rollout_busy_times, train_busy_times, utils, total_time = sim.simulate_run(self.simulate_steps)
                num_scale_up_rollout_nodes = 1 if not double_rollout_resource else 2
                cost = self.cost_func(jobs_in_group, len(all_rollout_nodes) + num_scale_up_rollout_nodes, train_busy_times, total_time, self.rollout_cost, self.train_cost)
                slowdowns = get_slowdowns(jobs_in_group, total_time, train_busy_times)
                # print(f"Case-2, {rollout_node=}, {train_node=}, {cost=}")
                if cost - self.group_costs[job_group_name] < best_cost_delta:
                    best_cost_delta = cost - self.group_costs[job_group_name]
                    best_rollout_nodes = rollout_nodes
                    best_train_node = train_node
                    best_group = job_group
                    best_slowdowns = slowdowns
                    best_utils = utils
        # Case-3: form a new group
        tmp_job = deepcopy(job)
        tmp_job.rollout_nodes = ["0"] if not double_rollout_resource else ["0", "1"]
        tmp_job.train_nodes = ["TN"]
        job_group = JobGroup(self.next_group_id(), [tmp_job])
        if double_rollout_resource:
            # Manually set last_rollout_node_id 
            job_group.last_rollout_node_id += 1
        sim = WeaveSimulator(job_group.jobs)
        rollout_busy_times, train_busy_times, utils, total_time = sim.simulate_run(self.simulate_steps)
        cost = self.cost_func(job_group.jobs, len(job_group.all_rollout_nodes), train_busy_times, total_time, self.rollout_cost, self.train_cost)
        slowdowns = get_slowdowns(job_group.jobs, total_time, train_busy_times)
        # print(f"Case-3, {cost=}")
        # If new group is the best
        if cost < best_cost_delta:
            best_cost_delta = cost
            best_rollout_nodes = job_group.all_rollout_nodes
            best_train_node = job_group.all_train_nodes[0]
            best_group = job_group
            best_slowdowns = slowdowns
            best_utils = utils
            self.job_groups[job_group.group_id] = job_group
            self.group_costs[job_group.group_id] = cost
        # If one existing group is the best, then assign this job to the group
        else:
            tmp_job = deepcopy(job)
            tmp_job.rollout_nodes = best_rollout_nodes
            tmp_job.train_nodes = [best_train_node]
            self.last_group_id -= 1  # new group is not the best, recall the added group ID
            self.job_groups[best_group.group_id].jobs.append(tmp_job)
            self.group_costs[best_group.group_id] += best_cost_delta
        # print(self.group_costs)
        # Update the slowdowns of relative jobs
        for jid, sld in best_slowdowns.items():
            self.jid_2_sld_trace.setdefault(jid, [])
            self.jid_2_sld_trace[jid].append((sld, timing))
        # Update the node count
        self.record_num_nodes(timing)
        # Update the utils
        self.record_utils_after_add(timing, best_group.group_id, best_utils)
        return best_rollout_nodes, best_train_node, best_group, best_cost_delta, {}

    def remove_job(self, job_id: str, timing: datetime):
        # To add the end timing for slowdown calculating.
        self.jid_2_sld_trace[job_id].append((None, timing))
        jg_id, utils = super().remove_job(job_id)
        # Update the node count
        self.record_num_nodes(timing)
        # Update the utils
        self.record_utils_after_remove(timing, jg_id, utils)

    def average_slowdown(self):
        '''
        slowdown_avg of a job = \sum \delta t_i * slowdown_i / (\sum \delta t_i)
        '''
        jid_2_avg_sld: Dict[str, float] = {}
        for jid, sld_trace in self.jid_2_sld_trace.items():
            assert sld_trace[-1][0] == None
            delta_ts: List[float] = []
            slds: List[float] = []
            for i in range(1, len(sld_trace)):
                delta_ts.append((sld_trace[i][1] - sld_trace[i - 1][1]).total_seconds())
                slds.append(sld_trace[i - 1][0])
            t = sum(delta_ts)
            assert t == (sld_trace[-1][1] - sld_trace[0][1]).total_seconds()
            jid_2_avg_sld[jid] = sum([delta_t * sld for delta_t, sld in zip(delta_ts, slds)]) / t
        return jid_2_avg_sld

    def average_utils(self):
        '''
        util_avg = \sum \delta t_i * util_i / (\sum \delta t_i)
        util_i = \sum #nodes_jg * util_jg / \sum #nodes_jg
        '''
        delta_ts: List[float] = []
        utils: List[Tuple[float, float]] = []
        for i in range(1, len(self.util_trace)):
            delta_ts.append((self.util_trace[i]['time'] - self.util_trace[i - 1]['time']).total_seconds())
            # average util. across job groups
            rollout_util = sum([r_info['u'] * r_info['n'] for r_info in self.util_trace[i - 1]['r']]) \
                / sum([r_info['n'] for r_info in self.util_trace[i - 1]['r']])
            train_util = sum([t_info['u'] * t_info['n'] for t_info in self.util_trace[i - 1]['t']]) \
                / sum([t_info['n'] for t_info in self.util_trace[i - 1]['t']])
            utils.append((rollout_util, train_util))
        t = sum(delta_ts)
        avg_r_util = sum([delta_t * r_util for delta_t, (r_util, _) in zip(delta_ts, utils)]) / t
        avg_t_util = sum([delta_t * t_util for delta_t, (_, t_util) in zip(delta_ts, utils)]) / t
        return {'rollout': avg_r_util, 'train': avg_t_util}


if __name__ == "__main__":
    sched = WeaveScheduler(per_time_cost, 3)
    for i in range(5):
        print("="*80)
        best_rollout_node, best_train_node, best_group, best_cost = sched.add_job(Job(f"{i}", 30, 10, 1.1))
        print(i, best_rollout_node, best_train_node, best_group, best_cost)
