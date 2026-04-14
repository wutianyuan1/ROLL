import argparse
import json
import math
import random
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from global_scheduler.brute_force_solver_new import BruteForceSolver
from global_scheduler.baselines import BaselineScheduler, RandomScheduler, MostIdleScheduler
from global_scheduler.new_simulator import WeaveSimulator
from global_scheduler.structs import Job, JobGroup
from global_scheduler.weave_scheduler import WeaveScheduler, per_time_cost


class DriftWeaveScheduler(WeaveScheduler):
    def record_num_nodes(self, timing):
        return

    def record_utils_after_add(self, timing, jg_id, utils):
        return

    def record_utils_after_remove(self, timing, jg_id, utils):
        return

    def remove_job(self, job_id: str, timing):
        return BaselineScheduler.remove_job(self, job_id)


def parse_trace_kind(trace_fn: str) -> str:
    with open(trace_fn, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return 'wild' if len(line.split(', ')) == 4 else 'parsed'
    raise ValueError(f'Empty trace: {trace_fn}')


def read_profile(profile_fn: str, profile_location: str) -> Dict[str, Dict[str, float]]:
    with open(profile_fn, 'r') as f:
        profile = json.load(f)
    return profile[profile_location]


def read_trace(trace_fn: str, profile_fn: Optional[str] = None,
               profile_location: str = 'disagg', default_slo: float = 1.5) -> List[Dict]:
    trace_kind = parse_trace_kind(trace_fn)
    profile = read_profile(profile_fn, profile_location) if trace_kind == 'wild' else None
    traces = []
    with open(trace_fn, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split(', ')
            if trace_kind == 'wild':
                jid, t, event, job_type = items
                event_time = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                event_type = int(event)
                record = {
                    'job_id': jid,
                    'time': event_time,
                    'event': event_type,
                    'job_type': job_type,
                }
                if event_type == 1:
                    record.update({
                        't_rollout': float(profile[job_type]['generate']),
                        't_train': float(profile[job_type]['train']),
                        'slo': default_slo,
                    })
                traces.append(record)
            else:
                if len(items) == 3:
                    jid, t, event = items
                    traces.append({
                        'job_id': jid,
                        'time': datetime.strptime(t, '%Y-%m-%d %H:%M:%S'),
                        'event': int(event),
                    })
                else:
                    jid, t, event, t_roll, t_train, slo = items
                    traces.append({
                        'job_id': jid,
                        'time': datetime.strptime(t, '%Y-%m-%d %H:%M:%S'),
                        'event': int(event),
                        't_rollout': float(t_roll),
                        't_train': float(t_train),
                        'slo': float(slo),
                    })
    return traces


def load_drift_profiles(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def resource_width(job_type: Optional[str]) -> int:
    return 2 if job_type and job_type.startswith('32B') else 1


def make_job(job_id: str, t_rollout: float, t_train: float, slo: float,
             rollout_step_trace: Optional[List[float]] = None,
             job_type: Optional[str] = None) -> Job:
    width = resource_width(job_type)
    return Job(
        job_id, t_rollout, t_train, slo,
        rollout_step_trace=rollout_step_trace,
        rollout_width=width,
        train_width=width,
    )


def estimate_job_for_solver(job: Job) -> Job:
    observed_rollout = job.completed_rollout_durations[-1] if job.completed_rollout_durations else job.t_rollout
    observed_train = job.train_time_for_rollout(observed_rollout)
    return Job(
        job.job_id,
        observed_rollout,
        observed_train,
        job.slo,
        rollout_width=job.rollout_width,
        train_width=job.train_width,
    )


def build_seed_partition(snapshot_jobs: List[Job], current_groups: Dict[str, JobGroup]) -> Tuple[Tuple[int, ...], ...]:
    jid_to_idx = {job.job_id: idx for idx, job in enumerate(snapshot_jobs)}
    partition = []
    for group in current_groups.values():
        members = [jid_to_idx[job.job_id] for job in group.jobs if job.job_id in jid_to_idx]
        if members:
            partition.append(tuple(sorted(members)))
    return tuple(sorted(partition))


def node_counts(job_groups: Dict[str, JobGroup]) -> Tuple[int, int]:
    rollout_nodes = sum(len(group.all_rollout_nodes) for group in job_groups.values())
    train_nodes = sum(len(group.all_train_nodes) for group in job_groups.values())
    return rollout_nodes, train_nodes


def total_busy_time(simulators: List[WeaveSimulator]) -> Tuple[float, float]:
    rollout_busy = 0.0
    train_busy = 0.0
    for sim in simulators:
        for jobs_data in sim.rollout_busy_times.values():
            for intervals in jobs_data.values():
                rollout_busy += sum(end - start for start, end in intervals if end > start)
        for jobs_data in sim.train_busy_times.values():
            for intervals in jobs_data.values():
                train_busy += sum(end - start for start, end in intervals if end > start)
    return rollout_busy, train_busy


def set_scheduler_groups_from_solution(scheduler: WeaveScheduler, solved_groups: List[JobGroup]):
    scheduler.job_groups = {}
    scheduler.group_costs = {}
    scheduler.group_invalid_jobs = {}
    scheduler.group_utils = {}
    scheduler.last_group_id = len(solved_groups) - 1
    for idx, solved_group in enumerate(solved_groups):
        cloned_jobs = []
        for job in solved_group.jobs:
            cloned_jobs.append(deepcopy(job))
        group = JobGroup(f'Group-{idx}', cloned_jobs)
        scheduler.job_groups[group.group_id] = group
        scheduler.group_costs[group.group_id] = (
            len(group.all_train_nodes) * scheduler.train_cost +
            len(group.all_rollout_nodes) * scheduler.rollout_cost
        )
        scheduler.group_invalid_jobs[group.group_id] = {}


def estimate_group_layout_cost(job_groups: Dict[str, JobGroup], rollout_cost: float, train_cost: float,
                               simulate_steps: int) -> float:
    total_cost = 0.0
    for group in job_groups.values():
        sim = WeaveSimulator(group.jobs)
        _, train_busy_times, _, total_time = sim.simulate_run(simulate_steps)
        cost = per_time_cost(
            group.jobs,
            len(group.all_rollout_nodes),
            train_busy_times,
            total_time,
            rollout_cost,
            train_cost,
        )
        if math.isinf(cost):
            return float("inf")
        total_cost += cost
    return total_cost


def next_external_event_time(events: List[Dict], event_idx: int) -> float:
    if event_idx >= len(events):
        return math.inf
    return events[event_idx]['relative_time']


class DriftExperimentRunner:
    def __init__(self, trace_fn: str, drift_profile_fn: str, max_group_size: int,
                 regroup_interval_sec: float, planning_penalty_sec: float,
                 regroup_pause_sec: float, exact_search_threshold: int,
                 max_search_steps: int, profile_fn: Optional[str] = None,
                 profile_location: str = 'disagg', default_slo: float = 1.5,
                 first_regroup_delay_sec: Optional[float] = None):
        self.trace = read_trace(trace_fn, profile_fn, profile_location, default_slo)
        self.trace_kind = parse_trace_kind(trace_fn)
        self.start_time = self.trace[0]['time']
        self.events = []
        for record in self.trace:
            event = deepcopy(record)
            event['relative_time'] = (record['time'] - self.start_time).total_seconds()
            self.events.append(event)
        self.drift_profiles = load_drift_profiles(drift_profile_fn).get('jobs', {})
        self.max_group_size = max_group_size
        self.regroup_interval_sec = regroup_interval_sec
        self.planning_penalty_sec = planning_penalty_sec
        self.regroup_pause_sec = regroup_pause_sec
        self.exact_search_threshold = exact_search_threshold
        self.max_search_steps = max_search_steps
        self.first_regroup_delay_sec = (
            min(600.0, regroup_interval_sec / 2.0)
            if first_regroup_delay_sec is None else first_regroup_delay_sec
        )
        self._trace_end_time = self.events[-1]['relative_time'] if self.events else 0.0
        self.random_seed = 2345

    def _report_progress(self, strategy_name: str, event_idx: int, current_time: float,
                         active_jobs: Dict[str, Job], metrics: Dict[str, float],
                         note: str = '') -> None:
        event_total = len(self.events)
        event_pct = 100.0 * event_idx / event_total if event_total else 100.0
        if self._trace_end_time > 0:
            sim_pct = min(100.0, 100.0 * current_time / self._trace_end_time)
        else:
            sim_pct = 100.0
        suffix = f", {note}" if note else ""
        print(
            f"[progress][{strategy_name}] "
            f"events={event_idx}/{event_total} ({event_pct:.1f}%), "
            f"sim_time={current_time:.1f}s ({sim_pct:.1f}%), "
            f"active_jobs={len(active_jobs)}, "
            f"regroup_checks={metrics['num_regroup_checks']}, "
            f"regroups={metrics['num_regroups']}{suffix}"
        )

    def _advance_regroup_time(self, current_time: float, next_regroup_time: float) -> float:
        if self.regroup_interval_sec <= 0:
            return math.inf
        if math.isinf(next_regroup_time):
            return math.inf
        if next_regroup_time >= current_time - 1e-9:
            return next_regroup_time
        steps = math.floor((current_time - next_regroup_time) / self.regroup_interval_sec) + 1
        return next_regroup_time + steps * self.regroup_interval_sec

    def _make_runtime_job(self, record: Dict) -> Job:
        profile = self.drift_profiles.get(record['job_id'], {})
        rollout_steps = profile.get('rollout_steps')
        return make_job(
            record['job_id'],
            record['t_rollout'],
            record['t_train'],
            record['slo'],
            rollout_step_trace=rollout_steps,
            job_type=record.get('job_type'),
        )

    def _make_scheduler_job(self, record: Dict) -> Job:
        return make_job(
            record['job_id'],
            record['t_rollout'],
            record['t_train'],
            record['slo'],
            job_type=record.get('job_type'),
        )

    def _cancel_and_archive(self, simulators: Dict[str, WeaveSimulator], archived_sims: List[WeaveSimulator]):
        for sim in simulators.values():
            sim.cancel_inflight_work(restart_incomplete=True)
            archived_sims.append(sim)
        simulators.clear()

    def _rebuild_simulators(self, scheduler: WeaveScheduler, active_jobs: Dict[str, Job],
                            current_time: float) -> Dict[str, WeaveSimulator]:
        simulators: Dict[str, WeaveSimulator] = {}
        for group_id, group in scheduler.job_groups.items():
            runtime_jobs = []
            for placed_job in group.jobs:
                runtime_job = active_jobs[placed_job.job_id]
                runtime_job.rollout_nodes = list(placed_job.rollout_nodes)
                runtime_job.train_nodes = list(placed_job.train_nodes)
                runtime_jobs.append(runtime_job)
            sim = WeaveSimulator(runtime_jobs)
            sim.bootstrap(start_time=current_time, reset_runtime=False, queue_from_state=True)
            simulators[group_id] = sim
        return simulators

    def _accumulate_interval(self, scheduler: WeaveScheduler, delta_t: float,
                             metrics: Dict[str, float]):
        if delta_t <= 0:
            return
        rollout_nodes, train_nodes = node_counts(scheduler.job_groups)
        metrics['total_cost'] += delta_t * sum(scheduler.group_costs.values())
        metrics['rollout_capacity_time'] += delta_t * rollout_nodes
        metrics['train_capacity_time'] += delta_t * train_nodes

    def _search_regroup_solution(self, active_jobs: Dict[str, Job]) -> List[JobGroup]:
        if not active_jobs:
            return [], float("inf")
        snapshot_jobs = [estimate_job_for_solver(job) for job in active_jobs.values()]
        current_partition = build_seed_partition(snapshot_jobs, self.current_scheduler_groups)
        solver = BruteForceSolver(snapshot_jobs, self.max_group_size, n_iters=20)
        best_cost, _, best_groups = solver.solve(
            max_search_steps=self.max_search_steps,
            force_enum_all=len(snapshot_jobs) <= self.exact_search_threshold,
            seed_partitions=[current_partition],
            verbose=False,
        )
        if best_groups is None:
            best_groups = []
            for idx, job in enumerate(snapshot_jobs):
                job.rollout_nodes = [str(i) for i in range(job.rollout_width)]
                job.train_nodes = [f'TN{i}' for i in range(job.train_width)]
                best_groups.append(JobGroup(f'fallback-{idx}', [job]))
            best_cost = float("inf")
        return best_groups, best_cost

    def _final_metrics(self, all_jobs: Dict[str, Job], simulators: List[WeaveSimulator],
                       metrics: Dict[str, float]) -> Dict:
        all_slowdowns = []
        total_iters = 0
        slo_violations = 0
        per_job_avg = {}
        for job in all_jobs.values():
            job_slowdowns = []
            for cycle_time, solo_time in zip(job.completed_cycle_durations, job.completed_solo_durations):
                if solo_time <= 0:
                    continue
                slowdown = cycle_time / solo_time
                job_slowdowns.append(slowdown)
                all_slowdowns.append(slowdown)
                total_iters += 1
                if slowdown > job.slo:
                    slo_violations += 1
            if job_slowdowns:
                per_job_avg[job.job_id] = sum(job_slowdowns) / len(job_slowdowns)

        rollout_busy, train_busy = total_busy_time(simulators)
        return {
            'total_cost': metrics['total_cost'],
            'average_slowdown': sum(all_slowdowns) / len(all_slowdowns) if all_slowdowns else 0.0,
            'average_job_slowdown': sum(per_job_avg.values()) / len(per_job_avg) if per_job_avg else 0.0,
            'slo_violation_ratio': slo_violations / total_iters if total_iters else 0.0,
            'rollout_utilization': rollout_busy / metrics['rollout_capacity_time'] if metrics['rollout_capacity_time'] else 0.0,
            'train_utilization': train_busy / metrics['train_capacity_time'] if metrics['train_capacity_time'] else 0.0,
            'num_regroups': metrics['num_regroups'],
            'num_regroup_checks': metrics['num_regroup_checks'],
            'planning_penalty_time': metrics['planning_penalty_time'],
            'planning_penalty_cost': metrics['planning_penalty_cost'],
            'regroup_pause_time': metrics['regroup_pause_time'],
            'regroup_pause_cost': metrics['regroup_pause_cost'],
            'completed_iterations': total_iters,
        }

    def _make_scheduler(self, strategy_name: str):
        if strategy_name in {'static_weave', 'dynamic_regroup'}:
            return DriftWeaveScheduler(per_time_cost, max_group_size=self.max_group_size)
        if strategy_name == 'random':
            return RandomScheduler(per_time_cost, max_group_size=self.max_group_size)
        if strategy_name == 'most_idle':
            return MostIdleScheduler(per_time_cost, max_group_size=self.max_group_size)
        raise ValueError(f'Unknown strategy: {strategy_name}')

    def run_strategy(self, strategy_name: str) -> Dict:
        dynamic_regroup = strategy_name == 'dynamic_regroup'
        scheduler = self._make_scheduler(strategy_name)
        np_random = None
        if strategy_name in {'random', 'most_idle'}:
            import numpy as np
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            np_random = np
        self.current_scheduler_groups = scheduler.job_groups
        active_jobs: Dict[str, Job] = {}
        all_jobs: Dict[str, Job] = {}
        simulators: Dict[str, WeaveSimulator] = {}
        archived_sims: List[WeaveSimulator] = []
        event_idx = 0
        current_time = 0.0
        next_regroup_time = self.first_regroup_delay_sec if dynamic_regroup else math.inf
        metrics = {
            'total_cost': 0.0,
            'rollout_capacity_time': 0.0,
            'train_capacity_time': 0.0,
            'num_regroups': 0,
            'num_regroup_checks': 0,
            'planning_penalty_time': 0.0,
            'planning_penalty_cost': 0.0,
            'regroup_pause_time': 0.0,
            'regroup_pause_cost': 0.0,
        }
        next_event_report_idx = 0
        next_sim_report_time = 0.0
        event_report_stride = max(1, len(self.events) // 20) if self.events else 1
        sim_report_stride = max(600.0, self.regroup_interval_sec / 4.0)

        self._report_progress(strategy_name, event_idx, current_time, active_jobs, metrics, note='started')

        while True:
            changed = False
            if event_idx < len(self.events) and self.events[event_idx]['relative_time'] <= current_time:
                self._cancel_and_archive(simulators, archived_sims)
            while event_idx < len(self.events) and self.events[event_idx]['relative_time'] <= current_time:
                record = self.events[event_idx]
                if record['event'] == 1:
                    runtime_job = self._make_runtime_job(record)
                    scheduler_job = self._make_scheduler_job(record)
                    active_jobs[record['job_id']] = runtime_job
                    all_jobs[record['job_id']] = runtime_job
                    if strategy_name in {'static_weave', 'dynamic_regroup'}:
                        scheduler.add_job(scheduler_job)
                    else:
                        scheduler.add_job(scheduler_job)
                else:
                    active_jobs.pop(record['job_id'], None)
                    if strategy_name in {'static_weave', 'dynamic_regroup'}:
                        scheduler.remove_job(record['job_id'], self.start_time)
                    else:
                        scheduler.remove_job(record['job_id'])
                changed = True
                event_idx += 1
            if changed:
                self.current_scheduler_groups = scheduler.job_groups
                simulators = self._rebuild_simulators(scheduler, active_jobs, current_time)
                if event_idx >= next_event_report_idx:
                    self._report_progress(strategy_name, event_idx, current_time, active_jobs, metrics, note='external events processed')
                    next_event_report_idx = event_idx + event_report_stride

            if dynamic_regroup and next_regroup_time < current_time - 1e-9:
                next_regroup_time = self._advance_regroup_time(current_time, next_regroup_time)

            if dynamic_regroup and active_jobs and current_time >= next_regroup_time - 1e-9:
                self._cancel_and_archive(simulators, archived_sims)
                metrics['num_regroup_checks'] += 1
                self._report_progress(strategy_name, event_idx, current_time, active_jobs, metrics, note='running regroup check')
                planning_cost = self.planning_penalty_sec * sum(scheduler.group_costs.values())
                self._accumulate_interval(scheduler, self.planning_penalty_sec, metrics)
                current_time += self.planning_penalty_sec
                metrics['planning_penalty_time'] += self.planning_penalty_sec
                metrics['planning_penalty_cost'] += planning_cost
                current_layout_cost = estimate_group_layout_cost(
                    scheduler.job_groups,
                    scheduler.rollout_cost,
                    scheduler.train_cost,
                    scheduler.simulate_steps,
                )
                regroup_groups, regroup_cost = self._search_regroup_solution(active_jobs)
                next_event_time = next_external_event_time(self.events, event_idx)
                stable_horizon = min(
                    self.regroup_interval_sec,
                    max(0.0, next_event_time - current_time),
                )
                regroup_pause_cost = self.regroup_pause_sec * sum(
                    len(group.all_train_nodes) * scheduler.train_cost +
                    len(group.all_rollout_nodes) * scheduler.rollout_cost
                    for group in regroup_groups
                )
                amortized_gain = (current_layout_cost - regroup_cost) * stable_horizon
                if regroup_cost < current_layout_cost and amortized_gain > regroup_pause_cost:
                    set_scheduler_groups_from_solution(scheduler, regroup_groups)
                    self.current_scheduler_groups = scheduler.job_groups
                    simulators = self._rebuild_simulators(scheduler, active_jobs, current_time)
                    self._accumulate_interval(scheduler, self.regroup_pause_sec, metrics)
                    current_time += self.regroup_pause_sec
                    metrics['regroup_pause_time'] += self.regroup_pause_sec
                    metrics['regroup_pause_cost'] += regroup_pause_cost
                    metrics['num_regroups'] += 1
                    for sim in simulators.values():
                        sim.apply_pause(self.regroup_pause_sec)
                    self._report_progress(strategy_name, event_idx, current_time, active_jobs, metrics, note='regroup applied')
                else:
                    simulators = self._rebuild_simulators(scheduler, active_jobs, current_time)
                    self._report_progress(strategy_name, event_idx, current_time, active_jobs, metrics, note='regroup skipped')
                next_regroup_time = self._advance_regroup_time(
                    current_time,
                    next_regroup_time + self.regroup_interval_sec,
                )
                continue

            next_external_time = self.events[event_idx]['relative_time'] if event_idx < len(self.events) else math.inf
            next_action_time = min(
                next_external_time,
                next_regroup_time if dynamic_regroup and active_jobs else math.inf,
            )
            if math.isinf(next_action_time):
                break

            self._accumulate_interval(scheduler, next_action_time - current_time, metrics)
            for sim in simulators.values():
                sim.run_until(target_time=next_action_time)
            current_time = next_action_time
            if current_time >= next_sim_report_time:
                self._report_progress(strategy_name, event_idx, current_time, active_jobs, metrics)
                next_sim_report_time = current_time + sim_report_stride

        archived_sims.extend(simulators.values())
        self._report_progress(strategy_name, event_idx, current_time, active_jobs, metrics, note='finished')
        return self._final_metrics(all_jobs, archived_sims, metrics)

    def run_all(self) -> Dict:
        return {
            'static_weave': self.run_strategy('static_weave'),
            'dynamic_regroup': self.run_strategy('dynamic_regroup'),
        }

    def run_selected(self, methods: List[str]) -> Dict:
        return {method: self.run_strategy(method) for method in methods}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run workload-drift experiments for the scheduler simulator.')
    parser.add_argument('--trace', required=True, help='Base trace file.')
    parser.add_argument('--drift-prefix', required=True, help='Prefix of generated drift JSON files.')
    parser.add_argument('--output', required=True, help='Output JSON summary path.')
    parser.add_argument('--profile', help='Profile JSON for wild.trace inputs.')
    parser.add_argument('--profile-location', default='disagg')
    parser.add_argument('--default-slo', type=float, default=1.5)
    parser.add_argument('--max-group-size', type=int, default=3)
    parser.add_argument(
        '--regroup-interval-sec',
        type=float,
        default=3600.0,
        help='Global wallclock interval between regroup checks in simulated seconds.',
    )
    parser.add_argument(
        '--first-regroup-delay-sec',
        type=float,
        help='Initial global wallclock offset for the first regroup check; defaults to min(600, interval/2).',
    )
    parser.add_argument('--planning-penalty-sec', type=float, default=120.0)
    parser.add_argument('--regroup-pause-sec', type=float, default=120.0)
    parser.add_argument('--exact-search-threshold', type=int, default=8)
    parser.add_argument('--max-search-steps', type=int, default=10000)
    parser.add_argument(
        '--methods',
        default='static_weave,dynamic_regroup',
        help='Comma-separated methods from {static_weave,dynamic_regroup,random,most_idle}.',
    )
    parser.add_argument(
        '--scenarios',
        default='increasing,decreasing,mixed',
        help='Comma-separated scenarios to run from {increasing,decreasing,mixed}.',
    )
    args = parser.parse_args()

    scenarios = [item.strip() for item in args.scenarios.split(',') if item.strip()]
    allowed = {'increasing', 'decreasing', 'mixed'}
    invalid = [item for item in scenarios if item not in allowed]
    if invalid:
        raise ValueError(f'Unsupported scenarios: {invalid}')
    methods = [item.strip() for item in args.methods.split(',') if item.strip()]
    allowed_methods = {'static_weave', 'dynamic_regroup', 'random', 'most_idle'}
    invalid_methods = [item for item in methods if item not in allowed_methods]
    if invalid_methods:
        raise ValueError(f'Unsupported methods: {invalid_methods}')

    results = {}
    for scenario in scenarios:
        runner = DriftExperimentRunner(
            trace_fn=args.trace,
            drift_profile_fn=f'{args.drift_prefix}_{scenario}.json',
            max_group_size=args.max_group_size,
            regroup_interval_sec=args.regroup_interval_sec,
            planning_penalty_sec=args.planning_penalty_sec,
            regroup_pause_sec=args.regroup_pause_sec,
            exact_search_threshold=args.exact_search_threshold,
            max_search_steps=args.max_search_steps,
            profile_fn=args.profile,
            profile_location=args.profile_location,
            default_slo=args.default_slo,
            first_regroup_delay_sec=args.first_regroup_delay_sec,
        )
        results[scenario] = runner.run_selected(methods)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
