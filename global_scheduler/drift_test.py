from global_scheduler.new_simulator import WeaveSimulator
from global_scheduler.structs import Job


def assert_close(a, b, eps=1e-6):
    assert abs(a - b) <= eps, (a, b)


# Static compatibility sanity check.
static_jobs = [
    Job('A', 10, 5, 1.2, ['R1'], ['TN']),
    Job('B', 8, 6, 1.2, ['R2'], ['TN']),
]
static_sim = WeaveSimulator(static_jobs)
_, _, utils, total_time = static_sim.simulate_run(3)
assert total_time > 0
assert 'R1' in utils['rollout'] and 'R2' in utils['rollout']

# Drift consumption and proportional train drift.
drift_job = Job('D', 10, 5, 1.2, ['R1'], ['TN'], rollout_step_trace=[10, 15, 20])
drift_sim = WeaveSimulator([drift_job])
drift_sim.simulate_run(3)
assert drift_job.completed_rollout_durations == [10, 15, 20]
assert_close(drift_job.completed_train_durations[0], 5.0)
assert_close(drift_job.completed_train_durations[1], 7.5)
assert_close(drift_job.completed_train_durations[2], 10.0)

# Tail handling repeats the last value.
repeat_job = Job('R', 10, 5, 1.2, ['R1'], ['TN'], rollout_step_trace=[12, 18])
repeat_sim = WeaveSimulator([repeat_job])
repeat_sim.simulate_run(4)
assert repeat_job.completed_rollout_durations == [12, 18, 18, 18]

# Resume correctness across a boundary with no regroup.
resume_jobs = [
    Job('X', 10, 5, 1.2, ['R1'], ['TN']),
    Job('Y', 9, 6, 1.2, ['R2'], ['TN']),
]
full_sim = WeaveSimulator([Job('X', 10, 5, 1.2, ['R1'], ['TN']), Job('Y', 9, 6, 1.2, ['R2'], ['TN'])])
_, _, _, full_total = full_sim.simulate_run(3)

resume_sim = WeaveSimulator(resume_jobs)
resume_sim.reset_state(0.0)
resume_sim.bootstrap(start_time=0.0, reset_runtime=True, queue_from_state=True)
resume_sim.run_until(target_time=20)
resume_sim.run_until(n_iters=3)
assert_close(resume_sim.current_time, full_total)

# Cancel/restart truncates in-flight work and preserves completed history.
cancel_job = Job('C', 10, 5, 1.2, ['R1'], ['TN'])
cancel_sim = WeaveSimulator([cancel_job])
cancel_sim.reset_state(0.0)
cancel_sim.bootstrap(start_time=0.0, reset_runtime=True, queue_from_state=True)
cancel_sim.run_until(target_time=3)
cancel_sim.cancel_inflight_work(restart_incomplete=True)
assert cancel_sim.current_time == 3
assert cancel_job.iterations_done == 0
assert cancel_job.next_phase == 'rollout'
assert cancel_job.completed_cycle_durations == []

print('drift_test.py passed')
