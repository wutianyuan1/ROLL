import json
import os
import tempfile

from global_scheduler.drift_experiment import DriftExperimentRunner, make_job, parse_trace_kind, read_trace
from global_scheduler.generate_drift_traces import build_profiles, read_wild_trace_jobs


def assert_equal(a, b):
    assert a == b, (a, b)


with tempfile.TemporaryDirectory() as tmpdir:
    trace_path = os.path.join(tmpdir, 'wild.trace')
    profile_path = os.path.join(tmpdir, 'profile.json')
    with open(trace_path, 'w') as f:
        f.write('job0, 2025-01-01 00:00:00, 1, 32B_8k\n')
        f.write('job1, 2025-01-01 00:00:00, 1, 7B_8k\n')
        f.write('job0, 2025-01-01 00:20:00, -1, 32B_8k\n')
        f.write('job1, 2025-01-01 00:20:00, -1, 7B_8k\n')
    with open(profile_path, 'w') as f:
        json.dump({
            'disagg': {
                '32B_8k': {'generate': 100.0, 'train': 80.0},
                '7B_8k': {'generate': 40.0, 'train': 20.0},
            }
        }, f)

    assert_equal(parse_trace_kind(trace_path), 'wild')
    trace_jobs = read_wild_trace_jobs(trace_path, profile_path, 'disagg')
    profiles_a = build_profiles(trace_jobs, 'mixed', [0.25, 0.5, 0.75], 20, 3, 0.25, 2345, 'wild')
    profiles_b = build_profiles(trace_jobs, 'mixed', [0.25, 0.5, 0.75], 20, 3, 0.25, 2345, 'wild')
    assert_equal(profiles_a['job_assignments'], profiles_b['job_assignments'])
    assert set(profiles_a['job_type_templates']['32B_8k']['increasing'].keys()) == {'0.25', '0.5', '0.75'}

    for scenario in ['increasing', 'decreasing', 'mixed']:
        out_path = os.path.join(tmpdir, f'drift_{scenario}.json')
        with open(out_path, 'w') as f:
            json.dump(build_profiles(trace_jobs, scenario, [0.25, 0.5, 0.75], 20, 3, 0.25, 2345, 'wild'), f)

    records = read_trace(trace_path, profile_path, 'disagg', 1.5)
    assert records[0]['job_type'] == '32B_8k'
    assert records[0]['t_rollout'] == 100.0
    assert records[0]['t_train'] == 80.0

    job = make_job('job0', 100.0, 80.0, 1.5, job_type='32B_8k')
    assert_equal(job.rollout_width, 2)
    assert_equal(job.train_width, 2)

    runner = DriftExperimentRunner(
        trace_fn=trace_path,
        drift_profile_fn=os.path.join(tmpdir, 'drift_mixed.json'),
        max_group_size=2,
        regroup_interval_sec=300.0,
        planning_penalty_sec=0.0,
        regroup_pause_sec=0.0,
        exact_search_threshold=6,
        max_search_steps=100,
        profile_fn=profile_path,
        profile_location='disagg',
        default_slo=1.5,
    )
    results = runner.run_all()
    assert 'static_weave' in results and 'dynamic_regroup' in results
    assert results['static_weave']['completed_iterations'] >= 0
    assert results['dynamic_regroup']['completed_iterations'] >= 0

print('wild_drift_test.py passed')
