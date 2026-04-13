import json
import os
import tempfile

from global_scheduler.drift_experiment import DriftExperimentRunner, make_job, parse_trace_kind, read_trace
from global_scheduler.generate_drift_traces import (
    TEMPLATE_NAMES,
    build_profiles,
    read_wild_trace_jobs,
)


def assert_equal(a, b):
    assert a == b, (a, b)


def assert_close(a, b, eps=1e-6):
    assert abs(a - b) <= eps, (a, b)


with tempfile.TemporaryDirectory() as tmpdir:
    trace_path = os.path.join(tmpdir, 'wild.trace')
    profile_path = os.path.join(tmpdir, 'profile.json')
    reference_dir = os.path.join(tmpdir, 'drift_reference')
    os.makedirs(reference_dir, exist_ok=True)

    with open(trace_path, 'w') as f:
        f.write('job0, 2025-01-01 00:00:00, 1, 32B_8k\n')
        f.write('job1, 2025-01-01 00:00:00, 1, 7B_8k\n')
        f.write('job2, 2025-01-01 00:00:00, 1, 3B_8k\n')
        f.write('job3, 2025-01-01 00:00:00, 1, 14B_4k\n')
        f.write('job0, 2025-01-01 00:20:00, -1, 32B_8k\n')
        f.write('job1, 2025-01-01 00:20:00, -1, 7B_8k\n')
        f.write('job2, 2025-01-01 00:20:00, -1, 3B_8k\n')
        f.write('job3, 2025-01-01 00:20:00, -1, 14B_4k\n')
    with open(profile_path, 'w') as f:
        json.dump({
            'disagg': {
                '32B_8k': {'generate': 100.0, 'train': 80.0},
                '7B_8k': {'generate': 40.0, 'train': 20.0},
                '3B_8k': {'generate': 20.0, 'train': 10.0},
                '14B_4k': {'generate': 60.0, 'train': 30.0},
            }
        }, f)

    # reference ratios end at 0.5 for easy assertions
    reference_rows = {
        '7b.csv': [100.0, 75.0, 50.0],
        '14b.csv': [120.0, 90.0, 60.0],
        '32b.csv': [200.0, 150.0, 100.0],
    }
    for filename, values in reference_rows.items():
        with open(os.path.join(reference_dir, filename), 'w') as f:
            for idx, value in enumerate(values):
                f.write(f'{idx},{value}\n')

    assert_equal(parse_trace_kind(trace_path), 'wild')
    trace_jobs = read_wild_trace_jobs(trace_path, profile_path, 'disagg')
    profiles_a = build_profiles(trace_jobs, 'mixed', 2345, 'wild', reference_dir=reference_dir)
    profiles_b = build_profiles(trace_jobs, 'mixed', 2345, 'wild', reference_dir=reference_dir)
    assert_equal(profiles_a['job_assignments'], profiles_b['job_assignments'])

    assert_equal(profiles_a['reference_mode'], 'real_trace_scaled')
    assert_equal(profiles_a['template_names'], TEMPLATE_NAMES)
    assert_equal(set(profiles_a['job_type_templates']['32B_8k'].keys()), set(TEMPLATE_NAMES))

    # mapping checks
    assert_equal(profiles_a['job_type_templates']['32B_8k']['no_drift']['reference_source'], '32b.csv')
    assert_equal(profiles_a['job_type_templates']['14B_4k']['no_drift']['reference_source'], '14b.csv')
    assert_equal(profiles_a['job_type_templates']['7B_8k']['no_drift']['reference_source'], '7b.csv')
    assert_equal(profiles_a['job_type_templates']['3B_8k']['no_drift']['reference_source'], '7b.csv')
    assert_close(profiles_a['job_type_templates']['3B_8k']['no_drift']['reference_amplitude_scale'], 0.5)
    assert_close(profiles_a['job_type_templates']['7B_8k']['no_drift']['reference_amplitude_scale'], 1.0)

    # no-drift should stay flat at base rollout
    assert_equal(profiles_a['job_type_templates']['7B_8k']['no_drift']['rollout_steps'], [40.0, 40.0, 40.0])

    # reference end ratio is 0.5; check scaled endpoints for 7B base=40
    seven_b_templates = profiles_a['job_type_templates']['7B_8k']
    assert_close(seven_b_templates['decreasing_1.0x']['rollout_steps'][-1], 20.0)
    assert_close(seven_b_templates['increasing_1.0x']['rollout_steps'][-1], 60.0)
    assert_close(seven_b_templates['decreasing_0.5x']['rollout_steps'][-1], 30.0)
    assert_close(seven_b_templates['increasing_1.5x']['rollout_steps'][-1], 70.0)

    # 3B uses 7B trend but amplitude halved: adjusted end ratio is 0.75, base=20
    three_b_templates = profiles_a['job_type_templates']['3B_8k']
    assert_close(three_b_templates['decreasing_1.0x']['rollout_steps'][-1], 15.0)
    assert_close(three_b_templates['increasing_1.0x']['rollout_steps'][-1], 25.0)

    # scenario pools
    increasing_profiles = build_profiles(trace_jobs, 'increasing', 2345, 'wild', reference_dir=reference_dir)
    decreasing_profiles = build_profiles(trace_jobs, 'decreasing', 2345, 'wild', reference_dir=reference_dir)
    for assignment in increasing_profiles['job_assignments'].values():
        assert assignment['template_name'] in {'no_drift', 'increasing_0.5x', 'increasing_1.0x', 'increasing_1.5x'}
    for assignment in decreasing_profiles['job_assignments'].values():
        assert assignment['template_name'] in {'no_drift', 'decreasing_0.5x', 'decreasing_1.0x', 'decreasing_1.5x'}

    for scenario in ['increasing', 'decreasing', 'mixed']:
        out_path = os.path.join(tmpdir, f'drift_{scenario}.json')
        with open(out_path, 'w') as f:
            json.dump(build_profiles(trace_jobs, scenario, 2345, 'wild', reference_dir=reference_dir), f)

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
