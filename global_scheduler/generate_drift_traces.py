import argparse
import json
import os
import random
from typing import Dict, List


def parse_trace_kind(trace_fn: str) -> str:
    with open(trace_fn, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split(', ')
            return 'wild' if len(items) == 4 else 'parsed'
    raise ValueError(f'Empty trace: {trace_fn}')


def read_profile(profile_fn: str, profile_location: str) -> Dict[str, Dict[str, float]]:
    with open(profile_fn, 'r') as f:
        profile = json.load(f)
    return profile[profile_location]


def read_parsed_trace_jobs(trace_fn: str) -> List[Dict]:
    jobs = []
    with open(trace_fn, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split(', ')
            if len(items) != 6:
                continue
            jid, _, event, t_roll, t_train, slo = items
            if int(event) != 1:
                continue
            jobs.append({
                'job_id': jid,
                't_rollout': float(t_roll),
                't_train': float(t_train),
                'slo': float(slo),
            })
    return jobs


def read_wild_trace_jobs(trace_fn: str, profile_fn: str, profile_location: str) -> List[Dict]:
    profile = read_profile(profile_fn, profile_location)
    jobs = []
    with open(trace_fn, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jid, _, event, job_type = line.split(', ')
            if int(event) != 1:
                continue
            jobs.append({
                'job_id': jid,
                'job_type': job_type,
                't_rollout': float(profile[job_type]['generate']),
                't_train': float(profile[job_type]['train']),
            })
    return jobs


def make_piecewise_steps(base_rollout: float, extent: float, pattern: str,
                         segment_len: int, num_segments: int, min_scale: float) -> List[float]:
    steps: List[float] = []
    for seg_idx in range(num_segments):
        if pattern == 'increasing':
            scale = 1.0 + extent * seg_idx
        elif pattern == 'decreasing':
            scale = max(min_scale, 1.0 - extent * seg_idx)
        else:
            raise ValueError(f'Unsupported pattern: {pattern}')
        steps.extend([base_rollout * scale] * segment_len)
    return steps


def build_job_type_templates(jobs: List[Dict], extents: List[float], segment_len: int,
                             num_segments: int, min_scale: float) -> Dict[str, Dict[str, Dict]]:
    templates: Dict[str, Dict[str, Dict]] = {}
    for job in jobs:
        job_type = job.get('job_type')
        if job_type is None or job_type in templates:
            continue
        templates[job_type] = {'increasing': {}, 'decreasing': {}}
        for extent in extents:
            extent_key = str(extent)
            templates[job_type]['increasing'][extent_key] = {
                'pattern': 'increasing',
                'extent': extent,
                'rollout_steps': make_piecewise_steps(
                    job['t_rollout'], extent, 'increasing', segment_len, num_segments, min_scale
                ),
            }
            templates[job_type]['decreasing'][extent_key] = {
                'pattern': 'decreasing',
                'extent': extent,
                'rollout_steps': make_piecewise_steps(
                    job['t_rollout'], extent, 'decreasing', segment_len, num_segments, min_scale
                ),
            }
    return templates


def sample_assignment(job: Dict, scenario: str, extents: List[float], rng: random.Random) -> Dict:
    extent = rng.choice(extents)
    if scenario == 'increasing':
        pattern = 'increasing'
    elif scenario == 'decreasing':
        pattern = 'decreasing'
    elif scenario == 'mixed':
        pattern = rng.choice(['increasing', 'decreasing'])
    else:
        raise ValueError(f'Unsupported scenario: {scenario}')
    return {'pattern': pattern, 'extent': extent, 'extent_key': str(extent)}


def build_profiles(jobs: List[Dict], scenario: str, extents: List[float], segment_len: int,
                   num_segments: int, min_scale: float, seed: int, trace_kind: str) -> Dict:
    rng = random.Random(seed)
    profiles = {
        'trace_kind': trace_kind,
        'scenario': scenario,
        'seed': seed,
        'k': len(extents),
        'extents': extents,
        'segment_len': segment_len,
        'num_segments': num_segments,
        'min_scale': min_scale,
        'jobs': {},
    }

    if trace_kind == 'wild':
        templates = build_job_type_templates(jobs, extents, segment_len, num_segments, min_scale)
        profiles['job_type_templates'] = templates
        profiles['job_assignments'] = {}
        for job in jobs:
            assignment = sample_assignment(job, scenario, extents, rng)
            profiles['job_assignments'][job['job_id']] = {
                'job_type': job['job_type'],
                **assignment,
            }
            template = templates[job['job_type']][assignment['pattern']][assignment['extent_key']]
            profiles['jobs'][job['job_id']] = {
                'job_type': job['job_type'],
                'pattern': assignment['pattern'],
                'extent': assignment['extent'],
                'segment_len': segment_len,
                'rollout_steps': template['rollout_steps'],
            }
    else:
        for job in jobs:
            assignment = sample_assignment(job, scenario, extents, rng)
            profiles['jobs'][job['job_id']] = {
                'pattern': assignment['pattern'],
                'extent': assignment['extent'],
                'segment_len': segment_len,
                'rollout_steps': make_piecewise_steps(
                    job['t_rollout'], assignment['extent'], assignment['pattern'],
                    segment_len, num_segments, min_scale,
                ),
            }
    return profiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate workload-drift traces for scheduler simulation.')
    parser.add_argument('--trace', required=True, help='Base trace file used for job arrivals.')
    parser.add_argument('--output-prefix', required=True, help='Output prefix for generated JSON sidecars.')
    parser.add_argument('--profile', help='Profile JSON for wild.trace inputs.')
    parser.add_argument('--profile-location', default='disagg', help='Profile location inside profile.json.')
    parser.add_argument('--k', type=int, default=3, help='Number of increasing/decreasing extents per job type.')
    parser.add_argument('--extents', default='0.25,0.5,0.75', help='Comma-separated drift extents.')
    parser.add_argument('--segment-len', type=int, default=20, help='Number of steps per piecewise-constant segment.')
    parser.add_argument('--num-segments', type=int, default=3, help='Number of drift segments per job.')
    parser.add_argument('--min-scale', type=float, default=0.25, help='Minimum scale for decreasing drift.')
    parser.add_argument('--seed', type=int, default=2345, help='Random seed for deterministic sampling.')
    parser.add_argument(
        '--scenarios',
        default='increasing,decreasing,mixed',
        help='Comma-separated scenarios to generate from {increasing,decreasing,mixed}.',
    )
    args = parser.parse_args()

    extents = [float(item) for item in args.extents.split(',') if item]
    if len(extents) != args.k:
        raise ValueError(f'Expected {args.k} extents, got {len(extents)}')

    trace_kind = parse_trace_kind(args.trace)
    if trace_kind == 'wild':
        if args.profile is None:
            raise ValueError('--profile is required for wild.trace inputs')
        trace_jobs = read_wild_trace_jobs(args.trace, args.profile, args.profile_location)
    else:
        trace_jobs = read_parsed_trace_jobs(args.trace)

    scenarios = [item.strip() for item in args.scenarios.split(',') if item.strip()]
    allowed = {'increasing', 'decreasing', 'mixed'}
    invalid = [item for item in scenarios if item not in allowed]
    if invalid:
        raise ValueError(f'Unsupported scenarios: {invalid}')

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for scenario in scenarios:
        profiles = build_profiles(
            jobs=trace_jobs,
            scenario=scenario,
            extents=extents,
            segment_len=args.segment_len,
            num_segments=args.num_segments,
            min_scale=args.min_scale,
            seed=args.seed,
            trace_kind=trace_kind,
        )
        output_path = f'{args.output_prefix}_{scenario}.json'
        with open(output_path, 'w') as f:
            json.dump(profiles, f, indent=2)
        print(f'Wrote {output_path}')
