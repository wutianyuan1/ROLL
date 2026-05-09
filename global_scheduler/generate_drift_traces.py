import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional


TEMPLATE_NAMES = [
    'no_drift',
    'decreasing_0.5x',
    'decreasing_1.0x',
    'decreasing_1.5x',
    'increasing_0.5x',
    'increasing_1.0x',
    'increasing_1.5x',
]
SCENARIO_TEMPLATE_POOLS = {
    'increasing': ['no_drift', 'increasing_0.5x', 'increasing_1.0x', 'increasing_1.5x'],
    'decreasing': ['no_drift', 'decreasing_0.5x', 'decreasing_1.0x', 'decreasing_1.5x'],
    'mixed': TEMPLATE_NAMES,
}
TEMPLATE_CONFIGS = {
    'no_drift': ('no_drift', 0.0),
    'decreasing_0.5x': ('decreasing', 0.5),
    'decreasing_1.0x': ('decreasing', 1.0),
    'decreasing_1.5x': ('decreasing', 1.5),
    'increasing_0.5x': ('increasing', 0.5),
    'increasing_1.0x': ('increasing', 1.0),
    'increasing_1.5x': ('increasing', 1.5),
}
REFERENCE_FILENAMES = {
    '7b': '7b.csv',
    '14b': '14b.csv',
    '32b': '32b.csv',
}
DEFAULT_REFERENCE_DIR = '/root/workspace/weave/ROLL/global_scheduler/drift_reference'


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


def read_reference_curve(path: str) -> List[float]:
    values = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            values.append(float(row[1]))
    if not values:
        raise ValueError(f'Empty reference curve: {path}')
    return values


def normalize_reference_curve(values: List[float]) -> List[float]:
    first = values[0]
    if first == 0:
        raise ValueError('Reference curve cannot start at zero')
    return [value / first for value in values]


def load_reference_ratios(reference_dir: str) -> Dict[str, List[float]]:
    ratios = {}
    for key, filename in REFERENCE_FILENAMES.items():
        ratios[key] = normalize_reference_curve(
            read_reference_curve(str(Path(reference_dir) / filename))
        )
    return ratios


def infer_reference_key(job_type: Optional[str]) -> str:
    if not job_type:
        return '7b'
    prefix = job_type.split('_', 1)[0].upper()
    if prefix.startswith('32B'):
        return '32b'
    if prefix.startswith('14B'):
        return '14b'
    return '7b'


def reference_amplitude_scale(job_type: Optional[str]) -> float:
    if not job_type:
        return 1.0
    prefix = job_type.split('_', 1)[0].upper()
    if prefix.startswith('3B') or prefix.startswith('4B'):
        return 0.5
    return 1.0


def adjusted_reference_ratios(job_type: Optional[str], base_reference_ratios: Dict[str, List[float]]) -> tuple[str, float, List[float]]:
    ref_key = infer_reference_key(job_type)
    amp_scale = reference_amplitude_scale(job_type)
    base_ratios = base_reference_ratios[ref_key]
    adjusted = [1.0 + amp_scale * (ratio - 1.0) for ratio in base_ratios]
    return ref_key, amp_scale, adjusted


def build_template_rollout_steps(base_rollout: float, reference_ratios: List[float], template_name: str) -> List[float]:
    pattern, extent = TEMPLATE_CONFIGS[template_name]
    steps = []
    for ratio in reference_ratios:
        if pattern == 'no_drift':
            scale = 1.0
        elif pattern == 'decreasing':
            scale = 1.0 + extent * (ratio - 1.0)
        elif pattern == 'increasing':
            scale = 1.0 - extent * (ratio - 1.0)
        else:
            raise ValueError(f'Unsupported template pattern: {pattern}')
        steps.append(base_rollout * scale)
    return steps


def build_job_type_templates(jobs: List[Dict], reference_dir: str) -> Dict[str, Dict[str, Dict]]:
    base_reference_ratios = load_reference_ratios(reference_dir)
    templates: Dict[str, Dict[str, Dict]] = {}
    for job in jobs:
        job_type = job.get('job_type')
        if job_type is None or job_type in templates:
            continue
        ref_key, amp_scale, ratios = adjusted_reference_ratios(job_type, base_reference_ratios)
        templates[job_type] = {}
        for template_name in TEMPLATE_NAMES:
            pattern, extent = TEMPLATE_CONFIGS[template_name]
            templates[job_type][template_name] = {
                'pattern': pattern,
                'extent': extent,
                'reference_source': REFERENCE_FILENAMES[ref_key],
                'reference_amplitude_scale': amp_scale,
                'rollout_steps': build_template_rollout_steps(job['t_rollout'], ratios, template_name),
            }
    return templates


def sample_assignment(job: Dict, scenario: str, rng: random.Random) -> Dict:
    template_name = rng.choice(SCENARIO_TEMPLATE_POOLS[scenario])
    pattern, extent = TEMPLATE_CONFIGS[template_name]
    return {
        'pattern': pattern,
        'extent': extent,
        'template_name': template_name,
    }


def build_profiles(jobs: List[Dict], scenario: str, seed: int, trace_kind: str,
                   reference_dir: str = DEFAULT_REFERENCE_DIR) -> Dict:
    rng = random.Random(seed)
    base_reference_ratios = load_reference_ratios(reference_dir)
    profiles = {
        'trace_kind': trace_kind,
        'scenario': scenario,
        'seed': seed,
        'reference_mode': 'real_trace_scaled',
        'template_names': TEMPLATE_NAMES,
        'reference_sources': REFERENCE_FILENAMES,
        'reference_curve_length': len(next(iter(base_reference_ratios.values()))),
        'jobs': {},
    }

    if trace_kind == 'wild':
        templates = build_job_type_templates(jobs, reference_dir)
        profiles['job_type_templates'] = templates
        profiles['job_assignments'] = {}
        for job in jobs:
            assignment = sample_assignment(job, scenario, rng)
            template = templates[job['job_type']][assignment['template_name']]
            profiles['job_assignments'][job['job_id']] = {
                'job_type': job['job_type'],
                'template_name': assignment['template_name'],
                'pattern': assignment['pattern'],
                'extent': assignment['extent'],
            }
            profiles['jobs'][job['job_id']] = {
                'job_type': job['job_type'],
                'template_name': assignment['template_name'],
                'pattern': assignment['pattern'],
                'extent': assignment['extent'],
                'reference_source': template['reference_source'],
                'reference_amplitude_scale': template['reference_amplitude_scale'],
                'rollout_steps': template['rollout_steps'],
            }
    else:
        profiles['job_assignments'] = {}
        for job in jobs:
            assignment = sample_assignment(job, scenario, rng)
            ref_key, amp_scale, ratios = adjusted_reference_ratios(job.get('job_type'), base_reference_ratios)
            rollout_steps = build_template_rollout_steps(job['t_rollout'], ratios, assignment['template_name'])
            profiles['job_assignments'][job['job_id']] = {
                'template_name': assignment['template_name'],
                'pattern': assignment['pattern'],
                'extent': assignment['extent'],
            }
            profiles['jobs'][job['job_id']] = {
                'template_name': assignment['template_name'],
                'pattern': assignment['pattern'],
                'extent': assignment['extent'],
                'reference_source': REFERENCE_FILENAMES[ref_key],
                'reference_amplitude_scale': amp_scale,
                'rollout_steps': rollout_steps,
            }
    return profiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate workload-drift traces for scheduler simulation.')
    parser.add_argument('--trace', required=True, help='Base trace file used for job arrivals.')
    parser.add_argument('--output-prefix', required=True, help='Output prefix for generated JSON sidecars.')
    parser.add_argument('--profile', help='Profile JSON for wild.trace inputs.')
    parser.add_argument('--profile-location', default='disagg', help='Profile location inside profile.json.')
    parser.add_argument('--reference-dir', default=DEFAULT_REFERENCE_DIR, help='Directory that contains 7b/14b/32b reference CSV files.')
    parser.add_argument('--k', type=int, default=3, help='Ignored in reference mode; kept for CLI compatibility.')
    parser.add_argument('--extents', default='0.25,0.5,0.75', help='Ignored in reference mode; kept for CLI compatibility.')
    parser.add_argument('--segment-len', type=int, default=20, help='Ignored in reference mode; kept for CLI compatibility.')
    parser.add_argument('--num-segments', type=int, default=3, help='Ignored in reference mode; kept for CLI compatibility.')
    parser.add_argument('--min-scale', type=float, default=0.25, help='Ignored in reference mode; kept for CLI compatibility.')
    parser.add_argument('--seed', type=int, default=2345, help='Random seed for deterministic sampling.')
    parser.add_argument(
        '--scenarios',
        default='increasing,decreasing,mixed',
        help='Comma-separated scenarios to generate from {increasing,decreasing,mixed}.',
    )
    args = parser.parse_args()

    trace_kind = parse_trace_kind(args.trace)
    if trace_kind == 'wild':
        if args.profile is None:
            raise ValueError('--profile is required for wild.trace inputs')
        trace_jobs = read_wild_trace_jobs(args.trace, args.profile, args.profile_location)
    else:
        trace_jobs = read_parsed_trace_jobs(args.trace)

    scenarios = [item.strip() for item in args.scenarios.split(',') if item.strip()]
    allowed = set(SCENARIO_TEMPLATE_POOLS.keys())
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
            seed=args.seed,
            trace_kind=trace_kind,
            reference_dir=args.reference_dir,
        )
        output_path = f'{args.output_prefix}_{scenario}.json'
        with open(output_path, 'w') as f:
            json.dump(profiles, f, indent=2)
        print(f'Wrote {output_path}')
