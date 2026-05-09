# Weave: Efficient Co-Scheduling for Disaggregated RL Post-Training

This is the repo for OSDI'26 paper "Weave: Efficient Co-Scheduling for Disaggregated RL Post-Training".

Weave is a cross-cluster scheduling framework for disaggregated RL post-training, 
it fills dependency bubbles inherent to one job by another concurrent job’s active phase.
The implementation is based on [ROLL](https://github.com/alibaba/ROLL).

## Getting Started
Prerequisites: redis-server, redis python bindings, and requirements in `requirements_torch260_vllm.txt`.
To run a minimal demo, you should launch the scheduler first (`eval/migration_ablation/sch.sh`):
```bash
NG=8 NT=16 GDA="NVIDIA H20" TDA="NVIDIA H800" MAPFN="examples/migration_ablation/mapping.txt" N_JOB=2 RATIO=0.875 python worker_scheduler/scheduler.py 
```
Here, `NG` is the total number of rollout GPUs, `NT` is the total number of train GPUs, `GDA` is the GPU device affinity of rollout, `TDA` is the GPU device affinity of train, `MAPFN` is an optional mapping file that enforces job mapping (if this is set, then the job is forced to run on the specified GPUs), `N_JOB` is the maximum number of jobs to run, `RATIO` is the migration threshold ratio.

Then to launch a demo job (Qwen2.5-7B), try
```bash
MODEL_SIZE=7B RATIO=0.875 ROLLOUT_BATCH_SIZE=128 N=1 GA_STEPS=16 HF_ENDPOINT=https://hf-mirror.com VLLM_USE_V1=0 PYTHONPATH=. python examples/start_rlvr_pipeline.py --config_path migration_ablation --config_name rlvr_config_mig
```

## Evaluation Scripts
See `eval/` for all experiment scripts
