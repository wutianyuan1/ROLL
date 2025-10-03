export VLLM_USE_V1=0
export PYTHONPATH=.
export SCHEDULER_ADDR=slogin-02
export SCHEDULER_PORT=9969
export MASTER_ADDR=dgx-22
export MASTER_PORT=6379
export USE_EXISTING_RAY_CLUSTER=1
python examples/start_rlvr_pipeline.py --config_path qwen2.5-7B-rlvr_megatron --config_name rlvr_disagg_2n8g