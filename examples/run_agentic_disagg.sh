# 设置 vLLM v0 模式和环境变量
export VLLM_USE_V1=0
export RAY_DEDUP_LOGS=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# 设置 Ray 相关环境变量以输出抢占信息
export RAY_DISABLE_DEDUP_WARNING=1
export RAY_DISABLE_IMPORT_WARNING=1

# 设置日志级别
export RAY_LOG_LEVEL=INFO

# 添加 CUDA 调试环境变量
export CUDA_LAUNCH_BLOCKING=1  # 同步 CUDA 调用，便于调试
export TORCH_USE_CUDA_DSA=1    # 启用设备端断言
export CUDA_DEVICE_MAX_CONNECTIONS=1  # 限制 CUDA 连接数

echo "=== vLLM v0 模式环境变量配置 ==="
echo "VLLM_USE_V1: $VLLM_USE_V1"
echo "RAY_DEDUP_LOGS: $RAY_DEDUP_LOGS"
echo "VLLM_ALLOW_LONG_MAX_MODEL_LEN: $VLLM_ALLOW_LONG_MAX_MODEL_LEN"
echo "RAY_LOG_LEVEL: $RAY_LOG_LEVEL"
echo "CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "TORCH_USE_CUDA_DSA: $TORCH_USE_CUDA_DSA"
echo "================================"
export NCCL_NVLS_ENABLE=0

export PYTHONPATH=.
export SCHEDULER_ADDR=slogin-02
export SCHEDULER_PORT=9969
export MASTER_ADDR=`hostname`
export MASTER_PORT=6379
export USE_EXISTING_RAY_CLUSTER=1


## MATH
python examples/start_agentic_pipeline.py --config_path agentic_disagg  --config_name gem_math_orz_57k 

## SWE
# python examples_lixing/start_agentic_pipeline.py --config_path swe-agentic  --config_name agent_swe_debug_2.5_0.5_local
