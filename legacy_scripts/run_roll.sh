# LOCAL_IMAGE=/home/twubt/containers/roll.sqsh
# VLLM_USE_PRECOMPILED=1 pip install --editable .
NCG_IMAGE=roll-registry.cn-hangzhou.cr.aliyuncs.com\#roll/pytorch:nvcr-24.05-py3-torch260-vllm084
LOCAL_IMAGE=/home/twubt/containers/roll_new.sqsh
srun --nodes 1 --gpus-per-node 4 --no-container-mount-home --container-remap-root --container-mounts=/home/twubt/.cache/huggingface:/root/.cache/huggingface,/home/twubt/workspace:/workspace --container-workdir=/workspace/ROLL --container-writable --container-image $LOCAL_IMAGE --pty zsh
