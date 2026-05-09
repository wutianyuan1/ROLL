cd /root/workspace/weave

PYTHONPATH=/root/workspace/weave/ROLL \
/root/workspace/weave/.venv/bin/python \
/root/workspace/weave/ROLL/global_scheduler/generate_drift_traces.py \
  --trace /root/workspace/weave/ROLL/global_scheduler/trace/wild.trace \
  --profile /root/workspace/weave/ROLL/global_scheduler/trace/profile.json \
  --profile-location disagg \
  --k 3 \
  --seed 2345 \
  --scenarios mixed,increasing,decreasing \
  --output-prefix /root/workspace/weave/ROLL/global_scheduler/generated_drift/wild_drift
