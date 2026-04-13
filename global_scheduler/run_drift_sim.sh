cd /root/workspace/weave

PYTHONPATH=/root/workspace/weave/ROLL \
/root/workspace/weave/.venv/bin/python \
/root/workspace/weave/ROLL/global_scheduler/drift_experiment.py \
  --trace /root/workspace/weave/ROLL/global_scheduler/trace/wild.trace \
  --profile /root/workspace/weave/ROLL/global_scheduler/trace/profile.json \
  --profile-location disagg \
  --drift-prefix /root/workspace/weave/ROLL/global_scheduler/generated_drift/wild_drift \
  --output /root/workspace/weave/ROLL/global_scheduler/results/wild_mixed_results.json \
  --scenarios mixed \
  --default-slo 1.5 \
  --max-group-size 3 \
  --regroup-interval-sec 3600 \
  --first-regroup-delay-sec 600 \
  --planning-penalty-sec 120 \
  --regroup-pause-sec 120 \
  --exact-search-threshold 8 \
  --max-search-steps 10000