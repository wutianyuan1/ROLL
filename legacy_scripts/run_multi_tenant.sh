#!/bin/bash

# export MASTER_ADDR=localhost
export SCHEDULER_PORT=9969

# Start the scheduler in the background, capture its PID
MASTER_ADDR=localhost python -m roll.multi_tenant.scheduler &
scheduler_pid=$!

# Start two RLVR superpod jobs with different MASTER_PORTs, capture their PIDs
MASTER_PORT=6124 ./start_rlvr.sh &
rlvr_pid_1=$!
MASTER_PORT=1935 ./start_rlvr.sh &
rlvr_pid_2=$!

# Cleanup function to kill jobs in order: first rlvr jobs, then scheduler
cleanup() {
    echo "Caught Ctrl+C. Terminating RLVR jobs and then the scheduler..."
    kill $rlvr_pid_1 2>/dev/null
    kill $rlvr_pid_2 2>/dev/null
    wait $rlvr_pid_1 2>/dev/null
    wait $rlvr_pid_2 2>/dev/null
    kill $scheduler_pid 2>/dev/null
    wait $scheduler_pid 2>/dev/null
    ray stop --force
    echo "All processes terminated."
    exit 0
}

# Trap Ctrl+C (SIGINT)
trap cleanup SIGINT

# Wait for background jobs
wait