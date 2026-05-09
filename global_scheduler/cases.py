import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from global_scheduler.new_simulator import WeaveSimulator
from global_scheduler.structs import Job

plt.figure(figsize=(12, 3))

# Case-1
# job_A = Job('A', 7, 7, rollout_nodes=['RN'], train_nodes=['TN'])
# job_B = Job('B', 7, 7, rollout_nodes=['RN'], train_nodes=['TN'])
# simulator = WeaveSimulator([job_A, job_B])

# Case-2
# job_A = Job('A', 14, 14, rollout_nodes=['RN'], train_nodes=['TN'])
# job_B = Job('B', 7, 7, rollout_nodes=['RN'], train_nodes=['TN'])
# job_C = Job('C', 7, 7, rollout_nodes=['RN'], train_nodes=['TN'])
# simulator = WeaveSimulator([job_A, job_B, job_C])

# Case-3
# job_A = Job('A', 2.5, 1, rollout_nodes=['RN-1'], train_nodes=['TN'])
# job_B = Job('B', 2.5, 1, rollout_nodes=['RN-2'], train_nodes=['TN'])
# job_C = Job('C', 2.5, 1, rollout_nodes=['RN-3'], train_nodes=['TN'])
# simulator = WeaveSimulator([job_A, job_B, job_C])

job_A = Job('A', 6, 1, rollout_nodes=['RN-1'], train_nodes=['TN'])
job_B = Job('B', 2.5, 1, rollout_nodes=['RN-2'], train_nodes=['TN'])
job_C = Job('C', 2.5, 1, rollout_nodes=['RN-3'], train_nodes=['TN'])
simulator = WeaveSimulator([job_A, job_B, job_C])

# Case-5
# job_A = Job('A', 2.5, 2.5, rollout_nodes=['RN-1', 'RN-2'], train_nodes=['TN'])
# job_B = Job('B', 2.5, 1, rollout_nodes=['RN-1'], train_nodes=['TN'])
# job_C = Job('C', 2.5, 1, rollout_nodes=['RN-2'], train_nodes=['TN'])
# simulator = WeaveSimulator([job_A, job_B, job_C])

color_settings = {
    "A": ("#ff9999", "#ff9999", ""),
    "B": ("#99dd99", "#99dd99", ""),
    "C": ("#bbbbbb", "#bbbbbb", ""),
}

# Failure-recovery demo:
# 1. Run normally for a while.
# 2. Fail A in the middle of its second rollout.
# 3. Let B/C continue.
# 4. Recover A and observe it rejoin the interleaving pattern.
simulator.reset_state(start_time=0.0)
simulator.bootstrap(start_time=0.0, reset_runtime=True, queue_from_state=True)
simulator.run_until(target_time=10.0)
failure_time = simulator.current_time
simulator.fail_job("A", restart_incomplete=True)
simulator.run_until(target_time=40.0)
recovery_time = simulator.current_time
simulator.recover_job("A", reset_runtime=False)
simulator.run_until(target_time=85.0)

simulator.plot_recorded_history(
    color_settings=color_settings,
    export_path="/root/workspace/weave/ROLL/global_scheduler/sim_failure_recovery.png",
)

plt.axvline(failure_time, color="red", linestyle="--", linewidth=1.2)
plt.axvline(recovery_time, color="blue", linestyle="--", linewidth=1.2)
plt.text(failure_time + 0.2, 3.65, "A fails", color="red", fontsize=11)
plt.text(recovery_time + 0.2, 3.65, "A recovers", color="blue", fontsize=11)
plt.savefig("/root/workspace/weave/ROLL/global_scheduler/sim_failure_recovery.png")
