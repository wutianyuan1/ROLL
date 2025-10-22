import matplotlib.pyplot as plt
from global_scheduler.new_simulator import WeaveSimulator
from global_scheduler.structs import Job


plt.figure(figsize=(12, 3))

# Case-1
# job_A = Job('A', 7, 7, ['RN'], ['TN'])
# job_B = Job('B', 7, 7, ['RN'], ['TN'])
# simulator = WeaveSimulator([job_A, job_B])

# Case-2
# job_A = Job('A', 14, 14, ['RN'], ['TN'])
# job_B = Job('B', 7, 7, ['RN'], ['TN'])
# job_C = Job('C', 7, 7, ['RN'], ['TN'])
# simulator = WeaveSimulator([job_A, job_B, job_C])

# Case-3
# job_A = Job('A', 2.5, 1, ['RN-1'], ['TN'])
# job_B = Job('B', 2.5, 1, ['RN-2'], ['TN'])
# job_C = Job('C', 2.5, 1, ['RN-3'], ['TN'])
# simulator = WeaveSimulator([job_A, job_B, job_C])

# Case-4
# job_A = Job('A', 6, 1, ['RN-1'], ['TN'])
# job_B = Job('B', 2.5, 1, ['RN-2'], ['TN'])
# job_C = Job('C', 2.5, 1, ['RN-3'], ['TN'])
# simulator = WeaveSimulator([job_A, job_B, job_C])

# Case-5
job_A = Job('A', 2.5, 2.5, ['RN-1', 'RN-2'], ['TN'])
job_B = Job('B', 2.5, 1, ['RN-1'], ['TN'])
job_C = Job('C', 2.5, 1, ['RN-2'], ['TN'])
simulator = WeaveSimulator([job_A, job_B, job_C])

simulator.plot(10, "global_scheduler/sim.png")
