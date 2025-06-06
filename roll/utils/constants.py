import enum
import os
import uuid

# RAY_NAMESPACE = "roll"
_ENV_NS = os.environ.get("ROLL_NAMESPACE")
if _ENV_NS:
    RAY_NAMESPACE = _ENV_NS
else:
    RAY_NAMESPACE = f"roll-{uuid.uuid4().hex[:8]}"

STORAGE_NAME = "SHARED_STORAGE_ACTOR"
GENERATE_SCHEDULER_NAME = "GENERATE_SCHEDULER_ACTOR"
REWARD_SCHEDULER_NAME = "REWARD_SCHEDULER_ACTOR"

CHECKPOINT_MANAGER_NAME = "CHECKPOINT_MANAGER_ACTOR"

SCHEDULER_NAME = "scheduler.pt"
OPTIMIZER_NAME = "optimizer.pt"
DIST_OPTIMIZER_DIR = "dist_optimizer"
RNG_STATE_DIR = "rng_state"

CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "roll")
