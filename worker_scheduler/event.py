import re
import redis
import enum
from typing import Dict, Optional, Any


class EventType(enum.Enum):
    CREATED = 'created'
    COMPLETED = 'completed'
    CRASHED = 'crashed'
    DONE = 'done'
    RELEASE_GPU = 'release_gpu'
    READY = 'ready'
    STATUS = 'status'
    UNKNOWN = 'unknown'

    @classmethod
    def match(cls, event_type: str) -> 'EventType':
        name_map = {k.lower(): v for k, v in cls._member_map_.items()}
        return name_map.get(event_type, cls.UNKNOWN)


class EventLevel(enum.Enum):
    WORKER = 'worker'
    PHASE = 'phase'
    JOB = 'job'


class Phase(enum.Enum):
    INIT = 'init'
    GENERATE = 'generate'
    TRAIN = 'train'
    UPDATE = 'update'

    @classmethod
    def match(cls, phase: Optional[str]) -> Optional['Phase']:
        name_map = {k.lower(): v for k, v in cls._member_map_.items()}
        return name_map.get(phase, None)


class Event:
    """
    An event, it can be one of the following:
        - Job level: e.g., job1 created
        - Phase (init/generate/train/update) level: e.g., job1's generation phase done
        - Worker (rank) level: e.g., worker 1 of job1's generation phase done
    Event type can be one of the following:
        - created (job level):
            * value: None
        - completed (job level)
            * value: None
        - crashed (job level):
            * value: None
        - done (job/phase/worker finished):
            * value (optional): (int) step count 
        - release_gpu:
            * value (mandatory): (str) device_affinity + (list[int]) gpu IDs, e.g. [NVIDIA H20,0,1,2,3]
    """
    def __init__(self, job_name: str,
                 event_type: EventType, 
                 phase: Optional[Phase] = None, 
                 worker_id: Optional[str] = None, 
                 value: Optional[Any] = None):
        self.job_name: str = job_name
        self.phase: Phase = phase
        self.worker_id: str = worker_id
        self.event_type: EventType = event_type
        self.value: str = value

    @property
    def level(self) -> EventLevel:
        """Get level of this event"""
        if self.worker_id is not None:
            return EventLevel.WORKER
        if self.phase is not None:
            return EventLevel.PHASE
        return EventLevel.JOB
    
    def __repr__(self) -> str:
        parts = [f"job={self.job_name}"]
        if self.phase:
            parts.append(f"phase={self.phase.value}")
        if self.worker_id:
            parts.append(f"worker={self.worker_id}")
        parts.append(f"type={self.event_type.value}")
        if self.value is not None:
            parts.append(f"value={self.value}")
        return f"Event({', '.join(parts)})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_name": self.job_name,
            "phase": self.phase,
            "worker_id": self.worker_id,
            "event_type": self.event_type,
            "value": self.value
        }

    def execute(self, shared_storage: redis.StrictRedis):
        key = self.job_name
        if self.phase is not None:
            key = key + ":" + str(self.phase.value)
        if self.worker_id is not None:
            key = key + ":" + str(self.worker_id)
        key = key + ":" + self.event_type.value
        shared_storage.set(key, self.value)


class EventParser:
    '''
    Parse an input event received from redis pubsub, the format is:
    job:(phase)?:(worker)?:event_type([event_value1, event_value2, ...])?
    '''
    EVENT_PATTERNS = [
        # Worker-level events: job:phase:worker:event_type
        re.compile(r"^(?P<job_name>[^:]+):(?P<phase>[^:]+):(?P<worker_id>[^:]+):(?P<event_type>[^:]+)$"),
        # Phase-level events: job:phase:event_type
        re.compile(r"^(?P<job_name>[^:]+):(?P<phase>[^:]+):(?P<event_type>[^:]+)$"),
        # Job-level events: job:event_type
        re.compile(r"^(?P<job_name>[^:]+):(?P<event_type>[^:]+)$"),
    ]
    EVENT_INFO_PATTERN = re.compile(r'^(?P<event_type>[a-zA-Z_]+)(?:\[(?P<event_value>([a-zA-Z0-9_\s,]+)?[0-9,]+)\])?$')

    @classmethod
    def parse(cls, event_key: str) -> Optional[Event]:
        """Try all patterns from low level to high level and returns the first matched pattern"""
        for pattern in cls.EVENT_PATTERNS:
            match = pattern.match(event_key)
            if match:
                groups = match.groupdict()
                value_match = cls.EVENT_INFO_PATTERN.match(groups['event_type']).groupdict()
                return Event(
                    job_name=groups["job_name"],
                    phase=Phase.match(groups.get("phase")),
                    worker_id=groups.get("worker_id"),
                    event_type=EventType.match(value_match["event_type"]),
                    value=value_match["event_value"]
                )
        return None


if __name__ == "__main__":
    parser = EventParser()
    print(parser.parse("job1:created"))
    print(parser.parse("job1:generate:done[0]"))
    # print(parser.parse("job1:init:release_gpu[0,1,2,3]"))
    print(parser.parse("job1:init:release_gpu[NVIDIA H20,0,1,2,3]"))
    # print(parser.parse("job1:generate:1:release_gpu[0,1,2,3]"))
    print(parser.parse("job1:generate:1:release_gpu[NVIDIA H20,0,1,2,3]"))
