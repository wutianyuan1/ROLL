import redis
import enum
import time
import dataclasses
import threading
from typing import Optional
from cluster import Cluster


class EventType(enum.Enum):
    # Job-level status
    CREATED = enum.auto()
    COMPLETED = enum.auto()
    CRASHED = enum.auto()
    # Phase (init, gen, train) done
    DONE = enum.auto()
    # Default is unknown
    UNKNOWN = enum.auto()


class JobPhase(enum.Enum):
    INIT = enum.auto()
    GEN = enum.auto()
    TRAIN = enum.auto()


@dataclasses.dataclass
class Event:
    job_name: str = dataclasses.field(default="")
    event_type: EventType = dataclasses.field(default=EventType.UNKNOWN)
    job_phase: Optional[JobPhase] = dataclasses.field(default=None)
    iteration: Optional[int] = dataclasses.field(default=None)

    def __post_init__(self):
        if self.event_type == EventType.DONE:
            assert self.job_phase is not None
            if self.job_phase != JobPhase.INIT:
                assert self.iteration is not None


class Scheduler:
    def __init__(self, shared_storage: redis.StrictRedis):
        self.shared_storage = shared_storage
        self.msg_channel = self.shared_storage.pubsub()
        self.msg_channel.subscribe("tenant_events")
        self.msg_channel.listen()
        self.cluster = Cluster(self.shared_storage, train_all_devices=[4, 5, 6, 7], gen_all_devices=[0, 1, 2, 3])
        self.event_queue = []
        self.lock = threading.Lock()

    def listen_events(self):
        for message in self.msg_channel.listen():
            print(f"==== listen: {message}")
            if message['type'] != 'message':
                print(f"**** Unexpected type: {message['type']}")
                continue
            data = message['data'].decode()
            if data == 'stop_server':
                print("Listen thread: exiting...")
                break
            job_name, *info = data.split(":")
            if "done" not in info[0]:
                assert info[0] in ['created', 'completed', 'crashed']
                type_map = {name.lower(): event_type for name, event_type in EventType._member_map_.items()}
                with self.lock:
                    self.event_queue.append(
                        Event(job_name=job_name, event_type=type_map[info[0]], job_phase=None, iteration=None)
                    )
            else:
                if 'init' in info[0]:
                    with self.lock:
                        self.event_queue.append(
                            Event(job_name=job_name, event_type=EventType.DONE, job_phase=JobPhase.INIT, iteration=None)
                        )
                else:
                    assert len(info) == 2
                    phase, iteration = info
                    print("info", info)
                    phase = phase[:-5]
                    print(phase)
                    assert phase in ['train', 'gen']
                    phase_map = {name.lower(): job_phase for name, job_phase in JobPhase._member_map_.items()}
                    with self.lock:
                        self.event_queue.append(
                            Event(job_name=job_name, event_type=EventType.DONE, job_phase=phase_map[phase], iteration=int(iteration))
                        )

    def run(self):
        event_monitor_thread = threading.Thread(
            target=self.listen_events, name='scheduler-listen'
        )
        event_monitor_thread.start()
        while True:
            try:
                if len(self.event_queue) == 0:
                    continue
                with self.lock:
                    event: Event = self.event_queue.pop(0)  # TODO: priority selection?
                if event.event_type == EventType.CREATED:
                    self.shared_storage.set(f"{event.job_name}_status", "initializing")
                elif event.event_type == EventType.DONE:
                    # After init done or train done, schedule generation tasks
                    if event.job_phase == JobPhase.INIT or event.job_phase == JobPhase.TRAIN:
                        for i in range(4):
                            self.shared_storage.set(f"{event.job_name}_actor_infer-{i}_status", "running")
                    # After generation done, schedule training tasks
                    elif event.job_phase == JobPhase.GEN:
                        self.shared_storage.set(f"{event.job_name}_train_status", "running")
                else:
                    print(f"*** Completion/Crash Event: {event}")
            except KeyboardInterrupt:
                self.shared_storage.publish("tenant_events", "stop_server")
                event_monitor_thread.join()
                print("Shutdown gracefully...")
                break


if __name__ == '__main__':
    shared_storage = redis.StrictRedis(host='localhost', port=9969, db=0)
    scheduler = Scheduler(shared_storage)
    scheduler.run()
