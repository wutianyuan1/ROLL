import os
import redis
import time
import logging
import threading
from copy import deepcopy
from typing import Callable, List, Optional
from resource_manager import ResourceManager
from event import Event, EventParser, EventType, EventLevel, Phase, ExecuteEventType 
from router import EventRouter


PolicyFuncType = Callable[[List[Event], ResourceManager], Optional[List[Event]]]


class FCFSPolicy:
    def __call__(self, event_queue: List[Event], resource_manager: ResourceManager) -> Optional[List[Event]]:
        if len(event_queue) == 0:
            return None
        event_to_run: Event = event_queue[0]
        print(f"[Policy] candidate event is {event_to_run}")
        assert event_to_run.event_type == EventType.READY
        if event_to_run.level == EventLevel.JOB:
            # If there are enough resource to schedule an init now
            if len(resource_manager.available_devices) == resource_manager.cluster_size:
                resource_manager.register_job(event_to_run.job_name)
                resource_manager.update_phase(event_to_run.job_name, Phase.INIT)
                resource_manager.allocate_all(event_to_run.job_name)
                event_queue.pop(0)
                return [Event(event_to_run.job_name, ExecuteEventType.STATUS, value='initializing')]
            else:
                return None
        elif event_to_run.phase == Phase.GENERATE:
            # HACK: simple implementation: just start this phase and run all 4 workers
            assert event_to_run.level == EventLevel.PHASE
            if len(resource_manager.available_devices) >= 4:
                event_queue.pop(0)
                events_to_execute = []
                events_to_execute.append(
                    Event(event_to_run.job_name,
                        ExecuteEventType.STATUS,
                        phase=Phase.GENERATE,
                        value='running'
                    )
                )
                for i in range(4):
                    resource_manager.allocate_worker(event_to_run.job_name, 1)
                    events_to_execute.append(
                        Event(event_to_run.job_name,
                            ExecuteEventType.STATUS,
                            phase=Phase.GENERATE,
                            worker_id=str(i),
                            value='running'
                        )
                    )
                return events_to_execute
            else:
                return None
        elif event_to_run.phase == Phase.TRAIN:
            if len(resource_manager.available_devices) >= 4:
                event_queue.pop(0)
                return [Event(event_to_run.job_name, ExecuteEventType.STATUS, phase=Phase.TRAIN, value='running')]
            else:
                return None
        elif event_to_run.phase == Phase.UPDATE:
            if len(resource_manager.available_devices) >= 4:
                event_queue.pop(0)
                return [Event(event_to_run.job_name, ExecuteEventType.STATUS, phase=Phase.UPDATE, value='running')]
            else:
                return None
        else:
            assert False, f"Unexpected event in policy: {event_to_run}"


class Scheduler:
    def __init__(self, shared_storage: redis.StrictRedis, policy: PolicyFuncType):
        self.shared_storage = shared_storage
        self.msg_channel = self.shared_storage.pubsub()
        self.msg_channel.subscribe("tenant_events")
        self.msg_channel.listen()
        self.resource_manager = ResourceManager([0, 1, 2, 3])
        self.ready_queue = []
        self.lock = threading.Lock()
        self.select_policy = policy
        self.backend_router = self.create_backend_router()

    def create_backend_router(self):

        def handle_job_complete(event: Event) -> None:
            '''on job complete or crash: cleanup all its allocated GPUs'''
            assert event.level == EventLevel.JOB
            self.resource_manager.cleanup_by_name(event.job_name)

        def handle_gpu_release(event: Event) -> None:
            '''on GPU release: ask the resource manager to free corresponding GPUs'''
            assert event.value is not None
            self.resource_manager.release(event.job_name, [int(i) for i in event.value.split(",")])

        def handle_job_creation(event: Event) -> None:
            '''on job creation: simply forward this message to ready queue'''
            assert event.level == EventLevel.JOB, f"{event}"
            ready_event = deepcopy(event)
            ready_event.event_type = EventType.READY
            self.ready_queue.append(ready_event)

        def handle_finish_event(event: Event) -> None:
            assert event.event_type == EventType.DONE
            if event.level == EventLevel.WORKER:
                # worker finished: do nothing
                return
            elif event.level == EventLevel.PHASE:
                # if init done or update done, then next step's generation is ready to run
                if event.phase == Phase.INIT or event.phase == Phase.UPDATE:
                    ready_event = Event(
                        job_name=event.job_name,
                        event_type=EventType.READY,
                        phase=Phase.GENERATE)
                    self.ready_queue.append(ready_event)
                # if generate done, then its training is ready to begin
                elif event.phase == Phase.GENERATE:
                    ready_event = Event(
                        job_name=event.job_name,
                        event_type=EventType.READY,
                        phase=Phase.TRAIN)
                    self.ready_queue.append(ready_event)
                # if training done, then its parameter update is ready to begin
                elif event.phase == Phase.TRAIN:
                    ready_event = Event(
                        job_name=event.job_name,
                        event_type=EventType.READY,
                        phase=Phase.UPDATE)
                    self.ready_queue.append(ready_event)
                else:
                    assert False, f"Unexpected finish event: {event}"
            elif event.level == EventLevel.JOB:
                self.resource_manager.cleanup_by_name(event.job_name)
            else:
                assert False, f"Unexpected event level, event={event}"

        router = EventRouter()
        router.register(EventType.CREATED, handle_job_creation)
        router.register(EventType.COMPLETED, handle_job_complete)
        router.register(EventType.CRASHED, handle_job_complete)
        router.register(EventType.RELEASE_GPU, handle_gpu_release)
        router.register(EventType.DONE, handle_finish_event)
        return router

    def listen_events(self):
        for message in self.msg_channel.listen():
            if message['type'] != 'message':
                logging.error(f"**** Unexpected type: {message['type']}")
                continue
            data = message['data'].decode()
            if data == 'stop_server':
                logging.info("Listen thread: exiting...")
                break
            event: Event = EventParser.parse(data)
            with self.lock:
                self.backend_router.handle_event(event)  

    def run(self):
        event_monitor_thread = threading.Thread(
            target=self.listen_events, name='scheduler-listen'
        )
        event_monitor_thread.start()
        while True:
            try:
                with self.lock:
                    events_to_execute = self.select_policy(self.ready_queue, self.resource_manager)
                if events_to_execute is None:
                    time.sleep(0.5)
                    continue
                else:
                    print(f"[Scheduler] will execute {events_to_execute}")
                assert len(events_to_execute) != 0
                for e in events_to_execute:
                    e.execute(self.shared_storage)
            except KeyboardInterrupt:
                self.shared_storage.publish("tenant_events", "stop_server")
                event_monitor_thread.join()
                logging.info("Shutdown gracefully...")
                break
            time.sleep(0.5)


if __name__ == '__main__':
    shared_storage = redis.StrictRedis(
        host=os.environ.get("MASTER_ADDR", "localhost"),
        port=int(os.environ.get("SCHEDULER_PORT", "9969")),
        db=0
    )
    policy = FCFSPolicy()
    scheduler = Scheduler(shared_storage, policy)
    scheduler.run()
