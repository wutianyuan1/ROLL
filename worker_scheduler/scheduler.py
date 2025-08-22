import os
import redis
import time
import logging
import threading
import subprocess
from copy import deepcopy
from typing import Callable, List, Optional
from resource_manager import ResourceManager, JobStatus
from event import Event, EventParser, EventType, EventLevel, Phase 
from router import EventRouter


PolicyFuncType = Callable[[List[Event], ResourceManager], Optional[List[Event]]]


def get_init_job_status(job_name: str, shared_storage: redis.StrictRedis) -> JobStatus:
    # gpus per worker: default = 1
    gpus_per_gen_worker = shared_storage.get(f'{job_name}:gpus_per_gen_worker')
    gpus_per_gen_worker = int(gpus_per_gen_worker.decode()) if gpus_per_gen_worker is not None else 1
    # max_train_gpus: default = 1
    max_train_gpus = shared_storage.get(f'{job_name}:max_train_gpus')
    max_train_gpus = int(max_train_gpus.decode()) if max_train_gpus is not None else 1
    # max_gen_gpus: default = 1
    max_gen_gpus = shared_storage.get(f'{job_name}:max_gen_gpus')
    max_gen_gpus = int(max_gen_gpus.decode()) if max_gen_gpus is not None else 1
    job_status = JobStatus(
        job_name=job_name,
        phase=Phase.INIT,
        gpus_per_gen_worker=gpus_per_gen_worker,
        max_gen_gpus=max_gen_gpus,
        max_train_gpus=max_train_gpus,
        allocated_gpus=[]
    )
    return job_status


class FCFSPolicy:
    def __init__(self, shared_storage: redis.StrictRedis, min_concurrency_ratio: float = 0.5):
        self.min_concurrency_ratio = min_concurrency_ratio
        self.shared_storage = shared_storage
        self.prev_candidate = None

    def __call__(self, event_queue: List[Event], resource_manager: ResourceManager) -> Optional[List[Event]]:
        if len(event_queue) == 0:
            return None
        event_to_run: Event = event_queue[0]
        if event_to_run != self.prev_candidate:
            # Only log the candidate event if it changes
            logging.info(f"[Policy] candidate event is {event_to_run}")
            self.prev_candidate = event_to_run
        assert event_to_run.event_type == EventType.READY
        if event_to_run.level == EventLevel.JOB:
            # If there are enough resource to schedule an init now
            if resource_manager.num_available_devices == resource_manager.cluster_size:
                resource_manager.register_job(
                    get_init_job_status(event_to_run.job_name, self.shared_storage)
                )
                resource_manager.allocate_all(event_to_run.job_name)
                event_queue.pop(0)
                return [Event(event_to_run.job_name, EventType.STATUS, value='initializing')]
            else:
                return None
        elif event_to_run.phase == Phase.GENERATE:
            assert event_to_run.level == EventLevel.PHASE
            job_status = resource_manager.job_mapping[event_to_run.job_name]
            if len(resource_manager.gen_available_devices) / job_status.max_gen_gpus >= self.min_concurrency_ratio:
                event_queue.pop(0)
                events_to_execute = []
                events_to_execute.append(
                    Event(event_to_run.job_name,
                        EventType.STATUS,
                        phase=Phase.GENERATE,
                        value='running'
                    )
                )
                for i in range(len(resource_manager.gen_available_devices) // job_status.gpus_per_gen_worker):
                    workers = resource_manager.allocate_worker(event_to_run.job_name, 'gen')
                    logging.info(f"[ResourceManager] Alloc worker: {workers}")
                    events_to_execute.append(
                        Event(event_to_run.job_name,
                            EventType.STATUS,
                            phase=Phase.GENERATE,
                            worker_id=str(i),
                            value='running'
                        )
                    )
                return events_to_execute
            else:
                return None
        elif event_to_run.phase == Phase.TRAIN:
            job_status = resource_manager.job_mapping[event_to_run.job_name]
            if len(resource_manager.train_available_devices) >= job_status.max_train_gpus:
                event_queue.pop(0)
                resource_manager.allocate_worker(event_to_run.job_name, 'train')
                return [Event(event_to_run.job_name, EventType.STATUS, phase=Phase.TRAIN, value='running')]
            else:
                return None
        elif event_to_run.phase == Phase.UPDATE:
            # Madoka: we observe that update do not require any resources, so it can be executed immediately
            event_queue.pop(0)
            return [Event(event_to_run.job_name, EventType.STATUS, phase=Phase.UPDATE, value='running')]
        else:
            assert False, f"Unexpected event in policy: {event_to_run}"


class Scheduler:
    def __init__(self, shared_storage: redis.StrictRedis, policy: PolicyFuncType):
        self.shared_storage = shared_storage
        self.msg_channel = self.shared_storage.pubsub()
        self.msg_channel.subscribe("tenant_events")
        self.msg_channel.listen()
        self.resource_manager = ResourceManager(gen_device_ids=list(range(0, 8)), train_device_ids=list(range(8, 16)))
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
                if event.phase == Phase.INIT:
                    ready_event = Event(
                        job_name=event.job_name,
                        event_type=EventType.READY,
                        phase=Phase.UPDATE)
                    self.ready_queue.append(ready_event)
                # if update done, then free all its resources next step's generation is ready to run
                elif event.phase == Phase.UPDATE:
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
                # if training done, then free its train resources and its parameter update is ready to begin
                elif event.phase == Phase.TRAIN:
                    ready_event = Event(
                        job_name=event.job_name,
                        event_type=EventType.READY,
                        phase=Phase.UPDATE)
                    self.resource_manager.cleanup_by_name(event.job_name)
                    self.ready_queue.append(ready_event)
                else:
                    assert False, f"Unexpected finish event: {event}"
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
            logging.info(f"[Router] event: {event}, available_devices_before: (gen){self.resource_manager.gen_available_devices} (train){self.resource_manager.train_available_devices},")
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
                    logging.info(f"[Scheduler] will execute {events_to_execute}")
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S'
    )

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    scheduler_port = int(os.environ.get("SCHEDULER_PORT", "9969"))
    redis_server_proc = subprocess.Popen(
        f"redis-server --bind {master_addr} --port {scheduler_port} --save \"\"",
        shell=True
    )
    shared_storage = redis.StrictRedis(
        host=master_addr,
        port=scheduler_port,
        db=0
    )
    # wait the redis ready
    while True:
        try:
            shared_storage.set("test", 1)
            break
        except:
            continue
    logging.info("[Scheduler] connected to redis.")
    policy = FCFSPolicy(shared_storage=shared_storage, min_concurrency_ratio=1.0)
    scheduler = Scheduler(shared_storage, policy)
    scheduler.run()
    redis_server_proc.terminate()
