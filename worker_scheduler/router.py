import logging
from event import Event, EventLevel, EventType
from typing import Any, Optional, Dict, Callable


class EventRouter:
    def __init__(self) -> None:
        self.event_handler_map: Dict[EventType, Callable[[Event], Any]] = {
            EventType.COMPLETED: None,
            EventType.CRASHED: None,
            EventType.CREATED: None,
            EventType.RECOVERED: None,
            EventType.DONE: None,
            EventType.RELEASE_GPU: None,
            EventType.UNKNOWN: None
        }

    def register(self, event_type: EventType, handler: Callable[[Event], Any]) -> None:
        self.event_handler_map[event_type] = handler

    def handle_event(self, event: Event) -> Optional[Any]:
        if event.event_type in self.event_handler_map:
            return self.event_handler_map[event.event_type](event)
        else:
            logging.warning(f"Unhandled event: {event}")
            return None
