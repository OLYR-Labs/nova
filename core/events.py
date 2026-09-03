import queue
import threading
from dataclasses import dataclass
from datetime import datetime


@dataclass
class NovaEvent:
    event_type: str
    data: dict
    timestamp: str


class EventBus:
    """
    Thread-safe event bus for NOVA.

    Allows the conversation layer, autonomous agent,
    proactive mind, voice system, and future GUI to
    communicate without directly depending on each other.
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._subscribers = []
        self._lock = threading.Lock()

    def publish(
        self,
        event_type: str,
        data: dict | None = None,
    ):
        event = NovaEvent(
            event_type=event_type,
            data=data or {},
            timestamp=datetime.now().isoformat(),
        )

        self._queue.put(event)

        with self._lock:
            subscribers = list(self._subscribers)

        for subscriber in subscribers:
            try:
                subscriber(event)
            except Exception as error:
                print(
                    f"[NOVA EVENT ERROR] {error}"
                )

    def subscribe(self, callback):
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback):
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def get(self, timeout=None):
        return self._queue.get(
            timeout=timeout
        )

    def empty(self):
        return self._queue.empty()