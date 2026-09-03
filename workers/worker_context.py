import threading
from queue import Queue, Empty


class WorkerContext:

    def __init__(self):
        self.events = Queue()
        self.running = False
        self.thread = None

    def emit(self, event_type, data=None):
        self.events.put({
            "type": event_type,
            "data": data,
        })

    def start(self, target):
        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=target,
            args=(self,),
            daemon=True,
            name="NOVA-AutonomousWorker",
        )

        self.thread.start()

    def stop(self):
        self.running = False

    def get_event(self):
        try:
            return self.events.get_nowait()
        except Empty:
            return None