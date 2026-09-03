import random
import threading
import time


class ProactiveMind:

    def __init__(
        self,
        orchestrator,
        event_bus,
        mode="auto",
        minimum_interval=120,
        maximum_interval=300,
    ):

        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.mode = mode

        self.minimum_interval = minimum_interval
        self.maximum_interval = maximum_interval

        self.running = False
        self.thread = None

    def start(self):

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="NOVA-ProactiveMind",
        )

        self.thread.start()

    def stop(self):

        self.running = False

    def _loop(self):

        while self.running:

            delay = random.randint(
                self.minimum_interval,
                self.maximum_interval,
            )

            time.sleep(delay)

            if not self.running:
                break

            try:

                message = (
                    self.orchestrator.proactive_message(
                        mode=self.mode
                    )
                )

                if not message:
                    continue

                self.event_bus.publish(
                    "nova.proactive_message",
                    {
                        "message": message,
                        "source": "proactive_mind",
                    },
                )

            except Exception as error:

                self.event_bus.publish(
                    "nova.error",
                    {
                        "source": "proactive_mind",
                        "error": str(error),
                    },
                )