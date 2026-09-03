import threading
import time
from datetime import datetime


class AutonomyController:

    def __init__(self, nova, state):
        self.nova = nova
        self.state = state

        self.running = False
        self.thread = None

        self.idle_threshold = 30
        self.cooldown = 180

        self.last_initiation = None

    def start(self):

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="NOVA-Autonomy",
        )

        self.thread.start()

        print("[AUTONOMY] Controller started.")

    def stop(self):

        self.running = False

        print("[AUTONOMY] Controller stopped.")

    def _user_idle(self):

        if self.state.user_active:
            return False

        if not self.state.last_user_input:
            return True

        elapsed = (
            datetime.now() -
            self.state.last_user_input
        ).total_seconds()

        return elapsed >= self.idle_threshold

    def _cooldown_finished(self):

        if self.last_initiation is None:
            return True

        elapsed = (
            datetime.now() -
            self.last_initiation
        ).total_seconds()

        return elapsed >= self.cooldown

    def _loop(self):

        while self.running:

            time.sleep(2)

            if not self._user_idle():
                continue

            if not self._cooldown_finished():
                continue

            if self.state.user_speaking:
                continue

            if self.state.nova_speaking:
                continue

            try:

                message = self.nova.proactive_message()

                if not message:
                    continue

                self.state.queue_message(message)

                self.last_initiation = datetime.now()

            except Exception as error:

                print(
                    f"[AUTONOMY ERROR] {error}"
                )