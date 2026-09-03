import threading
import time


class TerminalController:

    def __init__(self):

        self.input_active = False
        self.lock = threading.Lock()

    def begin_input(self):

        with self.lock:
            self.input_active = True

    def end_input(self):

        with self.lock:
            self.input_active = False

    def can_interrupt(self):

        with self.lock:
            return not self.input_active

    def wait_until_safe(self):

        while True:

            if self.can_interrupt():
                return

            time.sleep(0.1)