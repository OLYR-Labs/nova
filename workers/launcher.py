import signal
import sys
import time
import subprocess


class NOVALauncher:

    def __init__(self):

        self.process = None
        self.shutting_down = False

    # =========================================================
    # START
    # =========================================================

    def start_worker(self):

        if (
            self.process is not None
            and self.process.poll() is None
        ):

            print(
                "[NOVA] Worker already running."
            )

            return self.process

        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "workers.autonomous_worker",
            ],
            creationflags=(
                subprocess.CREATE_NEW_CONSOLE
            ),
        )

        print(
            f"[NOVA] Autonomous worker started."
        )

        print(
            f"[NOVA] PID: {self.process.pid}"
        )

        return self.process

    # =========================================================
    # SAFE SHUTDOWN
    # =========================================================

    def safe_shutdown(self):

        if self.shutting_down:
            return

        self.shutting_down = True

        print(
            "\n[NOVA] Requesting safe shutdown..."
        )

        if (
            self.process is None
            or self.process.poll() is not None
        ):

            print(
                "[NOVA] Worker already stopped."
            )

            return

        # -----------------------------------------------------
        # First attempt graceful CTRL+C.
        # -----------------------------------------------------

        try:

            self.process.send_signal(
                signal.CTRL_BREAK_EVENT
            )

        except Exception as error:

            print(
                f"[NOVA] Graceful signal failed: {error}"
            )

        # -----------------------------------------------------
        # Wait.
        # -----------------------------------------------------

        try:

            self.process.wait(
                timeout=10
            )

            print(
                "[NOVA] Worker exited safely."
            )

            return

        except subprocess.TimeoutExpired:

            print(
                "[NOVA] Worker did not exit "
                "during graceful shutdown."
            )

        # -----------------------------------------------------
        # Last resort.
        # -----------------------------------------------------

        print(
            "[NOVA] Escalating shutdown."
        )

        try:

            self.process.terminate()

            self.process.wait(
                timeout=5
            )

        except subprocess.TimeoutExpired:

            print(
                "[NOVA] Force termination required."
            )

            self.process.kill()

            self.process.wait()

    # =========================================================
    # RUN
    # =========================================================

    def run(self):

        self.start_worker()

        try:

            while (
                self.process is not None
                and self.process.poll() is None
            ):

                time.sleep(
                    1
                )

        except KeyboardInterrupt:

            self.safe_shutdown()

        finally:

            if (
                self.process is not None
                and self.process.poll() is None
            ):

                self.safe_shutdown()


if __name__ == "__main__":

    launcher = NOVALauncher()

    launcher.run()