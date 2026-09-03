import threading
import time


class AutonomousScheduler:

    def __init__(
        self,
        nova,
        state,
        goals,
    ):

        self.nova = nova
        self.state = state
        self.goals = goals

        self.running = False
        self.thread = None

        self.interval = 15

    def start(self):

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="NOVA-Scheduler",
        )

        self.thread.start()

    def stop(self):

        self.running = False

    def _loop(self):

        while self.running:

            time.sleep(self.interval)

            if not self.running:
                break

            if self.state.current_task:
                continue

            goal = self.goals.next_goal()

            if not goal:
                continue

            self.state.current_goal = (
                goal.description
            )

            self._execute_goal(goal)

    def _execute_goal(self, goal):

        try:

            result = self.nova.autonomous_task(
                goal.description
            )

            self.state.current_task = result

        except Exception as error:

            print(
                f"[SCHEDULER ERROR] {error}"
            )

        finally:

            self.state.current_task = None