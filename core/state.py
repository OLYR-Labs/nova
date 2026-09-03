from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock, Event


@dataclass
class AgentState:
    """
    Shared runtime state for NOVA.

    NOVA uses cooperative shutdown:
        RUNNING
            ↓
        STOP_REQUESTED
            ↓
        SAFE_STOPPING
            ↓
        STOPPED

    The agent never intentionally kills an operation halfway through.
    """

    # =========================================================
    # RUNTIME
    # =========================================================

    running: bool = True
    autonomous: bool = False
    cycle: int = 0

    # =========================================================
    # SAFE STOP
    # =========================================================

    stop_requested: bool = False
    safe_stopping: bool = False

    stop_reason: str | None = None

    # Event allows worker threads/process controllers to
    # observe a stop request without busy waiting.
    stop_event: Event = field(
        default_factory=Event,
        repr=False,
        compare=False,
    )

    # =========================================================
    # OBSERVATIONS
    # =========================================================

    observations: list[str] = field(
        default_factory=list
    )

    max_observations: int = field(
        default=100,
        repr=False,
        compare=False,
    )

    # =========================================================
    # USER / VOICE
    # =========================================================

    user_active: bool = False
    user_speaking: bool = False
    nova_speaking: bool = False

    # =========================================================
    # CURRENT WORK
    # =========================================================

    current_goal: str | None = None
    current_task: str | None = None
    current_operation: str | None = None

    # =========================================================
    # TIMESTAMPS
    # =========================================================

    last_user_input: datetime | None = None
    last_nova_output: datetime | None = None

    # =========================================================
    # MESSAGE QUEUE
    # =========================================================

    queued_messages: list[str] = field(
        default_factory=list
    )

    # =========================================================
    # THREAD SAFETY
    # =========================================================

    lock: Lock = field(
        default_factory=Lock,
        repr=False,
        compare=False,
    )

    # =========================================================
    # CYCLE
    # =========================================================

    def next_cycle(self) -> int:
        with self.lock:
            self.cycle += 1
            return self.cycle

    # =========================================================
    # OBSERVATION
    # =========================================================

    def observe(self, observation) -> None:

        if observation is None:
            return

        observation = str(
            observation
        ).strip()

        if not observation:
            return

        with self.lock:

            self.observations.append(
                observation
            )

            if (
                self.max_observations > 0
                and len(self.observations)
                > self.max_observations
            ):
                self.observations = (
                    self.observations[
                        -self.max_observations:
                    ]
                )

    # =========================================================
    # GET OBSERVATIONS
    # =========================================================

    def get_observations(
        self,
        limit: int | None = None,
    ) -> list[str]:

        with self.lock:

            if limit is None:
                return list(
                    self.observations
                )

            try:
                limit = int(limit)
            except (
                TypeError,
                ValueError,
            ):
                limit = 10

            if limit <= 0:
                return []

            return list(
                self.observations[
                    -limit:
                ]
            )

    # =========================================================
    # CLEAR OBSERVATIONS
    # =========================================================

    def clear_observations(self) -> None:

        with self.lock:
            self.observations.clear()

    # =========================================================
    # USER ACTIVITY
    # =========================================================

    def set_user_active(
        self,
        active: bool,
    ) -> None:

        with self.lock:

            self.user_active = bool(
                active
            )

            if self.user_active:
                self.last_user_input = (
                    datetime.now()
                )

    # =========================================================
    # USER SPEAKING
    # =========================================================

    def set_user_speaking(
        self,
        speaking: bool,
    ) -> None:

        with self.lock:

            self.user_speaking = bool(
                speaking
            )

            if self.user_speaking:

                self.user_active = True

                self.last_user_input = (
                    datetime.now()
                )

    # =========================================================
    # NOVA SPEAKING
    # =========================================================

    def set_nova_speaking(
        self,
        speaking: bool,
    ) -> None:

        with self.lock:

            self.nova_speaking = bool(
                speaking
            )

            if self.nova_speaking:

                self.last_nova_output = (
                    datetime.now()
                )

    # =========================================================
    # AUTONOMOUS
    # =========================================================

    def set_autonomous(
        self,
        enabled: bool,
    ) -> None:

        with self.lock:
            self.autonomous = bool(
                enabled
            )

    # =========================================================
    # GOAL
    # =========================================================

    def set_goal(
        self,
        goal: str | None,
    ) -> None:

        with self.lock:

            if goal is None:
                self.current_goal = None
                return

            goal = str(
                goal
            ).strip()

            self.current_goal = (
                goal
                if goal
                else None
            )

    # =========================================================
    # TASK
    # =========================================================

    def set_task(
        self,
        task: str | None,
    ) -> None:

        with self.lock:

            if task is None:
                self.current_task = None
                return

            task = str(
                task
            ).strip()

            self.current_task = (
                task
                if task
                else None
            )

    # =========================================================
    # CURRENT OPERATION
    # =========================================================

    def set_operation(
        self,
        operation: str | None,
    ) -> None:

        with self.lock:
            self.current_operation = operation

    # =========================================================
    # SAFE STOP REQUEST
    # =========================================================

    def request_stop(
        self,
        reason: str = "User requested stop.",
    ) -> None:

        with self.lock:

            self.stop_requested = True
            self.stop_reason = str(
                reason
            )

            self.safe_stopping = True

            self.stop_event.set()

    # =========================================================
    # STOP CHECK
    # =========================================================

    def should_stop(self) -> bool:

        return self.stop_event.is_set()

    # =========================================================
    # MARK STOPPED
    # =========================================================

    def mark_stopped(self) -> None:

        with self.lock:

            self.running = False
            self.autonomous = False
            self.safe_stopping = False

            self.current_operation = None

    # =========================================================
    # LEGACY STOP
    # =========================================================

    def stop(self) -> None:

        self.request_stop(
            "Runtime stop requested."
        )

    # =========================================================
    # START
    # =========================================================

    def start(self) -> None:

        with self.lock:

            self.running = True
            self.autonomous = False

            self.stop_requested = False
            self.safe_stopping = False
            self.stop_reason = None

            self.stop_event.clear()

    # =========================================================
    # INTERRUPTION SAFETY
    # =========================================================

    def can_interrupt(self) -> bool:

        with self.lock:

            return (
                self.running
                and not self.stop_requested
                and not self.user_active
                and not self.user_speaking
                and not self.nova_speaking
            )

    # =========================================================
    # MESSAGE QUEUE
    # =========================================================

    def queue_message(
        self,
        message: str,
    ) -> None:

        if message is None:
            return

        message = str(
            message
        ).strip()

        if not message:
            return

        with self.lock:
            self.queued_messages.append(
                message
            )

    # =========================================================
    # POP MESSAGE
    # =========================================================

    def pop_message(self) -> str | None:

        with self.lock:

            if not self.queued_messages:
                return None

            return self.queued_messages.pop(
                0
            )

    # =========================================================
    # CLEAR MESSAGES
    # =========================================================

    def clear_messages(self) -> None:

        with self.lock:
            self.queued_messages.clear()

    # =========================================================
    # SNAPSHOT
    # =========================================================

    def snapshot(self) -> dict:

        with self.lock:

            return {
                "running": self.running,

                "autonomous": self.autonomous,

                "cycle": self.cycle,

                "stop_requested": (
                    self.stop_requested
                ),

                "safe_stopping": (
                    self.safe_stopping
                ),

                "stop_reason": (
                    self.stop_reason
                ),

                "user_active": (
                    self.user_active
                ),

                "user_speaking": (
                    self.user_speaking
                ),

                "nova_speaking": (
                    self.nova_speaking
                ),

                "current_goal": (
                    self.current_goal
                ),

                "current_task": (
                    self.current_task
                ),

                "current_operation": (
                    self.current_operation
                ),

                "last_user_input": (
                    self.last_user_input.isoformat()
                    if self.last_user_input
                    else None
                ),

                "last_nova_output": (
                    self.last_nova_output.isoformat()
                    if self.last_nova_output
                    else None
                ),

                "observations": list(
                    self.observations
                ),

                "queued_messages": list(
                    self.queued_messages
                ),
            }


NOVAState = AgentState