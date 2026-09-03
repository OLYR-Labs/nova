from dataclasses import dataclass
from typing import Optional


@dataclass
class CommandResult:
    success: bool
    output: str = ""
    error: str = ""


class NOVACommandCenter:
    """
    Central coordination layer for NOVA.

    Responsibilities:
    - receive autonomous goals
    - inspect current state
    - decide whether work is allowed
    - execute bounded tasks
    - verify results
    - report outcomes
    """

    def __init__(
        self,
        nova,
        state,
        permissions,
        environment=None,
    ):
        self.nova = nova
        self.state = state
        self.permissions = permissions
        self.environment = environment

    # =========================================================
    # STATUS
    # =========================================================

    def status(self) -> dict:
        return {
            "current_goal": getattr(
                self.state,
                "current_goal",
                None,
            ),
            "current_task": getattr(
                self.state,
                "current_task",
                None,
            ),
            "user_speaking": getattr(
                self.state,
                "user_speaking",
                False,
            ),
            "nova_speaking": getattr(
                self.state,
                "nova_speaking",
                False,
            ),
        }

    # =========================================================
    # OBSERVE
    # =========================================================

    def observe(self) -> dict:
        if self.environment is None:
            return {}

        try:
            return self.environment.snapshot()
        except Exception as error:
            return {
                "error": str(error),
            }

    # =========================================================
    # ACCEPT GOAL
    # =========================================================

    def accept_goal(
        self,
        goal: str,
    ) -> bool:

        goal = str(goal).strip()

        if not goal:
            return False

        self.state.current_goal = goal

        return True

    # =========================================================
    # CLEAR GOAL
    # =========================================================

    def clear_goal(self):
        self.state.current_goal = None
        self.state.current_task = None

    # =========================================================
    # RUN BOUNDED REASONING
    # =========================================================

    def reason_about_goal(
        self,
        goal: str,
    ) -> Optional[str]:

        try:
            return self.nova.autonomous_task(
                goal
            )

        except Exception as error:

            return (
                f"Autonomous reasoning failed: "
                f"{error}"
            )