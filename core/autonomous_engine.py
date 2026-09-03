from core.autonomous_state import (
    AutonomousState,
)


class AutonomousEngine:

    def __init__(
        self,
        nova,
        state,
        logger,
        command_center,
        development_cycle,
    ):
        self.nova = nova
        self.state = state
        self.logger = logger
        self.command_center = command_center
        self.development_cycle = (
            development_cycle
        )

        self.running = False

    # =========================================================
    # START
    # =========================================================

    def start(self):
        self.running = True

    # =========================================================
    # STOP
    # =========================================================

    def stop(self):
        self.running = False

        self.state.set(
            AutonomousState.IDLE
        )

    # =========================================================
    # RUN GOAL
    # =========================================================

    def run_goal(
        self,
        goal: str,
    ):

        if not self.running:
            return {
                "success": False,
                "error": (
                    "Autonomous engine is stopped."
                ),
            }

        self.logger.info(
            "Autonomous goal started.",
            {
                "goal": goal,
            },
        )

        try:

            self.state.set(
                AutonomousState.OBSERVE
            )

            observation = (
                self.command_center.observe()
            )

            self.state.set(
                AutonomousState.PLAN
            )

            reasoning = (
                self.command_center
                .reason_about_goal(goal)
            )

            self.state.set(
                AutonomousState.REPORT
            )

            self.logger.success(
                "Autonomous reasoning completed.",
                {
                    "goal": goal,
                },
            )

            return {
                "success": True,
                "goal": goal,
                "observation": observation,
                "reasoning": reasoning,
            }

        except Exception as error:

            self.state.set(
                AutonomousState.ERROR
            )

            self.logger.error(
                "Autonomous goal failed.",
                {
                    "goal": goal,
                    "error": str(error),
                },
            )

            return {
                "success": False,
                "error": str(error),
            }

        finally:

            self.state.set(
                AutonomousState.IDLE
            )