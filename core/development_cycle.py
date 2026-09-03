from core.autonomous_state import (
    AutonomousState,
)


class DevelopmentCycle:

    def __init__(
        self,
        state,
        logger,
        executor,
        verifier,
    ):
        self.state = state
        self.logger = logger
        self.executor = executor
        self.verifier = verifier

    # =========================================================
    # EXECUTE
    # =========================================================

    def execute(
        self,
        command: str,
    ):

        self.state.set(
            AutonomousState.EXECUTE
        )

        self.logger.info(
            "Executing autonomous command.",
            {
                "command": command,
            },
        )

        result = self.executor.run(
            command,
            autonomous=True,
        )

        return result

    # =========================================================
    # VERIFY
    # =========================================================

    def verify(
        self,
        command: str,
        result: dict,
    ):

        self.state.set(
            AutonomousState.VERIFY
        )

        if not result.get("success"):
            self.logger.error(
                "Execution failed.",
                result,
            )

            return False

        self.logger.info(
            "Execution completed successfully.",
            {
                "command": command,
            },
        )

        return True