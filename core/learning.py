class LearningEngine:

    def __init__(
        self,
        memory,
        logger,
    ):
        self.memory = memory
        self.logger = logger

    def learn(
        self,
        task: str,
        reflection: str,
    ):

        if not reflection:
            return False

        procedure = {
            "reflection": reflection,
        }

        saved = self.memory.learn_procedure(
            task,
            procedure,
        )

        if saved:

            self.logger.success(
                "New procedure learned.",
                {
                    "task": task,
                },
            )

        return saved