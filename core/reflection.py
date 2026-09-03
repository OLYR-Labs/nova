class ReflectionEngine:

    def __init__(
        self,
        nova,
        memory,
        logger,
    ):
        self.nova = nova
        self.memory = memory
        self.logger = logger

    def reflect(
        self,
        goal: str,
        result,
    ):

        prompt = f"""
You are NOVA's reflection subsystem.

Review the autonomous task below.

GOAL:
{goal}

RESULT:
{result}

Determine:

1. What worked?
2. What failed?
3. What should NOVA learn?
4. Is there a reusable procedure?
5. What should happen next?

Do not claim actions that did not happen.

Return concise structured reasoning.
"""

        try:

            response = self.nova.router.select(
                "auto"
            ).chat([
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": (
                        "Reflect on this task."
                    ),
                },
            ])

            self.logger.info(
                "Reflection completed.",
                {
                    "goal": goal,
                },
            )

            return response

        except Exception as error:

            self.logger.error(
                "Reflection failed.",
                {
                    "error": str(error),
                },
            )

            return None