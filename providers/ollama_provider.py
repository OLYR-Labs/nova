from ollama import chat

from providers.base import AIProvider


class OllamaProvider(AIProvider):

    def __init__(
        self,
        model: str = "gpt-oss:20b",
    ):
        self.model = model

    def chat(
        self,
        messages: list[dict],
    ) -> str:

        response = chat(
            model=self.model,
            messages=messages,
        )

        return response.message.content or ""