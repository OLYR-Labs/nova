from abc import ABC, abstractmethod


class AIProvider(ABC):

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        raise NotImplementedError