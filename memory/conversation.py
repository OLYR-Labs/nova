class ConversationMemory:
    """
    Short-term conversation memory for NOVA.

    Stores the messages exchanged during the current session.
    """

    def __init__(self, max_messages: int = 40):
        self.max_messages = max_messages
        self.messages: list[dict] = []

    def add_user_message(self, content: str):
        self.messages.append({
            "role": "user",
            "content": content,
        })

        self._trim()

    def add_assistant_message(self, content: str):
        self.messages.append({
            "role": "assistant",
            "content": content,
        })

        self._trim()

    def add_system_message(self, content: str):
        self.messages.append({
            "role": "system",
            "content": content,
        })

        self._trim()

    def get_messages(self) -> list[dict]:
        return list(self.messages)

    def clear(self):
        self.messages.clear()

    def last_message(self):
        if not self.messages:
            return None

        return self.messages[-1]

    def _trim(self):
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]