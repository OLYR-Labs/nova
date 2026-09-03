class Router:

    def __init__(self, local_provider, groq_provider):
        self.local = local_provider
        self.groq = groq_provider

    def select(self, mode: str):

        mode = mode.lower()

        if mode == "local":
            return self.local

        if mode == "groq":
            return self.groq

        if mode == "auto":
            return self.groq

        raise ValueError(
            f"Unknown mode: {mode}"
        )