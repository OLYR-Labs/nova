class WorkingMemory:

    def __init__(self, max_items=20):

        self.max_items = max_items
        self.items = []

    def add(self, item):

        self.items.append(item)

        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]

    def recent(self):

        return list(self.items)

    def clear(self):

        self.items.clear()

    def context(self):

        if not self.items:
            return ""

        return "\n".join(
            f"- {item}"
            for item in self.items
        )