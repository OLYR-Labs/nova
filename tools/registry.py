from pathlib import Path


class ToolRegistry:
    """
    Central registry for NOVA tools.

    Tools are explicitly registered.
    Nothing is executable unless it exists in this registry.
    """

    def __init__(
        self,
        sandbox_root=None,
    ):

        self.tools = {}

        if sandbox_root is None:
            self.sandbox_root = None
        else:
            self.sandbox_root = Path(
                sandbox_root
            ).resolve()

    # =========================================================
    # REGISTER
    # =========================================================

    def register(
        self,
        name,
        function,
        description="",
    ):

        name = str(
            name
        ).strip()

        if not name:
            raise ValueError(
                "Tool name cannot be empty."
            )

        if not callable(function):
            raise TypeError(
                f"Tool '{name}' must be callable."
            )

        self.tools[name] = {
            "function": function,
            "description": str(
                description or ""
            ).strip(),
        }

    # =========================================================
    # GET
    # =========================================================

    def get(
        self,
        name,
    ):

        return self.tools.get(
            name
        )

    # =========================================================
    # LIST
    # =========================================================

    def list(self):

        return {
            name: data["description"]
            for name, data
            in self.tools.items()
        }

    # =========================================================
    # EXECUTE
    # =========================================================

    def execute(
        self,
        name,
        **kwargs,
    ):

        tool = self.tools.get(
            name
        )

        if not tool:

            raise ValueError(
                f"Unknown tool: {name}"
            )

        return tool["function"](
            **kwargs
        )

    # =========================================================
    # NAMES
    # =========================================================

    def names(self):

        return list(
            self.tools.keys()
        )

    # =========================================================
    # HAS
    # =========================================================

    def has(
        self,
        name,
    ):

        return name in self.tools

    # =========================================================
    # SANDBOX
    # =========================================================

    def get_sandbox_root(self):

        return self.sandbox_root

    # =========================================================
    # COUNT
    # =========================================================

    def count(self):

        return len(
            self.tools
        )