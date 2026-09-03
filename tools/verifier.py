from pathlib import Path


class VerificationEngine:

    def __init__(
        self,
        executor,
        project_root=".",
    ):
        self.executor = executor
        self.project_root = Path(
            project_root
        )

    # =========================================================
    # PYTHON COMPILE
    # =========================================================

    def compile_python(
        self,
        filename: str,
    ):

        path = Path(filename)

        if not path.exists():
            return {
                "success": False,
                "error": (
                    f"File does not exist: "
                    f"{filename}"
                ),
            }

        command = (
            f'python -m py_compile "{filename}"'
        )

        return self.executor.run(
            command,
            autonomous=True,
        )

    # =========================================================
    # IMPORT TEST
    # =========================================================

    def import_module(
        self,
        module: str,
    ):

        command = (
            f'python -c '
            f'"import {module}; '
            f'print(\'IMPORT_OK\')"'
        )

        return self.executor.run(
            command,
            autonomous=True,
        )

    # =========================================================
    # GIT DIFF
    # =========================================================

    def git_diff(self):

        return self.executor.run(
            "git diff",
            autonomous=True,
        )

    # =========================================================
    # GIT STATUS
    # =========================================================

    def git_status(self):

        return self.executor.run(
            "git status --short",
            autonomous=True,
        )

    # =========================================================
    # GENERIC
    # =========================================================

    def verify_command(
        self,
        command: str,
    ):

        return self.executor.run(
            command,
            autonomous=True,
        )