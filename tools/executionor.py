import subprocess
from typing import Optional


class ToolExecutor:
    """
    Controlled command execution.

    This is deliberately separated from the autonomous
    reasoning layer so permissions can be enforced before
    anything reaches the operating system.
    """

    def __init__(
        self,
        permissions,
        timeout: int = 30,
    ):
        self.permissions = permissions
        self.timeout = timeout

    def run(
        self,
        command: str,
        autonomous: bool = False,
    ):

        command = str(command).strip()

        if not command:
            return {
                "success": False,
                "output": "",
                "error": "Empty command.",
            }

        if not self.permissions.allow(
            command,
            autonomous=autonomous,
        ):
            return {
                "success": False,
                "output": "",
                "error": (
                    "Command rejected by "
                    "NOVA permission policy."
                ),
            }

        try:

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            return {
                "success": (
                    result.returncode == 0
                ),
                "output": result.stdout.strip(),
                "error": result.stderr.strip(),
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:

            return {
                "success": False,
                "output": "",
                "error": (
                    "Command timed out."
                ),
            }

        except Exception as error:

            return {
                "success": False,
                "output": "",
                "error": str(error),
            }