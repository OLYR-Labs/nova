import subprocess


class ShellTool:

    def __init__(self, timeout=60):
        self.timeout = timeout

    def execute(self, command: str):

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }