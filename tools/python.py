import subprocess
import sys


class PythonTool:

    def __init__(self, timeout=60):
        self.timeout = timeout

    def execute(self, code: str):

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }