import subprocess


class ProcessManager:

    def __init__(self):
        self.processes = {}

    def start(self, name, command):

        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.processes[name] = process

        return process.pid

    def stop(self, name):

        process = self.processes.get(name)

        if not process:
            return False

        process.terminate()

        del self.processes[name]

        return True

    def status(self, name):

        process = self.processes.get(name)

        if not process:
            return None

        return process.poll()

    def all(self):

        return {
            name: process.poll()
            for name, process in self.processes.items()
        }