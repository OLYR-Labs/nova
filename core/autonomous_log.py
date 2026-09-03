import json
from datetime import datetime
from pathlib import Path


class AutonomousLogger:

    def __init__(
        self,
        filename: str = "logs/autonomous.jsonl",
    ):
        self.path = Path(filename)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def log(
        self,
        event_type: str,
        message: str,
        data=None,
    ):

        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "message": message,
            "data": data or {},
        }

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                )
                + "\n"
            )

    def info(
        self,
        message: str,
        data=None,
    ):
        self.log(
            "info",
            message,
            data,
        )

    def error(
        self,
        message: str,
        data=None,
    ):
        self.log(
            "error",
            message,
            data,
        )

    def success(
        self,
        message: str,
        data=None,
    ):
        self.log(
            "success",
            message,
            data,
        )