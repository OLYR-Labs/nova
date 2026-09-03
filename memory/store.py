import json
from pathlib import Path


class MemoryStore:
    """
    Persistent local memory for NOVA.

    Data is stored locally on the computer.
    """

    def __init__(self, filename: str = "memory/data.json"):
        self.path = Path(filename)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.path.exists():
            self._save({})

    def _load(self) -> dict:
        try:
            with open(
                self.path,
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)

        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save(self, data: dict):
        with open(
            self.path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )

    def set(self, key: str, value):
        data = self._load()

        data[key] = value

        self._save(data)

    def get(self, key: str, default=None):
        data = self._load()

        return data.get(
            key,
            default,
        )

    def delete(self, key: str):
        data = self._load()

        if key in data:
            del data[key]

            self._save(data)

    def all(self) -> dict:
        return self._load()

    def clear(self):
        self._save({})