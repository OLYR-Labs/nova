from dataclasses import dataclass, field
from typing import List
import uuid


@dataclass
class AutonomousTask:
    id: str
    goal: str
    description: str
    priority: int = 5
    status: str = "pending"
    commands: List[str] = field(
        default_factory=list
    )
    results: List[str] = field(
        default_factory=list
    )

    @classmethod
    def create(
        cls,
        goal: str,
        description: str,
        priority: int = 5,
    ):
        return cls(
            id=str(uuid.uuid4()),
            goal=goal,
            description=description,
            priority=priority,
        )

    def complete(self):
        self.status = "completed"

    def fail(self, reason: str = ""):
        self.status = "failed"

        if reason:
            self.results.append(
                reason
            )

    def add_command(
        self,
        command: str,
    ):
        self.commands.append(
            command
        )

    def add_result(
        self,
        result: str,
    ):
        self.results.append(
            result
        )