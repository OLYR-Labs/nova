from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class Goal:
    id: str

    description: str

    status: str = "pending"

    priority: int = 5

    created_at: str = field(
        default_factory=lambda:
        datetime.now().isoformat()
    )

    steps: list[str] = field(
        default_factory=list
    )

    completed_steps: list[str] = field(
        default_factory=list
    )


class GoalManager:

    def __init__(self):
        self.goals = []

    # =========================================================
    # CREATE
    # =========================================================

    def create(
        self,
        description,
        priority=5,
        steps=None,
    ):

        description = str(
            description or ""
        ).strip()

        if not description:
            raise ValueError(
                "Goal description cannot be empty."
            )

        goal = Goal(
            id=str(uuid.uuid4()),
            description=description,
            priority=priority,
            steps=list(steps or []),
        )

        self.goals.append(goal)

        return goal

    # =========================================================
    # ADD
    #
    # Backward-compatible alias used by AutonomousAgent.
    # =========================================================

    def add(
        self,
        description,
        priority=5,
        steps=None,
    ):

        return self.create(
            description=description,
            priority=priority,
            steps=steps,
        )

    # =========================================================
    # GET ALL
    # =========================================================

    def get_goals(self):

        return [
            goal.description
            for goal in self.goals
            if goal.status == "pending"
        ]

    # =========================================================
    # ACTIVE
    # =========================================================

    def active(self):

        return [
            goal
            for goal in self.goals
            if goal.status == "pending"
        ]

    # =========================================================
    # GET
    # =========================================================

    def get(
        self,
        goal_id,
    ):

        for goal in self.goals:

            if goal.id == goal_id:
                return goal

        return None

    # =========================================================
    # COMPLETE STEP
    # =========================================================

    def complete_step(
        self,
        goal_id,
        step,
    ):

        for goal in self.goals:

            if goal.id != goal_id:
                continue

            if step not in goal.completed_steps:

                goal.completed_steps.append(
                    step
                )

            if (
                goal.steps
                and
                len(goal.completed_steps)
                >= len(goal.steps)
            ):

                goal.status = "completed"

            return True

        return False

    # =========================================================
    # COMPLETE GOAL
    # =========================================================

    def complete(
        self,
        goal_id,
    ):

        goal = self.get(
            goal_id
        )

        if goal is None:
            return False

        goal.status = "completed"

        return True

    # =========================================================
    # NEXT GOAL
    # =========================================================

    def next_goal(self):

        active = self.active()

        if not active:
            return None

        return sorted(
            active,
            key=lambda goal: goal.priority,
        )[0]

    # =========================================================
    # REMOVE
    # =========================================================

    def remove(
        self,
        goal_id,
    ):

        for index, goal in enumerate(
            self.goals
        ):

            if goal.id == goal_id:

                self.goals.pop(
                    index
                )

                return True

        return False

    # =========================================================
    # CLEAR
    # =========================================================

    def clear(self):

        self.goals.clear()

    # =========================================================
    # COUNT
    # =========================================================

    def count(self):

        return len(
            self.goals
        )