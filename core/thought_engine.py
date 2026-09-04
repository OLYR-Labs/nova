from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ThoughtState:
    """Structured cognitive state for autonomous work.

    This is deliberately a summary/state model, not a dump of hidden model
    chain-of-thought. It gives NOVA durable, inspectable working memory.
    """

    goal: str | None = None
    phase: str = "IDLE"
    current_step: str | None = None
    observations: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    tests: list[dict] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    next_action: str | None = None
    confidence: float = 0.0
    attempt: int = 0
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ThoughtEngine:
    """Thread-safe cognitive-state manager with JSON persistence."""

    def __init__(self, root: str | Path, max_items: int = 50):
        self.root = Path(root).resolve()
        self.path = self.root / ".nova" / "cognitive_state.json"
        self.max_items = max(1, int(max_items))
        self.lock = threading.RLock()
        self.state = ThoughtState()
        self._load()

    def _trim(self, values: list):
        if len(values) > self.max_items:
            del values[:-self.max_items]

    def _touch(self):
        self.state.updated_at = datetime.now().isoformat()

    def _load(self):
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                known = {field.name for field in self.__dataclass_fields__.values()}
                self.state = ThoughtState(**{k: v for k, v in data.items() if k in known})
        except Exception:
            self.state = ThoughtState()

    def persist(self):
        with self.lock:
            self._touch()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(asdict(self.state), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(self.path)

    def reset(self, goal: str):
        with self.lock:
            self.state = ThoughtState(goal=str(goal).strip(), phase="OBSERVE")
            self.persist()

    def transition(self, phase: str, step: str | None = None, next_action: str | None = None):
        with self.lock:
            self.state.phase = str(phase).upper()
            self.state.current_step = step
            self.state.next_action = next_action
            self.persist()

    def observe(self, value: str):
        with self.lock:
            value = str(value).strip()
            if value:
                self.state.observations.append(value)
                self._trim(self.state.observations)
                self.persist()

    def decide(self, value: str, confidence: float | None = None):
        with self.lock:
            value = str(value).strip()
            if value:
                self.state.decisions.append(value)
                self._trim(self.state.decisions)
            if confidence is not None:
                self.state.confidence = max(0.0, min(1.0, float(confidence)))
            self.persist()

    def hypothesize(self, value: str):
        with self.lock:
            value = str(value).strip()
            if value:
                self.state.hypotheses.append(value)
                self._trim(self.state.hypotheses)
                self.persist()

    def record_failure(self, failure: dict | str):
        with self.lock:
            self.state.failures.append(failure if isinstance(failure, dict) else {"error": str(failure)})
            self._trim(self.state.failures)
            self.persist()

    def record_test(self, result: dict):
        with self.lock:
            self.state.tests.append(dict(result))
            self._trim(self.state.tests)
            self.persist()

    def learn(self, lesson: str):
        with self.lock:
            lesson = str(lesson).strip()
            if lesson:
                self.state.lessons.append(lesson)
                self._trim(self.state.lessons)
                self.persist()

    def begin_attempt(self, attempt: int):
        with self.lock:
            self.state.attempt = int(attempt)
            self.persist()

    def snapshot(self) -> dict:
        with self.lock:
            return asdict(self.state)
