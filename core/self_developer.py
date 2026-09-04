from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from ollama import generate

from core.safe_workspace import SafeWorkspace
from core.thought_engine import ThoughtEngine


class SelfDevelopmentEngine:
    """Bounded autonomous software-development loop for NOVA.

    Loop:
        observe -> plan -> modify -> verify -> diagnose -> repair -> reflect

    The model never receives unrestricted shell access. It proposes file
    changes; this controller validates paths, snapshots changes, applies them,
    runs bounded verification commands, and restores the snapshot on failure.
    """

    DEFAULT_MODEL = "qwen3-coder:30b"
    MAX_ATTEMPTS = 6
    COMMAND_TIMEOUT = 45

    BLOCKED_FILES = {
        ".env",
        ".env.local",
        ".env.production",
        "credentials.json",
        "secrets.json",
    }

    def __init__(self, root: str, model: str | None = None):
        self.root = Path(root).resolve()
        self.workspace = SafeWorkspace(self.root)
        self.thoughts = ThoughtEngine(self.root)
        self.model = model or os.getenv("NOVA_CODER_MODEL", self.DEFAULT_MODEL)
        self.last_plan: dict | None = None
        self.snapshot_root: Path | None = None

    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------

    def _allowed(self, path: str) -> bool:
        candidate = Path(path)
        name = candidate.name.lower()
        return name not in {item.lower() for item in self.BLOCKED_FILES}

    def _tree(self, limit: int = 250) -> list[str]:
        results = []
        blocked_dirs = {".git", ".venv", "venv", "__pycache__", ".nova_backups"}
        for item in sorted(self.root.rglob("*"), key=lambda p: str(p).lower()):
            try:
                relative = item.relative_to(self.root)
            except ValueError:
                continue
            if any(part in blocked_dirs for part in relative.parts):
                continue
            if item.is_file() and self._allowed(str(relative)):
                results.append(str(relative))
                if len(results) >= limit:
                    break
        return results

    def _read_files(self, paths: list[str]) -> dict[str, str]:
        output = {}
        for raw_path in paths[:20]:
            try:
                path = str(raw_path).replace("\\", "/")
                if not self._allowed(path):
                    continue
                target = self.workspace.safe_path(path)
                if target.is_file() and target.stat().st_size <= 150_000:
                    output[path] = target.read_text(encoding="utf-8")
            except Exception as error:
                output[str(raw_path)] = f"<READ ERROR: {error}>"
        return output

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------

    def _model(self, prompt: str) -> dict:
        response = generate(
            model=self.model,
            prompt=prompt,
            think=True,
            options={"temperature": 0.1},
        )
        content = (getattr(response, "response", "") or "").strip()
        thinking = (getattr(response, "thinking", "") or "").strip()
        return {"content": content, "thinking": thinking}

    @staticmethod
    def _json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            if text.endswith("```"):
                text = text[:-3]
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model did not return a JSON object.")
        value = json.loads(text[start:end + 1])
        if not isinstance(value, dict):
            raise ValueError("Model response is not a JSON object.")
        return value

    def _planning_prompt(self, goal: str, tree: list[str]) -> str:
        return f"""You are NOVA's software architect.

Goal:
{goal}

Project root:
{self.root}

Project files:
{json.dumps(tree, indent=2)}

Do not edit anything yet. Determine the smallest safe set of files that must
be inspected, the intended implementation, and how it should be verified.
Return JSON only:
{{
  "summary": "...",
  "files_to_read": ["relative/path.py"],
  "plan": ["step 1", "step 2"],
  "verification": ["python -m pytest -q"]
}}
Never request secrets, .env files, .git internals, or files outside the root.
"""

    def _coding_prompt(self, goal: str, plan: dict, files: dict[str, str], failure: dict | None) -> str:
        return f"""You are NOVA's autonomous coding engineer.

Goal:
{goal}

Architecture plan:
{json.dumps(plan, indent=2, ensure_ascii=False)}

Actual source files:
{json.dumps(files, indent=2, ensure_ascii=False)}

Previous verification failure (if any):
{json.dumps(failure or {}, indent=2, ensure_ascii=False)}

Return JSON only with this schema:
{{
  "summary": "what you are changing",
  "changes": [
    {{"path": "relative/file.py", "action": "create|replace", "content": "COMPLETE FILE CONTENT"}}
  ],
  "tests": ["python -m pytest -q"]
}}

Rules:
- Work only inside the project root.
- Never touch .env, secrets, credentials, .git, virtual environments, or keys.
- For action=replace, content must be the COMPLETE replacement file.
- Prefer the smallest change that solves the goal.
- Preserve existing behavior unless the goal requires changing it.
- Do not invent dependencies that are not already installed unless the goal
  explicitly requires a dependency.
- Tests must be non-destructive and bounded.
- Do not include markdown outside the JSON object.
"""

    # ------------------------------------------------------------------
    # SNAPSHOT / ROLLBACK
    # ------------------------------------------------------------------

    def _snapshot(self, paths: list[str]) -> Path:
        snapshot = self.root / ".nova" / "snapshots" / f"attempt_{self.thoughts.state.attempt}"
        snapshot.mkdir(parents=True, exist_ok=True)
        for raw_path in paths:
            path = self.workspace.safe_path(raw_path)
            if path.exists() and path.is_file():
                destination = snapshot / path.relative_to(self.root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        self.snapshot_root = snapshot
        return snapshot

    def _rollback(self, changed_paths: list[str]):
        if self.snapshot_root is None:
            return
        for raw_path in changed_paths:
            target = self.workspace.safe_path(raw_path)
            saved = self.snapshot_root / target.relative_to(self.root)
            if saved.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(saved, target)
            elif target.exists():
                target.unlink()

    # ------------------------------------------------------------------
    # APPLY / VERIFY
    # ------------------------------------------------------------------

    def _apply(self, changes: list[dict]) -> list[str]:
        changed = []
        for change in changes:
            path = str(change.get("path", "")).strip()
            action = str(change.get("action", "")).lower().strip()
            content = change.get("content")
            if not path or not self._allowed(path):
                raise PermissionError(f"Blocked or invalid path: {path}")
            if action not in {"create", "replace"}:
                raise ValueError(f"Unsupported change action: {action}")
            if not isinstance(content, str):
                raise ValueError(f"Missing complete file content for {path}")
            target = self.workspace.safe_path(path)
            if action == "create" and target.exists():
                raise FileExistsError(f"File already exists: {path}")
            self.workspace.write_file(path, content)
            changed.append(path)
        return changed

    def _run_command(self, command: str) -> dict:
        if not command or len(command) > 500:
            return {"success": False, "error": "Invalid verification command."}
        forbidden = ["&&", "||", ">", "<", "|", "git push", "git reset --hard", "del /", "rm -rf"]
        if any(token in command.lower() for token in forbidden):
            return {"success": False, "error": "Verification command contains a blocked shell construct."}
        try:
            completed = subprocess.run(
                command,
                cwd=self.root,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.COMMAND_TIMEOUT,
            )
            return {
                "success": completed.returncode == 0,
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-12_000:],
                "stderr": completed.stderr[-12_000:],
            }
        except subprocess.TimeoutExpired as error:
            return {"success": False, "command": command, "error": f"timeout: {error}"}
        except Exception as error:
            return {"success": False, "command": command, "error": str(error)}

    def _verify(self, commands: list[str]) -> dict:
        results = []
        for command in commands[:8]:
            result = self._run_command(command)
            results.append(result)
            self.thoughts.record_test(result)
            if not result.get("success"):
                return {"success": False, "results": results, "failure": result}
        return {"success": True, "results": results}

    # ------------------------------------------------------------------
    # PUBLIC LOOP
    # ------------------------------------------------------------------

    def run(self, goal: str, max_attempts: int | None = None) -> dict:
        goal = str(goal or "").strip()
        if not goal:
            return {"success": False, "error": "Goal cannot be empty."}

        attempts = max(1, min(int(max_attempts or self.MAX_ATTEMPTS), self.MAX_ATTEMPTS))
        self.thoughts.reset(goal)
        tree = self._tree()
        self.thoughts.observe(f"Discovered {len(tree)} project files.")
        self.thoughts.transition("PLAN", next_action="inspect relevant source files")

        try:
            plan = self._json(self._model(self._planning_prompt(goal, tree))["content"])
            self.last_plan = plan
            self.thoughts.decide(plan.get("summary", "Initial implementation plan"), 0.75)
            files = self._read_files(plan.get("files_to_read", []))
            self.thoughts.observe(f"Inspected {len(files)} files selected by the planner.")
        except Exception as error:
            self.thoughts.record_failure({"phase": "PLAN", "error": str(error)})
            self.thoughts.transition("FAILED")
            return {"success": False, "phase": "PLAN", "error": str(error), "thoughts": self.thoughts.snapshot()}

        failure = None
        changed_paths: list[str] = []

        for attempt in range(1, attempts + 1):
            self.thoughts.begin_attempt(attempt)
            self.thoughts.transition("CODE", next_action="generate and apply a minimal patch")
            try:
                coding = self._json(self._model(self._coding_prompt(goal, plan, files, failure))["content"])
                changes = coding.get("changes", [])
                if not isinstance(changes, list) or not changes:
                    raise ValueError("Coder returned no changes.")

                paths = [str(item.get("path", "")) for item in changes]
                self._snapshot(paths)
                self.thoughts.observe(f"Attempt {attempt}: snapshot created for {len(paths)} files.")
                changed_paths = self._apply(changes)

                self.thoughts.transition("VERIFY", next_action="run verification")
                verification = self._verify(coding.get("tests") or plan.get("verification") or ["python -m compileall -q ."])
                if verification["success"]:
                    self.thoughts.transition("REFLECT", next_action="record the successful lesson")
                    self.thoughts.learn(f"Goal succeeded on attempt {attempt}: {coding.get('summary', 'code change applied')}")
                    self.thoughts.transition("DONE")
                    return {
                        "success": True,
                        "goal": goal,
                        "attempts": attempt,
                        "changes": changed_paths,
                        "verification": verification,
                        "thoughts": self.thoughts.snapshot(),
                    }

                failure = verification.get("failure") or verification
                self.thoughts.record_failure(failure)
                self.thoughts.hypothesize("The implementation failed verification; diagnose the concrete test output and repair only the affected code.")
                self.thoughts.transition("DIAGNOSE", next_action="feed actual failure into repair pass")
                # Keep the failed changes in place so the next coder can inspect
                # the exact broken state. Roll back only after the final attempt.
                files = self._read_files(paths)
            except Exception as error:
                failure = {"error": str(error), "attempt": attempt}
                self.thoughts.record_failure(failure)
                self.thoughts.transition("DIAGNOSE", next_action="repair the failed operation")
                if changed_paths:
                    files = self._read_files(changed_paths)

        self.thoughts.transition("ROLLBACK", next_action="restore last known good snapshot")
        self._rollback(changed_paths)
        self.thoughts.learn("Autonomous change was rolled back because bounded verification did not pass.")
        self.thoughts.transition("FAILED")
        return {
            "success": False,
            "goal": goal,
            "attempts": attempts,
            "changes_rolled_back": changed_paths,
            "failure": failure,
            "thoughts": self.thoughts.snapshot(),
        }
