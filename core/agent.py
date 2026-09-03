import hashlib
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from pathlib import Path

from ollama import generate

from core.state import AgentState
from core.goals import GoalManager
from core.safe_workspace import SafeWorkspace
from tools.registry import ToolRegistry

from voice.voice import NOVAVoice


class AgentSessionState:

    def __init__(self):

        self.read_files = {}

        self.current_phase = "INSPECT"

        self.active_plan = None

    # =========================================================
    # READ LEDGER
    # =========================================================

    def record_read(
        self,
        path,
        content,
    ):

        file_hash = hashlib.sha256(
            content.encode(
                "utf-8"
            )
        ).hexdigest()

        self.read_files[path] = file_hash

    def has_read(
        self,
        path,
    ):

        return path in self.read_files

    def get_hash(
        self,
        path,
    ):

        return self.read_files.get(
            path
        )

    def clear(self):

        self.read_files.clear()

        self.active_plan = None

        self.current_phase = "INSPECT"


class AutonomousAgent:

    def __init__(
        self,
        orchestrator,
        interval=10,
        sandbox_root=r"D:\AI\NOVA",
    ):

        self.orchestrator = orchestrator

        self.interval = interval

        self.state = AgentState()

        self.goals = GoalManager()

        self.session = (
            AgentSessionState()
        )

        self.tools = ToolRegistry(
            sandbox_root
        )

        self.workspace = SafeWorkspace(
            sandbox_root
        )

        # =====================================================
        # NOVA VOICE
        # =====================================================

        self.voice = NOVAVoice(
            state=self.state,
            model_name=(
                "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
            ),
            speaker="Ryan",
            device="cuda:0",
            output_dir=r"D:\AI\NOVA\data\voice",
        )

        # -----------------------------------------------------
        # REGISTER TOOLS
        # -----------------------------------------------------

        self.register_builtin_tools()

        # -----------------------------------------------------
        # CODING MODEL
        # -----------------------------------------------------

        self.model = "qwen3-coder:30b"

        self.max_operations_per_cycle = 8

        self.operation_history = []

        self.current_process = None

    # =========================================================
    # GOALS
    # =========================================================

    def add_goal(
        self,
        goal,
    ):

        goal = str(
            goal or ""
        ).strip()

        if not goal:
            return None

        created = self.goals.create(
            description=goal
        )

        self.state.set_goal(
            goal
        )

        print(
            f"\n[NOVA AGENT] Goal added: {goal}"
        )

        return created

    # =========================================================
    # OBSERVE
    # =========================================================

    def observe(self):

        self.state.observe(
            f"Cycle: {self.state.cycle}"
        )

        self.state.observe(
            f"Goals: {self.goals.get_goals()}"
        )

    # =========================================================
    # THINK
    # =========================================================

    def think(self):

        goal = self.goals.next_goal()

        if goal is None:
            return None

        self.state.set_goal(
            goal.description
        )

        self.state.set_task(
            "autonomous software development"
        )

        return goal.description

    # =========================================================
    # SYSTEM PROMPT
    # =========================================================

    def build_system_prompt(self):

        return r"""
You are NOVA, an autonomous software-development agent.

You operate inside:

D:\AI\NOVA

You are responsible for diagnosing and repairing software.

You can create new modules, folders and files when required.

============================================================
CORE DEVELOPMENT LOOP
============================================================

INSPECT
  ↓
UNDERSTAND
  ↓
PLAN
  ↓
MODIFY
  ↓
VERIFY
  ↓
TEST
  ↓
REPAIR IF NECESSARY
  ↓
DONE

Do NOT assume that every problem can be solved by editing
an existing file.

If a missing module is required:

1. Create the directory.
2. Create the module.
3. Connect it to the appropriate existing module.
4. Verify syntax.
5. Test the result.

============================================================
AVAILABLE COMMANDS
============================================================

LIST FILES

<nova_cmd>
<action>LIST_FILES</action>
<path>.</path>
</nova_cmd>


READ FILE

<nova_cmd>
<action>READ_FILE</action>
<path>core/example.py</path>
</nova_cmd>


WRITE FILE

<nova_cmd>
<action>WRITE_FILE</action>
<path>core/new_module.py</path>
<content>
complete file contents
</content>
</nova_cmd>


EDIT FILE

<nova_cmd>
<action>EDIT_FILE</action>
<path>core/example.py</path>
<old_text>
exact existing text
</old_text>
<new_text>
replacement text
</new_text>
</nova_cmd>


PYTHON CHECK

<nova_cmd>
<action>CHECK_PYTHON</action>
<path>core/example.py</path>
</nova_cmd>


TEST

<nova_cmd>
<action>TEST_PROJECT</action>
</nova_cmd>


DONE

<nova_cmd>
<action>DONE</action>
</nova_cmd>

============================================================
RULES
============================================================

1. Inspect before modifying.

2. Never invent existing file contents.

3. Before editing an existing file, READ_FILE first.

4. WRITE_FILE may create a new file.

5. WRITE_FILE automatically creates missing directories.

6. When a missing module is needed, CREATE IT.

7. After creating a module, connect it to the application.

8. Use CHECK_PYTHON after Python modifications.

9. Use TEST_PROJECT after meaningful changes.

10. If a test fails, inspect the failure and repair it.

11. Never repeat an already successful operation.

12. Never claim that something was executed unless the
controller returned an actual result.

13. Work incrementally.

14. One command per response.

15. Your response MUST contain exactly one
<nova_cmd>...</nova_cmd> block.

============================================================
SAFE STOP
============================================================

The controller may request a shutdown while you are working.

A stop request means:

FINISH CURRENT SAFE FILESYSTEM OPERATION
→ SAVE STATE
→ DO NOT START ANOTHER OPERATION
→ STOP

Never intentionally corrupt or partially write a file because
the runtime is stopping.
"""

    # =========================================================
    # PROMPT
    # =========================================================

    def build_prompt(
        self,
        goal,
        previous_result=None,
    ):

        prompt = (
            self.build_system_prompt()
            + "\n\n"
        )

        prompt += (
            "============================================================\n"
            "CURRENT TASK\n"
            "============================================================\n\n"
        )

        prompt += (
            f"GOAL:\n{goal}\n\n"
        )

        prompt += (
            f"CYCLE:\n{self.state.cycle}\n\n"
        )

        prompt += (
            "PHASE:\n"
            f"{self.session.current_phase}\n\n"
        )

        prompt += (
            "AVAILABLE TOOLS:\n"
        )

        for name in self.tools.names():

            prompt += (
                f"- {name}\n"
            )

        prompt += "\n"

        if self.session.read_files:

            prompt += (
                "FILES READ:\n"
            )

            for path, file_hash in (
                self.session.read_files.items()
            ):

                prompt += (
                    f"- {path} "
                    f"(sha256={file_hash})\n"
                )

            prompt += "\n"

        if self.operation_history:

            prompt += (
                "OPERATIONS ALREADY EXECUTED:\n"
            )

            for operation in (
                self.operation_history
            ):

                prompt += (
                    f"- {operation}\n"
                )

            prompt += "\n"

        observations = (
            self.state.get_observations(
                limit=10
            )
        )

        if observations:

            prompt += (
                "RECENT OBSERVATIONS:\n"
            )

            for observation in observations:

                prompt += (
                    f"- {observation}\n"
                )

            prompt += "\n"

        if previous_result is not None:

            prompt += (
                "ACTUAL CONTROLLER RESULT:\n"
            )

            try:

                prompt += json.dumps(
                    previous_result,
                    indent=2,
                    ensure_ascii=False,
                )

            except Exception:

                prompt += str(
                    previous_result
                )

            prompt += "\n\n"

            prompt += (
                "Use the actual result above.\n"
                "Do not repeat the completed operation.\n\n"
            )

        if self.state.should_stop():

            prompt += (
                "SAFE STOP REQUESTED.\n"
                "Do not start another operation.\n"
                "Return DONE.\n\n"
            )

        prompt += (
            "Return exactly ONE "
            "<nova_cmd>...</nova_cmd> command."
        )

        return prompt

    # =========================================================
    # MODEL
    # =========================================================

    def call_model(
        self,
        prompt,
        think=True,
    ):

        try:

            response = generate(
                model=self.model,
                prompt=prompt,
                think=think,
                options={
                    "temperature": 0.1,
                },
            )

        except Exception as error:

            print(
                "\n[NOVA MODEL ERROR]"
            )

            print(error)

            return {
                "content": "",
                "thinking": "",
                "tool_calls": None,
                "error": str(error),
            }

        content = (
            getattr(
                response,
                "response",
                "",
            )
            or ""
        ).strip()

        thinking = (
            getattr(
                response,
                "thinking",
                "",
            )
            or ""
        ).strip()

        tool_calls = getattr(
            response,
            "tool_calls",
            None,
        )

        return {
            "content": content,
            "thinking": thinking,
            "tool_calls": tool_calls,
        }

    # =========================================================
    # PARSE
    # =========================================================

    def parse_operation(
        self,
        response,
    ):

        if not response:
            return None

        match = re.search(
            r"<nova_cmd>\s*(.*?)\s*</nova_cmd>",
            response,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        if not match:
            return None

        command_body = (
            match.group(1)
            .strip()
        )

        if not command_body:
            return None

        try:

            root = ET.fromstring(
                f"<nova_cmd>{command_body}</nova_cmd>"
            )

            action_elem = root.find(
                "action"
            )

            if (
                action_elem is None
                or not action_elem.text
            ):
                return None

            action = (
                action_elem.text
                .strip()
                .lower()
            )

            if action == "done":

                return {
                    "action": "done"
                }

            path_elem = root.find(
                "path"
            )

            path = (
                path_elem.text.strip()
                if (
                    path_elem is not None
                    and path_elem.text
                )
                else "."
            )

            if action == "list_files":

                return {
                    "action": "tool",
                    "name": "list_files",
                    "arguments": {
                        "path": path
                    },
                }

            if action == "read_file":

                return {
                    "action": "tool",
                    "name": "read_file",
                    "arguments": {
                        "path": path
                    },
                }

            if action == "write_file":

                content_elem = root.find(
                    "content"
                )

                content = (
                    content_elem.text
                    if (
                        content_elem is not None
                        and content_elem.text is not None
                    )
                    else ""
                )

                return {
                    "action": "tool",
                    "name": "write_file",
                    "arguments": {
                        "path": path,
                        "content": content,
                    },
                }

            if action == "edit_file":

                old_elem = root.find(
                    "old_text"
                )

                new_elem = root.find(
                    "new_text"
                )

                old_text = (
                    old_elem.text
                    if (
                        old_elem is not None
                        and old_elem.text is not None
                    )
                    else ""
                )

                new_text = (
                    new_elem.text
                    if (
                        new_elem is not None
                        and new_elem.text is not None
                    )
                    else ""
                )

                if not old_text:
                    return None

                return {
                    "action": "tool",
                    "name": "edit_file",
                    "arguments": {
                        "path": path,
                        "old_text": old_text,
                        "new_text": new_text,
                    },
                }

            if action == "check_python":

                return {
                    "action": "tool",
                    "name": "check_python",
                    "arguments": {
                        "path": path
                    },
                }

            if action == "test_project":

                return {
                    "action": "tool",
                    "name": "test_project",
                    "arguments": {},
                }

        except ET.ParseError:

            return None

        return None

    # =========================================================
    # EXECUTE
    # =========================================================

    def execute_tool(
        self,
        operation,
    ):

        name = operation.get(
            "name"
        )

        arguments = operation.get(
            "arguments",
            {},
        )

        if name not in self.tools.names():

            return {
                "success": False,
                "error": (
                    f"Unknown operation: {name}"
                ),
            }

        self.state.set_operation(
            name
        )

        print(
            f"\n[NOVA TOOL] {name}"
        )

        try:

            result = self.tools.execute(
                name,
                **arguments,
            )

            self.state.observe(
                f"{name}: {result}"
            )

            return result

        except Exception as error:

            result = {
                "success": False,
                "error": str(error),
            }

            self.state.observe(
                f"{name} failed: {error}"
            )

            return result

        finally:

            self.state.set_operation(
                None
            )

    # =========================================================
    # OPERATION SIGNATURE
    # =========================================================

    def operation_signature(
        self,
        operation,
    ):

        name = operation.get(
            "name"
        )

        arguments = operation.get(
            "arguments",
            {},
        )

        try:

            serialized = json.dumps(
                arguments,
                sort_keys=True,
                ensure_ascii=False,
            )

        except Exception:

            serialized = str(
                arguments
            )

        return (
            f"{name}:{serialized}"
        )

    # =========================================================
    # AUTONOMOUS EXECUTION
    # =========================================================

    def execute(
        self,
        goal,
    ):

        previous_result = None

        self.operation_history = []

        self.session.clear()

        self.state.set_autonomous(
            True
        )

        self.state.set_goal(
            goal
        )

        try:

            for operation_number in range(
                1,
                self.max_operations_per_cycle + 1,
            ):

                # -------------------------------------------------
                # SAFE STOP BEFORE NEW OPERATION
                # -------------------------------------------------

                if self.state.should_stop():

                    print(
                        "\n[NOVA] Safe stop requested."
                    )

                    print(
                        "[NOVA] No new operation will start."
                    )

                    break

                print(
                    f"\n[NOVA] Operation "
                    f"{operation_number}/"
                    f"{self.max_operations_per_cycle}"
                )

                prompt = self.build_prompt(
                    goal=goal,
                    previous_result=previous_result,
                )

                model_result = self.call_model(
                    prompt
                )

                content = model_result.get(
                    "content",
                    "",
                )

                print(
                    "\n[NOVA DECISION]"
                )

                print(content)

                operation = (
                    self.parse_operation(
                        content
                    )
                )

                if operation is None:

                    print(
                        "\n[NOVA] Invalid command."
                    )

                    self.speak(
                        "I received an invalid command from my coding model.",
                        blocking=False,
                    )

                    return

                if operation[
                    "action"
                ] == "done":

                    print(
                        "\n[NOVA DONE]"
                    )

                    self.speak(
                        "Task completed.",
                        blocking=False,
                    )

                    break

                signature = (
                    self.operation_signature(
                        operation
                    )
                )

                if signature in (
                    self.operation_history
                ):

                    print(
                        "\n[NOVA] Duplicate operation."
                    )

                    self.speak(
                        "I detected a repeated operation, so I stopped safely.",
                        blocking=False,
                    )

                    break

                self.operation_history.append(
                    signature
                )

                # -------------------------------------------------
                # EXECUTE EXACTLY ONE SAFE OPERATION
                # -------------------------------------------------

                result = self.execute_tool(
                    operation
                )

                previous_result = result

                # -------------------------------------------------
                # AFTER OPERATION
                # -------------------------------------------------

                if self.state.should_stop():

                    print(
                        "\n[NOVA] Current operation completed."
                    )

                    print(
                        "[NOVA] Entering safe stopping state."
                    )

                    break

        finally:

            self.state.set_autonomous(
                False
            )

            self.state.set_task(
                None
            )

            self.state.set_operation(
                None
            )

    # =========================================================
    # CYCLE
    # =========================================================

    def cycle_once(self):

        self.state.next_cycle()

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"NOVA AUTONOMOUS CYCLE "
            f"#{self.state.cycle}"
        )

        print(
            f"{'=' * 60}"
        )

        self.observe()

        goal = self.think()

        if goal is None:

            print(
                "\nNOVA: No active goals."
            )

            return

        print(
            f"\nGoal: {goal}"
        )

        self.execute(
            goal
        )

    # =========================================================
    # RUN
    # =========================================================

    def run(self):

        self.state.start()

        self.state.set_autonomous(
            True
        )

        print(
            "\n[NOVA] Autonomous mode started."
        )

        try:

            while (
                self.state.running
                and not self.state.should_stop()
            ):

                try:

                    self.cycle_once()

                    # Don't start another cycle after stop.
                    if self.state.should_stop():
                        break

                    for _ in range(
                        self.interval
                    ):

                        if self.state.should_stop():
                            break

                        time.sleep(
                            1
                        )

                except KeyboardInterrupt:

                    self.request_safe_stop(
                        "Keyboard interrupt."
                    )

                except Exception as error:

                    print(
                        "\n[NOVA ERROR]"
                    )

                    print(error)

                    self.state.observe(
                        f"Runtime error: {error}"
                    )

                    if self.state.should_stop():
                        break

                    time.sleep(
                        self.interval
                    )

        finally:

            self.state.set_autonomous(
                False
            )

            self.state.mark_stopped()

            print(
                "\n[NOVA] Runtime safely stopped."
            )

    # =========================================================
    # SAFE STOP
    # =========================================================

    def request_safe_stop(
        self,
        reason="User requested shutdown.",
    ):

        print(
            "\n[NOVA] SAFE STOP REQUESTED"
        )

        self.state.request_stop(
            reason
        )

        self.state.observe(
            f"Safe stop requested: {reason}"
        )

        # We deliberately do NOT kill a filesystem operation.
        #
        # execute_tool() completes its current atomic operation,
        # then execute() sees stop_requested and exits.

    # =========================================================
    # VOICE
    # =========================================================

    def speak(
        self,
        text: str,
        blocking: bool = False,
    ):
        """
        Make NOVA speak using the local Qwen3-TTS engine.

        blocking=False keeps the autonomous agent responsive
        while speech generation and playback happen in a
        background thread.
        """

        if not text:
            return None

        try:

            return self.voice.speak(
                text=text,
                blocking=blocking,
            )

        except Exception as error:

            print(
                f"\n[NOVA VOICE ERROR] {error}"
            )

            self.state.observe(
                f"Voice error: {error}"
            )

            return {
                "success": False,
                "error": str(error),
            }

    # =========================================================
    # BUILT-IN TOOLS
    # =========================================================

    def register_builtin_tools(self):

        self.tools.register(
            "list_files",
            self._list_files,
            "List files/directories.",
        )

        self.tools.register(
            "read_file",
            self._read_file,
            "Read UTF-8 text file.",
        )

        self.tools.register(
            "write_file",
            self._write_file,
            "Create or replace a UTF-8 file.",
        )

        self.tools.register(
            "edit_file",
            self._edit_file,
            "Perform exact safe text replacement.",
        )

        self.tools.register(
            "check_python",
            self._check_python,
            "Compile-check a Python file.",
        )

        self.tools.register(
            "test_project",
            self._test_project,
            "Run the project's controlled test suite.",
        )

        print(
            "\n[NOVA TOOLS]"
        )

        for name in self.tools.names():

            print(
                f"  - {name}"
            )

    # =========================================================
    # SAFE PATH
    # =========================================================

    def _safe_path(
        self,
        path,
    ):

        return self.workspace.safe_path(
            path
        )

    # =========================================================
    # LIST
    # =========================================================

    def _list_files(
        self,
        path=".",
    ):

        target = self._safe_path(
            path
        )

        if not target.exists():

            return {
                "success": False,
                "error": (
                    f"Path does not exist: {path}"
                ),
            }

        if not target.is_dir():

            return {
                "success": False,
                "error": (
                    f"Not a directory: {path}"
                ),
            }

        root = self.workspace.root

        results = []

        for item in sorted(
            target.iterdir(),
            key=lambda p: (
                not p.is_dir(),
                p.name.lower(),
            ),
        ):

            results.append(
                {
                    "name": item.name,
                    "path": str(
                        item.relative_to(
                            root
                        )
                    ),
                    "type": (
                        "directory"
                        if item.is_dir()
                        else "file"
                    ),
                }
            )

        return {
            "success": True,
            "items": results,
        }

    # =========================================================
    # READ
    # =========================================================

    def _read_file(
        self,
        path,
    ):

        target = self._safe_path(
            path
        )

        if not target.exists():

            return {
                "success": False,
                "error": (
                    f"File does not exist: {path}"
                ),
            }

        if not target.is_file():

            return {
                "success": False,
                "error": (
                    f"Not a file: {path}"
                ),
            }

        try:

            content = target.read_text(
                encoding="utf-8"
            )

            self.session.record_read(
                path=str(
                    target.relative_to(
                        self.workspace.root
                    )
                ),
                content=content,
            )

            return content

        except UnicodeDecodeError:

            return {
                "success": False,
                "error": (
                    "File is not UTF-8 text."
                ),
            }

    # =========================================================
    # WRITE
    # =========================================================

    def _write_file(
        self,
        path,
        content,
    ):

        return self.workspace.write_file(
            path,
            content,
        )

    # =========================================================
    # EDIT
    # =========================================================

    def _edit_file(
        self,
        path,
        old_text,
        new_text,
    ):

        return self.workspace.edit_file(
            path,
            old_text,
            new_text,
        )

    # =========================================================
    # PYTHON CHECK
    # =========================================================

    def _check_python(
        self,
        path,
    ):

        target = self._safe_path(
            path
        )

        if not target.exists():

            return {
                "success": False,
                "error": (
                    f"File does not exist: {path}"
                ),
            }

        if target.suffix.lower() != ".py":

            return {
                "success": False,
                "error": (
                    "CHECK_PYTHON requires a .py file."
                ),
            }

        try:

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "py_compile",
                    str(target),
                ],
                cwd=str(
                    self.workspace.root
                ),
                capture_output=True,
                text=True,
                timeout=30,
            )

            success = (
                result.returncode == 0
            )

            return {
                "success": success,
                "action": "check_python",
                "path": str(
                    target.relative_to(
                        self.workspace.root
                    )
                ),
                "returncode": (
                    result.returncode
                ),
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            }

        except subprocess.TimeoutExpired:

            return {
                "success": False,
                "error": (
                    "Python syntax check timed out."
                ),
            }

    # =========================================================
    # PROJECT TEST
    # =========================================================

    def _test_project(self):

        root = self.workspace.root

        tests_dir = root / "tests"

        if tests_dir.exists():

            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
            ]

        else:

            # Fallback: compile all Python files.
            command = [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                ".",
            ]

        try:

            self.current_process = subprocess.Popen(
                command,
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            while (
                self.current_process.poll()
                is None
            ):

                # Cooperative stop.
                if self.state.should_stop():

                    print(
                        "\n[NOVA TEST] "
                        "Stop requested."
                    )

                    # We do not instantly kill it.
                    #
                    # Give the test process a short grace
                    # period to exit naturally.
                    try:

                        self.current_process.wait(
                            timeout=3
                        )

                    except subprocess.TimeoutExpired:

                        # Only now terminate the controlled
                        # child process.
                        self.current_process.terminate()

                        try:

                            self.current_process.wait(
                                timeout=2
                            )

                        except subprocess.TimeoutExpired:

                            self.current_process.kill()

                            self.current_process.wait()

                    break

                time.sleep(
                    0.2
                )

            stdout, stderr = (
                self.current_process.communicate()
            )

            returncode = (
                self.current_process.returncode
            )

            return {
                "success": (
                    returncode == 0
                    and not self.state.should_stop()
                ),
                "action": "test_project",
                "returncode": returncode,
                "command": command,
                "stdout": stdout[-8000:],
                "stderr": stderr[-8000:],
                "stopped": self.state.should_stop(),
            }

        except Exception as error:

            return {
                "success": False,
                "error": str(error),
            }

        finally:

            self.current_process = None