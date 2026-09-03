import re
import time
import json

from ollama import generate

from core.state import AgentState
from core.goals import GoalManager
from tools.registry import ToolRegistry


class AutonomousAgent:

    def __init__(
        self,
        orchestrator,
        interval: int = 10,
        sandbox_root: str = r"D:\AI\NOVA",
    ):

        self.orchestrator = orchestrator
        self.interval = interval

        self.state = AgentState()
        self.goals = GoalManager()

        # NOVA works directly inside its own project.
        self.tools = ToolRegistry(
            sandbox_root
        )

        self.model = "gpt-oss:20b"

        # Maximum real operations per autonomous cycle.
        self.max_operations_per_cycle = 5

        # Operations already executed during the current cycle.
        self.operation_history = []

    # =========================================================
    # GOALS
    # =========================================================

    def add_goal(self, goal: str):

        self.goals.add(goal)

        print(
            f"\n[NOVA AGENT] Goal added: {goal}"
        )

    # =========================================================
    # OBSERVATION
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

        goals = self.goals.get_goals()

        if not goals:
            return None

        return goals[
            (self.state.cycle - 1)
            % len(goals)
        ]

    # =========================================================
    # SYSTEM PROMPT
    # =========================================================

    def build_system_prompt(self):

        return r"""
You are NOVA, an autonomous software-development agent.

You operate inside your own project.

PROJECT ROOT:

D:\AI\NOVA

Your objective is to inspect, understand, improve, modify,
and test your own software.

The controller executes operations for you.

============================================================
CRITICAL MODEL RULES
============================================================

You are NOT connected to native tools.

NEVER call:

container.exec
shell
bash
powershell
python tools
external functions
native function calling

NEVER output:

JSON
Markdown
code fences
tool_call blocks
function calls

The ONLY executable instruction is:

<nova_cmd>...</nova_cmd>

Your visible response MUST contain exactly ONE command.

============================================================
AVAILABLE COMMANDS
============================================================

LIST_FILES <path>

READ_FILE <path>

WRITE_FILE <path> <content>

EDIT_FILE <path> OLD: <old_text> NEW: <new_text>

DONE

Examples:

<nova_cmd>LIST_FILES .</nova_cmd>

<nova_cmd>LIST_FILES core</nova_cmd>

<nova_cmd>READ_FILE core/agent.py</nova_cmd>

<nova_cmd>DONE</nova_cmd>

============================================================
DEVELOPMENT RULES
============================================================

1. Inspect before modifying.

2. Never invent file contents.

3. Never pretend an operation was executed.

4. Use the actual controller result.

5. Do not repeat an operation that already succeeded.

6. Before editing a file, read it.

7. After modifying code, verify the change.

8. Work incrementally.

9. Make useful progress toward the current goal.

10. Your command must appear in visible output.

IMPORTANT:

Your reasoning is private.

The controller can only execute commands that appear inside
<nova_cmd>...</nova_cmd> in your visible response.
"""

    # =========================================================
    # PROMPT
    # =========================================================

    def build_prompt(
        self,
        goal: str,
        previous_result=None,
    ):

        prompt = (
            self.build_system_prompt()
            + "\n\n"
        )

        prompt += (
            "============================================================\n"
            "CURRENT AUTONOMOUS TASK\n"
            "============================================================\n\n"
        )

        prompt += (
            f"GOAL:\n{goal}\n\n"
        )

        prompt += (
            f"CYCLE:\n{self.state.cycle}\n\n"
        )

        # -----------------------------------------------------
        # Previous operations
        # -----------------------------------------------------

        if self.operation_history:

            prompt += (
                "OPERATIONS ALREADY EXECUTED THIS CYCLE:\n"
            )

            for operation in self.operation_history:

                prompt += (
                    f"- {operation}\n"
                )

            prompt += "\n"

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------

        if self.state.observations:

            prompt += (
                "RECENT OBSERVATIONS:\n"
            )

            for observation in (
                self.state.observations[-10:]
            ):

                prompt += (
                    f"- {observation}\n"
                )

            prompt += "\n"

        # -----------------------------------------------------
        # Actual controller result
        # -----------------------------------------------------

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
                "The operation above has already been executed.\n"
                "Do NOT repeat it.\n"
                "Use its actual result to select the next operation.\n\n"
            )

        prompt += (
            "Now choose the next operation.\n"
            "Return exactly one <nova_cmd>...</nova_cmd> command."
        )

        return prompt

    # =========================================================
    # COMMAND RECOVERY PROMPT
    # =========================================================

    def build_command_recovery_prompt(
        self,
        goal: str,
        previous_result,
        thinking: str,
    ):

        """
        GPT-OSS sometimes finishes its reasoning without placing
        the command into visible output.

        This second request does NOT execute anything.

        It converts the model's previous reasoning + actual result
        into one explicit visible NOVA command.

        Only the XML command is subsequently parsed and executed.
        """

        prompt = r"""
You are NOVA's command formatter.

You are NOT an execution engine.

Do NOT call native tools.

Do NOT call container.exec.

Do NOT call shell commands.

Do NOT output JSON.

Do NOT output Markdown.

Your ONLY job is to output ONE executable NOVA command.

The command MUST be inside:

<nova_cmd>...</nova_cmd>

AVAILABLE COMMANDS:

LIST_FILES <path>

READ_FILE <path>

WRITE_FILE <path> <content>

EDIT_FILE <path> OLD: <old_text> NEW: <new_text>

DONE

Return exactly ONE <nova_cmd>...</nova_cmd> command.

Do not explain anything.
"""

        prompt += "\n\nCURRENT GOAL:\n"
        prompt += str(goal)

        prompt += "\n\nACTUAL CONTROLLER RESULT:\n"

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

        if thinking:

            prompt += (
                "\n\nMODEL'S PREVIOUS REASONING:\n"
            )

            prompt += thinking

        prompt += (
            "\n\nIMPORTANT:\n"
            "Convert the intended next step into a visible command.\n"
            "Return ONLY the XML command."
        )

        return prompt

    # =========================================================
    # MODEL REQUEST
    # =========================================================

    def call_model(
        self,
        prompt: str,
        think: bool = True,
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

            print(
                error
            )

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

        print(
            "\n[DEBUG OLLAMA RESPONSE]"
        )

        print(
            f"Content: {content!r}"
        )

        print(
            f"Thinking: {thinking!r}"
        )

        print(
            f"Tool calls: {tool_calls!r}"
        )

        if tool_calls:

            print(
                "\n[NOVA SECURITY] "
                "Native tool calls detected."
            )

            print(
                "[NOVA SECURITY] "
                "Native tool calls will NOT be executed."
            )

        return {
            "content": content,
            "thinking": thinking,
            "tool_calls": tool_calls,
        }

    # =========================================================
    # ASK AGENT
    # =========================================================

    def ask_agent(
        self,
        prompt: str,
    ):

        return self.call_model(
            prompt,
            think=True,
        )

    # =========================================================
    # COMMAND FORMATTER
    # =========================================================

    def recover_command(
        self,
        goal: str,
        previous_result,
        thinking: str,
    ):

        print(
            "\n[NOVA AGENT] "
            "Visible command missing."
        )

        print(
            "[NOVA AGENT] "
            "Requesting explicit command formatting."
        )

        recovery_prompt = (
            self.build_command_recovery_prompt(
                goal=goal,
                previous_result=previous_result,
                thinking=thinking,
            )
        )

        result = self.call_model(
            recovery_prompt,
            think=False,
        )

        content = result.get(
            "content",
            "",
        )

        if not content:

            print(
                "\n[NOVA AGENT] "
                "Command recovery returned no content."
            )

            return ""

        print(
            "\n[NOVA RECOVERY RESPONSE]"
        )

        print(
            content
        )

        return content

    # =========================================================
    # NORMALIZE RESPONSE
    # =========================================================

    def normalize_response(
        self,
        response: str,
    ):

        if not response:

            return ""

        response = response.strip()

        response = re.sub(
            r"^```(?:xml|text)?\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )

        response = re.sub(
            r"\s*```$",
            "",
            response,
            flags=re.IGNORECASE,
        )

        return response.strip()

    # =========================================================
    # EXTRACT XML COMMAND
    # =========================================================

    def extract_command(
        self,
        response: str,
    ):

        if not response:

            return None

        match = re.search(
            r"<nova_cmd>\s*(.*?)\s*</nova_cmd>",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not match:

            return None

        command = match.group(
            1
        ).strip()

        if not command:

            return None

        return command

    # =========================================================
    # PARSE OPERATION
    # =========================================================

    def parse_operation(
        self,
        response: str,
    ):

        response = self.normalize_response(
            response
        )

        if not response:

            return None

        command = self.extract_command(
            response
        )

        if not command:

            print(
                "\n[NOVA AGENT] "
                "No <nova_cmd> found."
            )

            print(
                f"Response: {response!r}"
            )

            return None

        # -----------------------------------------------------
        # DONE
        # -----------------------------------------------------

        if command.upper() == "DONE":

            return {
                "action": "done"
            }

        # -----------------------------------------------------
        # LIST_FILES
        # -----------------------------------------------------

        match = re.fullmatch(
            r"LIST_FILES\s+(.+)",
            command,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match:

            return {
                "action": "tool",
                "name": "list_files",
                "arguments": {
                    "path":
                        match.group(1).strip()
                },
            }

        # -----------------------------------------------------
        # READ_FILE
        # -----------------------------------------------------

        match = re.fullmatch(
            r"READ_FILE\s+(.+)",
            command,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match:

            return {
                "action": "tool",
                "name": "read_file",
                "arguments": {
                    "path":
                        match.group(1).strip()
                },
            }

        # -----------------------------------------------------
        # WRITE_FILE
        # -----------------------------------------------------

        match = re.fullmatch(
            r"WRITE_FILE\s+(\S+)\s+([\s\S]+)",
            command,
            flags=re.IGNORECASE,
        )

        if match:

            return {
                "action": "tool",
                "name": "write_file",
                "arguments": {
                    "path":
                        match.group(1).strip(),

                    "content":
                        match.group(2),
                },
            }

        # -----------------------------------------------------
        # EDIT_FILE
        # -----------------------------------------------------

        match = re.fullmatch(
            r"EDIT_FILE\s+([^\s]+)\s+OLD:\s*(.*?)\s+NEW:\s*(.*)",
            command,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match:

            return {
                "action": "tool",
                "name": "edit_file",
                "arguments": {
                    "path":
                        match.group(1).strip(),

                    "old_text":
                        match.group(2),

                    "new_text":
                        match.group(3),
                },
            }

        print(
            "\n[NOVA AGENT] "
            "Unknown NOVA command:"
        )

        print(
            repr(command)
        )

        return None

    # =========================================================
    # EXECUTE TOOL
    # =========================================================

    def execute_tool(
        self,
        operation: dict,
    ):

        name = operation.get(
            "name"
        )

        arguments = operation.get(
            "arguments",
            {},
        )

        allowed_tools = (
            self.tools.available_tools()
        )

        if name not in allowed_tools:

            print(
                "\n[NOVA SECURITY] "
                f"Blocked unknown operation: {name}"
            )

            return {
                "error":
                    f"Unknown operation: {name}"
            }

        print(
            f"\n[NOVA TOOL] {name}"
        )

        print(
            f"Arguments: {arguments}"
        )

        try:

            result = self.tools.execute(
                name,
                arguments,
            )

            print(
                "\n[NOVA TOOL RESULT]"
            )

            print(
                result
            )

            return result

        except Exception as error:

            print(
                "\n[NOVA TOOL ERROR]"
            )

            print(
                error
            )

            return {
                "error": str(error)
            }

    # =========================================================
    # OPERATION SIGNATURE
    # =========================================================

    def operation_signature(
        self,
        operation: dict,
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
        goal: str,
    ):

        previous_result = None

        self.operation_history = []

        for operation_number in range(
            1,
            self.max_operations_per_cycle + 1,
        ):

            print(
                f"\n[NOVA AGENT] "
                f"Operation "
                f"{operation_number}/"
                f"{self.max_operations_per_cycle}"
            )

            # -------------------------------------------------
            # Ask model to reason and choose command.
            # -------------------------------------------------

            prompt = self.build_prompt(
                goal=goal,
                previous_result=previous_result,
            )

            model_result = self.ask_agent(
                prompt
            )

            content = model_result.get(
                "content",
                "",
            )

            thinking = model_result.get(
                "thinking",
                "",
            )

            # -------------------------------------------------
            # If GPT-OSS reasoned but didn't emit content,
            # perform an isolated command-formatting pass.
            # -------------------------------------------------

            if not content:

                content = self.recover_command(
                    goal=goal,
                    previous_result=previous_result,
                    thinking=thinking,
                )

            print(
                "\nNOVA DECISION:"
            )

            print(
                content
            )

            # -------------------------------------------------
            # Still nothing.
            # -------------------------------------------------

            if not content:

                print(
                    "\n[NOVA AGENT] "
                    "No response from model."
                )

                return

            # -------------------------------------------------
            # Parse.
            # -------------------------------------------------

            operation = self.parse_operation(
                content
            )

            if operation is None:

                print(
                    "\n[NOVA AGENT] "
                    "No valid operation produced."
                )

                return

            # -------------------------------------------------
            # DONE.
            # -------------------------------------------------

            if operation["action"] == "done":

                print(
                    "\n[NOVA DONE]"
                )

                print(
                    "Autonomous reasoning completed "
                    "this cycle."
                )

                return

            # -------------------------------------------------
            # Duplicate protection.
            # -------------------------------------------------

            signature = (
                self.operation_signature(
                    operation
                )
            )

            if signature in self.operation_history:

                print(
                    "\n[NOVA AGENT] "
                    "Duplicate operation detected."
                )

                print(
                    signature
                )

                print(
                    "[NOVA AGENT] "
                    "Stopping to prevent an autonomous loop."
                )

                return

            self.operation_history.append(
                signature
            )

            # -------------------------------------------------
            # Execute.
            # -------------------------------------------------

            result = self.execute_tool(
                operation
            )

            self.state.observe(
                f"{operation['name']} "
                f"returned: {result}"
            )

            previous_result = result

        print(
            "\n[NOVA AGENT] "
            "Maximum operations reached "
            "for this cycle."
        )

    # =========================================================
    # SINGLE CYCLE
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

        self.state.running = True

        print(
            "\n[NOVA AGENT] "
            "Autonomous mode started."
        )

        while self.state.running:

            try:

                self.cycle_once()

                time.sleep(
                    self.interval
                )

            except KeyboardInterrupt:

                self.stop()

            except Exception as error:

                print(
                    "\n[NOVA AGENT ERROR]"
                )

                print(
                    error
                )

                time.sleep(
                    self.interval
                )

    # =========================================================
    # STOP
    # =========================================================

    def stop(self):

        self.state.running = False

        print(
            "\n[NOVA AGENT] Stopped."
        )