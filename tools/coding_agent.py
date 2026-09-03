import json
from pathlib import Path

from ollama import generate


class CodingAgent:
    """
    NOVA's dedicated software-engineering agent.

    Uses Qwen3-Coder locally through Ollama.

    This agent does NOT directly modify files.
    It analyzes the task and returns a structured coding plan
    for NOVA's main controller to execute safely.
    """

    def __init__(
        self,
        model: str = "qwen3-coder:30b",
        project_root: str = r"D:\AI\NOVA",
    ):
        self.model = model
        self.project_root = Path(project_root).resolve()

    # =========================================================
    # SYSTEM PROMPT
    # =========================================================

    def build_system_prompt(self) -> str:
        return r"""
You are NOVA CODER, an advanced autonomous software-engineering
agent powered by Qwen3-Coder.

Your job is to analyze software projects, diagnose problems,
design solutions, create missing files, modify existing files,
and improve architecture.

You operate under the control of NOVA's main agent.

============================================================
CORE RULES
============================================================

1. NEVER assume a file exists.
2. NEVER invent existing source code.
3. Work from the actual files and information supplied by NOVA.
4. Prefer small, reliable changes over unnecessary rewrites.
5. Preserve existing functionality unless the task requires
   changing it.
6. When a missing module/file is required, explicitly identify it.
7. When a directory is required, explicitly identify it.
8. When imports must change, explicitly identify them.
9. Always consider dependencies between files.
10. Always verify the resulting architecture logically.
11. Never claim that code was executed unless NOVA reports that
    it was actually executed.
12. Never access the user's computer directly.
13. Never execute shell commands yourself.
14. Never use native tools.
15. NOVA controls all actual filesystem operations.

============================================================
PROJECT
============================================================

PROJECT ROOT:

D:\AI\NOVA

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

The JSON must have this structure:

{
    "status": "continue",
    "summary": "short explanation",
    "analysis": "technical analysis",
    "files_to_create": [
        {
            "path": "relative/path.py",
            "reason": "why this file is required",
            "content": "complete file contents"
        }
    ],
    "files_to_modify": [
        {
            "path": "relative/path.py",
            "reason": "why it must change",
            "old_text": "exact existing text",
            "new_text": "exact replacement"
        }
    ],
    "directories_to_create": [
        "relative/directory"
    ],
    "verification": [
        "what should be verified after changes"
    ]
}

If no changes are required:

{
    "status": "done",
    "summary": "explanation",
    "analysis": "analysis",
    "files_to_create": [],
    "files_to_modify": [],
    "directories_to_create": [],
    "verification": []
}

============================================================
IMPORTANT
============================================================

For every new file:

- provide COMPLETE source code
- include imports
- include classes/functions
- include required error handling
- do not provide placeholders such as:
  "add code here"
  "etc."
  "..."
  "implement this"

For every modification:

- old_text MUST match the supplied file exactly
- new_text MUST contain the complete replacement

If a new file is needed to fix an import such as:

from core.voice import VoiceEngine

and core/voice.py does not exist, explicitly create:

core/voice.py

If a new package is needed, explicitly create the directory
and the required __init__.py file.

You are a coding engineer, not merely a code explainer.
"""


    # =========================================================
    # BUILD PROMPT
    # =========================================================

    def build_prompt(
        self,
        task: str,
        project_context: str = "",
    ) -> str:

        prompt = self.build_system_prompt()

        prompt += """

============================================================
CURRENT CODING TASK
============================================================

"""

        prompt += f"TASK:\n{task}\n\n"

        prompt += """
============================================================
AVAILABLE PROJECT CONTEXT
============================================================

"""

        if project_context:
            prompt += project_context
        else:
            prompt += "No project files supplied yet.\n"

        prompt += """

============================================================
FINAL INSTRUCTION
============================================================

Analyze the supplied project.

Determine exactly what must be created or modified.

If multiple files are required, include ALL required files.

Pay particular attention to:

- imports
- package structure
- missing modules
- circular dependencies
- class interfaces
- function interfaces
- configuration
- integration points
- error handling
- backward compatibility

Return ONLY the required JSON object.
"""

        return prompt


    # =========================================================
    # CALL MODEL
    # =========================================================

    def call_model(
        self,
        prompt: str,
    ):

        try:

            response = generate(
                model=self.model,
                prompt=prompt,
                think=True,
                options={
                    "temperature": 0.1,
                },
            )

        except Exception as error:

            return {
                "success": False,
                "error": str(error),
                "result": None,
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

        if not content:

            return {
                "success": False,
                "error": "Coding model returned no response.",
                "result": None,
                "thinking": thinking,
            }

        parsed = self.parse_json(content)

        if parsed is None:

            return {
                "success": False,
                "error": "Coding model returned invalid JSON.",
                "raw_response": content,
                "thinking": thinking,
                "result": None,
            }

        return {
            "success": True,
            "result": parsed,
            "thinking": thinking,
            "raw_response": content,
        }


    # =========================================================
    # JSON PARSER
    # =========================================================

    def parse_json(
        self,
        content: str,
    ):

        content = content.strip()

        try:
            return json.loads(content)

        except json.JSONDecodeError:
            pass

        # Attempt to recover JSON surrounded by text/code fences.

        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1:
            return None

        candidate = content[start:end + 1]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            return None


    # =========================================================
    # ANALYZE
    # =========================================================

    def analyze(
        self,
        task: str,
        project_context: str = "",
    ):

        prompt = self.build_prompt(
            task=task,
            project_context=project_context,
        )

        return self.call_model(
            prompt
        )


    # =========================================================
    # PLAN CHANGES
    # =========================================================

    def plan_changes(
        self,
        task: str,
        project_context: str = "",
    ):

        result = self.analyze(
            task=task,
            project_context=project_context,
        )

        if not result.get("success"):

            return result

        plan = result.get(
            "result"
        )

        if not isinstance(
            plan,
            dict,
        ):

            return {
                "success": False,
                "error": "Invalid coding plan returned.",
                "result": None,
            }

        plan.setdefault(
            "files_to_create",
            [],
        )

        plan.setdefault(
            "files_to_modify",
            [],
        )

        plan.setdefault(
            "directories_to_create",
            [],
        )

        plan.setdefault(
            "verification",
            [],
        )

        return {
            "success": True,
            "result": plan,
            "thinking": result.get(
                "thinking",
                "",
            ),
        }


    # =========================================================
    # VALIDATE PLAN
    # =========================================================

    def validate_plan(
        self,
        plan,
    ):

        if not isinstance(
            plan,
            dict,
        ):

            return {
                "valid": False,
                "errors": [
                    "Plan must be a dictionary."
                ],
            }

        errors = []

        required_fields = [
            "files_to_create",
            "files_to_modify",
            "directories_to_create",
            "verification",
        ]

        for field in required_fields:

            if field not in plan:

                errors.append(
                    f"Missing field: {field}"
                )

        if not isinstance(
            plan.get(
                "files_to_create",
                [],
            ),
            list,
        ):

            errors.append(
                "files_to_create must be a list."
            )

        if not isinstance(
            plan.get(
                "files_to_modify",
                [],
            ),
            list,
        ):

            errors.append(
                "files_to_modify must be a list."
            )

        for item in plan.get(
            "files_to_create",
            [],
        ):

            if not isinstance(
                item,
                dict,
            ):

                errors.append(
                    "Invalid files_to_create item."
                )
                continue

            if not item.get("path"):
                errors.append(
                    "Created file is missing path."
                )

            if "content" not in item:
                errors.append(
                    f"Created file has no content: "
                    f"{item.get('path')}"
                )

        for item in plan.get(
            "files_to_modify",
            [],
        ):

            if not isinstance(
                item,
                dict,
            ):

                errors.append(
                    "Invalid files_to_modify item."
                )
                continue

            if not item.get("path"):
                errors.append(
                    "Modified file is missing path."
                )

            if "old_text" not in item:
                errors.append(
                    f"Modification missing old_text: "
                    f"{item.get('path')}"
                )

            if "new_text" not in item:
                errors.append(
                    f"Modification missing new_text: "
                    f"{item.get('path')}"
                )

        return {
            "valid": not errors,
            "errors": errors,
        }


    # =========================================================
    # SAFE PATH VALIDATION
    # =========================================================

    def safe_path(
        self,
        path: str,
    ) -> Path:

        requested = Path(
            str(path)
        )

        if requested.is_absolute():

            target = requested.resolve()

        else:

            target = (
                self.project_root / requested
            ).resolve()

        try:

            target.relative_to(
                self.project_root
            )

        except ValueError:

            raise PermissionError(
                f"Coding agent path outside project "
                f"root blocked: {path}"
            )

        return target


    # =========================================================
    # FORMAT PLAN FOR NOVA
    # =========================================================

    def format_plan(
        self,
        plan,
    ) -> str:

        validation = self.validate_plan(
            plan
        )

        if not validation["valid"]:

            return (
                "INVALID CODING PLAN:\n"
                + "\n".join(
                    validation["errors"]
                )
            )

        lines = []

        lines.append(
            "NOVA CODING PLAN"
        )

        lines.append(
            "=" * 60
        )

        lines.append(
            f"Status: {plan.get('status', 'continue')}"
        )

        lines.append(
            f"Summary: {plan.get('summary', '')}"
        )

        lines.append("")

        directories = plan.get(
            "directories_to_create",
            [],
        )

        if directories:

            lines.append(
                "DIRECTORIES TO CREATE:"
            )

            for directory in directories:

                lines.append(
                    f"  + {directory}"
                )

            lines.append("")

        files = plan.get(
            "files_to_create",
            [],
        )

        if files:

            lines.append(
                "FILES TO CREATE:"
            )

            for item in files:

                lines.append(
                    f"  + {item.get('path')}"
                )

            lines.append("")

        modifications = plan.get(
            "files_to_modify",
            [],
        )

        if modifications:

            lines.append(
                "FILES TO MODIFY:"
            )

            for item in modifications:

                lines.append(
                    f"  ~ {item.get('path')}"
                )

            lines.append("")

        verification = plan.get(
            "verification",
            [],
        )

        if verification:

            lines.append(
                "VERIFICATION:"
            )

            for item in verification:

                lines.append(
                    f"  - {item}"
                )

        return "\n".join(
            lines
        )