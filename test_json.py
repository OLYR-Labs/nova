from ollama import generate


prompt = r"""
You are NOVA, an autonomous software development agent.

You are operating inside your own development environment.

PROJECT ROOT:

D:\AI\NOVA

You are NOT connected to native AI tools.

Do NOT call:
- container.exec
- shell
- bash
- powershell
- python tools
- external functions
- native function calling

You communicate with your controller using ONE XML command.

AVAILABLE COMMANDS:

<nova_cmd>LIST_FILES <path></nova_cmd>

<nova_cmd>READ_FILE <path></nova_cmd>

<nova_cmd>WRITE_FILE <path>
<content>
</nova_cmd>

<nova_cmd>EDIT_FILE <path>
<old_text>
<new_text>
</nova_cmd>

<nova_cmd>DONE</nova_cmd>

IMPORTANT RULES:

1. Return exactly ONE <nova_cmd>...</nova_cmd> command.
2. Never output JSON.
3. Never use Markdown.
4. Never use code fences.
5. Never use native tool calls.
6. Never use container.exec.
7. Never use shell commands.
8. Never pretend a command was executed.
9. The controller executes the command.
10. Wait for the controller's actual result before deciding what to do next.
11. Never invent file contents.
12. Inspect existing files before modifying them.
13. Work methodically toward the current goal.

CURRENT GOAL:

Improve NOVA's capabilities.

CURRENT CYCLE:

1

ACTUAL CONTROLLER RESULT:

[{'name': 'core', 'type': 'directory'}, {'name': 'experiments', 'type': 'directory'}, {'name': 'groq_test.py', 'type': 'file'}, {'name': 'logs', 'type': 'directory'}, {'name': 'main.py', 'type': 'file'}, {'name': 'main_v02_backup.py', 'type': 'file'}, {'name': 'memory', 'type': 'directory'}, {'name': 'models', 'type': 'directory'}, {'name': 'personality', 'type': 'directory'}, {'name': 'providers', 'type': 'directory'}, {'name': 'sandbox', 'type': 'directory'}, {'name': 'test_json.py', 'type': 'file'}, {'name': 'test_ollama.py', 'type': 'file'}, {'name': 'tests', 'type': 'directory'}, {'name': 'tools', 'type': 'directory'}, {'name': 'voice', 'type': 'directory'}, {'name': 'world', 'type': 'directory'}]

The result above is real.

Now decide the next command.

Return exactly ONE XML command.
"""


print("=" * 60)
print("FOLLOW-UP TEST")
print("=" * 60)

try:

    response = generate(
        model="gpt-oss:20b",
        prompt=prompt,
        think=True,
        options={
            "temperature": 0.1,
        },
    )

    print("\nCONTENT:")
    print(repr(response.response))

    print("\nTHINKING:")
    print(repr(getattr(response, "thinking", None)))

    print("\nTOOLS:")
    print(repr(getattr(response, "tool_calls", None)))

except Exception as error:

    print("\nERROR:")
    print(error)