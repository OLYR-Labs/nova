import os
import signal
import sys

from core.agent import AutonomousAgent
from core.self_developer import SelfDevelopmentEngine


agent = None


def handle_shutdown(
    signum,
    frame,
):

    global agent

    print(
        "\n[NOVA WORKER] "
        "Shutdown signal received."
    )

    if agent is not None and hasattr(agent, "request_safe_stop"):
        agent.request_safe_stop(
            "External shutdown signal."
        )


def main():

    global agent

    signal.signal(
        signal.SIGINT,
        handle_shutdown,
    )

    try:
        signal.signal(
            signal.SIGBREAK,
            handle_shutdown,
        )
    except AttributeError:
        pass

    root = os.getenv(
        "NOVA_ROOT",
        r"D:\AI\NOVA",
    )

    # One-shot autonomous software-development mode:
    #   python -m workers.autonomous_worker "Fix the failing tests"
    # or set NOVA_SELF_DEVELOP_GOAL.
    goal = " ".join(sys.argv[1:]).strip()
    goal = goal or os.getenv("NOVA_SELF_DEVELOP_GOAL", "").strip()

    if goal:
        print("\n[NOVA WORKER]")
        print("Autonomous self-development mode started.")
        print(f"Goal: {goal}")

        engine = SelfDevelopmentEngine(root)
        result = engine.run(goal)

        print("\n[NOVA SELF-DEVELOPMENT RESULT]")
        print(result)
        return 0 if result.get("success") else 1

    agent = AutonomousAgent(
        orchestrator=None,
        interval=5,
        sandbox_root=root,
    )

    print(
        "\n[NOVA WORKER]"
    )

    print(
        "Safe autonomous worker initialized."
    )

    agent.run()

    print(
        "\n[NOVA WORKER]"
    )

    print(
        "Worker exited cleanly."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
