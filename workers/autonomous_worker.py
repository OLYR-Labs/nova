import signal
import threading

from core.agent import AutonomousAgent


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

    if agent is not None:

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

    agent = AutonomousAgent(
        orchestrator=None,
        interval=5,
        sandbox_root=r"D:\AI\NOVA",
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


if __name__ == "__main__":

    main()