import threading

from core.events import EventBus
from core.proactive import ProactiveMind
from core.agent import AutonomousAgent

from dotenv import load_dotenv
from queue import Queue

from providers.ollama_provider import OllamaProvider
from providers.groq_provider import GroqProvider
from core.router import Router
from core.orchestrator import Orchestrator
from workers.launcher import start_worker


load_dotenv()


def main():

    # ---------------------------------------------------------
    # EVENT SYSTEM
    # ---------------------------------------------------------

    event_bus = EventBus()

    # Proactive messages are queued in terminal mode.
    # They must never interrupt input().
    pending_proactive_messages = Queue()

    # ---------------------------------------------------------
    # AI PROVIDERS
    # ---------------------------------------------------------

    local = OllamaProvider(
        model="gpt-oss:20b"
    )

    groq = GroqProvider(
        model="openai/gpt-oss-20b"
    )

    # ---------------------------------------------------------
    # ROUTER
    # ---------------------------------------------------------

    router = Router(
        local_provider=local,
        groq_provider=groq,
    )

    # ---------------------------------------------------------
    # NOVA CORE
    # ---------------------------------------------------------

    nova = Orchestrator(router)

    # ---------------------------------------------------------
    # AUTONOMOUS AGENT
    # ---------------------------------------------------------

    agent = AutonomousAgent(
        orchestrator=nova,
        interval=10,
    )

    # ---------------------------------------------------------
    # PROACTIVE MIND
    # ---------------------------------------------------------

    proactive = ProactiveMind(
        orchestrator=nova,
        event_bus=event_bus,
        mode="auto",
        minimum_interval=120,
        maximum_interval=300,
    )

    # ---------------------------------------------------------
    # EVENT HANDLERS
    # ---------------------------------------------------------

    def handle_nova_event(event):

        if event.event_type == "nova.proactive_message":

            message = event.data.get(
                "message",
                "",
            )

            if message:
                pending_proactive_messages.put(message)

        elif event.event_type == "nova.error":

            print(
                "\n\n[NOVA EVENT ERROR]"
            )

            print(
                event.data.get(
                    "error",
                    "Unknown error",
                )
            )

            print(
                "\n[You] ",
                end="",
                flush=True,
            )

    event_bus.subscribe(
        handle_nova_event
    )

    # ---------------------------------------------------------
    # START PROACTIVE MIND
    # ---------------------------------------------------------

    proactive.start()

    # ---------------------------------------------------------
    # CURRENT MODE
    # ---------------------------------------------------------

    mode = "auto"

    # ---------------------------------------------------------
    # AUTONOMOUS AGENT STATE
    # ---------------------------------------------------------

    agent_thread = None

    # ---------------------------------------------------------
    # STARTUP
    # ---------------------------------------------------------

    print("=" * 60)
    print("NOVA v0.6")
    print("DUAL-BRAIN + MEMORY + EVENT BUS")
    print("PROACTIVE MIND + AUTONOMOUS DEVELOPMENT")
    print("=" * 60)

    print("""
Commands:

/local      → Use local GPT-OSS 20B
/groq       → Use Groq cloud brain
/auto       → Automatic routing

/agent      → Start autonomous development
/agent-stop → Stop autonomous development

/memory     → Show saved long-term memory
/remember   → Save something to long-term memory
/forget     → Delete something from long-term memory
/clear      → Clear current conversation

/exit       → Exit NOVA
""")

    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------

    while True:

        try:

            user_input = input(
                f"\n[{mode}] You: "
            ).strip()

        except (KeyboardInterrupt, EOFError):

            print(
                "\n\nNOVA: Shutting down..."
            )

            proactive.stop()
            agent.stop()

            break

        if not user_input:
            continue

        # -----------------------------------------------------
        # SURFACE QUEUED PROACTIVE MESSAGES
        # -----------------------------------------------------

        while not pending_proactive_messages.empty():

            try:
                message = pending_proactive_messages.get_nowait()
            except Exception:
                break

            print(
                "\n\n[NOVA - PROACTIVE]"
            )

            print(message)

        command = user_input.lower()

        # -----------------------------------------------------
        # START AUTONOMOUS AGENT
        # -----------------------------------------------------

        if command == "/agent":

            if (
                agent_thread is not None
                and agent_thread.poll() is None
            ):

                print(
                    "\nNOVA: Autonomous agent is already running."
                )

                continue

            agent.add_goal(
                "Improve NOVA's capabilities."
            )

            print(
                "\n[NOVA AGENT] "
                "Starting autonomous development in background..."
            )

            agent_thread = start_worker()

            print(
                f"[NOVA AGENT] Autonomous worker started in Terminal 2. "
                f"PID: {agent_thread.pid}"
            )

            continue

        # -----------------------------------------------------
        # STOP AUTONOMOUS AGENT
        # -----------------------------------------------------

        if command == "/agent-stop":

            if (
                agent_thread is None
                or agent_thread.poll() is not None
            ):

                print(
                    "\nNOVA: Autonomous worker is not running."
                )

                continue

            agent_thread.terminate()
            agent_thread = None

            print(
                "\nNOVA: Autonomous worker stop requested."
            )

            continue

        # -----------------------------------------------------
        # EXIT
        # -----------------------------------------------------

        if command == "/exit":

            print(
                "\nNOVA: Shutting down..."
            )

            proactive.stop()

            if (
                agent_thread is not None
                and agent_thread.poll() is None
            ):

                print(
                    "[NOVA] Stopping autonomous worker..."
                )

                agent_thread.terminate()

                try:
                    agent_thread.wait(
                        timeout=2
                    )
                except Exception:
                    agent_thread.kill()

                agent_thread = None

            print(
                "NOVA: Goodbye."
            )

            break

        # -----------------------------------------------------
        # PROVIDER MODES
        # -----------------------------------------------------

        if command == "/local":

            mode = "local"

            print(
                "NOVA: Switched to LOCAL brain."
            )

            continue

        if command == "/groq":

            mode = "groq"

            print(
                "NOVA: Switched to GROQ cloud brain."
            )

            continue

        if command == "/auto":

            mode = "auto"

            print(
                "NOVA: Automatic routing enabled."
            )

            continue

        # -----------------------------------------------------
        # CLEAR CONVERSATION
        # -----------------------------------------------------

        if command == "/clear":

            nova.conversation.clear()

            print(
                "NOVA: Current conversation cleared."
            )

            continue

        # -----------------------------------------------------
        # SHOW MEMORY
        # -----------------------------------------------------

        if command == "/memory":

            memories = nova.memory.all()

            if not memories:

                print(
                    "NOVA: I don't have any saved memories."
                )

            else:

                print(
                    "\nNOVA MEMORY:"
                )

                for key, value in memories.items():

                    print(
                        f"  {key}: {value}"
                    )

            continue

        # -----------------------------------------------------
        # REMEMBER
        # -----------------------------------------------------

        if command.startswith("/remember "):

            content = user_input[
                len("/remember "):
            ].strip()

            if not content:

                print(
                    "NOVA: Tell me what you want me to remember."
                )

                continue

            key = (
                f"memory_"
                f"{len(nova.memory.all()) + 1}"
            )

            nova.memory.set(
                key,
                content,
            )

            print(
                "NOVA: Saved to long-term memory."
            )

            continue

        # -----------------------------------------------------
        # FORGET
        # -----------------------------------------------------

        if command.startswith("/forget "):

            key = user_input[
                len("/forget "):
            ].strip()

            if nova.memory.get(key) is None:

                print(
                    f"NOVA: I couldn't find '{key}'."
                )

            else:

                nova.memory.delete(
                    key
                )

                print(
                    f"NOVA: Forgot '{key}'."
                )

            continue

        # -----------------------------------------------------
        # NORMAL AI CONVERSATION
        # -----------------------------------------------------

        try:

            response = nova.run(
                message=user_input,
                mode=mode,
            )

            print(
                f"\nNOVA: {response}"
            )

        except Exception as error:

            print(
                f"\nNOVA ERROR: {error}"
            )


# -------------------------------------------------------------
# PROGRAM ENTRY POINT
# -------------------------------------------------------------

if __name__ == "__main__":
    main()