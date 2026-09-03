import os

from dotenv import load_dotenv

from providers.ollama_provider import OllamaProvider
from providers.groq_provider import GroqProvider
from core.router import Router
from core.orchestrator import Orchestrator


load_dotenv()


def main():

    local = OllamaProvider(
        model="gpt-oss:20b"
    )

    groq = GroqProvider(
        model="openai/gpt-oss-20b"
    )

    router = Router(
        local_provider=local,
        groq_provider=groq,
    )

    nova = Orchestrator(router)

    mode = "auto"

    print("=" * 60)
    print("NOVA v0.2")
    print("DUAL-BRAIN AI SYSTEM")
    print("=" * 60)

    print("""
Commands:

/local  → Use local GPT-OSS 20B
/groq   → Use Groq cloud brain
/auto   → Automatic routing

/exit   → Exit NOVA
""")

    while True:

        user_input = input(f"\n[{mode}] You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("\nNOVA: Goodbye.")
            break

        if user_input.lower() == "/local":
            mode = "local"
            print("NOVA: Switched to LOCAL brain.")
            continue

        if user_input.lower() == "/groq":
            mode = "groq"
            print("NOVA: Switched to GROQ cloud brain.")
            continue

        if user_input.lower() == "/auto":
            mode = "auto"
            print("NOVA: Automatic routing enabled.")
            continue

        try:

            response = nova.run(
                message=user_input,
                mode=mode,
            )

            print(f"\nNOVA: {response}")

        except Exception as error:

            print(f"\nNOVA ERROR: {error}")


if __name__ == "__main__":
    main()