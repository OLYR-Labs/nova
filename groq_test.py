import os
import time
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is not set.")

client = Groq(api_key=api_key)

start = time.perf_counter()

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Explain quantum computing in simple terms."
        }
    ],
)

elapsed = time.perf_counter() - start

print("\n" + "=" * 60)
print("GROQ TEST")
print("=" * 60)
print(response.choices[0].message.content)
print("=" * 60)
print(f"Time: {elapsed:.2f} seconds")

if response.usage:
    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Completion tokens: {response.usage.completion_tokens}")
    print(f"Total tokens: {response.usage.total_tokens}")

    if elapsed > 0:
        print(
            f"Approx. generation speed: "
            f"{response.usage.completion_tokens / elapsed:.1f} tokens/sec"
        )