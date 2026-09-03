from ollama import chat

response = chat(
    model="gpt-oss:20b",
    messages=[
        {
            "role": "user",
            "content": 'Reply with exactly this JSON and nothing else: {"action":"done","reason":"test successful"}',
        }
    ],
)

print("RAW:", repr(response.message.content))
