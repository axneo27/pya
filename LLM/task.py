import json
import requests

MODEL_NAME = "qwen2.5:14b-instruct-q4_K_M"
API_URL = "http://localhost:11434/api/generate"


def generate_response(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(API_URL, json=payload, timeout=120)
    response.raise_for_status()
    text = response.text.strip()
    lines = [line for line in text.splitlines() if line.strip()]
    data = json.loads(lines[-1])
    return data["response"].strip()


while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "bye":
        print("Bot: Bye!")
        break
    
    answer = generate_response(user_input)

    print(f"Bot: {answer}\n")
