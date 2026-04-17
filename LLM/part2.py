import requests
import json

payload = {
    "model": "llama3.1:8b",
    "prompt": "What is JS language?",
    "stream": True,
}

with requests.post(
    "http://localhost:11434/api/generate", json=payload, stream=True
) as response:

    print("Status code:", response.status_code)
    print("\nResponse from LLaMA 3.1:")

    for line in response.iter_lines():
        if line:
            content = line.decode("utf-8").removeprefix("data: ")
            try:
                chunk = json.loads(content)
                print(chunk.get("response", ""), end="", flush=True)
            except Exception as e:
                print(f"\n[Error parsing line]: {content}")
