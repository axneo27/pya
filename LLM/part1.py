import requests

payload = {
    "model": "llama3.1:8b",
    "prompt": "What is JS?",
    "stream": False,
}

# Send request
response = requests.post("http://localhost:11434/api/generate", json=payload)

# Print output
print("Status code:", response.status_code)
# print("Raw response:", response.text)

data = response.json()
print("\nResponse from LLaMA 3.1:")
print(data["response"])
