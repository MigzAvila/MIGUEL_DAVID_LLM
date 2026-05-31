import os
import requests
import json
import time

url = "https://integrate.api.nvidia.com/v1/chat/completions"

# Key 1: from .env (the one that timed out)
key_env = "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"

# Key 2: working key from user prompt
key_working = "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"

payload = {
    "model": "minimaxai/minimax-m2.7",
    "messages": [{"role": "user", "content": "Say OK"}],
    "max_tokens": 3,
    "temperature": 0,
}

def test_key(name, key):
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    print(f"\n--- Testing {name} ---")
    start = time.time()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        elapsed = time.time() - start
        print(f"Status Code: {resp.status_code} (took {elapsed:.2f}s)")
        print(f"Response: {resp.text[:300]}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"Failed with exception: {e} (took {elapsed:.2f}s)")

test_key("ENV KEY", key_env)
test_key("WORKING KEY", key_working)
