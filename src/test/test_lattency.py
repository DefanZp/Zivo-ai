import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
api_key = os.getenv("OLLAMA_API_KEY", "")

headers = {
    "Authorization": f"Bearer {api_key}",
}

start = time.perf_counter()

try:
    response = httpx.post(
        f"{url}/api/chat",
        headers=headers,
        json={
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "user",
                    "content": "Jawab singkat: siapa kamu?"
                }
            ],
            "think": False,
            "stream": False,
            "options": {
                "num_predict": 128
            }
        },
        timeout=180.0,
    )

    print("HTTP status:", response.status_code)
    print("Durasi:", round(time.perf_counter() - start, 2), "detik")
    print("Response:", response.text[:2000])

except httpx.HTTPError as error:
    print("Request gagal:", repr(error))
    print("Durasi:", round(time.perf_counter() - start, 2), "detik")