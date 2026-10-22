import os

import requests

from dotenv import load_dotenv


load_dotenv()

ollama_base_url = os.getenv("OLLAMA_BASE_URL")
ollama_api_key = os.getenv("OLLAMA_API_KEY")


headers = {
    "Authorization": f"Bearer {ollama_api_key}"
}


response = requests.get(
    f"{ollama_base_url}/api/tags",
    headers=headers,
    timeout=30
)

print("Status:", response.status_code)
print("Response:", response.text)