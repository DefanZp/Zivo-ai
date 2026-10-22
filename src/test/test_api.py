import os 

import requests

from dotenv import load_dotenv

load_dotenv()

backend_url = os.getenv('BACKEND_URL')

try:
    response = requests.get(
        f"{backend_url}/api/ai/products"
    )

    response.raise_for_status()

    data = response.json()

    print(data)

except requests.RequestException as error:
    print(f"Request gagal: {error}")
