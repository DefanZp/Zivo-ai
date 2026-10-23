import os

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()

base_url = os.getenv("OLLAMA_BASE_URL")

if not base_url:
    raise ValueError("OLLAMA_BASE_URL belum dikonfigurasi.")

embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.getenv("OLLAMA_API_KEY")}"
        }
    },
)

text = "Customers can request a return within 7 days."

vector = embeddings.embed_query(text)

print("Embedding berhasil dibuat.")
print("Jumlah dimensi:", len(vector))
print("Lima nilai pertama:", vector[:5])