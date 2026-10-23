# Get project directory
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_PATH = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

if not OLLAMA_BASE_URL:
    raise ValueError("OLLAMA_BASE_URL belum dikonfigurasi.")


# Connect to the same embedding model.
embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
        }
    },
)

# Load the existing vector store.
vector_store = Chroma(
    collection_name="zivo_returns_policy",
    embedding_function=embeddings,
    persist_directory=str(VECTOR_STORE_PATH),
)

# Ask a question
query = "Can I return after 5 days?"


# Find the two most relevant chunks.
results = vector_store.similarity_search(
    query=query,
    k=2,
)


# Display the retrieved document
print(f"Question: {query}")

for index, document in enumerate(results, start=1):
    print(f"\nResult {index}:")
    print(document.page_content)
    print("Source:", document.metadata.get("source"))