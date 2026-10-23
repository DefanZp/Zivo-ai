import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_PATH = BASE_DIR / "documents" / "returns_policy.txt"
VECTOR_STORE_PATH = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR.parent / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

if not OLLAMA_BASE_URL:
    raise ValueError("OLLAMA_BASE_URL belum dikonfigurasi.")

if not DOCUMENT_PATH.exists():
    raise FileNotFoundError(
        f"Dokumen tidak ditemukan: {DOCUMENT_PATH}"
    )

# 1. Read the document
text = DOCUMENT_PATH.read_text(encoding="utf-8")

if not text.strip():
    raise ValueError("Dokumen kosong.")

print("Dokumen berhasil dibaca.")


# 2. Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=180,
    chunk_overlap=30,
)

chunks = text_splitter.create_documents(
    [text],
    metadatas=[{"source": DOCUMENT_PATH.name}],
)

print(f"Jumlah chunk: {len(chunks)}")

for index, chunk in enumerate(chunks):
    print(f"\nChunk {index + 1}:")
    print(chunk.page_content)


# 3. Create the embedding model.
embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBEDDINGS_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
        }
    },
)

# 4. Connect to a persistent Chroma vector store.
vector_store = Chroma(
    collection_name="zivo_returns_policy",
    embedding_function=embeddings,
    persist_directory=str(VECTOR_STORE_PATH),
)

# 5. Store chunks and their embeddings.
chunk_ids = [
    f"returns-policy-{index}"
    for index in range(len(chunks))
]

vector_store.add_documents(
    documents=chunks,
    ids=chunk_ids,
)

print("\nIngestion selesai.")
print(f"Vector store tersimpan di: {VECTOR_STORE_PATH}")