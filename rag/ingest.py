import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_PATH = BASE_DIR / "documents" / "returns_policy.txt"
VECTOR_STORE_PATH = BASE_DIR / "chroma_gemini_db"

load_dotenv(BASE_DIR.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY belum dikonfigurasi.")

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

# 3. Create the embedding model.
embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "models/gemini-embedding-001",
    ),
    google_api_key=GEMINI_API_KEY,
    task_type="RETRIEVAL_DOCUMENT",
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
print(f"Embedding model: {embeddings.model}")
print(f"Vector store tersimpan di: {VECTOR_STORE_PATH}")
print(f"Total dokumen di-index: {len(chunks)}")