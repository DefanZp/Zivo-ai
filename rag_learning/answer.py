import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Get the project directory.
BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_PATH = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL")

if not OLLAMA_BASE_URL:
    raise ValueError("OLLAMA_BASE_URL belum dikonfigurasi.")

# 1. Connect to embedding model.    
embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
        }
    },
)

# 2. Load the existing vector store.
vector_store = Chroma(
    collection_name="zivo_returns_policy",
    embedding_function=embeddings,
    persist_directory=str(VECTOR_STORE_PATH),
)

# 3. Initialize the language model.
llm = ChatOllama(
    model=OLLAMA_CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
        }
    }
)


# 4. Define the question.
question = "Can I return a sale item if I lost the receipt?"


# 5. Retrieve relevant documents.
documents = vector_store.similarity_search(
    query=question,
    k=2,
)

# 6. Combine retrieved documents into context.
context = "\n\n".join(
    (
        f"Source: {document.metadata.get('source', 'Unknown')}\n"
        f"Content:\n{document.page_content}"
    )
    for document in documents
)

# 7. Create the prompt.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a customer support assistant for Zivo.

            Answer the user's question using only the
            information provided in the context.

            If the context does not contain enough
            information, say that you do not have
            enough information to answer.

            Do not invent policies or requirements.

            Context:
            {context}
            """,
        ),
        (
            "human",
            "{question}",
        ),
    ]
)

# 8. Connect the prompt to the language model
chain = prompt | llm

# 9. Generate the answer.
response = chain.invoke(
    {
        "context": context,
        "question": question,
    }
)

print(f"Question: {question}")
print(f"\nAnswer:\n{response.content}")

print("\nRetrieved sources:")

sources = sorted(
    {
        document.metadata.get("source", "Unknown")
        for document in documents
    }
)

for source in sources:
    print(f"- {source}")
