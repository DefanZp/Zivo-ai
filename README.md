# Zivo AI Service

The Python AI service behind **Zivo's AI Shopping Assistant**. It exposes chat endpoints through FastAPI and orchestrates a LangGraph agent that uses Google Gemini, MCP tools, Chroma retrieval, and the Laravel REST API.

The service helps customers discover products using natural language, retrieve store-policy information, inspect current product details, and perform supported cart operations for authenticated users.

## Responsibilities

- Provide non-streaming and streaming chat APIs.
- Execute the shopping agent with LangGraph state and runtime context.
- Retrieve product candidates and policy information from Chroma.
- Call Laravel for current product and cart data through MCP tools.
- Verify optional user Bearer tokens through the Laravel profile API.
- Persist conversation checkpoints in PostgreSQL.

Laravel remains the source of truth for current product prices, stock, user authorization, and cart contents. RAG results are retrieval references, not transactional data.

## Architecture

```text
Next.js Chat Widget
        |
        v
Next.js Chat Route Handler
        |
        v
FastAPI (/chat or /chat/stream)
        |
        v
ZivoAgentService
        |
        v
LangGraph Agent
   |       |        |
 Gemini   MCP     PostgreSQL
          |       Checkpointer
          |
          +---- Laravel API
          |
          +---- Chroma Collections
                    |
                    v
              Ollama Embeddings
```

At application startup, FastAPI opens the asynchronous PostgreSQL checkpointer, initializes checkpoint tables, connects to the local MCP server through `MCPAdapter`, discovers the MCP tools, and creates `ZivoAgentService`.

## AI Shopping Agent

The agent uses LangChain's `create_agent` with LangGraph execution and a Google Gemini chat model. The inspected configuration defaults to `gemini-3.1-flash-lite`; it reads `GEMINI_API_KEY` and optionally `GEMINI_MODEL`. The model configuration in the uploaded source uses temperature `1.0`, timeout `180` seconds, and one retry.

The agent follows a product discovery flow:

1. Retrieve product candidates using product knowledge search when attribute-based discovery is needed.
2. Evaluate candidates against the customer's requirements.
3. Set an unambiguous selection in graph state.
4. Retrieve current product information from Laravel before presenting live details.
5. Use cart tools only when the required product selection and authentication context are available.

Conversation instructions require the agent to avoid inventing product IDs, prices, stock, cart state, and tool outcomes. The system prompt is behavioral guidance; Laravel authorization and validation remain the security boundary.

A `SummarizationMiddleware` is configured to trigger at 4,000 tokens and retain the last six messages.

## RAG Architecture

The active MCP server opens two persistent Chroma collections under `rag/chroma_db`:

| Collection | Purpose | Retrieval size |
|---|---|---:|
| `zivo_product_catalog` | Retrieve product candidates by attributes and description | 3 |
| `zivo_returns_policy` | Retrieve store return-policy information | 2 |

### Product ingestion

`rag/ingest_products.py` retrieves product data from Laravel's `/api/ai/products` endpoint, converts product name, category, description, and specifications into documents, and stores embeddings in `zivo_product_catalog`. Deterministic IDs use `product-{id}`. The current script assumes the endpoint returns the full catalog in one response and does not implement pagination.

Price and stock are not embedded as live transactional fields. They must be fetched from Laravel when needed.

### Policy ingestion

`rag/ingest.py` reads `rag/documents/returns_policy.txt`, splits the text into chunks with a 180-character chunk size and 30-character overlap, creates embeddings, and stores them in `zivo_returns_policy`.

The current policy ingestion script uses deterministic chunk IDs but does not reconcile obsolete chunk IDs after the source policy document changes.

The Chroma database is generated locally and is excluded from Git. It must be populated in the target environment before retrieval can return indexed content.

## MCP Integration

The MCP server in `src/mcp_server.py` exposes ten tools:

| Tool | Responsibility |
|---|---|
| `search_products` | Search products through Laravel |
| `get_product` | Retrieve current product details from Laravel |
| `search_product_knowledge` | Retrieve product candidates from Chroma |
| `search_policy_knowledge` | Retrieve policy documents from Chroma |
| `get_cart` | Retrieve the authenticated user's cart |
| `add_to_cart` | Add an item to the cart |
| `increase_quantity` | Increase a cart item's quantity |
| `decrease_quantity` | Decrease a cart item's quantity |
| `remove_from_cart` | Remove a cart item |
| `clear_cart` | Clear the cart |

The agent also has two local state tools, `set_current_product` and `clear_current_product`. These are agent tools, not additional MCP server tools.

The agent wraps cart-related MCP tools to pass `selected_product_id` from graph state and `cart_api_token` from runtime context. Laravel still validates the token, ownership, stock, and cart operation.

**Known wrapper defect:** in the uploaded source, the `increase_quantity` and `decrease_quantity` wrappers call MCP `.ainvoke()` without `await`. Their schemas also still declare a `product_id` argument even though the wrapper reads the selected product from agent state. These operations need to be fixed and tested before they are described as fully working.

## Agent State and Runtime Context

`src/state.py` defines:

- `selected_product_id`: graph state used to remember the currently selected product.
- `cart_api_token`: per-request runtime context used by wrappers for authenticated cart operations.

The FastAPI service verifies an optional Bearer token by calling Laravel's `/api/user/profile`. Authenticated thread IDs are namespaced by user ID; guest threads use a separate `guest:` namespace.

## Conversation Persistence

The service uses LangGraph's `AsyncPostgresSaver` with `LANGGRAPH_DB_URI` for checkpoint persistence. The FastAPI lifespan initializes the checkpointer and the MCP connection once when the application starts.

The current activity registry records authenticated threads. Guest checkpoint IDs are not inserted into that registry, so the existing activity-based cleanup does not cover guest checkpoints. Review retention and cleanup before operating this service with long-lived guest conversations.

## API Reference

### `GET /health`

Returns a basic service health response:

```json
{"status":"ok"}
```

This endpoint indicates that the HTTP application responds; it is not a complete readiness check for the model provider, vector store, or Laravel integration.

### `POST /chat`

Request:

```json
{
  "thread_id": "conversation-123",
  "user_message": "Find an oversized black hoodie"
}
```

Response:

```json
{
  "thread_id": "conversation-123",
  "response": "..."
}
```

### `POST /chat/stream`

Accepts the same request body and returns Server-Sent Events:

| Event | Payload |
|---|---|
| `token` | `{"content":"..."}` text chunk |
| `done` | `{}` when streaming completes |
| `error` | `{"message":"..."}` generic error |

An optional `Authorization: Bearer <token>` header is used for authenticated cart actions. Without it, the request is treated as a guest conversation. The Next.js Route Handler forwards the Bearer header when present.

## Technology Stack

| Area | Technology |
|---|---|
| API framework | FastAPI |
| Agent orchestration | LangChain `create_agent`, LangGraph |
| LLM | Google Gemini via `ChatGoogleGenerativeAI` |
| Tool protocol | MCP via `MCPAdapter` |
| Vector database | Chroma |
| Embeddings | Ollama `OllamaEmbeddings` |
| Persistence | PostgreSQL via `AsyncPostgresSaver` |
| E-commerce integration | Laravel REST API |

The uploaded repository does not include an authoritative active dependency manifest. `requirements-before-pruning.txt` is an earlier dependency snapshot and should not be treated as the supported install specification.

## Environment Variables

The active source reads these variables:

| Variable | Purpose |
|---|---|
| `LANGGRAPH_DB_URI` | PostgreSQL connection string for LangGraph checkpointing |
| `BACKEND_URL` | Laravel backend base URL |
| `GEMINI_API_KEY` | Google Gemini API key |
| `GEMINI_MODEL` | Optional model override; default is `gemini-3.1-flash-lite` |
| `OLLAMA_BASE_URL` | Ollama API base URL for embeddings |
| `OLLAMA_EMBEDDINGS_MODEL` | Embedding model name |
| `OLLAMA_API_KEY` | Optional/required by the configured Ollama endpoint for authorization header |

Other Ollama/OpenRouter variables found in the source belong to commented or legacy model experiments and are not part of the active Gemini model configuration.

Create a local `.env` file and keep all credentials out of Git. The repository currently does not contain an `.env.example`; add one with placeholder values before publishing setup instructions.

## Installation and Running

The repository snapshot does not declare a supported Python version and does not include a current `requirements.txt` or `pyproject.toml`. A reproducible installation guide cannot be finalized until the active dependencies and Python version are declared.

Once a maintained dependency manifest is added, the intended service entry point is `src/api.py`. Run the application from the `src` directory so its current module imports resolve, for example:

```bash
cd src
python -m uvicorn api:app --host 0.0.0.0 --port 8001
```

The AI service also requires:

- A reachable PostgreSQL database configured by `LANGGRAPH_DB_URI`.
- A reachable Laravel backend configured by `BACKEND_URL`.
- A valid Google Gemini API key.
- A reachable Ollama endpoint and configured embedding model.
- Product and policy Chroma collections populated under `rag/chroma_db`.

### RAG ingestion

After dependencies and environment variables are configured, the current scripts are:

```bash
python rag/ingest_products.py
python rag/ingest.py
```

Run them from the repository root. The product ingestion script expects the Laravel AI product endpoint to return the product catalog. The policy script reads `rag/documents/returns_policy.txt`.

## Testing and Known Limitations

- The source audit parsed all 28 Python files with Python 3.13.5 AST parsing. This does not verify imports, dependency compatibility, startup, provider access, database persistence, MCP execution, or end-to-end behavior.
- Existing scripts under `src/test/` are primarily manual scripts and do not provide reliable automated coverage of the active MCP wrapper path.
- `increase_quantity` and `decrease_quantity` wrappers are missing `await` and have schema/prompt disagreement.
- Full tool arguments and tool results are logged in the non-streaming chat endpoint; cart results may contain private user data. Replace raw-content logging with sanitized operational metadata.
- Synchronous HTTPX and Psycopg operations are called from async request/lifespan paths; evaluate an async client/pool or a deliberate threadpool boundary.
- Guest checkpoints are not included in the activity-based cleanup registry.
- `src/main.py` is an outdated/manual runner inconsistent with the active service constructor and async API. Use `src/api.py` as the web entry point.
- Product ingestion assumes an unpaginated full-catalog response; policy ingestion does not remove obsolete chunk IDs.

## Related Repositories

- [Zivo Frontend](https://github.com/DefanZp/Zivo-frontEnd-next)
- [Zivo Backend](https://github.com/DefanZp/Zivo-backEnd)
- [Zivo n8n](https://github.com/DefanZp/Zivo-n8n)

## Author

**Defan Zain Pratama** — Full-Stack Developer interested in web applications, backend systems, and AI-powered applications.

- GitHub: [@DefanZp](https://github.com/DefanZp)
