# Zivo AI Service

The Python AI service behind Zivo's AI Shopping Assistant. It exposes chat endpoints through FastAPI and coordinates specialized LangGraph agents that use Google Gemini, MCP tools, Chroma retrieval, and the Laravel REST API.

The service supports natural-language product discovery, documented store-policy questions, current product verification, and authenticated shopping-cart operations.

Laravel remains the source of truth for current product details, prices, stock, user authorization, and cart contents. RAG results are retrieval references, not transactional data.

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
Supervisor Agent
   |         |          |
   v         v          v
Product     Cart      Knowledge
Agent       Agent      Agent
   |         |          |
   +---------+----------+
             |
         MCP tools
       /     |      \
      v      v       v
  Laravel  Chroma  PostgreSQL
   API    Vector DB  Checkpoints
```

The FastAPI lifespan initializes the asynchronous PostgreSQL checkpointer, opens the MCP adapter, discovers MCP tools, and creates `ZivoAgentService`. The Supervisor delegates work to three specialized worker agents:

- **Product Agent** — searches the catalog, retrieves product candidates, verifies current product details, and manages the selected-product state.
- **Cart Agent** — retrieves and changes the authenticated user's cart. It receives the selected product ID from graph state and the cart token through runtime context.
- **Knowledge Agent** — retrieves documented store policies through `search_policy_knowledge` and returns policy findings to the Supervisor.

The Supervisor coordinates the conversation and delegates dependent product/cart tasks sequentially. The worker agents are specialized; the Supervisor supplies relevant context when invoking them.

## Product Discovery and Verification

For attribute-based discovery, Product Agent uses product knowledge retrieval to identify candidates. It then verifies a clearly selected candidate through Laravel's current product-detail tool before setting the selected product. If multiple candidates are similarly suitable, it asks the user to choose.

RAG is not authoritative for live prices or stock. Current product information is retrieved from Laravel before it is presented as current data.

## Cart Operations and Security

Cart operations are delegated to Cart Agent. The selected product ID is stored in LangGraph state, while `cart_api_token` is passed through request-scoped runtime context. The model does not generate the user's authentication token.

Laravel remains responsible for authorization, product ownership checks, stock validation, and cart business rules. Agent prompts guide behavior but are not a security boundary.

## Model Configuration

The source configures `ChatGoogleGenerativeAI` using:

| Setting | Source configuration |
|---|---|
| Model | `GEMINI_MODEL`, default `gemini-3.1-flash-lite` |
| Temperature | `0.0` |
| Timeout | 180 seconds |
| Retries | 1 |
| Summarization trigger | 4,000 tokens |
| Messages retained by summarization | Last 6 messages |

The uploaded archive does not include `.env`; therefore, a runtime override of `GEMINI_MODEL` cannot be confirmed from this snapshot.

## RAG Architecture

The active MCP server opens two Chroma collections in the same persistent directory:

| Collection | Purpose | Retrieval size |
|---|---|---:|
| `zivo_product_catalog` | Product candidates based on product attributes and descriptions | 3 |
| `zivo_returns_policy` | Documented store-policy retrieval | 2 |

**Active vector-store directory:** `rag/chroma_gemini_db/`

**Embedding implementation:** `GoogleGenerativeAIEmbeddings`

**Default embedding model:** `models/gemini-embedding-001`, configurable with `GEMINI_EMBEDDING_MODEL`.

### Product ingestion

Run `python rag/ingest_products.py` from the repository root. The script retrieves the catalog from Laravel `GET /api/ai/products`, converts product name, category, description, and specifications into documents, and stores them using deterministic IDs in the form `product-{id}`. It also removes stored product IDs that are no longer present in the returned catalog.

The script assumes the endpoint returns the complete catalog in one response; it does not implement pagination. Current price and stock are not embedded as authoritative transactional values and must be retrieved from Laravel when needed.

### Policy ingestion

Run `python rag/ingest.py` from the repository root. The script reads `rag/documents/returns_policy.txt`, splits the text with `chunk_size=180` and `chunk_overlap=30`, and writes to `zivo_returns_policy` using IDs in the form `returns-policy-{index}`.

The current policy ingestion script does not reconcile obsolete chunk IDs when the source policy changes. Review/rebuild the policy collection carefully after policy-document changes.

The Chroma database is generated locally and must be populated in each target environment before retrieval can return indexed content. The current `.gitignore` still ignores `rag/chroma_db/`, while the active directory is `rag/chroma_gemini_db/`; correct and verify the ignore rule before committing generated vector data.

## MCP Tools

The MCP server in `src/mcp_server.py` exposes ten tools:

| Tool | Responsibility |
|---|---|
| `search_products` | Search the Laravel product catalog |
| `get_product` | Retrieve current product details from Laravel |
| `search_product_knowledge` | Retrieve product candidates from Chroma |
| `search_policy_knowledge` | Retrieve documented policies from Chroma |
| `get_cart` | Retrieve the authenticated user's cart |
| `add_to_cart` | Add the selected product to the cart |
| `increase_quantity` | Increase the selected cart item's quantity |
| `decrease_quantity` | Decrease the selected cart item's quantity |
| `remove_from_cart` | Remove the selected product from the cart |
| `clear_cart` | Clear the authenticated user's cart |

`set_current_product` and `clear_current_product` are local agent state tools, not MCP server tools.

Cart Agent wraps cart MCP operations to pass the selected product ID from graph state and the authentication token from runtime context. The quantity wrappers use awaited MCP calls and derive the product ID from the selected-product state.

## State and Runtime Context

`src/state.py` separates graph state from request-scoped context:

- `selected_product_id: NotRequired[int | None]` — optional graph-state field for the currently selected product.
- `cart_api_token: str | None` — request-scoped context used by authenticated cart operations.

FastAPI verifies an optional Bearer token by calling Laravel `GET /api/user/profile`. Authenticated thread IDs are namespaced by user ID; guest thread IDs use a `guest:` namespace.

## Conversation Persistence and Cleanup

LangGraph checkpoints are stored in PostgreSQL through `AsyncPostgresSaver`, configured by `LANGGRAPH_DB_URI`. The FastAPI lifespan opens the checkpointer and initializes its schema.

The current thread-activity registry records authenticated threads. Guest checkpoints use the `guest:` namespace but are not written to the activity registry used by the current cleanup script. Guest checkpoint cleanup is therefore an open retention issue.

## API Reference

### `GET /health`

Returns a basic HTTP application health response:

```json
{"status":"ok"}
```

This is not a readiness check for Gemini, Chroma, PostgreSQL, MCP, or Laravel.

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

| Event | Payload | Meaning |
|---|---|---|
| `token` | `{"content":"..."}` | Assistant text chunk |
| `done` | `{}` | Stream completed |
| `error` | `{"message":"..."}` | Stream failed |

An optional `Authorization: Bearer <token>` header is used for authenticated cart actions. Requests without a token are treated as guest conversations. The Next.js Route Handler forwards the Bearer header when present.

## Environment Variables

The active application and RAG source read these variables:

| Variable | Required | Purpose / default |
|---|---|---|
| `LANGGRAPH_DB_URI` | Yes | PostgreSQL connection string for LangGraph checkpoints |
| `BACKEND_URL` | Yes | Laravel backend base URL |
| `GEMINI_API_KEY` | Yes | Google Gemini API key for chat and embeddings |
| `GEMINI_MODEL` | No | Chat model override; default `gemini-3.1-flash-lite` |
| `GEMINI_EMBEDDING_MODEL` | No | Embedding model override; default `models/gemini-embedding-001` |

Smoke-test scripts also read `CART_SMOKE_TEST_TOKEN` and `SMOKE_TEST_PRODUCT_ID`. These are only for manual smoke tests, not normal API startup. Keep real tokens and API keys in a local `.env` file and never commit them.

The archive does not contain `.env.example`. A placeholder-only draft is included in this proposal package for review.

## Installation and Running

### Dependency setup status

The repository currently has no authoritative `requirements.txt`, `pyproject.toml`, or dependency lock file, and it does not declare a supported Python version. `requirements-before-pruning.txt` is an older UTF-16 dependency snapshot, not a supported installation manifest. Do not use it as the documented installation command without reviewing and regenerating the dependency set.

A reproducible installation guide should be finalized only after a clean environment is used to create and validate an authoritative dependency manifest and supported Python version.

### Runtime prerequisites

Before starting the API, the environment needs:

- A supported Python version, to be declared in the repository.
- The validated project dependencies, once an authoritative manifest is added.
- PostgreSQL reachable through `LANGGRAPH_DB_URI`.
- Laravel reachable through `BACKEND_URL`.
- A valid `GEMINI_API_KEY`.
- Both Chroma collections populated under `rag/chroma_gemini_db/`.

### Start the API

The active FastAPI entry point is `src/api.py`. Run Uvicorn from the `src` directory so the current module imports resolve:

```bash
cd src
python -m uvicorn api:app --host 0.0.0.0 --port 8001
```

This command assumes the project's dependencies and environment variables have already been installed/configured.

### Populate the vector store

Run these commands from the repository root after dependencies and environment variables are ready:

```bash
python rag/ingest_products.py
python rag/ingest.py
```

The product ingestion requires the Laravel AI products endpoint. The policy ingestion reads the repository's `rag/documents/returns_policy.txt`.

## Tests

The repository contains eight pytest modules under `src/test/`, covering:

- active cart MCP wrappers;
- Product, Cart, and Knowledge Agent wrappers;
- non-streaming and streaming API endpoints;
- cart request error mapping;
- thread activity persistence helpers.

Run from the `src` directory after installing the validated test dependencies:

```bash
cd src
python -m pytest -q
```

The 2026-11-13 source audit found 29 Python files and all 29 passed Python 3.13 AST parsing. Pytest collection was attempted in the audit environment but stopped with eight import errors because project dependencies were not installed there. No test body completed in that environment. A prior conversation-reported result of `111 passed, 3 warnings` on 2026-11-08 is historical and was not reproduced against this archive.

## Known Limitations and Follow-ups

- Guest LangGraph checkpoints are not covered by the current activity-based cleanup path.
- `src/agents/knowledge_agent.py` logs the task text and policy retrieval content; review raw-content logging and log retention.
- `.gitignore` excludes `rag/chroma_db/`, not the active `rag/chroma_gemini_db/` directory. Verify the Git index and update the ignore rule before generated Chroma data is committed.
- Policy ingestion does not remove obsolete chunks after the source policy document changes.
- No authoritative dependency manifest or supported Python version is declared.
- MCP cart handlers use synchronous `requests.request()`; performance impact has not been measured.
- Product ingestion assumes the Laravel endpoint returns the full catalog without pagination.

## Related Repositories

- [Zivo Frontend](https://github.com/DefanZp/Zivo-frontEnd-next)
- [Zivo Backend](https://github.com/DefanZp/Zivo-backEnd)
- [Zivo n8n](https://github.com/DefanZp/Zivo-n8n)

## Author

**Defan Zain Pratama** — Full-Stack Developer interested in web applications, backend systems, and AI-powered applications.

- GitHub: [@DefanZp](https://github.com/DefanZp)
