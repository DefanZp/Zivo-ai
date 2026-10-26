import os
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from fastapi import (
    Depends, 
    FastAPI, 
    Header,
    HTTPException, 
    status
)
from dotenv import load_dotenv
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import httpx
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, Field

from service import ZivoAgentService
from thread_activity import ensure_thread_activity_table, record_thread_activity
from langchain.mcp import MCPAdapter
logger = logging.getLogger("uvicorn.error")

load_dotenv()

db_uri = os.getenv("LANGGRAPH_DB_URI")
backend_url = os.getenv("BACKEND_URL")

if not db_uri:
    raise ValueError("LANGGRAPH_DB_URI belum ditemukan di .env")

if not backend_url:
    raise ValueError("BACKEND_URL belum ditemukan di .env")

security = HTTPBearer(auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Siapkan tabel aktivitas percakapan.
    ensure_thread_activity_table(db_uri)

    # Lokasi file MCP server.
    server_path = Path(__file__).resolve().with_name(
        "mcp_server.py"
    )

    # Buka koneksi PostgreSQL.
    async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:

        # Siapkan tabel checkpoint jika belum tersedia.
        await checkpointer.setup()

        # Buka koneksi MCP.
        async with MCPAdapter(server_path) as adapter:

            # Ambil tools dari MCP server.
            mcp_tools = await adapter.list_tools()

            logger.info(
                "MCP tools berhasil dimuat: %s",
                [tool.name for tool in mcp_tools],
            )

            # Buat agent menggunakan PostgreSQL checkpoint
            # dan MCP tools yang tersedia.
            app.state.agent_service = ZivoAgentService(
                checkpointer,
                mcp_tools,
            )

            # Aplikasi mulai menerima request.
            yield

app = FastAPI(
    title="Zivo Ai Service",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }   

class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100)
    user_message: str = Field(min_length=1, max_length=5000)

def verify_user_token(token: str) -> int:
    try:
        response = httpx.get(
            f"{backend_url}/api/user/profile",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            timeout=10.0,
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is unavailable.",
        )

    if response.status_code in (401, 403):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to verify user authentication.",
        )

    try:
        data = response.json()["data"]
        user_id = data["id"]

        if isinstance(user_id, bool) or not isinstance(user_id, int):
            raise ValueError("Invalid user ID")

        return user_id

    except (ValueError, KeyError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Invalid authentication response.",
        )

@app.post("/chat")
async def chat(
        request: ChatRequest,
        authorization: str | None = Header(default=None),
        credentials: HTTPAuthorizationCredentials | None = Depends(security),
        ):

        service = app.state.agent_service

        # Tolak header Authorization yang formatnya tidak valid.
        if authorization and credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header.",
            )

        user_id: int | None = None
        cart_api_token: str | None = None

        if credentials is not None:
            cart_api_token = credentials.credentials
            user_id = verify_user_token(cart_api_token)

        if user_id is not None:
            # Namespace percakapan user yang sudah login.
            effective_thread_id = (
                f"user:{user_id}:thread:{request.thread_id}"
            )

            # Aktivitas dicatat menggunakan mekanisme yang sudah ada.
            record_thread_activity(
                db_uri=db_uri,
                effective_thread_id=effective_thread_id,
                user_id=user_id,
                client_thread_id=request.thread_id,
            )
        else:
            # Namespace guest harus terpisah dari namespace user.
            effective_thread_id = (
                f"guest:thread:{request.thread_id}"
            )


        result = await service.chat(
            effective_thread_id,
            request.user_message,
            cart_api_token,
        )

        # Log nama tool yang dipanggil oleh agent.
        for message in result.get("messages", []):
            tool_calls = getattr(message, "tool_calls", [])

            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "unknown")

                logger.info(
                    "AI Agent memanggil tool: %s",
                    tool_name,
                )

        last_message = result["messages"][-1]
        content = last_message.content

        # cek jika response berupa list normalisasi ke text
        if isinstance(content, str):
            response_text = content
        elif isinstance(content, list):
            response_text = "\n".join(
                item["text"]
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            )
        else:
            response_text = ""

        return {
            "thread_id": request.thread_id,
            "response": response_text,
        }