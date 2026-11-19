import json
import os
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Literal
from fastapi import (
    Depends, 
    FastAPI, 
    Header,
    HTTPException, 
    status
)
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import httpx
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import ToolMessage
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
    await ensure_thread_activity_table(db_uri)

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

class ChatResumeRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100)
    interrupt_id: str = Field(min_length=1, max_length=200)
    decision: Literal["approve", "reject"]

async def verify_user_token(token: str) -> int:
    try:
        async with httpx.AsyncClient(
            timeout=10.0
        ) as client:

            response = await client.get(
                f"{backend_url}/api/user/profile",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
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

    if authorization and credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header.",
        )

    user_id: int | None = None
    cart_api_token: str | None = None

    if credentials is not None:
        cart_api_token = credentials.credentials
        user_id = await verify_user_token(cart_api_token)

    if user_id is not None:
        effective_thread_id = (
            f"user:{user_id}:thread:{request.thread_id}"
        )

        await record_thread_activity(
            db_uri=db_uri,
            effective_thread_id=effective_thread_id,
            user_id=user_id,
            client_thread_id=request.thread_id,
        )
    else:
        effective_thread_id = (
            f"guest:thread:{request.thread_id}"
        )

    # Jalankan AI agent.
    try:
        result = await service.chat(
            effective_thread_id,
            request.user_message,
            cart_api_token,
        )

    except Exception:
        logger.exception("[CHAT] AI agent execution failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process your message. Please try again.",
        )

    # Validasi hasil agent.
    if not isinstance(result, dict):
        logger.error("[CHAT] Agent returned an invalid result.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to generate AI response.",
        )

    messages = result.get("messages")

    if not isinstance(messages, list) or not messages:
        logger.error("[CHAT] Agent returned no messages.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to generate AI response.",
        )

    # Log metadata tool tanpa mencatat data sensitif.
    for message in messages:
        tool_calls = getattr(message, "tool_calls", [])

        for tool_call in tool_calls:
            logger.info(
                "[TRACE] AI TOOL CALL | name=%s | id=%s",
                tool_call.get("name"),
                tool_call.get("id"),
            )

        if isinstance(message, ToolMessage):
            logger.info(
                "[TRACE] TOOL RESULT | name=%s | tool_call_id=%s",
                message.name or "unknown",
                message.tool_call_id,
            )

    # Ambil dan normalisasi response terakhir.
    last_message = messages[-1]
    content = getattr(last_message, "content", None)

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

    if not response_text.strip():
        logger.error("[CHAT] Agent returned empty response content.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to generate AI response.",
        )

    return {
        "thread_id": request.thread_id,
        "response": response_text,
    }


@app.post("/chat/stream")
async def chat_stream(
        request: ChatRequest,
        authorization: str | None = Header(default=None),
        credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> StreamingResponse:

    service = app.state.agent_service

    # Validasi header Authorization.
    if authorization and credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header.",
        )

    user_id: int | None = None
    cart_api_token: str | None = None

    if credentials is not None:
        cart_api_token = credentials.credentials
        user_id = await verify_user_token(cart_api_token)

    # Pisahkan thread user dan guest.
    if user_id is not None:
        effective_thread_id = (
            f"user:{user_id}:thread:{request.thread_id}"
        )

        await record_thread_activity(
            db_uri=db_uri,
            effective_thread_id=effective_thread_id,
            user_id=user_id,
            client_thread_id=request.thread_id,
        )
    else:
        effective_thread_id = (
            f"guest:thread:{request.thread_id}"
        )

    async def event_generator():
        has_pending_approval = False

        try:
            async for stream_event in service.chat_stream(
                effective_thread_id,
                request.user_message,
                cart_api_token,
            ):
                if not isinstance(stream_event, dict):
                    continue

                event_type = stream_event.get("type")

                # Handle token streaming.
                if event_type == "token":
                    content = stream_event.get("content")

                    if not isinstance(content, str) or not content:
                        continue

                    payload = json.dumps(
                        {"content": content},
                        ensure_ascii=False,
                    )

                    yield f"event: token\ndata: {payload}\n\n"

                # Handle approval request.
                elif event_type == "approval_required":
                    has_pending_approval = True

                    payload = json.dumps(
                        {
                            "interrupt_id": stream_event["interrupt_id"],
                            "approval": stream_event["approval"],
                        },
                        ensure_ascii=False,
                    )

                    yield f"event: approval_required\ndata: {payload}\n\n"

            # Beri tahu frontend bahwa streaming sudah selesai.
            done_payload = json.dumps({
                "status": (
                    "waiting_for_approval"
                    if has_pending_approval
                    else "completed"
                )
            })

            yield f"event: done\ndata: {done_payload}\n\n"

        except Exception:
            logger.exception("[STREAM] AI response stream failed")

            payload = json.dumps(
                {"message": "Unable to stream AI response."}
            )

            yield f"event: error\ndata: {payload}\n\n"


    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

@app.post("/chat/resume/stream")
async def chat_resume_stream(
    request: ChatResumeRequest,
    authorization: str | None = Header(default=None),
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> StreamingResponse:

    service = app.state.agent_service

    # Validasi header Authorization.
    if authorization and credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header.",
        )

    # Resume cart action hanya tersedia untuk pengguna login.
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required to resume cart actions.",
        )

    cart_api_token = credentials.credentials
    user_id = await verify_user_token(cart_api_token)

    # Gunakan thread ID yang sama dengan request awal.
    effective_thread_id = (
        f"user:{user_id}:thread:{request.thread_id}"
    )

    await record_thread_activity(
        db_uri=db_uri,
        effective_thread_id=effective_thread_id,
        user_id=user_id,
        client_thread_id=request.thread_id,
    )

    async def event_generator():
        has_pending_approval = False

        try:
            async for stream_event in service.resume_chat_stream(
                effective_thread_id,
                request.interrupt_id,
                request.decision,
                cart_api_token,
            ):
                if not isinstance(stream_event, dict):
                    continue

                event_type = stream_event.get("type")

                # Handle token streaming.
                if event_type == "token":
                    content = stream_event.get("content")

                    if not isinstance(content, str) or not content:
                        continue

                    payload = json.dumps(
                        {"content": content},
                        ensure_ascii=False,
                    )

                    yield f"event: token\ndata: {payload}\n\n"

                # Handle approval request berikutnya.
                elif event_type == "approval_required":
                    has_pending_approval = True

                    payload = json.dumps(
                        {
                            "interrupt_id": stream_event["interrupt_id"],
                            "approval": stream_event["approval"],
                        },
                        ensure_ascii=False,
                    )

                    yield f"event: approval_required\ndata: {payload}\n\n"

            # Beri tahu frontend bahwa stream sudah selesai.
            done_payload = json.dumps({
                "status": (
                    "waiting_for_approval"
                    if has_pending_approval
                    else "completed"
                )
            })

            yield f"event: done\ndata: {done_payload}\n\n"

        except Exception:
            logger.exception("[RESUME STREAM] AI response stream failed")

            payload = json.dumps(
                {"message": "Unable to resume AI response stream."}
            )

            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )