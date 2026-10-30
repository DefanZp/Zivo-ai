from collections.abc import AsyncIterator
from langchain_core.messages import AIMessageChunk

from agent import create_zivo_agent
import logging

logger = logging.getLogger(__name__)

class ZivoAgentService:

    def __init__(self, checkpointer, mcp_tools):
        self.agent = create_zivo_agent(checkpointer, mcp_tools)

    async def chat(
            self, 
            thread_id, 
            user_message, 
            cart_api_token
        ):
        
            config = {
                "configurable": {
                    "thread_id": thread_id  
                }
            }

            result = await self.agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_message
                        }
                    ]
                },
                config,
                context={
                    "cart_api_token": cart_api_token
                }
            )


            return result

    async def chat_stream(
            self,
            thread_id,
            user_message,
            cart_api_token,
    ) -> AsyncIterator[str]: 
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # Baca output model secara bertahap.
        async for chunk in self.agent.astream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            },
            config,
            context={
                "cart_api_token": cart_api_token,
            },
            stream_mode="messages",
            version="v2",
        ):
            # Abaikan event yang bukan output pesan model.
            if chunk["type"] != "messages":
                continue

            message_chunk, metadata = chunk["data"]

            # Hanya teruskan output dari node model.
            if metadata.get("langgraph_node") != "model":
                continue

            # Abaikan pesan yang bukan potongan output AI.
            if not isinstance(message_chunk, AIMessageChunk):
                continue

            # Jangan teruskan pemanggilan tools ke frontend.
            if getattr(message_chunk, "tool_call_chunks", None):
                continue

            if getattr(message_chunk, "tool_calls", None):
                continue

            content = message_chunk.content

            # Output berupa teks biasa.
            if isinstance(content, str):
                if content:
                    yield content

            # Dukungan jika output berbentuk blok teks.
            elif isinstance(content, list):
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "text"
                        and isinstance(item.get("text"), str)
                    ):
                        yield item["text"]


