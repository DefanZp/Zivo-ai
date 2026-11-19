from collections.abc import AsyncIterator
from langchain_core.messages import AIMessageChunk
from langgraph.types import Command

from agent import create_zivo_agent

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
    ) -> AsyncIterator[dict]: 
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
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            if chunk["type"] == "updates":
                update_data = chunk["data"]

                interrupt_events = update_data.get(
                    "__interrupt__",
                    (),
                )

                for interrupt_event in interrupt_events:
                    yield {
                        "type": "approval_required",
                        "interrupt_id": interrupt_event.id,
                        "approval": interrupt_event.value,
                    }

                continue

            # Abaikan event selain output pesan model.
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
                    yield {
                        "type": "token",
                        "content": content,
                    }

            # jika output berbentuk blok teks.
            elif isinstance(content, list):
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "text"
                        and isinstance(item.get("text"), str)
                    ):
                        yield {
                            "type": "token",
                            "content": item["text"],
                        }

    async def resume_chat_stream(
        self,
        thread_id,
        interrupt_id,
        decision,
        cart_api_token,
    ) -> AsyncIterator[dict]:

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # Siapkan keputusan untuk melanjutkan interrupt.
        resume_command = Command(
            resume={
                interrupt_id: {
                    "decision": decision,
                }
            }
        )

        # Lanjutkan graph dengan checkpoint sebelumnya.
        async for chunk in self.agent.astream(
            resume_command,
            config,
            context={
                "cart_api_token": cart_api_token,
            },
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            # Handle update dari graph.
            if chunk["type"] == "updates":
                update_data = chunk["data"]

                interrupt_events = update_data.get(
                    "__interrupt__",
                    (),
                )

                for interrupt_event in interrupt_events:
                    yield {
                        "type": "approval_required",
                        "interrupt_id": interrupt_event.id,
                        "approval": interrupt_event.value,
                    }

                continue

            # Abaikan event selain output pesan model.
            if chunk["type"] != "messages":
                continue

            message_chunk, metadata = chunk["data"]

            # Hanya teruskan output dari node model.
            if metadata.get("langgraph_node") != "model":
                continue

            if not isinstance(message_chunk, AIMessageChunk):
                continue

            # Jangan teruskan pemanggilan tools ke frontend.
            if getattr(message_chunk, "tool_call_chunks", None):
                continue

            if getattr(message_chunk, "tool_calls", None):
                continue

            content = message_chunk.content

            if isinstance(content, str):
                if content:
                    yield {
                        "type": "token",
                        "content": content,
                    }

            elif isinstance(content, list):
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "text"
                        and isinstance(item.get("text"), str)
                    ):
                        yield {
                            "type": "token",
                            "content": item["text"],
                        }
