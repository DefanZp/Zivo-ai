import os
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

from agent import create_zivo_agent
from service import ZivoAgentService


load_dotenv()

# fungsi untuk create run agent
def run_agent(agent, thread_id, user_message):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    return agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        },
        config,
    )

db_uri = os.getenv("LANGGRAPH_DB_URI")

if not db_uri:
    raise ValueError("LANGGRAPH_DB_URI belum ditemukan di .env") 

# Run agent dengan checkpointer postgres
with PostgresSaver.from_conn_string(db_uri) as checkpointer:

    service = ZivoAgentService(checkpointer)

    thread_id = "db7de6e7-3f25-4753-a4ec-39e3cea5ef37"

    result = service.chat(
        thread_id,
        "Pilih produk dengan ID 64."
    )

    print("Thread ID:")
    print(thread_id)

    print("\nSelected product:")
    print(result.get("selected_product_id"))