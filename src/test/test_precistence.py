import os

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

db_uri = os.getenv("LANGGRAPH_DB_URI")

if not db_uri:
    raise ValueError("LANGGRAPH_DB_URI belum ditemukan di .env")

config = {
    "configurable": {
        "thread_id": "zivo-state-memory-test"
    }
}

with PostgresSaver.from_conn_string(db_uri) as checkpointer:
    state = checkpointer.get(config)

    print("Checkpoint state dari PostgreSQL:")
    print(state)