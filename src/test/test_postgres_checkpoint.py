import os

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

db_uri = os.getenv("LANGGRAPH_DB_URI")

if not db_uri:
    raise ValueError("LANGGRAPH_DB_URI belum ditemukan di .env")

with PostgresSaver.from_conn_string(db_uri) as checkpointer:
    checkpointer.setup()

    print("PostgresSaver berhasil diinisialisasi.")
    print("Checkpoint tables berhasil dibuat.")