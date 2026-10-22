import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver


# locate main ai directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

load_dotenv()

DB_URI = os.getenv("LANGGRAPH_DB_URI")

if not DB_URI:
    raise ValueError("LANGGRAPH_DB_URI belum ditemukan di .env")


# Cleanup Configuration
RETENTION_DAYS = 30
BATCH_SIZE = 100


def get_inactive_threads(db_uri: str)-> list[tuple]:
    """Find threads that have been inactive beyond the retention period."""

    with psycopg.connect(db_uri) as connection:
        cursor = connection.execute(
            """
            SELECT
                thread_id,
                user_id,
                client_thread_id,
                last_activity_at
            FROM ai_thread_activity
            WHERE last_activity_at < (
                NOW() - (%s * INTERVAL '1 day')
            )
            ORDER BY last_activity_at ASC
            LIMIT %s
            """,
            (RETENTION_DAYS, BATCH_SIZE),
        )

        return cursor.fetchall()


def remove_activity_records(
        db_uri: str,
        thread_ids: list[str],
)-> None:
    """Remove activity metadata after checkpoint deletion."""

    with psycopg.connect(db_uri) as connection:
        connection.execute(
            """
            DELETE FROM ai_thread_activity
            WHERE thread_id = ANY(%s)
            """,
            (thread_ids,),
        )
    
def main()-> None:
    parser = argparse.ArgumentParser(
        description="Clean up inactive LangGraph threads."
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete expired threads.",
    )

    args = parser.parse_args()

    threads = get_inactive_threads(DB_URI)

    print("=== Thread Cleanup ===")
    print(f"Retention period: {RETENTION_DAYS} days")
    print(f"Inactive threads found: {len(threads)}")

    if not threads:
        print("\nNo inactive threads found.")
        return

    for thread_id, user_id, client_thread_id, last_activity_at in threads:
        print(
            f"\nThread: {thread_id}\n"
            f"User: {user_id}\n"
            f"Client thread: {client_thread_id}\n"
            f"Last activity: {last_activity_at}"
        )

    if not args.execute:
        print("\nDry run only. No data was deleted.")
        print("Use --execute to delete these threads.")
        return

    thread_ids = [thread[0] for thread in threads]

    # Delete checkpoint data through the supported API.
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.prune(
            thread_ids,
            strategy="delete",
        )

    # Remove metadata only after checkpoint pruning succeeds.
    remove_activity_records(DB_URI, thread_ids)

    print(f"\nDeleted {len(thread_ids)} inactive threads.")

if __name__ == "__main__":
    main()
