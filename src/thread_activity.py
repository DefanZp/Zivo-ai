import psycopg


def ensure_thread_activity_table(db_uri: str)-> None:
    """Create the thread activity table if it does not exist."""

    with psycopg.connect(db_uri) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_thread_activity (
                thread_id TEXT PRIMARY KEY,
                user_id BIGINT NOT NULL,
                client_thread_id TEXT NOT NULL,
                last_activity_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ai_thread_activity_last_activity
            ON ai_thread_activity (last_activity_at)
            """
        )


def record_thread_activity(
        db_uri: str,
        effective_thread_id: str,
        user_id: int,
        client_thread_id: str,
)->None:
    """Insert or update the last activity timestamp for a thread."""

    with psycopg.connect(db_uri) as connection:
        connection.execute(
            """
            INSERT INTO ai_thread_activity (
                thread_id,
                user_id,
                client_thread_id,
                last_activity_at
            )
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (thread_id)
            DO UPDATE SET
                user_id = EXCLUDED.user_id,
                client_thread_id = EXCLUDED.client_thread_id,
                last_activity_at = NOW()
            """,
            (
                effective_thread_id,
                user_id,
                client_thread_id,
            ),
        )