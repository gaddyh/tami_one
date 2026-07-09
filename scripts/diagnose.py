"""Diagnostic script: connect to prod DB and run all health checks."""

import os
import sys

from sqlalchemy import create_engine, text


QUERIES = [
    ("1. Messages ingested by chat", """
        SELECT chat_id, direction,
               count(*) AS total,
               count(*) FILTER (WHERE processed_at IS NOT NULL) AS processed,
               count(*) FILTER (WHERE processed_at IS NULL) AS unprocessed,
               count(*) FILTER (WHERE extraction_attempts > 0) AS attempted,
               max(sent_at) AS latest_message
        FROM chatmessage
        GROUP BY chat_id, direction
        ORDER BY latest_message DESC
        LIMIT 20
    """),
    ("2. Conversations created", """
        SELECT c.chat_id, left(c.id, 8) AS conv_id, c.summary,
               c.last_message_at, c.started_at,
               count(m.id) AS message_count
        FROM conversation c
        LEFT JOIN chatmessage m ON m.conversation_id = c.id
        GROUP BY c.chat_id, c.id, c.summary, c.last_message_at, c.started_at
        ORDER BY c.last_message_at DESC
        LIMIT 20
    """),
    ("3. Total commitments", """
        SELECT count(*) AS total_commitments FROM commitmentitem
    """),
    ("4. Commitments by status and chat", """
        SELECT chat_id, status, count(*) AS cnt, max(created_at) AS latest
        FROM commitmentitem
        GROUP BY chat_id, status
        ORDER BY latest DESC
    """),
    ("5. Unprocessed messages (stuck / failing)", """
        SELECT chat_id, provider_message_id, direction,
               left(text, 60) AS text_preview, sent_at,
               extraction_attempts, left(conversation_id, 8) AS conv_id
        FROM chatmessage
        WHERE processed_at IS NULL
        ORDER BY sent_at DESC
        LIMIT 30
    """),
    ("6. Dead letters (hit extraction attempt cap)", """
        SELECT chat_id, provider_message_id,
               left(text, 60) AS text_preview,
               extraction_attempts, sent_at
        FROM chatmessage
        WHERE extraction_attempts >= 3
        ORDER BY sent_at DESC
        LIMIT 20
    """),
    ("7. Conversation_id coverage on messages", """
        SELECT count(*) AS total,
               count(*) FILTER (WHERE conversation_id IS NULL) AS no_conversation,
               count(*) FILTER (WHERE conversation_id IS NOT NULL) AS has_conversation
        FROM chatmessage
    """),
    ("8. Contacts", """
        SELECT count(*) AS total_contacts, max(created_at) AS latest FROM contact
    """),
    ("9. Tables in schema", """
        SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename
    """),
    ("10. Messages processed in last 24h", """
        SELECT chat_id, count(*) AS processed_count, max(processed_at) AS last_processed
        FROM chatmessage
        WHERE processed_at > now() - interval '24 hours'
        GROUP BY chat_id
        ORDER BY last_processed DESC
    """),
]


def main():
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        print("ERROR: Set DATABASE_URL env var first.")
        print("  DATABASE_URL='postgresql://...' python scripts/diagnose.py")
        sys.exit(1)

    engine = create_engine(db_url)

    with engine.connect() as conn:
        for title, sql in QUERIES:
            print(f"\n{'=' * 70}")
            print(f"  {title}")
            print(f"{'=' * 70}")

            try:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                cols = list(result.keys())

                if not rows:
                    print("  (no rows)")
                    continue

                # Print header
                print("  " + " | ".join(f"{c:<20}" for c in cols))
                print("  " + "-" * (22 * len(cols)))

                for row in rows:
                    vals = []
                    for v in row:
                        if v is None:
                            vals.append("NULL")
                        elif isinstance(v, str) and len(v) > 20:
                            vals.append(v[:20])
                        else:
                            vals.append(str(v))
                    print("  " + " | ".join(f"{v:<20}" for v in vals))

            except Exception as e:
                print(f"  ERROR: {e}")

    engine.dispose()


if __name__ == "__main__":
    main()
