# db.py
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import psycopg
from psycopg_pool import ConnectionPool

HIVE_DB_URL = os.getenv("HIVE_DB_URL")

if not HIVE_DB_URL:
    raise RuntimeError("HIVE_DB_URL is not set")

# Simple global connection pool
pool = ConnectionPool(
    conninfo=HIVE_DB_URL,
    min_size=1,
    max_size=5,
    kwargs={"autocommit": True},
)


@dataclass
class HiveKnowledgeRow:
    id: str
    user_id: Optional[str]
    title: str
    content: str
    metadata: Dict[str, Any]
    created_at: str
    score: float | None = None


def search_hive_knowledge(
    query: str,
    user_id: Optional[str] = None,
    limit: int = 5,
) -> List[HiveKnowledgeRow]:
    """
    Fuzzy search over hive_knowledge.content using pg_trgm similarity.
    If user_id is provided, prefer that user's rows; otherwise search all.
    """
    if not query.strip():
        return []

    sql = """
        select
            id::text,
            user_id,
            title,
            content,
            metadata,
            created_at,
            similarity(content, %(q)s) as score
        from hive_knowledge
        where content %% %(q)s
          and (%(uid)s is null or user_id = %(uid)s)
        order by score desc, created_at desc
        limit %(limit)s
    """

    params = {"q": query, "uid": user_id, "limit": limit}

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    results: List[HiveKnowledgeRow] = []
    for r in rows:
        results.append(
            HiveKnowledgeRow(
                id=r[0],
                user_id=r[1],
                title=r[2],
                content=r[3],
                metadata=r[4] or {},
                created_at=r[5].isoformat() if hasattr(r[5], "isoformat") else str(r[5]),
                score=float(r[6]) if r[6] is not None else None,
            )
        )
    return results


def insert_hive_knowledge(
    *,
    user_id: Optional[str],
    title: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Insert a new knowledge row. Returns the id.
    """
    metadata = metadata or {}

    sql = """
        insert into hive_knowledge (user_id, title, content, metadata)
        values (%(user_id)s, %(title)s, %(content)s, %(metadata)s)
        returning id::text
    """

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            sql,
            {
                "user_id": user_id,
                "title": title,
                "content": content,
                "metadata": metadata,
            },
        )
        row = cur.fetchone()
        return row[0]
