# src/db/supabase_client.py
import os
from typing import Any, Dict
from supabase import create_client, Client

_supabase: Client | None = None


def get_supabase() -> Client:
    """
    Lazily create and return a Supabase client using the service role key.
    """
    global _supabase
    if _supabase is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        _supabase = create_client(url, key)
    return _supabase


def save_interaction(
    user_id: str,
    user_message: str,
    result: Dict[str, Any],
) -> None:
    """
    Persist one Meera interaction into Supabase.

    Expected table schema (can adjust later):

      meera_interactions(
        id uuid primary key default gen_random_uuid(),
        user_id text,
        message text,
        response text,
        intent text,
        memory_ids jsonb,
        created_at timestamptz default now()
      )
    """
    supabase = get_supabase()

    data = {
        "user_id": user_id,
        "message": user_message,
        "response": result.get("response"),
        "intent": result.get("intent"),
        "memory_ids": result.get("memory_ids"),
    }

    supabase.table("meera_interactions").insert(data).execute()
