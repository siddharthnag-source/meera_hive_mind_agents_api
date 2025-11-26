"""Memory retrieval logic for Vishnu Agent."""

import structlog
from typing import List, Optional, Dict, Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.memory.storage import MemoryStorage
from src.memory.nodes import MemoryNode, UserIdentity
from src.utils.config import settings
from src.db.supabase_client import supabase  # <-- uses the same client as save_interaction

logger = structlog.get_logger()


class MemoryRetriever:
    """Handles intelligent memory retrieval for context building."""

    def __init__(self, storage: MemoryStorage):
        """Initialize memory retriever with storage backend."""
        self.storage = storage
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
        )
        logger.info("Memory retriever initialized")

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------
    def get_user_identity(self, user_id: str) -> Optional[UserIdentity]:
        """Retrieve user identity profile."""
        return self.storage.get_user_identity(user_id)

    # -------------------------------------------------------------------------
    # Personal memories (vector + recent fallback)
    # -------------------------------------------------------------------------
    def retrieve_personal_memories(
        self,
        user_id: str,
        query: str,
        limit: Optional[int] = None,
    ) -> List[MemoryNode]:
        """Retrieve relevant personal memories for a user query."""
        if limit is None:
            limit = settings.max_personal_memories

        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Search for relevant memories
            memories = self.storage.search_memories(
                query_embedding=query_embedding,
                user_id=user_id,
                is_hive_mind=False,
                limit=limit,
            )

            # If vector search returns few results, fall back to recent memories
            if len(memories) < limit:
                recent = self.storage.get_recent_memories(
                    user_id=user_id,
                    is_hive_mind=False,
                    limit=limit - len(memories),
                )
                # Merge and deduplicate
                existing_ids = {m.memory_id for m in memories}
                for mem in recent:
                    if mem.memory_id not in existing_ids:
                        memories.append(mem)

            logger.info(
                "Personal memories retrieved",
                user_id=user_id,
                count=len(memories),
                query_preview=query[:50],
            )
            return memories[:limit]

        except Exception as e:
            logger.error(
                "Failed to retrieve personal memories",
                error=str(e),
                user_id=user_id,
            )
            # Fallback to recent memories
            return self.storage.get_recent_memories(
                user_id=user_id,
                is_hive_mind=False,
                limit=limit,
            )

    # -------------------------------------------------------------------------
    # Hive mind memories (vector + recent fallback, same as before)
    # -------------------------------------------------------------------------
    def retrieve_hive_mind_memories(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> List[MemoryNode]:
        """Retrieve relevant hive mind (shared) memories via vector store."""
        if limit is None:
            limit = settings.max_hive_mind_memories

        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Search for relevant hive mind memories
            memories = self.storage.search_memories(
                query_embedding=query_embedding,
                user_id=None,
                is_hive_mind=True,
                limit=limit,
            )

            # If vector search returns few results, fall back to recent memories
            if len(memories) < limit:
                recent = self.storage.get_recent_memories(
                    user_id=None,
                    is_hive_mind=True,
                    limit=limit - len(memories),
                )
                # Merge and deduplicate
                existing_ids = {m.memory_id for m in memories}
                for mem in recent:
                    if mem.memory_id not in existing_ids:
                        memories.append(mem)

            logger.info(
                "Hive mind memories retrieved",
                count=len(memories),
                query_preview=query[:50],
            )
            return memories[:limit]

        except Exception as e:
            logger.error(
                "Failed to retrieve hive mind memories",
                error=str(e),
            )
            # Fallback to recent memories
            return self.storage.get_recent_memories(
                user_id=None,
                is_hive_mind=True,
                limit=limit,
            )

    # -------------------------------------------------------------------------
    # NEW: Direct Supabase retrieval from hive_knowledge table
    # -------------------------------------------------------------------------
    def search_hive_knowledge(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve internal knowledge entries from the hive_knowledge table.

        - Global entries: user_id is null
        - User-specific entries: user_id = given user
        """
        try:
            logger.info(
                "hive_knowledge_search_start",
                user_id=user_id,
                query=query[:80],
            )

            resp = (
                supabase.table("hive_knowledge")
                .select("id, user_id, title, content, metadata, created_at")
                .or_(f"user_id.eq.{user_id},user_id.is.null")
                .ilike("content", f"%{query}%")
                .limit(limit)
                .execute()
            )

            rows = resp.data or []
            logger.info(
                "hive_knowledge_search_done",
                user_id=user_id,
                query=query[:80],
                count=len(rows),
            )
            return rows

        except Exception as e:
            logger.error(
                "hive_knowledge_search_failed",
                user_id=user_id,
                error=str(e),
            )
            return []
