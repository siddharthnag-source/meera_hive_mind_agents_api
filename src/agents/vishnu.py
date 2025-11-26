"""Vishnu Agent: Dynamic system prompt builder."""

import structlog
from typing import Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from src.memory.retrieval import MemoryRetriever
from src.memory.nodes import UserIdentity
from src.prompts.templates import PromptBuilder
from src.utils.config import settings, config_loader

logger = structlog.get_logger()


class VishnuAgent:
    """
    Vishnu Agent: The dynamic system prompt builder.

    Responsibilities:
    - Detect user intent
    - Update user identity
    - Retrieve relevant memories (personal + hive mind + hive_knowledge table)
    - Build final dynamic system prompt
    - Pass prompt to Brahma
    """

    def __init__(self, memory_retriever: MemoryRetriever):
        """Initialize Vishnu Agent."""
        self.memory_retriever = memory_retriever
        self.prompt_builder = PromptBuilder()
        self.config = config_loader.load()
        self.agent_config = self.config.get("agents", {}).get("vishnu", {})

        # Intent detection LLM (lightweight)
        self.intent_llm = (
            ChatGoogleGenerativeAI(
                model="gemini-flash-latest",
                google_api_key=settings.gemini_api_key,
                temperature=0.3,
            )
            if self.agent_config.get("intent_detection", True)
            else None
        )

        logger.info("Vishnu Agent initialized")

    # -------------------------------------------------------------------------
    # Main process
    # -------------------------------------------------------------------------
    def process(self, user_id: str, user_message: str) -> Dict[str, Any]:
        """
        Main processing method for Vishnu Agent.

        Returns:
            Dictionary containing:
            - system_prompt: The complete dynamic system prompt
            - user_identity: Updated user identity
            - personal_memories: Retrieved personal memories
            - hive_mind_memories: Retrieved hive mind memories (vector store)
            - intent: Detected user intent
            - user_message: Original user message
            - hive_knowledge: Raw rows from hive_knowledge table
        """
        logger.info(
            "Vishnu processing started",
            user_id=user_id,
            message_preview=user_message[:50],
        )

        # Step 1: Detect intent
        intent = self._detect_intent(user_message) if self.intent_llm else None

        # Step 2: Get/Update user identity
        user_identity = self._get_or_create_identity(user_id)
        user_identity = self._update_identity(user_identity, user_message, intent)

        # Step 3: Retrieve memories (existing behaviour)
        personal_memories = self.memory_retriever.retrieve_personal_memories(
            user_id=user_id,
            query=user_message,
        )

        hive_mind_memories = self.memory_retriever.retrieve_hive_mind_memories(
            query=user_message,
        )

        # Step 3b: NEW — retrieve hive_knowledge rows from Supabase
        hive_knowledge_rows = self.memory_retriever.search_hive_knowledge(
            user_id=user_id,
            query=user_message,
            limit=5,
        )

        # Step 4: Build base dynamic system prompt using your existing template
        base_system_prompt = self.prompt_builder.build_system_prompt(
            user_identity=user_identity,
            personal_memories=personal_memories,
            hive_mind_memories=hive_mind_memories,
            user_query=user_message,
        )

        # Step 5: Append a clear Hive Knowledge block so Brahma MUST see it
        system_prompt = base_system_prompt
        if hive_knowledge_rows:
            lines = []
            for row in hive_knowledge_rows:
                title = row.get("title") or "Entry"
                content = row.get("content") or ""
                lines.append(f"- {title}: {content}")

            hive_block = (
                "\n[Hive Knowledge]\n"
                "The following internal knowledge entries are available to you. "
                "They are trusted internal data for this user. "
                "If the user's question is answered by these entries, you MUST "
                "use them directly in your response, even if they look like "
                "internal or confidential project details.\n"
                + "\n".join(lines)
                + "\n"
            )

            system_prompt = base_system_prompt + hive_block

        logger.info(
            "Vishnu processing completed",
            user_id=user_id,
            intent=intent,
            personal_memories_count=len(personal_memories),
            hive_mind_memories_count=len(hive_mind_memories),
            hive_knowledge_count=len(hive_knowledge_rows),
        )

        return {
            "system_prompt": system_prompt,
            "user_identity": user_identity,
            "personal_memories": personal_memories,
            "hive_mind_memories": hive_mind_memories,
            "intent": intent,
            "user_message": user_message,
            "hive_knowledge": hive_knowledge_rows,
        }

    # -------------------------------------------------------------------------
    # Intent detection
    # -------------------------------------------------------------------------
    def _detect_intent(self, user_message: str) -> Optional[str]:
        """Detect user intent from message."""
        if not self.intent_llm:
            return None

        try:
            prompt = f"""Analyze the following user message and identify the primary intent in one short phrase (e.g., "question about consciousness", "emotional support", "technical inquiry", "philosophical discussion").

User message: {user_message}

Intent:"""

            response = self.intent_llm.invoke(prompt)
            intent = (
                response.content.strip()
                if hasattr(response, "content")
                else str(response).strip()
            )

            logger.debug("Intent detected", intent=intent)
            return intent

        except Exception as e:
            # Don't fail the workflow if intent detection fails (e.g., quota issues)
            logger.warning(
                "Failed to detect intent, continuing without intent",
                error=str(e),
            )
            return None

    # -------------------------------------------------------------------------
    # Identity helpers
    # -------------------------------------------------------------------------
    def _get_or_create_identity(self, user_id: str) -> UserIdentity:
        """Get existing user identity or create a new one."""
        identity = self.memory_retriever.get_user_identity(user_id)

        if identity is None:
            identity = UserIdentity(user_id=user_id)
            logger.info("New user identity created", user_id=user_id)

        return identity

    def _update_identity(
        self,
        identity: UserIdentity,
        user_message: str,
        intent: Optional[str],
    ) -> UserIdentity:
        """Update user identity based on message and intent."""
        if not self.agent_config.get("identity_update", True):
            return identity

        from datetime import datetime

        # Basic timestamp update (you can extend this later)
        identity.updated_at = datetime.utcnow()
        return identity
