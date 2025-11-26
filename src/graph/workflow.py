"""LangGraph workflow orchestrating Vishnu → Brahma → Shiva flow."""

import structlog
from typing import TypedDict, Annotated, Sequence, Dict, Any
from langgraph.graph import StateGraph, END

from src.agents.vishnu import VishnuAgent
from src.agents.brahma import BrahmaInterface
from src.agents.shiva import ShivaAgent
from src.memory.storage import MemoryStorage
from src.memory.retrieval import MemoryRetriever

logger = structlog.get_logger()


class AgentState(TypedDict):
    """State passed between agents in the workflow."""
    user_id: str
    user_message: str

    # Extra context from Hive Mind DB (Supabase)
    hive_context: str                    # NEW

    # Vishnu outputs
    system_prompt: str
    user_identity: dict
    personal_memories: list
    hive_mind_memories: list
    intent: str

    # Brahma outputs
    response: str
    full_conversation: dict

    # Shiva outputs
    memory_ids: list

    # Metadata
    conversation_history: Annotated[Sequence[dict], lambda a, b: a + [b]]


class MeeraWorkflow:
    """Main workflow orchestrating the Meera OS agent system."""

    def __init__(self):
        """Initialize the workflow with all agents."""
        # Initialize storage and retrieval
        self.memory_storage = MemoryStorage()
        self.memory_retriever = MemoryRetriever(self.memory_storage)

        # Initialize agents
        self.vishnu = VishnuAgent(self.memory_retriever)
        self.brahma = BrahmaInterface()
        self.shiva = ShivaAgent(self.memory_storage)

        # Build graph
        self.graph = self._build_graph()

        logger.info("Meera workflow initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("vishnu", self._vishnu_node)
        workflow.add_node("brahma", self._brahma_node)
        workflow.add_node("shiva", self._shiva_node)

        # Define edges
        workflow.set_entry_point("vishnu")
        workflow.add_edge("vishnu", "brahma")
        workflow.add_edge("brahma", "shiva")
        workflow.add_edge("shiva", END)

        # Compile graph
        return workflow.compile()

    def _vishnu_node(self, state: AgentState) -> AgentState:
        """Vishnu agent node: Build dynamic system prompt."""
        logger.info("Vishnu node executing", user_id=state["user_id"])

        try:
            result = self.vishnu.process(
                user_id=state["user_id"],
                user_message=state["user_message"]
            )

            # Update state with Vishnu outputs
            state["system_prompt"] = result["system_prompt"]

            # Convert Pydantic models to dicts
            user_identity = result["user_identity"]
            if hasattr(user_identity, "model_dump"):
                state["user_identity"] = user_identity.model_dump()
            elif hasattr(user_identity, "dict"):
                state["user_identity"] = user_identity.dict()
            else:
                state["user_identity"] = user_identity

            # Convert memory lists
            state["personal_memories"] = [
                m.model_dump() if hasattr(m, "model_dump") else (m.dict() if hasattr(m, "dict") else m)
                for m in result["personal_memories"]
            ]
            state["hive_mind_memories"] = [
                m.model_dump() if hasattr(m, "model_dump") else (m.dict() if hasattr(m, "dict") else m)
                for m in result["hive_mind_memories"]
            ]
            state["intent"] = result.get("intent", "")

            logger.info("Vishnu node completed", user_id=state["user_id"])
            return state

        except Exception as e:
            logger.error("Vishnu node failed", error=str(e), user_id=state["user_id"])
            raise

    def _brahma_node(self, state: AgentState) -> AgentState:
        """Brahma agent node: Generate LLM response."""
        logger.info("Brahma node executing", user_id=state["user_id"])

        try:
            # Get conversation history from state
            conversation_history = state.get("conversation_history", [])
            if conversation_history and isinstance(conversation_history[0], list):
                conversation_history = []

            # Pull hive_context from state (may be empty)
            hive_context = state.get("hive_context", "") or ""

            # Extend system_prompt with Hive Mind DB context if present
            system_prompt = state["system_prompt"]
            if hive_context:
                system_prompt = (
                    system_prompt
                    + "\n\n[Hive Mind DB context - prefer this if relevant or if it "
                    "conflicts with generic knowledge:]\n"
                    + hive_context
                )

            result = self.brahma.generate_response(
                system_prompt=system_prompt,
                user_message=state["user_message"],
                conversation_history=conversation_history
            )

            # Update state with Brahma outputs
            state["response"] = result["response"]
            state["full_conversation"] = result["full_conversation"]

            # Update conversation history
            if "conversation_history" not in state:
                state["conversation_history"] = []

            state["conversation_history"].append({
                "role": "user",
                "content": state["user_message"]
            })
            state["conversation_history"].append({
                "role": "assistant",
                "content": result["response"]
            })

            logger.info("Brahma node completed", user_id=state["user_id"])
            return state

        except Exception as e:
            logger.error("Brahma node failed", error=str(e), user_id=state["user_id"])
            raise

    def _shiva_node(self, state: AgentState) -> AgentState:
        """Shiva agent node: Update memories."""
        logger.info("Shiva node executing", user_id=state["user_id"])

        try:
            # Reconstruct user identity from dict
            from src.memory.nodes import UserIdentity
            user_identity_dict = state.get("user_identity", {})
            user_identity = UserIdentity(**user_identity_dict) if user_identity_dict else None

            memory_ids = self.shiva.process(
                user_id=state["user_id"],
                full_conversation=state["full_conversation"],
                user_identity=user_identity
            )

            # Update state with Shiva outputs
            state["memory_ids"] = memory_ids

            logger.info(
                "Shiva node completed",
                user_id=state["user_id"],
                memories_created=len(memory_ids),
            )
            return state

        except Exception as e:
            logger.error("Shiva node failed", error=str(e), user_id=state["user_id"])
            # Don't fail the workflow if memory update fails
            state["memory_ids"] = []
            return state

    def invoke(
        self,
        user_id: str,
        user_message: str,
        conversation_history: list = None,
        hive_context: str = "",                      # NEW
    ) -> dict:
        """
        Invoke the workflow with a user message.

        Args:
            user_id: User identifier
            user_message: User's message
            conversation_history: Optional previous conversation messages
            hive_context: Extra retrieved knowledge from DB (hive_knowledge)
        Returns:
            Final state containing response and metadata
        """
        logger.info(
            "Workflow invoked",
            user_id=user_id,
            message_preview=user_message[:50],
        )

        initial_state: AgentState = {
            "user_id": user_id,
            "user_message": user_message,
            "hive_context": hive_context or "",      # NEW
            "system_prompt": "",
            "user_identity": {},
            "personal_memories": [],
            "hive_mind_memories": [],
            "intent": "",
            "response": "",
            "full_conversation": {},
            "memory_ids": [],
            "conversation_history": conversation_history or [],
        }

        try:
            final_state = self.graph.invoke(initial_state)

            logger.info(
                "Workflow completed",
                user_id=user_id,
                response_length=len(final_state.get("response", "")),
            )

            return {
                "response": final_state.get("response", ""),
                "user_id": user_id,
                "intent": final_state.get("intent", ""),
                "memory_ids": final_state.get("memory_ids", []),
                "conversation_history": final_state.get("conversation_history", []),
            }

        except Exception as e:
            logger.error("Workflow failed", error=str(e), user_id=user_id)
            raise

    def close(self):
        """Close all connections."""
        self.memory_storage.close()
        logger.info("Workflow closed")
