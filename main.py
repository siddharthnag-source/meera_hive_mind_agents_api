"""Main entry point for Meera OS."""

import logging
import structlog
import sys
from typing import Optional, Dict, Any

from src.graph.workflow import MeeraWorkflow
from src.db.supabase_client import save_interaction

# Configure logging
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=False,
)

logger = structlog.get_logger()

# Reusable global workflow
workflow: Optional[MeeraWorkflow] = None


def init_workflow() -> MeeraWorkflow:
    global workflow
    if workflow is None:
        workflow = MeeraWorkflow()
    return workflow


def run_meera(user_id: str, user_message: str) -> Dict[str, Any]:
    """
    Core entrypoint used by API and CLI.
    """
    wf = init_workflow()

    logger.info("Processing user message", user_id=user_id, message=user_message)

    result = wf.invoke(
        user_id=user_id,
        user_message=user_message
    )

    # Persist into Supabase
    try:
        save_interaction(user_id, user_message, result)
    except Exception as e:
        logger.error("Failed to save interaction to Supabase", error=str(e))

    return result


def main():
    """Main entry point for CLI usage."""
    logger.info("Meera OS starting up")

    try:
        if len(sys.argv) > 1:
            user_id = sys.argv[1]
            user_message = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "what is the impact of consciousness in 2030?"
        else:
            user_id = "39383"
            user_message = "what is the impact of consciousness in 2030?"

        result = run_meera(user_id, user_message)

        print("\n" + "=" * 80)
        print("MEERA RESPONSE:")
        print("=" * 80)
        print(result["response"])
        print("=" * 80)
        print(f"\nIntent: {result.get('intent', 'N/A')}")
        print(f"Memories created: {len(result.get('memory_ids', []))}")
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully")
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        sys.exit(1)
    finally:
        if workflow is not None:
            workflow.close()


if __name__ == "__main__":
    main()
