"""
Navigate node for structural document navigation.
"""
import logging
from typing import Dict, Any

from ..state import AgentState
from ..constants import NAVIGATOR_MAX_ITERATIONS

logger = logging.getLogger(__name__)


def create_navigate_structure(navigator_subgraph, tracer):
    """
    Create the navigate_structure node function.
    
    Args:
        navigator_subgraph: Compiled navigator sub-graph
        tracer: Langfuse tracer for observability
    
    Returns:
        Navigate structure node function
    """
    
    def navigate_structure(state: AgentState) -> Dict[str, Any]:
        """
        Navigate document structure using the navigator sub-graph.

        This wrapper:
        1. Checks if structural navigation is needed
        2. Transforms AgentState hints into NavigatorState input
        3. Invokes the compiled navigator sub-graph
        4. Returns structural_seeds to the parent graph
        """
        plan = state.get("plan") or {}
        hints = plan.get("structural_hints", {})

        if not hints.get("has_structural"):
            logger.info("[NAVIGATE] No structural intent detected, skipping navigation")
            return {"structural_seeds": []}

        logger.info(f"[NAVIGATE] Starting structural navigation with sub-graph")
        logger.info(f"[NAVIGATE] Hints: chapter={hints.get('chapter')}, position={hints.get('position')}, keywords={hints.get('section_keywords')}")

        with tracer.span("navigate_structure", input={"hints": hints}) as span:
            # Build NavigatorState input
            navigator_input: Dict[str, Any] = {
                "chapter": hints.get("chapter"),
                "position": hints.get("position"),
                "section_keywords": hints.get("section_keywords", []),
                "goal": "",  # Will be set by parse_goal node
                "messages": [],
                "iteration": 0,
                "max_iterations": NAVIGATOR_MAX_ITERATIONS,
                "scope_level": 0,
                "status": "searching",
                "explored_centers": [],
                "seen_nodes": {},
                "candidate_nodes": [],
                "found_nodes": [],
                "landmark_node_id": None,
                "current_position": "",
                "last_action": "",
                "reflection": "",
                "structural_seeds": [],
            }

            # Invoke sub-graph
            try:
                result = navigator_subgraph.invoke(navigator_input)
                structural_seeds = result.get("structural_seeds", [])

                logger.info(f"[NAVIGATE] Sub-graph completed: status={result.get('status')}, "
                           f"found {len(structural_seeds)} seeds, "
                           f"iterations={result.get('iteration')}, "
                           f"scope={result.get('scope_level')}")

                if span is not None:
                    span.update(output={
                        "found_count": len(structural_seeds),
                        "node_ids": [s.get("node_id") for s in structural_seeds],
                        "final_status": result.get("status"),
                        "iterations_used": result.get("iteration"),
                        "scope_level": result.get("scope_level"),
                    })

                return {"structural_seeds": structural_seeds}

            except Exception as e:
                logger.error(f"[NAVIGATE] Sub-graph failed: {e}", exc_info=True)
                if span is not None:
                    span.update(output={"error": str(e)})
                return {"structural_seeds": []}
    
    return navigate_structure
