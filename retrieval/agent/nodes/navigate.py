"""
Navigate node for structural document navigation.

Implements Anthropic's routing principle:
- structural_only: Use deterministic navigation (no LLM loop)
- hybrid: Use LLM-based navigator sub-graph
- semantic_only: Skip navigation entirely
"""
import logging
from typing import Dict, Any

from ..state import AgentState
from ..constants import NAVIGATOR_MAX_ITERATIONS
from ...tools import RetrievalTools

logger = logging.getLogger(__name__)


def create_navigate_structure(navigator_subgraph, tracer, retrieval_tools: RetrievalTools = None, text_collection: str = None):
    """
    Create the navigate_structure node function.
    
    Args:
        navigator_subgraph: Compiled navigator sub-graph
        tracer: Langfuse tracer for observability
        retrieval_tools: RetrievalTools instance for deterministic navigation
        text_collection: Collection name for deterministic navigation
    
    Returns:
        Navigate structure node function
    """
    
    def navigate_structure(state: AgentState) -> Dict[str, Any]:
        """
        Navigate document structure using routing-based approach.

        Routes (Anthropic best practice):
        - structural_only: Deterministic navigation (no LLM, fast & reliable)
        - hybrid: LLM-based navigator for complex cases
        - semantic_only: Skip navigation entirely
        """
        plan = state.get("plan") or {}
        route = plan.get("route", "semantic_only")
        hints = plan.get("structural_hints", {})

        # ROUTE: semantic_only - skip navigation entirely
        if route == "semantic_only":
            logger.info("[NAVIGATE] Route=semantic_only, skipping structural navigation")
            return {"structural_seeds": []}

        # Check if we have structural hints
        if not hints.get("has_structural") and route != "structural_only":
            logger.info("[NAVIGATE] No structural intent detected, skipping navigation")
            return {"structural_seeds": []}

        logger.info(f"[NAVIGATE] Route={route}, chapter={hints.get('chapter')}, "
                   f"position={hints.get('position')}, keywords={hints.get('section_keywords')}")

        # ROUTE: structural_only - use deterministic navigation (Anthropic: "start simple")
        if route == "structural_only" and retrieval_tools is not None:
            return _navigate_deterministic(state, hints, retrieval_tools, text_collection or state.get("text_collection"), tracer)

        # ROUTE: hybrid - use LLM-based navigator for complex queries
        return _navigate_with_llm(state, hints, navigator_subgraph, tracer)

    def _navigate_deterministic(state: AgentState, hints: Dict, tools: RetrievalTools, collection: str, tracer) -> Dict[str, Any]:
        """
        Deterministic structural navigation - no LLM loop needed.
        
        Anthropic principle: "Only introduce multi-step agentic systems 
        when simpler solutions are insufficient."
        """
        logger.info("[NAVIGATE] Using DETERMINISTIC navigation (no LLM loop)")
        
        with tracer.span("navigate_deterministic", input={"hints": hints}) as span:
            try:
                # Extract navigation parameters
                chapter = hints.get("chapter")
                position = hints.get("position")
                keywords = hints.get("section_keywords", [])
                
                # Call deterministic navigation
                matches = tools.navigate_structural_deterministic(
                    collection_name=collection,
                    chapter=chapter,
                    position=position,
                    keywords=keywords,
                )
                
                logger.info(f"[NAVIGATE] Deterministic nav found {len(matches)} matches")
                
                if span is not None:
                    span.update(output={
                        "method": "deterministic",
                        "found_count": len(matches),
                        "node_ids": [m.get("node_id") for m in matches[:10]],
                    })
                
                return {"structural_seeds": matches}
                
            except Exception as e:
                logger.error(f"[NAVIGATE] Deterministic nav failed: {e}", exc_info=True)
                if span is not None:
                    span.update(output={"error": str(e)})
                # Fall back to empty seeds (semantic search will handle it)
                return {"structural_seeds": []}

    def _navigate_with_llm(state: AgentState, hints: Dict, subgraph, tracer) -> Dict[str, Any]:
        """
        LLM-based navigation using the navigator sub-graph.
        
        Used for hybrid queries where deterministic approach isn't sufficient.
        """
        logger.info("[NAVIGATE] Using LLM-based navigator sub-graph")

        with tracer.span("navigate_llm", input={"hints": hints}) as span:
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
                result = subgraph.invoke(navigator_input)
                structural_seeds = result.get("structural_seeds", [])

                logger.info(f"[NAVIGATE] LLM sub-graph completed: status={result.get('status')}, "
                           f"found {len(structural_seeds)} seeds, "
                           f"iterations={result.get('iteration')}, "
                           f"scope={result.get('scope_level')}")

                if span is not None:
                    span.update(output={
                        "method": "llm_subgraph",
                        "found_count": len(structural_seeds),
                        "node_ids": [s.get("node_id") for s in structural_seeds],
                        "final_status": result.get("status"),
                        "iterations_used": result.get("iteration"),
                        "scope_level": result.get("scope_level"),
                    })

                return {"structural_seeds": structural_seeds}

            except Exception as e:
                logger.error(f"[NAVIGATE] LLM sub-graph failed: {e}", exc_info=True)
                if span is not None:
                    span.update(output={"error": str(e)})
                return {"structural_seeds": []}
    
    return navigate_structure
