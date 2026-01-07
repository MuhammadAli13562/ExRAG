"""
Loop runner node that invokes the RetrievalLoop sub-graph.
"""
import logging
from typing import Dict, Any

from ..state import AgentState
from ..constants import RETRIEVAL_MAX_ITERATIONS

logger = logging.getLogger(__name__)


def create_run_retrieval_loop(retrieval_loop_subgraph, text_collection: str, tracer):
    """
    Create the run_retrieval_loop node function.
    
    Args:
        retrieval_loop_subgraph: Compiled retrieval loop sub-graph
        text_collection: Name of the text-indexed collection
        tracer: Langfuse tracer for observability
    
    Returns:
        Run retrieval loop node function
    """
    
    def run_retrieval_loop(state: AgentState) -> Dict[str, Any]:
        """
        Run the RetrievalLoop sub-graph for ReAct-style seed processing.

        This wrapper:
        1. Transforms AgentState into RetrievalLoopState input
        2. Invokes the compiled retrieval loop sub-graph
        3. Returns results to the parent graph
        """
        plan = state.get("plan") or {}
        pending_seeds = state.get("pending_seeds") or []

        logger.info(f"[RETRIEVAL_LOOP] Starting with {len(pending_seeds)} seeds")

        with tracer.span("run_retrieval_loop", input={"num_seeds": len(pending_seeds)}) as span:
            # Build RetrievalLoopState input
            loop_input: Dict[str, Any] = {
                "user_query": state["user_query"],
                "subqueries": plan.get("subqueries", [state["user_query"]]),
                "text_collection": text_collection,
                "initial_seeds": pending_seeds,
                "pending_seeds": [],
                "visited_node_ids": [],
                "evidence_pool": [],
                "iteration": 0,
                "max_iterations": RETRIEVAL_MAX_ITERATIONS,
                "scope_level": 0,
                "status": "processing",
                "seeds_processed": 0,
                "total_seeds": len(pending_seeds),
                "reflection": "",
                "last_action": "",
                "high_grade_count": 0,
                "medium_grade_count": 0,
                "final_answer": "",
                "synthesis_attempts": 0,
                "validation": {},
                "synthesis_feedback": "",
                "result": {},
            }

            try:
                result = retrieval_loop_subgraph.invoke(loop_input)
                output = result.get("result", {})

                logger.info(f"[RETRIEVAL_LOOP] Sub-graph completed: status={output.get('status')}, "
                           f"synthesis_attempts={output.get('synthesis_attempts')}, "
                           f"scope_level={output.get('scope_level')}")

                if span is not None:
                    span.update(output={
                        "status": output.get("status"),
                        "evidence_count": len(output.get("evidence_pool", [])),
                        "synthesis_attempts": output.get("synthesis_attempts"),
                        "scope_level": output.get("scope_level"),
                    })

                return {
                    "final_answer": output.get("final_answer", ""),
                    "evidence_pool": output.get("evidence_pool", []),
                    "validation": output.get("validation", {}),
                    "processing_complete": True,
                }

            except Exception as e:
                logger.error(f"[RETRIEVAL_LOOP] Sub-graph failed: {e}", exc_info=True)
                if span is not None:
                    span.update(output={"error": str(e)})
                return {
                    "final_answer": f"Error during retrieval: {str(e)}",
                    "evidence_pool": [],
                    "validation": {"ok": False, "error": str(e)},
                    "processing_complete": True,
                }
    
    return run_retrieval_loop
