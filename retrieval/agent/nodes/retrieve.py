"""
Retrieve seeds node for the retrieval agent.
"""
import time
import logging
from typing import Dict, Any

from ..state import AgentState
from ...tools import RetrievalTools

logger = logging.getLogger(__name__)


def create_retrieve_seeds(
    retrieval_tools_instance: RetrievalTools,
    text_collection: str,
    tracer
):
    """
    Create the retrieve_seeds node function.
    
    Args:
        retrieval_tools_instance: RetrievalTools instance for document access
        text_collection: Name of the text-indexed collection
        tracer: Langfuse tracer for observability
    
    Returns:
        Retrieve seeds node function
    """
    
    def retrieve_seeds(state: AgentState) -> Dict[str, Any]:
        """
        Retrieve initial seed candidates (max 5) using all subqueries.
        Seeds are the starting points for per-seed exploration.

        Merges structural seeds (from navigation) with semantic search results.
        Structural seeds are prioritized (appear first).
        
        OPTIMIZATION: If structural navigation found high-quality seeds (relevance_grade="high"),
        skip semantic search to save API calls and time.
        """
        plan = state.get("plan") or {}
        subqueries = plan.get("subqueries", [state["user_query"]])
        top_k = int(plan.get("top_k") or 8)
        max_seeds = int(plan.get("max_seeds") or 5)
        structural_seeds = state.get("structural_seeds") or []

        logger.info(f"[STAGE] retrieve_seeds: {len(subqueries)} subqueries, top_k={top_k}, max_seeds={max_seeds}")
        logger.info(f"[STAGE] retrieve_seeds: {len(structural_seeds)} structural seeds from navigation")

        with tracer.span("retrieve_seeds", input={"subqueries": subqueries, "top_k": top_k, "num_structural": len(structural_seeds)}) as span:
            all_candidates: Dict[str, Dict[str, Any]] = {}  # node_id → node (dedup)

            # 1. Add structural seeds first (highest priority)
            for s in structural_seeds:
                node_id = s.get("node_id")
                if node_id:
                    all_candidates[node_id] = s

            # Check if structural seeds are high-quality (navigation successfully found target)
            has_high_quality_structural = any(
                s.get("relevance_grade") == "high" 
                for s in structural_seeds
            )

            # 2. Skip semantic search if we have high-quality structural seeds
            # This saves API calls when navigation successfully found the target
            if has_high_quality_structural and structural_seeds:
                logger.info(f"[RETRIEVE] High-quality structural seeds found, skipping semantic search to save API calls")
                tracer.event(
                    "retrieve_seeds.skip_semantic",
                    input={"reason": "high_quality_structural_seeds", "num_structural": len(structural_seeds)},
                )
            else:
                # 2. Do semantic search to complement structural seeds
                for query in subqueries:
                    t0 = time.time()
                    results = retrieval_tools_instance.search_by_text(query, text_collection, top_k=top_k)
                    dt = int((time.time() - t0) * 1000)

                    tracer.event(
                        "retrieve_seeds.subquery",
                        input={"query": query, "top_k": top_k, "collection": text_collection},
                        output={
                            "num_results": len(results),
                            "node_ids": [r.get("node_id") for r in results[:50]],
                        },
                        metadata={"duration_ms": dt},
                    )

                    for r in results:
                        node_id = r.get("node_id")
                        if node_id and node_id not in all_candidates:
                            r["source"] = "semantic"
                            all_candidates[node_id] = r

            # 3. Sort: structural first, then by similarity score
            sorted_candidates = sorted(
                all_candidates.values(),
                key=lambda x: (
                    0 if x.get("source") == "structural" else 1,
                    -(x.get("similarity_score") or 0)
                )
            )
            seeds = sorted_candidates[:max_seeds]

            logger.info(f"[RETRIEVE] Got {len(all_candidates)} unique candidates, selected {len(seeds)} seeds")
            structural_count = len([s for s in seeds if s.get("source") == "structural"])
            semantic_count = len(seeds) - structural_count
            logger.info(f"[RETRIEVE] Seeds breakdown: {structural_count} structural, {semantic_count} semantic")

            tracer.event(
                "retrieve_seeds.results",
                output={
                    "total_unique": len(all_candidates),
                    "selected_seeds": len(seeds),
                    "seed_ids": [s.get("node_id") for s in seeds],
                    "structural_count": structural_count,
                    "semantic_count": semantic_count,
                },
            )

            out = {
                "pending_seeds": seeds,
                "processing_complete": len(seeds) == 0,
            }
            if span is not None and hasattr(span, "update"):
                span.update(output={"num_seeds": len(seeds), "structural": structural_count, "semantic": semantic_count})
            return out
    
    return retrieve_seeds
