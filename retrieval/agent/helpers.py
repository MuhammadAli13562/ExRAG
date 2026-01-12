"""
Helper functions for the retrieval agent.
"""
import re
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def is_evidence_sufficient(
    high_count: int, 
    medium_count: int, 
    query_type: str = "conceptual",
    route: str = "semantic_only"
) -> bool:
    """
    Check if we have sufficient evidence for synthesis.
    
    Anthropic principle: Use explicit criteria based on query type
    instead of always asking LLM "is this enough?"
    
    Args:
        high_count: Number of high-grade evidence nodes
        medium_count: Number of medium-grade evidence nodes
        query_type: "structural", "conceptual", or "hybrid"
        route: "structural_only", "semantic_only", or "hybrid"
    
    Returns:
        True if evidence is sufficient for synthesis
    """
    # Structural queries (specific section lookup) need fewer nodes
    if query_type == "structural" or route == "structural_only":
        # Found the exact section we were looking for
        return high_count >= 1 or (high_count >= 0 and medium_count >= 2)
    
    # Hybrid queries need moderate coverage
    if query_type == "hybrid" or route == "hybrid":
        return (high_count >= 2) or (high_count >= 1 and medium_count >= 3)
    
    # Conceptual queries need broader coverage
    # (original thresholds for backward compatibility)
    return (high_count >= 3) or (high_count >= 2 and medium_count >= 3) or (medium_count >= 8)


def should_early_exit(
    high_count: int, 
    medium_count: int,
    query_type: str = "conceptual",
    route: str = "semantic_only"
) -> bool:
    """
    Check if we have excellent evidence and can skip remaining seeds.
    
    Anthropic principle: Set clear stopping points to control costs.
    
    Args:
        high_count: Number of high-grade evidence nodes
        medium_count: Number of medium-grade evidence nodes
        query_type: "structural", "conceptual", or "hybrid"
        route: "structural_only", "semantic_only", or "hybrid"
    
    Returns:
        True if we should exit early with current evidence
    """
    # Structural queries: exit early once we find the target
    if query_type == "structural" or route == "structural_only":
        return high_count >= 2 or (high_count >= 1 and medium_count >= 2)
    
    # Hybrid: slightly lower bar than conceptual
    if query_type == "hybrid" or route == "hybrid":
        return (high_count >= 3) or (high_count >= 2 and medium_count >= 3)
    
    # Conceptual: need more evidence before early exit
    return (high_count >= 5) or (high_count >= 3 and medium_count >= 4)


def count_evidence_grades(evidence_pool: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Count high and medium grade nodes in evidence pool."""
    high = len([n for n in evidence_pool if n.get("relevance_grade") == "high"])
    medium = len([n for n in evidence_pool if n.get("relevance_grade") == "medium"])
    return high, medium
