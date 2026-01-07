"""
Helper functions for the retrieval agent.
"""
import re
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def detect_structural_intent(query: str) -> Dict[str, Any]:
    """
    Detect structural patterns in a query: chapter refs, position hints, section keywords.

    Args:
        query: The user's query

    Returns:
        Dict with keys:
            - has_structural: bool - whether structural navigation should be used
            - chapter: int or None - chapter number if mentioned
            - position: str or None - "end" or "beginning"
            - section_keywords: List[str] - detected section type keywords
    """
    hints: Dict[str, Any] = {
        "has_structural": False,
        "chapter": None,
        "position": None,
        "section_keywords": [],
    }

    query_lower = query.lower()

    # Detect chapter reference (e.g., "chapter 4", "ch. 12", "chapter4")
    chapter_match = re.search(r'(?:chapter|ch\.?)\s*(\d+)', query_lower)
    if chapter_match:
        hints["chapter"] = int(chapter_match.group(1))
        hints["has_structural"] = True

    # Detect position hints
    if any(p in query_lower for p in ["end of", "at the end", "last", "final", "concluding"]):
        hints["position"] = "end"
        hints["has_structural"] = True
    elif any(p in query_lower for p in ["beginning of", "at the beginning", "start of", "first", "intro", "opening"]):
        hints["position"] = "beginning"
        hints["has_structural"] = True

    # Detect section type keywords
    section_keywords = [
        "review", "questions", "summary", "critical thinking",
        "key terms", "introduction", "exercises", "problems",
        "glossary", "vocabulary", "objectives", "learning outcomes",
        "test yourself", "self-assessment", "quiz", "practice"
    ]
    for kw in section_keywords:
        if kw in query_lower:
            hints["section_keywords"].append(kw)
            hints["has_structural"] = True

    if hints["has_structural"]:
        logger.info(f"[STRUCTURAL] Detected structural intent: chapter={hints['chapter']}, "
                    f"position={hints['position']}, keywords={hints['section_keywords']}")

    return hints


def is_evidence_sufficient(high_count: int, medium_count: int) -> bool:
    """
    Check if we have sufficient evidence for synthesis.
    
    Sufficient if:
    - 3+ high-grade nodes, OR
    - 2+ high + 3+ medium, OR
    - 8+ medium (for broad queries)
    """
    return (high_count >= 3) or (high_count >= 2 and medium_count >= 3) or (medium_count >= 8)


def should_early_exit(high_count: int, medium_count: int) -> bool:
    """
    Check if we have excellent evidence and can skip remaining seeds.
    
    Early exit if:
    - 5+ high-grade nodes, OR
    - 3+ high + 4+ medium
    """
    return (high_count >= 5) or (high_count >= 3 and medium_count >= 4)


def count_evidence_grades(evidence_pool: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Count high and medium grade nodes in evidence pool."""
    high = len([n for n in evidence_pool if n.get("relevance_grade") == "high"])
    medium = len([n for n in evidence_pool if n.get("relevance_grade") == "medium"])
    return high, medium
