"""
Grading functions for evaluating node relevance.
"""
import json
import re
import logging
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


def grade_single_node(node: Dict[str, Any], query: str, llm) -> str:
    """
    Grade ONE node individually for relevance to the query.

    Args:
        node: The node to grade (must have text and optionally title)
        query: The user's query
        llm: The LLM instance to use

    Returns:
        Grade string: 'high', 'medium', 'low', or 'irrelevant'
    """
    title = node.get("title") or node.get("metadata", {}).get("title", "")
    text = (node.get("text") or "")[:800]
    node_id = node.get("node_id", "unknown")

    prompt = f"""Grade this node's relevance to answering the query.

QUERY: {query}

NODE [{node_id}]:
Title: {title}
Text: {text}

Return ONLY one word: high, medium, low, or irrelevant

Criteria:
- high: Directly answers or is essential to answering the query
- medium: Provides useful context or partial information
- low: Tangentially related but not very useful
- irrelevant: Not related to the query at all"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, 'content') else str(response)
        grade = raw.strip().lower()
        # Normalize to valid grades
        if grade not in ("high", "medium", "low", "irrelevant"):
            grade = "medium"  # Default to medium if parse fails
        logger.debug(f"[GRADE_SINGLE] Node {node_id}: {grade}")
        return grade
    except Exception as e:
        logger.warning(f"[GRADE_SINGLE] Failed to grade node {node_id}: {e}")
        return "medium"


def grade_neighbors_batch(neighbors: List[Dict[str, Any]], query: str, llm) -> List[Dict[str, Any]]:
    """
    Grade multiple neighbors in a single LLM call for efficiency.

    Args:
        neighbors: List of nodes to grade
        query: The user's query
        llm: The LLM instance to use

    Returns:
        List of nodes with 'relevance_grade' added to each
    """
    if not neighbors:
        return []

    # Build batch grading prompt
    nodes_text = []
    for i, n in enumerate(neighbors[:15]):  # Limit to 15 per batch
        node_id = n.get("node_id", "unknown")
        title = n.get("title") or n.get("metadata", {}).get("title", "")
        text = (n.get("text") or "")[:400]
        nodes_text.append(f"[{i+1}] ID:{node_id} | Title:{title}\n{text}")

    prompt = f"""Grade each node's relevance to the query.

QUERY: {query}

NODES:
{chr(10).join(nodes_text)}

Return ONLY a valid JSON array with objects:
[{{"index": 1, "node_id": "XXXX", "relevance": "high|medium|low|irrelevant"}}, ...]

Criteria:
- high: Directly answers or is essential
- medium: Useful context or partial info
- low: Tangentially related
- irrelevant: Not related"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a relevance grader. Output only valid JSON array."),
            HumanMessage(content=prompt)
        ])
        raw = response.content if hasattr(response, 'content') else str(response)

        # Parse JSON response
        clean_raw = raw.strip()
        if clean_raw.startswith("```"):
            clean_raw = re.sub(r"^```(?:json)?\n?", "", clean_raw)
            clean_raw = re.sub(r"\n?```$", "", clean_raw)

        grades = json.loads(clean_raw)

        # Build grade map
        grade_map = {}
        for g in grades:
            nid = g.get("node_id")
            rel = g.get("relevance", "medium")
            if rel not in ("high", "medium", "low", "irrelevant"):
                rel = "medium"
            grade_map[nid] = rel

        # Apply grades to neighbors
        result = []
        for i, n in enumerate(neighbors[:15]):
            node_id = n.get("node_id")
            grade = grade_map.get(node_id, "medium")
            result.append({**n, "relevance_grade": grade})

        logger.info(f"[GRADE_BATCH] Graded {len(result)} neighbors")
        return result

    except Exception as e:
        logger.warning(f"[GRADE_BATCH] Failed to parse batch grades: {e}")
        # Fallback: assign medium to all
        return [{**n, "relevance_grade": "medium"} for n in neighbors[:15]]
