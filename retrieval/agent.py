"""
LangGraph-based agentic retrieval system.
"""
from typing import TypedDict, Annotated, List, Dict, Any
import operator
import json
import re
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

from .config import AGENT_MODEL, TEMPERATURE, OPENAI_API_KEY, MAX_ITERATIONS, RECURSION_LIMIT
from .tools import RetrievalTools
from .langfuse_tracing import get_tracer

import logging

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


# Initialize retrieval tools
logger.info("Initializing retrieval tools for agent")
retrieval_tools = RetrievalTools()

# Global flag for deep logging (set by query_agent)
_DEEP_LOG_ENABLED = False


# Define tools using LangChain's @tool decorator
@tool
def search_by_title(query: str, collection_name: str, top_k: int = 5) -> str:
    """
    Search for relevant chunks using title-based semantic search.
    Use this when you want to find sections based on their titles or headings.
    
    Args:
        query: The search query describing what you're looking for
        collection_name: Name of the title-indexed collection to search
        top_k: Number of top results to return (default: 5)
    
    Returns:
        JSON string containing matching chunks with node_ids, titles, and similarity scores
    """
    # Title search is intentionally disabled (it can be misleading for this corpus).
    logger.warning("[TOOL] search_by_title is disabled; use search_by_text instead")
    return "Error: search_by_title is disabled. Use search_by_text for semantic retrieval."


@tool
def search_by_text(query: str, collection_name: str, top_k: int = 5) -> str:
    """
    Search for relevant chunks using text-based semantic search.
    Use this when you want to find content based on the actual text/content of sections.
    
    Args:
        query: The search query describing what you're looking for
        collection_name: Name of the text-indexed collection to search
        top_k: Number of top results to return (default: 5)
    
    Returns:
        JSON string containing matching chunks with node_ids, text content, and similarity scores
    """
    import json
    logger.info(f"[TOOL] search_by_text called with query='{query[:50]}...', top_k={top_k}")
    
    if _DEEP_LOG_ENABLED:
        logger.info("="*80)
        logger.info("[DEEP LOG] search_by_text INPUT:")
        logger.info(f"  query: {query}")
        logger.info(f"  collection_name: {collection_name}")
        logger.info(f"  top_k: {top_k}")
        logger.info("="*80)
    
    try:
        results = retrieval_tools.search_by_text(query, collection_name, top_k)
        logger.info(f"[TOOL] search_by_text returned {len(results)} results")
        
        if _DEEP_LOG_ENABLED:
            logger.info("="*80)
            logger.info(f"[DEEP LOG] search_by_text OUTPUT ({len(results)} results):")
            for i, result in enumerate(results, 1):
                logger.info(f"\n  Result {i}:")
                logger.info(f"    node_id: {result.get('node_id', 'N/A')}")
                logger.info(f"    text: {result.get('text', 'N/A')[:200]}...")
                logger.info(f"    similarity_score: {result.get('similarity_score', 'N/A')}")
                if 'metadata' in result:
                    logger.info(f"    metadata: {result['metadata']}")
            logger.info("="*80)
        
        # Add citation reminder to output
        output = {
            "results": results,
            "CITATION_REMINDER": "🚨 MANDATORY: Use the node_id from these results to cite in your final answer. Format: [Node XXXX]"
        }
        return json.dumps(output, indent=2)
    except Exception as e:
        logger.error(f"[TOOL] search_by_text failed: {e}", exc_info=True)
        return f"Error: {str(e)}"


@tool
def explore_nodes(node_id: str, collection_name: str, direction: str = "both", count: int = 3) -> str:
    """
    Explore nodes above and/or below a specific node to get surrounding context.
    Use this when you find a relevant chunk and want to see what comes before or after it.
    
    Args:
        node_id: The ID of the node to explore around (e.g., "0042")
        collection_name: Name of the collection (to locate the source JSON)
        direction: Direction to explore - "up" (previous nodes), "down" (next nodes), or "both" (default: "both")
        count: Number of nodes to retrieve in each direction (default: 3, max: 10)
    
    Returns:
        JSON string containing the target node and surrounding nodes with full content
    """
    import json
    logger.info(f"[TOOL] explore_nodes called for node_id={node_id}, direction={direction}, count={count}")
    
    if _DEEP_LOG_ENABLED:
        logger.info("="*80)
        logger.info("[DEEP LOG] explore_nodes INPUT:")
        logger.info(f"  node_id: {node_id}")
        logger.info(f"  collection_name: {collection_name}")
        logger.info(f"  direction: {direction}")
        logger.info(f"  count: {count}")
        logger.info("="*80)
    
    try:
        results = retrieval_tools.explore_nodes(node_id, collection_name, direction, count)
        logger.info(f"[TOOL] explore_nodes returned context for node {node_id}")
        
        if _DEEP_LOG_ENABLED:
            logger.info("="*80)
            logger.info("[DEEP LOG] explore_nodes OUTPUT:")
            if 'target_node' in results:
                logger.info("\n  Target Node:")
                logger.info(f"    node_id: {results['target_node'].get('node_id', 'N/A')}")
                logger.info(f"    title: {results['target_node'].get('title', 'N/A')}")
            if 'nodes_above' in results:
                logger.info(f"\n  Nodes Above: {len(results['nodes_above'])} nodes")
                for node in results['nodes_above']:
                    logger.info(f"    - {node.get('node_id', 'N/A')}: {node.get('title', 'N/A')[:50]}...")
            if 'nodes_below' in results:
                logger.info(f"\n  Nodes Below: {len(results['nodes_below'])} nodes")
                for node in results['nodes_below']:
                    logger.info(f"    - {node.get('node_id', 'N/A')}: {node.get('title', 'N/A')[:50]}...")
            logger.info("="*80)
        
        return json.dumps(results, indent=2)
    except Exception as e:
        logger.error(f"[TOOL] explore_nodes failed: {e}", exc_info=True)
        return f"Error: {str(e)}"


@tool
def list_collections() -> str:
    """
    List all available collections in the vector database.
    Use this to discover what collections are available for searching.
    
    Returns:
        JSON string containing collection names, counts, and metadata
    """
    import json
    logger.info("[TOOL] list_collections called")
    try:
        results = retrieval_tools.list_collections()
        logger.info(f"[TOOL] list_collections returned {len(results)} collections")
        return json.dumps(results, indent=2)
    except Exception as e:
        logger.error(f"[TOOL] list_collections failed: {e}", exc_info=True)
        return f"Error: {str(e)}"


@tool
def get_node(node_id: str, collection_name: str) -> str:
    """
    Fetch a single node by node_id from the collection's backing JSON structure.

    Returns:
        JSON string with the node (node_id/title/text/line_num/...) for inspection/citation.
    """
    import json
    logger.info(f"[TOOL] get_node called for node_id={node_id}")
    try:
        node = retrieval_tools.get_node(node_id=node_id, collection_name=collection_name)
        return json.dumps({"node": node}, indent=2)
    except Exception as e:
        logger.error(f"[TOOL] get_node failed: {e}", exc_info=True)
        return f"Error: {str(e)}"


@tool
def get_nodes(node_ids: List[str], collection_name: str) -> str:
    """
    Fetch multiple nodes by node_id (batch).

    Returns:
        JSON string with nodes in the same order as requested (missing IDs omitted).
    """
    import json
    logger.info(f"[TOOL] get_nodes called for {len(node_ids)} node_id(s)")
    try:
        nodes = retrieval_tools.get_nodes(node_ids=node_ids, collection_name=collection_name)
        return json.dumps({"nodes": nodes, "count": len(nodes)}, indent=2)
    except Exception as e:
        logger.error(f"[TOOL] get_nodes failed: {e}", exc_info=True)
        return f"Error: {str(e)}"


@tool
def expand_around(node_id: str, collection_name: str, radius: int = 3) -> str:
    """
    Expand around a node by radius above/below (document order).
    """
    import json
    logger.info(f"[TOOL] expand_around called for node_id={node_id}, radius={radius}")
    try:
        results = retrieval_tools.expand_around(node_id=node_id, collection_name=collection_name, radius=radius)
        return json.dumps(results, indent=2)
    except Exception as e:
        logger.error(f"[TOOL] expand_around failed: {e}", exc_info=True)
        return f"Error: {str(e)}"


@tool
def expand_many(seed_node_ids: List[str], collection_name: str, radius: int = 3, max_nodes: int = 50) -> str:
    """
    Expand around multiple seed nodes, dedupe, and return a stitched evidence set.
    """
    import json
    logger.info(f"[TOOL] expand_many called for {len(seed_node_ids)} seed(s), radius={radius}, max_nodes={max_nodes}")
    try:
        results = retrieval_tools.expand_many(
            seed_node_ids=seed_node_ids, collection_name=collection_name, radius=radius, max_nodes=max_nodes
        )
        return json.dumps(results, indent=2)
    except Exception as e:
        logger.error(f"[TOOL] expand_many failed: {e}", exc_info=True)
        return f"Error: {str(e)}"


# Define the agent state
class AgentState(TypedDict):
    """State of the retrieval agent."""
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    title_collection: str
    text_collection: str
    final_answer: str
    plan: Dict[str, Any]
    validation: Dict[str, Any]

    # Structural navigation state
    structural_seeds: List[Dict[str, Any]]    # Nodes found via structural navigation

    # Per-seed processing state
    pending_seeds: List[Dict[str, Any]]       # Seeds waiting to be processed (max 5)
    visited_node_ids: List[str]               # MEMORY: don't re-process nodes
    evidence_pool: List[Dict[str, Any]]       # Accumulated relevant nodes (ranked)
    processing_complete: bool                 # True when all seeds processed


# Navigation system prompt for structural queries
NAVIGATION_SYSTEM_PROMPT = """You are a document navigation agent. Your job is to find specific sections WITHIN a specific chapter by exploring document structure.

CRITICAL RULES:
1. ALWAYS find the CHAPTER landmark FIRST (e.g., "Chapter 4")
2. NEVER search directly for section titles like "Review Questions" - they exist in EVERY chapter!
3. Navigate FROM the chapter landmark to find sections WITHIN that chapter
4. Sections belong to a chapter if they appear AFTER the chapter header and BEFORE the next chapter

You have these tools:
1. nav_search_title(query) - Search for chapter headers. Use ONLY for finding chapter landmarks.
2. nav_explore_titles(node_id, direction, radius) - See titles around a node
   - direction: "up" (earlier in doc), "down" (later), "both"
   - radius: how many nodes to explore (default 10)
3. nav_peek_content(node_id) - Read a node's content to verify it belongs to the right chapter
4. nav_mark_found(node_ids) - Call when you've found the correct section in the correct chapter

CORRECT STRATEGY for "Review Questions at end of Chapter 4":
1. Search: nav_search_title("Chapter 4") → find Chapter 4 header (e.g., node 0200)
2. Explore DOWN: nav_explore_titles("0200", direction="down", radius=15)
3. Look for "Review Questions" in the results that comes AFTER Chapter 4 but BEFORE Chapter 5
4. If not visible, explore further DOWN from the last node you saw
5. When found, verify it's before "Chapter 5" header, then call nav_mark_found()

WRONG STRATEGY (DO NOT DO THIS):
- Searching nav_search_title("Review Questions") - this finds ALL review questions from ALL chapters!
- Picking the first "Review Questions" you see without checking the chapter

The nav_status in tool results shows your position - use it to stay oriented!
"""


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


# System prompt for the agent
SYSTEM_PROMPT = """======================================================================================
🚨 MANDATORY CITATION REQUIREMENT - MOST IMPORTANT RULE 🚨
======================================================================================
YOU MUST CITE EVERY SINGLE PIECE OF INFORMATION WITH [Node XXXX] FORMAT.
NO EXCEPTIONS. NO INFORMATION WITHOUT A NODE CITATION. THIS IS ABSOLUTE.

Example of CORRECT answer:
"The two kinds of electric charges are positive and negative [Node 0005]. These were identified 
by Charles du Fay and named by Benjamin Franklin [Node 0005]. Like charges repel and unlike 
charges attract [Node 0006]."

Example of WRONG answer (NO CITATIONS - UNACCEPTABLE):
"The two kinds of electric charges are positive and negative."

======================================================================================

You are an intelligent retrieval agent that helps users find relevant information from a structured document.

You have access to tools that allow you to:
1. Search by title - Find sections based on their headings/titles (returns node_id in results)
2. Search by text - Find content based on actual text/content (returns node_id in results)
3. Explore nodes - View surrounding context (returns node_id in results)
4. List collections - See what collections are available

STRATEGY:
1. Start by understanding the user's query
2. Use both title and text search to find relevant candidates
3. Look at the node_id in each result - YOU WILL NEED THIS FOR CITATIONS
4. Identify the most promising chunks based on similarity scores
5. For good candidates, use explore_nodes to get surrounding context
6. Track all node_ids you use
7. Synthesize final answer WITH CITATIONS using the node_ids you collected

IMPORTANT:
- Node IDs are formatted as 4-digit strings (e.g., "0042", "0123")
- All nodes are in a flat structure (no hierarchy)
- Use explore_nodes to understand what comes before/after a node
- Don't stop after the first search - explore thoroughly
- Higher similarity_score values (closer to 1.0) indicate better matches
- EVERY tool result includes node_id - TRACK THESE FOR YOUR CITATIONS

CITATION FORMAT (REQUIRED FOR EVERY FACT):
Format: [Node XXXX] or [Node XXXX: Section Title]

Examples:
- "There are two kinds of charges: positive and negative [Node 0005]."
- "Like charges repel while unlike charges attract [Node 0006: Cross-link]."
- "The coulomb is the SI unit of charge [Node 0010]."

Reference style is also acceptable:
"There are two kinds of charges [1]. They repel or attract [2]."
Sources: [1] Node 0005, [2] Node 0006

REMEMBER: When tools return results, they include node_id. Use those node_ids in your final answer.
"""


def create_agent(title_collection: str, text_collection: str):
    """
    Create a LangGraph agent for retrieval.
    
    Args:
        title_collection: Name of the title-indexed collection
        text_collection: Name of the text-indexed collection
    
    Returns:
        Compiled LangGraph agent
    """
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model=AGENT_MODEL,
        temperature=TEMPERATURE,
        api_key=OPENAI_API_KEY
    )
    
    tracer = get_tracer()

    def _state_summary(state: AgentState) -> Dict[str, Any]:
        """Compact state summary for tracing (avoids dumping huge node texts)."""
        plan = state.get("plan") or {}
        return {
            "user_query": state.get("user_query"),
            "plan": plan,
            "pending_seeds": len(state.get("pending_seeds") or []),
            "num_visited": len(state.get("visited_node_ids") or []),
            "num_evidence": len(state.get("evidence_pool") or []),
            "processing_complete": state.get("processing_complete", False),
            "validation": state.get("validation") or {},
        }

    def _node_preview(nodes: List[Dict[str, Any]], max_nodes: int = 10, text_chars: int = 400) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for n in (nodes or [])[:max_nodes]:
            out.append(
                {
                    "node_id": n.get("node_id"),
                    "title": n.get("title"),
                    "line_num": n.get("line_num"),
                    "text_preview": ((n.get("text") or "")[:text_chars] + ("..." if (n.get("text") and len(n.get("text")) > text_chars) else "")),
                }
            )
        return out

    # =========================================================================
    # Navigation tools for structural queries
    # =========================================================================

    def nav_search_title(query: str, top_k: int = 5) -> str:
        """
        Search for sections by title (semantic search on headings).
        Use to find chapter landmarks or section headers.

        Args:
            query: e.g., "Chapter 4", "Review Questions"
            top_k: Number of results (default: 5)

        Returns:
            JSON with matches: [{node_id, title, similarity_score}, ...]
        """
        logger.info(f"[NAV] nav_search_title: query='{query}', top_k={top_k}")
        try:
            results = retrieval_tools.search_by_title(query, title_collection, top_k)
            # Simplify output for navigation
            simplified = [
                {
                    "node_id": r.get("node_id"),
                    "title": r.get("title"),
                    "similarity_score": round(r.get("similarity_score", 0), 3)
                }
                for r in results
            ]
            return json.dumps(simplified, indent=2)
        except Exception as e:
            logger.error(f"[NAV] nav_search_title failed: {e}")
            return json.dumps({"error": str(e)})

    def nav_explore_titles(node_id: str, direction: str = "both", radius: int = 10) -> str:
        """
        Get section titles around a node to understand document structure.

        Args:
            node_id: Center node to explore around
            direction: "up" (earlier in doc), "down" (later), "both"
            radius: How many nodes in each direction (default: 10)

        Returns:
            JSON with titles: [{node_id, title, position}, ...]
        """
        logger.info(f"[NAV] nav_explore_titles: node_id={node_id}, direction={direction}, radius={radius}")
        try:
            # Map direction values
            dir_map = {"up": "up", "down": "down", "both": "both"}
            actual_direction = dir_map.get(direction, "both")

            titles = retrieval_tools.explore_titles(node_id, text_collection, radius, actual_direction)
            return json.dumps(titles, indent=2)
        except Exception as e:
            logger.error(f"[NAV] nav_explore_titles failed: {e}")
            return json.dumps({"error": str(e)})

    def nav_peek_content(node_id: str) -> str:
        """
        Get the full content of a specific node to verify it matches.
        Use sparingly - only when title is ambiguous.

        Args:
            node_id: Node to peek at

        Returns:
            JSON with {node_id, title, text_preview (first 500 chars)}
        """
        logger.info(f"[NAV] nav_peek_content: node_id={node_id}")
        try:
            node = retrieval_tools.get_node(node_id, text_collection)
            if node:
                return json.dumps({
                    "node_id": node.get("node_id"),
                    "title": node.get("title"),
                    "text_preview": (node.get("text") or "")[:500]
                }, indent=2)
            return json.dumps({"error": f"Node {node_id} not found"})
        except Exception as e:
            logger.error(f"[NAV] nav_peek_content failed: {e}")
            return json.dumps({"error": str(e)})

    def nav_mark_found(node_ids: List[str]) -> str:
        """
        Mark nodes as the target. Call this when you've found what the user is looking for.
        This will end the navigation and return these nodes.

        Args:
            node_ids: List of node IDs that match the user's request

        Returns:
            Confirmation message
        """
        logger.info(f"[NAV] nav_mark_found: {node_ids}")
        return json.dumps({"status": "found", "node_ids": node_ids})

    # Navigation tool definitions for LLM binding
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field as PydanticField

    class NavSearchTitleInput(BaseModel):
        query: str = PydanticField(description="Search query for title, e.g. 'Chapter 4'")
        top_k: int = PydanticField(default=5, description="Number of results")

    class NavExploreTitlesInput(BaseModel):
        node_id: str = PydanticField(description="Center node ID to explore around")
        direction: str = PydanticField(default="both", description="Direction: 'up', 'down', or 'both'")
        radius: int = PydanticField(default=10, description="How many nodes in each direction")

    class NavPeekContentInput(BaseModel):
        node_id: str = PydanticField(description="Node ID to peek at")

    class NavMarkFoundInput(BaseModel):
        node_ids: List[str] = PydanticField(description="List of node IDs that match the user's request")

    nav_tools = [
        StructuredTool.from_function(
            func=nav_search_title,
            name="nav_search_title",
            description="Search for sections by title to find chapter landmarks or section headers",
            args_schema=NavSearchTitleInput
        ),
        StructuredTool.from_function(
            func=nav_explore_titles,
            name="nav_explore_titles",
            description="Get section titles around a node to understand document structure",
            args_schema=NavExploreTitlesInput
        ),
        StructuredTool.from_function(
            func=nav_peek_content,
            name="nav_peek_content",
            description="Get a node's content preview to verify it matches (use sparingly)",
            args_schema=NavPeekContentInput
        ),
        StructuredTool.from_function(
            func=nav_mark_found,
            name="nav_mark_found",
            description="Mark nodes as found. Call when you've located the target sections.",
            args_schema=NavMarkFoundInput
        ),
    ]

    def navigate_structure(state: AgentState) -> Dict[str, Any]:
        """
        Navigation sub-agent: iteratively explores document structure to find target sections.
        Only runs if structural intent was detected in planner.
        """
        plan = state.get("plan") or {}
        hints = plan.get("structural_hints", {})
        user_query = state["user_query"]

        if not hints.get("has_structural"):
            logger.info("[NAVIGATE] No structural intent detected, skipping navigation")
            return {"structural_seeds": []}

        logger.info(f"[NAVIGATE] Starting structural navigation for: {user_query}")
        logger.info(f"[NAVIGATE] Hints: chapter={hints.get('chapter')}, position={hints.get('position')}, keywords={hints.get('section_keywords')}")

        with tracer.span("navigate_structure", input={"hints": hints, "query": user_query}) as span:
            # Create navigation LLM with tools
            nav_llm = llm.bind_tools(nav_tools)

            # Build initial prompt
            chapter_num = hints.get('chapter')
            position = hints.get('position', 'any')
            keywords = hints.get('section_keywords', [])

            if chapter_num:
                user_msg = f"""Find: "{user_query}"

TARGET: Find "{' '.join(keywords)}" section in Chapter {chapter_num}
POSITION: {position} of the chapter (end = explore DOWN, beginning = explore UP)

STEP 1: Search for "Chapter {chapter_num}" to find the chapter landmark
STEP 2: Explore {position if position in ('down', 'up') else 'down' if position == 'end' else 'down'} from the chapter landmark
STEP 3: Find the section with keywords: {keywords}
STEP 4: VERIFY it's after Chapter {chapter_num} header and BEFORE Chapter {chapter_num + 1} header
STEP 5: Call nav_mark_found with the correct node IDs

DO NOT search for "{' '.join(keywords)}" directly - it will match wrong chapters!"""
            else:
                user_msg = f"""Find: "{user_query}"

Keywords to look for: {keywords}
Position hint: {position}

Start by searching for a relevant landmark, then explore to find the target section.
When you find the matching sections, call nav_mark_found with their node IDs."""

            # Navigation state tracking
            nav_state = {
                "landmark_node": None,       # The chapter/section landmark we started from
                "landmark_title": None,      # Title of the landmark
                "current_position": None,    # Current node we're exploring around
                "direction": hints.get("position", "both"),  # "end" → down, "beginning" → up
                "explored_centers": set(),   # Nodes we've explored around (don't re-explore)
                "seen_nodes": {},            # node_id → title (all nodes we've seen)
                "search_results": [],        # Results from title searches
            }

            def get_nav_status() -> str:
                """Build status string for LLM context."""
                status_lines = ["=== NAVIGATION STATUS ==="]
                if nav_state["landmark_node"]:
                    status_lines.append(f"Landmark: [{nav_state['landmark_node']}] {nav_state['landmark_title']}")
                if nav_state["current_position"]:
                    status_lines.append(f"Current position: {nav_state['current_position']}")
                status_lines.append(f"Direction hint: {nav_state['direction']} (end=down, beginning=up)")
                status_lines.append(f"Explored around {len(nav_state['explored_centers'])} nodes: {sorted(nav_state['explored_centers'])}")
                status_lines.append(f"Total nodes seen: {len(nav_state['seen_nodes'])}")
                return "\n".join(status_lines)

            messages = [
                SystemMessage(content=NAVIGATION_SYSTEM_PROMPT),
                HumanMessage(content=user_msg)
            ]

            found_nodes: List[str] = []
            max_iterations = 8

            for iteration in range(max_iterations):
                logger.info(f"[NAVIGATE] Iteration {iteration + 1}/{max_iterations}, explored centers: {len(nav_state['explored_centers'])}, seen: {len(nav_state['seen_nodes'])}")

                t0 = time.time()
                response = nav_llm.invoke(messages)
                dt = int((time.time() - t0) * 1000)

                messages.append(response)

                tracer.event(
                    "navigate.llm_call",
                    input={"iteration": iteration + 1, "explored": len(nav_state['explored_centers'])},
                    output={"has_tool_calls": bool(response.tool_calls)},
                    metadata={"duration_ms": dt},
                )

                # Check for tool calls
                if not response.tool_calls:
                    logger.info("[NAVIGATE] No tool calls, ending navigation")
                    break

                # Process each tool call
                from langchain_core.messages import ToolMessage
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    logger.info(f"[NAVIGATE] Tool call: {tool_name}({tool_args})")

                    # Execute tool with state tracking
                    if tool_name == "nav_mark_found":
                        found_nodes = tool_args.get("node_ids", [])
                        result = nav_mark_found(found_nodes)
                        logger.info(f"[NAVIGATE] Found target nodes: {found_nodes}")

                    elif tool_name == "nav_search_title":
                        result = nav_search_title(**tool_args)
                        # Track search results and set landmark
                        try:
                            search_data = json.loads(result)
                            if isinstance(search_data, list) and search_data:
                                nav_state["search_results"] = search_data
                                # First result becomes landmark if not set
                                if not nav_state["landmark_node"]:
                                    nav_state["landmark_node"] = search_data[0].get("node_id")
                                    nav_state["landmark_title"] = search_data[0].get("title")
                                    nav_state["current_position"] = nav_state["landmark_node"]
                                for r in search_data:
                                    nid = r.get("node_id")
                                    if nid:
                                        nav_state["seen_nodes"][nid] = r.get("title", "")
                        except:
                            pass

                    elif tool_name == "nav_explore_titles":
                        center_node = tool_args.get("node_id", "")

                        # Check if already explored this center
                        if center_node in nav_state["explored_centers"]:
                            result = json.dumps({
                                "error": f"Already explored around node {center_node}!",
                                "suggestion": "Pick a node at the EDGE of what you've seen to explore further.",
                                "explored_centers": sorted(nav_state["explored_centers"]),
                                "nav_status": get_nav_status()
                            })
                            logger.warning(f"[NAVIGATE] Blocked re-exploration of {center_node}")
                        else:
                            result = nav_explore_titles(**tool_args)
                            nav_state["explored_centers"].add(center_node)
                            nav_state["current_position"] = center_node

                            # Track all seen nodes from results
                            try:
                                titles_data = json.loads(result)
                                if isinstance(titles_data, list):
                                    for t in titles_data:
                                        nid = t.get("node_id")
                                        if nid:
                                            nav_state["seen_nodes"][nid] = t.get("title", "")
                                    # Append status to help LLM
                                    result = json.dumps({
                                        "titles": titles_data,
                                        "nav_status": get_nav_status()
                                    }, indent=2)
                            except:
                                pass

                    elif tool_name == "nav_peek_content":
                        result = nav_peek_content(**tool_args)
                    else:
                        result = json.dumps({"error": f"Unknown tool: {tool_name}"})

                    messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

                if found_nodes:
                    break

            # Log final status
            if not found_nodes:
                logger.warning(f"[NAVIGATE] Finished without finding target. Explored {len(nav_state['explored_centers'])} centers, saw {len(nav_state['seen_nodes'])} nodes")
            else:
                logger.info(f"[NAVIGATE] Found targets after exploring {len(nav_state['explored_centers'])} centers")

            # Fetch full content for found nodes
            structural_seeds: List[Dict[str, Any]] = []
            if found_nodes:
                try:
                    nodes = retrieval_tools.get_nodes(found_nodes, text_collection)
                    for i, node in enumerate(nodes):
                        structural_seeds.append({
                            **node,
                            "source": "structural",
                            "similarity_score": 1.0 - (i * 0.01),  # Slightly decrease for ordering
                            "relevance_grade": "high",  # Structural matches are high relevance
                        })
                    logger.info(f"[NAVIGATE] Retrieved {len(structural_seeds)} structural seeds")
                except Exception as e:
                    logger.error(f"[NAVIGATE] Failed to fetch found nodes: {e}")

            if span is not None:
                span.update(output={
                    "found_node_ids": found_nodes,
                    "num_structural_seeds": len(structural_seeds),
                    "iterations_used": iteration + 1,
                })

            return {"structural_seeds": structural_seeds}

    def planner(state: AgentState) -> Dict[str, Any]:
        """
        Plan sub-queries and retrieval parameters.
        Returns a dict in state['plan'] with keys:
          subqueries: list[str]
          top_k: int
          radius: int
          max_evidence_nodes: int
          max_seeds: int
        """
        user_query = state["user_query"]
        logger.info("[STAGE] planner: building retrieval plan")
        with tracer.span("planner", input={"state": _state_summary(state)}) as span:

            sys = SystemMessage(
                content=(
                    "You are planning retrieval against a flat list of nodes.\n"
                    "Return ONLY valid JSON with these keys:\n"
                    "- subqueries: array of 1-5 short search queries\n"
                    "- top_k: integer 3-15\n"
                    "- radius: integer 1-8 (neighbor expansion)\n"
                    "- max_evidence_nodes: integer 20-120\n"
                    "- max_seeds: integer 3-10\n"
                    "Optimize for answer quality and completeness. Keep subqueries diverse.\n"
                )
            )
            t0 = time.time()
            resp = llm.invoke([sys, HumanMessage(content=user_query)])
            dt = int((time.time() - t0) * 1000)
            raw = resp.content if isinstance(resp, AIMessage) else str(resp)
            tracer.event(
                "llm.planner",
                input={"messages": [sys.content, user_query]},
                output={"response": raw},
                metadata={"duration_ms": dt, "model": AGENT_MODEL},
            )
        plan: Dict[str, Any]
        try:
            plan = json.loads(raw)
        except Exception:
            # Safe fallback (keeps system running even if planner outputs non-JSON)
            plan = {
                "subqueries": [user_query],
                "top_k": 8,
                "radius": 3,
                "max_evidence_nodes": 60,
                "max_seeds": 5,
            }

        subqueries = plan.get("subqueries") or [user_query]
        if isinstance(subqueries, str):
            subqueries = [subqueries]
        subqueries = [str(s).strip() for s in subqueries if str(s).strip()][:5] or [user_query]

        def _clamp_int(val: Any, lo: int, hi: int, default: int) -> int:
            try:
                n = int(val)
            except Exception:
                return default
            return max(lo, min(hi, n))

        # Detect structural intent (chapter refs, position hints, section keywords)
        structural_hints = detect_structural_intent(user_query)

        plan_norm = {
            "subqueries": subqueries,
            "top_k": _clamp_int(plan.get("top_k"), 3, 15, 8),
            "radius": _clamp_int(plan.get("radius"), 1, 8, 3),
            "max_evidence_nodes": _clamp_int(plan.get("max_evidence_nodes"), 20, 120, 60),
            "max_seeds": _clamp_int(plan.get("max_seeds"), 3, 10, 5),
            "structural_hints": structural_hints,  # Add structural navigation hints
        }
        out = {
            "plan": plan_norm,
            # Small artifacts for tracing/debugging
            "validation": {"planner_raw_length": len(raw), "has_structural": structural_hints.get("has_structural", False)},
        }
        if span is not None and hasattr(span, "update"):
            span.update(output=out)  # type: ignore[attr-defined]
        return out

    def retrieve_seeds(state: AgentState) -> Dict[str, Any]:
        """
        Retrieve initial seed candidates (max 5) using all subqueries.
        Seeds are the starting points for per-seed exploration.

        Merges structural seeds (from navigation) with semantic search results.
        Structural seeds are prioritized (appear first).
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

            # 2. Also do semantic search
            for query in subqueries:
                t0 = time.time()
                results = retrieval_tools.search_by_text(query, text_collection, top_k=top_k)
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

    def process_seed(state: AgentState) -> Dict[str, Any]:
        """
        Process ONE seed at a time:
        1. Grade the seed individually
        2. If relevant: expand around it (radius 3), batch-grade neighbors
        3. Add relevant nodes to evidence pool
        4. Move to next seed
        """
        seeds = state.get("pending_seeds") or []
        if not seeds:
            logger.info("[PROCESS_SEED] No more seeds to process")
            return {"processing_complete": True}

        # Pop first seed
        current_seed = seeds[0]
        remaining_seeds = seeds[1:]
        visited = set(state.get("visited_node_ids") or [])
        evidence_pool = list(state.get("evidence_pool") or [])
        evidence_ids = {e.get("node_id") for e in evidence_pool}

        # Track seed order (first seed = 0, second = 1, etc.)
        # Calculate from how many seeds we've already processed
        plan = state.get("plan") or {}
        max_seeds = int(plan.get("max_seeds") or 5)
        seed_order = max_seeds - len(seeds)  # 0 for first, 1 for second, etc.

        seed_id = current_seed.get("node_id", "unknown")
        logger.info(f"[PROCESS_SEED] Processing seed {seed_id}, {len(remaining_seeds)} remaining")

        with tracer.span("process_seed", input={"seed_id": seed_id, "remaining": len(remaining_seeds)}) as span:
            # 1. Grade this ONE seed individually
            t0 = time.time()
            seed_grade = grade_single_node(current_seed, state["user_query"], llm)
            dt = int((time.time() - t0) * 1000)

            tracer.event(
                "process_seed.grade",
                input={"seed_id": seed_id},
                output={"grade": seed_grade},
                metadata={"duration_ms": dt},
            )

            logger.info(f"[PROCESS_SEED] Seed {seed_id} graded as: {seed_grade}")

            # Mark seed as visited
            new_visited = list(visited) + [seed_id]

            if seed_grade not in ("high", "medium"):
                # Seed not relevant → skip to next
                logger.info(f"[PROCESS_SEED] Seed {seed_id} not relevant, skipping")
                out = {
                    "pending_seeds": remaining_seeds,
                    "visited_node_ids": new_visited,
                    "processing_complete": len(remaining_seeds) == 0,
                }
                if span is not None and hasattr(span, "update"):
                    span.update(output={"action": "skipped", "grade": seed_grade})
                return out

            # 2. Seed is relevant → expand around it (radius 3)
            plan = state.get("plan") or {}
            radius = int(plan.get("radius") or 3)

            try:
                expanded = retrieval_tools.expand_around(
                    node_id=seed_id,
                    collection_name=text_collection,
                    radius=radius
                )
                # expand_around returns {"target_node": {...}, "nodes_above": [...], "nodes_below": [...]}
                nodes_above = expanded.get("nodes_above", [])
                nodes_below = expanded.get("nodes_below", [])
                all_expanded_nodes = nodes_above + nodes_below
                # Filter out visited nodes and the seed itself
                neighbors = [n for n in all_expanded_nodes
                             if n.get("node_id") not in visited
                             and n.get("node_id") != seed_id
                             and n.get("node_id") not in evidence_ids]

                logger.info(f"[PROCESS_SEED] Expanded seed {seed_id}: {len(all_expanded_nodes)} total, {len(neighbors)} new neighbors")
            except Exception as e:
                logger.warning(f"[PROCESS_SEED] Failed to expand around {seed_id}: {e}")
                neighbors = []

            # 3. Batch-grade neighbors
            graded_neighbors = []
            if neighbors:
                t0 = time.time()
                graded_neighbors = grade_neighbors_batch(neighbors, state["user_query"], llm)
                dt = int((time.time() - t0) * 1000)

                tracer.event(
                    "process_seed.grade_neighbors",
                    input={"seed_id": seed_id, "num_neighbors": len(neighbors)},
                    output={
                        "relevant_count": len([n for n in graded_neighbors if n.get("relevance_grade") in ("high", "medium")]),
                        "graded_count": len(graded_neighbors),
                    },
                    metadata={"duration_ms": dt},
                )

            # Filter to relevant neighbors only
            relevant_neighbors = [n for n in graded_neighbors if n.get("relevance_grade") in ("high", "medium")]

            # 4. Accumulate evidence with seed_order for ranking
            # Add seed with its grade and seed_order
            seed_with_grade = {
                **current_seed,
                "relevance_grade": seed_grade,
                "seed_order": seed_order,
            }
            if seed_id not in evidence_ids:
                evidence_pool.append(seed_with_grade)

            # Add relevant neighbors with seed_order (avoid duplicates)
            for n in relevant_neighbors:
                nid = n.get("node_id")
                if nid and nid not in evidence_ids:
                    evidence_pool.append({**n, "seed_order": seed_order})
                    evidence_ids.add(nid)

            # Sort evidence pool: high first, then by seed_order, then by similarity
            # This prioritizes nodes from earlier (higher-similarity) seeds
            evidence_pool.sort(key=lambda x: (
                0 if x.get("relevance_grade") == "high" else 1,
                x.get("seed_order", 999),  # Earlier seeds rank higher
                -(x.get("similarity_score") or 0)
            ))

            # Update visited with all neighbors we saw
            new_visited = list(visited) + [seed_id] + [n.get("node_id") for n in neighbors if n.get("node_id")]

            logger.info(f"[PROCESS_SEED] Seed {seed_id}: added {1 + len(relevant_neighbors)} to evidence (total: {len(evidence_pool)})")

            tracer.event(
                "process_seed.summary",
                input={"seed_id": seed_id},
                output={
                    "seed_grade": seed_grade,
                    "neighbors_expanded": len(neighbors),
                    "neighbors_relevant": len(relevant_neighbors),
                    "total_evidence": len(evidence_pool),
                },
            )

            out = {
                "pending_seeds": remaining_seeds,
                "evidence_pool": evidence_pool,
                "visited_node_ids": new_visited,
                "processing_complete": len(remaining_seeds) == 0,
            }
            if span is not None and hasattr(span, "update"):
                span.update(output={
                    "action": "expanded",
                    "grade": seed_grade,
                    "neighbors_added": len(relevant_neighbors),
                    "total_evidence": len(evidence_pool),
                })
            return out

    def route_seed_processing(state: AgentState) -> str:
        """Route based on whether there are more seeds to process."""
        if state.get("processing_complete"):
            logger.info("[ROUTE] All seeds processed, proceeding to synthesize")
            return "done"
        logger.info("[ROUTE] More seeds to process, continuing loop")
        return "continue"

    def synthesize_answer(state: AgentState) -> Dict[str, Any]:
        """Synthesize final answer from accumulated evidence pool with citations."""
        user_query = state["user_query"]
        # Use evidence_pool (already sorted by grade → seed_order → similarity)
        evidence_pool = state.get("evidence_pool") or []

        # Take top evidence nodes - limit to 10-12 to reduce noise
        # Prioritize high-grade nodes, limit medium-grade
        high_nodes = [n for n in evidence_pool if n.get("relevance_grade") == "high"]
        medium_nodes = [n for n in evidence_pool if n.get("relevance_grade") == "medium"]

        # Take all high-grade (up to 8) + top medium-grade (up to 4)
        max_high = 8
        max_medium = 4
        evidence_nodes = high_nodes[:max_high] + medium_nodes[:max_medium]

        logger.info(f"[STAGE] synthesize: using {len(evidence_nodes)} evidence nodes "
                    f"({len(high_nodes[:max_high])} high + {len(medium_nodes[:max_medium])} medium, "
                    f"from pool of {len(evidence_pool)})")

        with tracer.span("synthesize", input={"num_evidence": len(evidence_nodes)}) as span:
            evidence_lines: List[str] = []
            for n in evidence_nodes:
                nid = n.get("node_id")
                title = (n.get("title") or "").strip()
                text = (n.get("text") or "").strip()
                grade = n.get("relevance_grade", "unknown")
                # Keep context compact but useful
                snippet = text[:1200]
                evidence_lines.append(f"[Node {nid}: {title}] (grade: {grade})\n{snippet}")

            sys = SystemMessage(
                content=(
                    "You are a retrieval QA system.\n"
                    "You MUST cite every sentence with [Node XXXX] or [Node XXXX: Title].\n"
                    "Only use information present in the provided evidence nodes.\n"
                    "If evidence is insufficient, say what is missing and still cite the closest relevant nodes.\n"
                )
            )
            human = HumanMessage(
                content=("QUESTION:\n" + user_query + "\n\nEVIDENCE:\n" + "\n\n---\n\n".join(evidence_lines))
            )

            t0 = time.time()
            resp = llm.invoke([sys, human])
            dt = int((time.time() - t0) * 1000)
            answer = resp.content if isinstance(resp, AIMessage) else str(resp)
            tracer.event(
                "llm.synthesize",
                input={"prompt": human.content},
                output={"response": answer},
                metadata={
                    "duration_ms": dt,
                    "model": AGENT_MODEL,
                    "evidence_node_ids": [n.get("node_id") for n in evidence_nodes[:50]],
                    "evidence_preview": _node_preview(evidence_nodes, max_nodes=10, text_chars=250),
                },
            )
            out = {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "validation": {**(state.get("validation") or {}), "answer_length": len(answer)},
            }
            if span is not None and hasattr(span, "update"):
                span.update(
                    output={
                        "answer": answer,
                        "answer_length": len(answer),
                        "evidence_node_ids": [n.get("node_id") for n in evidence_nodes],
                    }
                )  # type: ignore[attr-defined]
            return out

    def validate_citations(state: AgentState) -> Dict[str, Any]:
        """
        Validate citation format. NO RETRY - just log results and proceed.
        """
        answer = state.get("final_answer") or ""
        evidence_pool = state.get("evidence_pool") or []
        evidence_ids = {e.get("node_id") for e in evidence_pool if e.get("node_id")}

        # Split into sentences (simple heuristic)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        citation_re = re.compile(r"\[Node\s+(\d{4})(?::[^\]]+)?\]")

        missing_citation: List[str] = []
        cited_ids: List[str] = []

        for sent in sentences:
            if len(re.sub(r"\W+", "", sent)) < 15:
                continue
            found = citation_re.findall(sent)
            if not found:
                missing_citation.append(sent)
            else:
                cited_ids.extend(found)

        invalid_ids = sorted({cid for cid in cited_ids if cid not in evidence_ids})
        ok = (len(missing_citation) == 0) and (len(invalid_ids) == 0) and (len(sentences) > 0)

        validation = {
            "ok": ok,
            "num_sentences": len(sentences),
            "missing_citation_count": len(missing_citation),
            "invalid_citation_node_ids": invalid_ids,
            "cited_ids": cited_ids[:20],
        }

        logger.info(f"[VALIDATE] ok={ok}, missing={len(missing_citation)}, invalid={len(invalid_ids)}")

        tracer.event(
            "validate_citations",
            input={"answer_length": len(answer), "num_evidence": len(evidence_pool)},
            output={
                "validation": validation,
                "missing_citation_examples": missing_citation[:3],
            },
        )

        return {
            "validation": {**(state.get("validation") or {}), **validation},
        }

    # Build the graph with per-seed processing loop
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner)
    workflow.add_node("navigate", navigate_structure)  # NEW: structural navigation
    workflow.add_node("retrieve", retrieve_seeds)
    workflow.add_node("process_seed", process_seed)
    workflow.add_node("synthesize", synthesize_answer)
    workflow.add_node("validate", validate_citations)

    # Set entry point
    workflow.set_entry_point("planner")

    # Main flow: planner → navigate → retrieve → process_seed loop → synthesize → validate
    workflow.add_edge("planner", "navigate")
    workflow.add_edge("navigate", "retrieve")
    workflow.add_edge("retrieve", "process_seed")

    # PER-SEED LOOP: process each seed individually
    workflow.add_conditional_edges(
        "process_seed",
        route_seed_processing,
        {
            "continue": "process_seed",  # LOOP: More seeds to process
            "done": "synthesize",        # EXIT: All seeds processed
        },
    )

    # NO RETRY after synthesis - validate goes straight to END
    workflow.add_edge("synthesize", "validate")
    workflow.add_edge("validate", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def query_agent(
    user_query: str,
    title_collection: str,
    text_collection: str,
    verbose: bool = True,
    deep_log: bool = False
) -> Dict[str, Any]:
    """
    Query the retrieval agent.
    
    Args:
        user_query: The user's question/query
        title_collection: Name of the title-indexed collection
        text_collection: Name of the text-indexed collection
        verbose: Whether to print intermediate steps
        deep_log: Whether to log all tool inputs and outputs in detail
    
    Returns:
        Dictionary containing the final answer and conversation history
    """
    # Declare global at the start of function
    global _DEEP_LOG_ENABLED
    
    # Set global deep log flag
    _DEEP_LOG_ENABLED = deep_log
    
    logger.info("="*80)
    logger.info(f"Starting agent query: '{user_query}'")
    logger.info(f"Title collection: {title_collection}")
    logger.info(f"Text collection: {text_collection}")
    if deep_log:
        logger.info("🔍 DEEP LOGGING ENABLED - All tool inputs/outputs will be logged")
    logger.info("="*80)
    
    # Create the agent
    logger.info("Creating agent graph")
    agent = create_agent(title_collection, text_collection)
    tracer = get_tracer()
    logger.info(
        f"Langfuse tracing: tracer={type(tracer).__name__}, enabled={getattr(tracer, 'enabled', False)} "
        f"(set EXRAG_TRACING=langfuse and install `langfuse` python package)"
    )
    tracer.start_trace(
        name="ExRAG_Query",
        input={"user_query": user_query},
        metadata={
            "title_collection": title_collection,
            "text_collection": text_collection,
            "max_attempts": MAX_ITERATIONS,
        },
    )
    # After start_trace, log whether we actually have an active trace.
    logger.info(
        "Langfuse trace status: enabled=%s trace_id=%s last_error=%s",
        getattr(tracer, "enabled", False),
        getattr(tracer, "trace_id", None),
        getattr(tracer, "last_error", None),
    )
    
    # Create initial state for per-seed processing
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "title_collection": title_collection,
        "text_collection": text_collection,
        "final_answer": "",
        "plan": {},
        "validation": {},
        # Structural navigation state
        "structural_seeds": [],
        # Per-seed processing state
        "pending_seeds": [],
        "visited_node_ids": [],
        "evidence_pool": [],
        "processing_complete": False,
    }
    
    # Run the agent
    if verbose:
        print(f"\n{'='*80}")
        print(f"QUERY: {user_query}")
        print(f"{'='*80}\n")
    
    try:
        # Ensure LangGraph recursion_limit is always comfortably above our own max-attempt cap.
        # Each retry traverses multiple nodes; recursion_limit must scale with MAX_ITERATIONS.
        effective_recursion_limit = max(RECURSION_LIMIT, (MAX_ITERATIONS + 5) * 10)

        logger.info(f"Invoking agent (recursion_limit={effective_recursion_limit}, max_attempts={MAX_ITERATIONS})")
        try:
            final_state = agent.invoke(
                initial_state,
                {
                    "recursion_limit": effective_recursion_limit,
                    # LangSmith-friendly metadata (visible when tracing is enabled)
                    "run_name": "ExRAG_RetrievalAgent",
                    "tags": ["exrag", "retrieval", "langgraph"],
                    "metadata": {
                        "title_collection": title_collection,
                        "text_collection": text_collection,
                        "deep_log": deep_log,
                        "max_attempts": MAX_ITERATIONS,
                    },
                },
            )
        except Exception as e:
            # Best-effort fallback: if recursion_limit is hit anyway, rerun with a higher recursion cap.
            # We still stop by our own max_attempts and return the best answer.
            if "Recursion limit" in str(e):
                boosted = effective_recursion_limit * 3
                logger.warning(f"Recursion limit hit; retrying once with recursion_limit={boosted}")
                final_state = agent.invoke(
                    initial_state,
                    {
                        "recursion_limit": boosted,
                        "run_name": "ExRAG_RetrievalAgent",
                        "tags": ["exrag", "retrieval", "langgraph"],
                        "metadata": {
                            "title_collection": title_collection,
                            "text_collection": text_collection,
                            "deep_log": deep_log,
                            "max_attempts": MAX_ITERATIONS,
                            "retry_after_recursion_limit": True,
                        },
                    },
                )
            else:
                raise
        
        # Extract final answer
        messages = final_state["messages"]
        final_message = messages[-1] if messages else AIMessage(content=final_state.get("final_answer", ""))
        evidence_pool = final_state.get("evidence_pool") or []
        num_evidence = len(evidence_pool)

        final_answer = final_state.get("final_answer") or (
            final_message.content if isinstance(final_message, AIMessage) else str(final_message)
        )

        logger.info(f"Agent completed successfully with {num_evidence} evidence nodes")
        logger.info(f"Answer length: {len(final_answer)} characters")
        tracer.end_trace(
            output={"answer": final_answer},
            metadata={"num_evidence": num_evidence, "validation": final_state.get("validation")},
        )
        tracer.flush()

        if verbose:
            print(f"\n{'='*80}")
            print("FINAL ANSWER:")
            print(f"{'='*80}")
            print(final_answer)
            print(f"\n{'='*80}")
            print(f"Evidence nodes: {num_evidence}")
            print(f"{'='*80}\n")

        # Reset deep log flag
        _DEEP_LOG_ENABLED = False

        return {
            "answer": final_answer,
            "messages": messages,
            "evidence_count": num_evidence
        }
        
    except Exception as e:
        error_msg = f"Error during agent execution: {str(e)}"
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        tracer.end_trace(output={"error": error_msg}, metadata={"failed": True})
        tracer.flush()
        
        # Reset deep log flag
        _DEEP_LOG_ENABLED = False
        
        if verbose:
            print(f"\nERROR: {error_msg}\n")
        return {
            "answer": error_msg,
            "messages": [],
            "evidence_count": 0
        }

