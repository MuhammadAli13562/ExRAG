"""
LangGraph-based agentic retrieval system.
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
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


# =========================================================================
# Navigator Sub-graph State and Constants
# =========================================================================

# Scope levels for navigator retry mechanism
SCOPE_PARAMS = {
    0: {"radius": 10, "top_k": 5, "name": "narrow"},    # Initial focused search
    1: {"radius": 15, "top_k": 8, "name": "medium"},    # After first failure
    2: {"radius": 25, "top_k": 12, "name": "broad"},    # Maximum expansion
}

MAX_SCOPE_LEVEL = 2
NAVIGATOR_MAX_ITERATIONS = 8  # Per scope level


class NavigatorState(TypedDict):
    """State for the navigator sub-graph."""
    # Input from parent (set by parse_goal)
    goal: str                                  # Natural language goal
    chapter: Optional[int]                     # Target chapter number (if specified)
    position: Optional[str]                    # "end", "beginning", or None
    section_keywords: List[str]                # Keywords to look for

    # Navigation tracking
    landmark_node_id: Optional[str]            # Chapter/section landmark found
    current_position: str                      # Current node being explored around
    explored_centers: List[str]                # Nodes explored (prevent re-exploration)
    seen_nodes: Dict[str, str]                 # node_id -> title mapping

    # Results
    candidate_nodes: List[Dict[str, Any]]      # Potential matches found
    found_nodes: List[str]                     # Confirmed found node IDs

    # Control flow
    messages: Annotated[List[BaseMessage], operator.add]  # Conversation history
    iteration: int                             # Current iteration count
    max_iterations: int                        # Max allowed per scope (default 8)
    scope_level: int                           # 0=narrow, 1=medium, 2=broad
    status: str                                # searching|verifying|found|failed|retry
    last_action: str                           # Description of last action taken
    reflection: str                            # Agent's reflection on progress

    # Output (for returning to parent graph)
    structural_seeds: List[Dict[str, Any]]     # Final output nodes


# Goal-oriented navigation prompt (replaces prescriptive prompt)
NAVIGATOR_GOAL_PROMPT = """You are a document navigator. Your goal is to find specific sections within a document's structure.

YOUR GOAL: {goal}

TOOLS AVAILABLE:
- nav_search_title(query, top_k): Semantic search on section titles. Returns nodes with similarity scores.
- nav_explore_titles(node_id, direction, radius): See titles around a node.
  - direction: "up" (earlier in doc), "down" (later), "both"
  - radius: how many nodes to explore (default based on scope)
- nav_peek_content(node_id): Read a node's content preview to verify it matches.
- nav_mark_found(node_ids): Mark nodes as found when you've located the target sections.

CURRENT STATE:
- Landmark: {landmark}
- Current position: {current_position}
- Explored {num_explored} centers, seen {num_seen} nodes
- Scope: {scope_name} (radius={radius}, top_k={top_k})
- Iteration: {iteration}/{max_iterations}

KEY INSIGHTS:
- For chapter-specific content: Find the chapter landmark first, then explore from there.
- Sections like "Review Questions" or "Summary" exist in EVERY chapter - verify you're in the right one.
- Position hints: "end of chapter" = explore DOWN from chapter header, "beginning" = explore UP.
- If not making progress, try exploring from edge nodes you've already seen.
- When confident you've found the target, call nav_mark_found() with the node IDs.

Think about your goal and current state. What action will make progress toward finding the target?"""


# Reflection prompt for navigator
NAVIGATOR_REFLECTION_PROMPT = """Reflect on your navigation progress.

GOAL: {goal}
ITERATION: {iteration}/{max_iterations}
SCOPE: {scope_name}
LANDMARK: {landmark}
CURRENT POSITION: {current_position}
EXPLORED CENTERS: {num_explored}
CANDIDATES FOUND: {num_candidates}
LAST ACTION: {last_action}

Evaluate your progress:
1. Are you making progress toward the goal? (yes/no/uncertain)
2. What should happen next? (continue/verify/expand_scope/give_up)
3. Brief reason (one sentence)

Answer in format: progress|action|reason"""


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


# =========================================================================
# Navigator Sub-graph Implementation
# =========================================================================

def create_navigator_subgraph(
    llm: ChatOpenAI,
    retrieval_tools_instance: RetrievalTools,
    title_collection: str,
    text_collection: str,
    tracer
):
    """
    Create the navigator sub-graph for structural document navigation.

    Args:
        llm: The LLM instance to use
        retrieval_tools_instance: RetrievalTools instance for document access
        title_collection: Name of the title-indexed collection
        text_collection: Name of the text-indexed collection
        tracer: Langfuse tracer for observability

    Returns:
        Compiled LangGraph sub-graph for navigation
    """
    from langchain_core.tools import StructuredTool
    from langchain_core.messages import ToolMessage
    from pydantic import BaseModel, Field as PydanticField

    # -------------------------------------------------------------------------
    # Navigation Tool Definitions (closures over collections)
    # -------------------------------------------------------------------------

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

    def nav_search_title(query: str, top_k: int = 5) -> str:
        """Search for sections by title (semantic search on headings)."""
        logger.info(f"[NAV] nav_search_title: query='{query}', top_k={top_k}")
        try:
            results = retrieval_tools_instance.search_by_title(query, title_collection, top_k)
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
        """Get section titles around a node to understand document structure."""
        logger.info(f"[NAV] nav_explore_titles: node_id={node_id}, direction={direction}, radius={radius}")
        try:
            titles = retrieval_tools_instance.explore_titles(node_id, text_collection, radius, direction)
            return json.dumps(titles, indent=2)
        except Exception as e:
            logger.error(f"[NAV] nav_explore_titles failed: {e}")
            return json.dumps({"error": str(e)})

    def nav_peek_content(node_id: str) -> str:
        """Get the full content of a specific node to verify it matches."""
        logger.info(f"[NAV] nav_peek_content: node_id={node_id}")
        try:
            node = retrieval_tools_instance.get_node(node_id, text_collection)
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
        """Mark nodes as the target. Call when you've found what you're looking for."""
        logger.info(f"[NAV] nav_mark_found: {node_ids}")
        return json.dumps({"status": "found", "node_ids": node_ids})

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

    # Bind tools to LLM
    nav_llm = llm.bind_tools(nav_tools)

    # -------------------------------------------------------------------------
    # Navigator Node Functions
    # -------------------------------------------------------------------------

    def parse_goal(state: NavigatorState) -> Dict[str, Any]:
        """
        Entry node: Parse structural hints into a natural language goal.
        Initialize navigation state.
        """
        chapter = state.get("chapter")
        position = state.get("position")
        keywords = state.get("section_keywords") or []

        # Build natural language goal
        goal_parts = []
        if keywords:
            goal_parts.append(f"Find the '{' '.join(keywords)}' section")
        if chapter is not None:
            goal_parts.append(f"in Chapter {chapter}")
        if position == "end":
            goal_parts.append("near the end of the chapter")
        elif position == "beginning":
            goal_parts.append("near the beginning of the chapter")

        goal = " ".join(goal_parts) if goal_parts else "Find relevant structural sections"

        logger.info(f"[NAVIGATOR] parse_goal: {goal}")

        return {
            "goal": goal,
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
            "last_action": "initialized",
            "reflection": "",
            "structural_seeds": [],
        }

    def execute_tools(state: NavigatorState) -> Dict[str, Any]:
        """
        Execute navigation tools based on LLM decisions.
        """
        scope = SCOPE_PARAMS.get(state.get("scope_level", 0), SCOPE_PARAMS[0])

        # Build the prompt with current state
        prompt = NAVIGATOR_GOAL_PROMPT.format(
            goal=state.get("goal", ""),
            landmark=state.get("landmark_node_id") or "Not found yet",
            current_position=state.get("current_position") or "Starting",
            num_explored=len(state.get("explored_centers", [])),
            num_seen=len(state.get("seen_nodes", {})),
            scope_name=scope["name"],
            radius=scope["radius"],
            top_k=scope["top_k"],
            iteration=state.get("iteration", 0) + 1,
            max_iterations=state.get("max_iterations", NAVIGATOR_MAX_ITERATIONS),
        )

        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=prompt))

        logger.info(f"[NAVIGATOR] execute_tools: iteration {state.get('iteration', 0) + 1}")

        t0 = time.time()
        response = nav_llm.invoke(messages)
        dt = int((time.time() - t0) * 1000)

        tracer.event(
            "navigator.llm_call",
            input={"iteration": state.get("iteration", 0) + 1},
            output={"has_tool_calls": bool(response.tool_calls)},
            metadata={"duration_ms": dt},
        )

        new_messages = [response]
        updates: Dict[str, Any] = {}

        # Track state changes
        explored_centers = list(state.get("explored_centers", []))
        seen_nodes = dict(state.get("seen_nodes", {}))
        landmark_node_id = state.get("landmark_node_id")
        current_position = state.get("current_position", "")
        candidate_nodes = list(state.get("candidate_nodes", []))
        action_descriptions = []

        if not response.tool_calls:
            # LLM didn't call any tools - might be stuck
            logger.warning("[NAVIGATOR] No tool calls from LLM")
            action_descriptions.append("No action taken")
        else:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                logger.info(f"[NAVIGATOR] Tool call: {tool_name}({tool_args})")

                # Execute tool
                if tool_name == "nav_search_title":
                    result = nav_search_title(**tool_args)
                    action_descriptions.append(f"Searched titles for '{tool_args.get('query', '')}'")

                    # Track results
                    try:
                        data = json.loads(result)
                        if isinstance(data, list) and data:
                            # First result becomes landmark if not set
                            if not landmark_node_id:
                                landmark_node_id = data[0].get("node_id")
                                current_position = landmark_node_id
                            for r in data:
                                nid = r.get("node_id")
                                if nid:
                                    seen_nodes[nid] = r.get("title", "")
                    except json.JSONDecodeError:
                        pass

                elif tool_name == "nav_explore_titles":
                    center_node = tool_args.get("node_id", "")

                    # Check re-exploration
                    if center_node in explored_centers:
                        result = json.dumps({
                            "warning": f"Already explored around node {center_node}",
                            "suggestion": "Try exploring from a different node at the edge of what you've seen."
                        })
                        action_descriptions.append(f"Blocked re-exploration of {center_node}")
                    else:
                        result = nav_explore_titles(**tool_args)
                        explored_centers.append(center_node)
                        current_position = center_node
                        action_descriptions.append(f"Explored titles around {center_node}")

                        # Track seen nodes
                        try:
                            data = json.loads(result)
                            if isinstance(data, list):
                                for t in data:
                                    nid = t.get("node_id")
                                    if nid:
                                        seen_nodes[nid] = t.get("title", "")
                        except json.JSONDecodeError:
                            pass

                elif tool_name == "nav_peek_content":
                    result = nav_peek_content(**tool_args)
                    action_descriptions.append(f"Peeked at content of {tool_args.get('node_id', '')}")

                elif tool_name == "nav_mark_found":
                    result = nav_mark_found(**tool_args)
                    found_ids = tool_args.get("node_ids", [])
                    action_descriptions.append(f"Marked found: {found_ids}")
                    updates["found_nodes"] = found_ids
                    updates["status"] = "verifying"

                else:
                    result = json.dumps({"error": f"Unknown tool: {tool_name}"})
                    action_descriptions.append(f"Unknown tool: {tool_name}")

                new_messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

        return {
            "messages": new_messages,
            "iteration": state.get("iteration", 0) + 1,
            "last_action": "; ".join(action_descriptions) if action_descriptions else "No action",
            "explored_centers": explored_centers,
            "seen_nodes": seen_nodes,
            "landmark_node_id": landmark_node_id,
            "current_position": current_position,
            "candidate_nodes": candidate_nodes,
            **updates
        }

    def reflect(state: NavigatorState) -> Dict[str, Any]:
        """
        Reflect on navigation progress and decide next action.
        """
        # CRITICAL: Preserve "found" status - don't override if verification already succeeded
        current_status = state.get("status", "searching")
        if current_status == "found":
            logger.info("[NAVIGATOR] reflect: Status already 'found', preserving it")
            return {
                "reflection": "Verification succeeded - nodes found",
                "status": "found",
            }
        
        scope = SCOPE_PARAMS.get(state.get("scope_level", 0), SCOPE_PARAMS[0])

        prompt = NAVIGATOR_REFLECTION_PROMPT.format(
            goal=state.get("goal", ""),
            iteration=state.get("iteration", 0),
            max_iterations=state.get("max_iterations", NAVIGATOR_MAX_ITERATIONS),
            scope_name=scope["name"],
            landmark=state.get("landmark_node_id") or "Not found",
            current_position=state.get("current_position") or "Starting",
            num_explored=len(state.get("explored_centers", [])),
            num_candidates=len(state.get("candidate_nodes", [])),
            last_action=state.get("last_action", "None"),
        )

        t0 = time.time()
        response = llm.invoke([
            SystemMessage(content="You are reflecting on document navigation progress. Be concise."),
            HumanMessage(content=prompt)
        ])
        dt = int((time.time() - t0) * 1000)

        raw = response.content if hasattr(response, 'content') else str(response)

        # Parse reflection: progress|action|reason
        parts = raw.strip().split("|")
        progress = parts[0].strip().lower() if len(parts) > 0 else "uncertain"
        action = parts[1].strip().lower() if len(parts) > 1 else "continue"
        reason = parts[2].strip() if len(parts) > 2 else ""

        logger.info(f"[NAVIGATOR] reflect: progress={progress}, action={action}, reason={reason}")

        tracer.event(
            "navigator.reflect",
            input={"iteration": state.get("iteration", 0)},
            output={"progress": progress, "action": action, "reason": reason},
            metadata={"duration_ms": dt},
        )

        # Determine new status based on reflection
        # If we're already verifying (found nodes marked), stay in verifying
        if current_status == "verifying":
            new_status = "verifying"
        elif action == "give_up":
            # If we haven't exhausted scope levels, retry with broader scope
            if state.get("scope_level", 0) < MAX_SCOPE_LEVEL:
                new_status = "retry"
            else:
                new_status = "failed"
        elif action == "expand_scope":
            if state.get("scope_level", 0) < MAX_SCOPE_LEVEL:
                new_status = "retry"
            else:
                new_status = "searching"  # Can't expand further, keep trying
        elif action == "verify" and state.get("candidate_nodes"):
            new_status = "verifying"
        else:
            new_status = "searching"

        return {
            "reflection": f"{progress}: {reason}",
            "status": new_status,
        }

    def verify(state: NavigatorState) -> Dict[str, Any]:
        """
        Verify that found nodes actually match the goal.
        """
        found_nodes = state.get("found_nodes", [])
        chapter = state.get("chapter")
        keywords = state.get("section_keywords", [])

        if not found_nodes:
            logger.info("[NAVIGATOR] verify: No found nodes to verify")
            return {"status": "searching"}

        logger.info(f"[NAVIGATOR] verify: Checking {len(found_nodes)} nodes")

        verified = []
        for node_id in found_nodes[:5]:  # Limit verification to 5 nodes
            try:
                node = retrieval_tools_instance.get_node(node_id, text_collection)
                if not node:
                    continue

                title = (node.get("title") or "").lower()
                text = (node.get("text") or "").lower()

                # Check if keywords match
                keyword_match = any(kw.lower() in title or kw.lower() in text for kw in keywords) if keywords else True

                # For now, trust that if keywords match, it's valid
                # (More sophisticated chapter verification could be added)
                if keyword_match:
                    verified.append(node_id)
                    logger.info(f"[NAVIGATOR] verify: Node {node_id} verified")
                else:
                    logger.info(f"[NAVIGATOR] verify: Node {node_id} rejected (keyword mismatch)")

            except Exception as e:
                logger.warning(f"[NAVIGATOR] verify: Error checking node {node_id}: {e}")

        if verified:
            return {
                "found_nodes": verified,
                "status": "found"
            }
        else:
            # Verification failed - continue searching
            return {
                "found_nodes": [],
                "status": "searching",
                "reflection": "Candidates did not verify - continuing search"
            }

    def retry_broader(state: NavigatorState) -> Dict[str, Any]:
        """
        Expand search scope and retry navigation.
        """
        current_scope = state.get("scope_level", 0)
        new_scope = min(current_scope + 1, MAX_SCOPE_LEVEL)

        scope_params = SCOPE_PARAMS.get(new_scope, SCOPE_PARAMS[MAX_SCOPE_LEVEL])

        logger.info(f"[NAVIGATOR] retry_broader: Expanding from scope {current_scope} to {new_scope} ({scope_params['name']})")

        return {
            "scope_level": new_scope,
            "status": "searching",
            "iteration": 0,  # Reset iteration count for new scope
            "explored_centers": [],  # Clear explored centers for fresh exploration
            "candidate_nodes": [],
            "found_nodes": [],
            "reflection": f"Expanding to {scope_params['name']} scope (radius={scope_params['radius']})",
            "messages": [HumanMessage(content=f"Previous search at narrower scope didn't find the target. Now using {scope_params['name']} scope. Goal: {state.get('goal', '')}")],
        }

    def exit_success(state: NavigatorState) -> Dict[str, Any]:
        """
        Exit with found nodes - fetch full content and return as structural_seeds.
        """
        found_nodes = state.get("found_nodes", [])

        logger.info(f"[NAVIGATOR] exit_success: Returning {len(found_nodes)} nodes")

        structural_seeds = []
        try:
            nodes = retrieval_tools_instance.get_nodes(found_nodes, text_collection)
            for i, node in enumerate(nodes):
                structural_seeds.append({
                    **node,
                    "source": "structural",
                    "similarity_score": 1.0 - (i * 0.01),  # Slightly decrease for ordering
                    "relevance_grade": "high",
                })
        except Exception as e:
            logger.error(f"[NAVIGATOR] exit_success: Failed to fetch nodes: {e}")

        return {"structural_seeds": structural_seeds, "status": "found"}

    def exit_failure(state: NavigatorState) -> Dict[str, Any]:
        """
        Exit after exhausting all options - return empty seeds.
        """
        logger.warning(
            f"[NAVIGATOR] exit_failure: Failed to find target after scope={state.get('scope_level')}, "
            f"iterations={state.get('iteration')}"
        )
        return {"structural_seeds": [], "status": "failed"}

    # -------------------------------------------------------------------------
    # Routing Function
    # -------------------------------------------------------------------------

    def route_navigator(state: NavigatorState) -> str:
        """Route based on navigation status."""
        status = state.get("status", "searching")
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", NAVIGATOR_MAX_ITERATIONS)
        scope_level = state.get("scope_level", 0)

        logger.debug(f"[NAVIGATOR] route: status={status}, iter={iteration}/{max_iter}, scope={scope_level}")

        if status == "found":
            return "exit_success"

        if status == "failed":
            return "exit_failure"

        if status == "retry":
            return "retry_broader"

        if status == "verifying":
            return "verify"

        # Still searching - check iteration limit
        if iteration >= max_iter:
            if scope_level < MAX_SCOPE_LEVEL:
                return "retry_broader"
            else:
                return "exit_failure"

        return "execute_tools"

    # -------------------------------------------------------------------------
    # Build Sub-graph
    # -------------------------------------------------------------------------

    navigator = StateGraph(NavigatorState)

    # Add nodes
    navigator.add_node("parse_goal", parse_goal)
    navigator.add_node("execute_tools", execute_tools)
    navigator.add_node("reflect", reflect)
    navigator.add_node("verify", verify)
    navigator.add_node("retry_broader", retry_broader)
    navigator.add_node("exit_success", exit_success)
    navigator.add_node("exit_failure", exit_failure)

    # Entry point
    navigator.set_entry_point("parse_goal")

    # Fixed edges
    navigator.add_edge("parse_goal", "execute_tools")
    navigator.add_edge("execute_tools", "reflect")
    # Verify routes conditionally based on status (success -> exit_success, failure -> reflect)
    navigator.add_conditional_edges(
        "verify",
        route_navigator,
        {
            "exit_success": "exit_success",
            "reflect": "reflect",
            "execute_tools": "execute_tools",
            "retry_broader": "retry_broader",
            "exit_failure": "exit_failure",
        }
    )
    navigator.add_edge("retry_broader", "execute_tools")

    # Conditional routing from reflect
    navigator.add_conditional_edges(
        "reflect",
        route_navigator,
        {
            "execute_tools": "execute_tools",
            "verify": "verify",
            "retry_broader": "retry_broader",
            "exit_success": "exit_success",
            "exit_failure": "exit_failure",
        }
    )

    # Terminal edges
    navigator.add_edge("exit_success", END)
    navigator.add_edge("exit_failure", END)

    return navigator.compile()


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
    # Navigator Sub-graph Integration
    # =========================================================================

    # Create the navigator sub-graph
    navigator_subgraph = create_navigator_subgraph(
        llm=llm,
        retrieval_tools_instance=retrieval_tools,
        title_collection=title_collection,
        text_collection=text_collection,
        tracer=tracer
    )

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
                
                # Don't truncate high-grade nodes - they're directly relevant and need full context
                # For medium-grade nodes, use a higher limit to preserve more context
                if grade == "high":
                    snippet = text  # Full text for high-quality nodes
                else:
                    snippet = text[:5000]  # Increased from 1200 to 5000 for medium nodes
                
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

