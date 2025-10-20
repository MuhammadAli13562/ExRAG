"""
LangGraph-based agentic retrieval system.
"""
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

from .config import AGENT_MODEL, TEMPERATURE, OPENAI_API_KEY, MAX_ITERATIONS, RECURSION_LIMIT
from .tools import RetrievalTools

import logging

logger = logging.getLogger(__name__)

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
    import json
    logger.info(f"[TOOL] search_by_title called with query='{query[:50]}...', top_k={top_k}")
    
    if _DEEP_LOG_ENABLED:
        logger.info("="*80)
        logger.info("[DEEP LOG] search_by_title INPUT:")
        logger.info(f"  query: {query}")
        logger.info(f"  collection_name: {collection_name}")
        logger.info(f"  top_k: {top_k}")
        logger.info("="*80)
    
    try:
        results = retrieval_tools.search_by_title(query, collection_name, top_k)
        logger.info(f"[TOOL] search_by_title returned {len(results)} results")
        
        if _DEEP_LOG_ENABLED:
            logger.info("="*80)
            logger.info(f"[DEEP LOG] search_by_title OUTPUT ({len(results)} results):")
            for i, result in enumerate(results, 1):
                logger.info(f"\n  Result {i}:")
                logger.info(f"    node_id: {result.get('node_id', 'N/A')}")
                logger.info(f"    title: {result.get('title', 'N/A')[:100]}...")
                logger.info(f"    similarity: {result.get('similarity', 'N/A')}")
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
        logger.error(f"[TOOL] search_by_title failed: {e}", exc_info=True)
        return f"Error: {str(e)}"


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
                logger.info(f"    similarity: {result.get('similarity', 'N/A')}")
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
                logger.info(f"\n  Target Node:")
                logger.info(f"    node_id: {results['target_node'].get('node_id', 'N/A')}")
                logger.info(f"    title: {results['target_node'].get('title', 'N/A')}")
            if 'previous_nodes' in results:
                logger.info(f"\n  Previous Nodes: {len(results['previous_nodes'])} nodes")
                for node in results['previous_nodes']:
                    logger.info(f"    - {node.get('node_id', 'N/A')}: {node.get('title', 'N/A')[:50]}...")
            if 'next_nodes' in results:
                logger.info(f"\n  Next Nodes: {len(results['next_nodes'])} nodes")
                for node in results['next_nodes']:
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


# Define the agent state
class AgentState(TypedDict):
    """State of the retrieval agent."""
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    title_collection: str
    text_collection: str
    final_answer: str
    iteration_count: int


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
- Higher similarity scores (closer to 1.0) indicate better matches
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
    
    # Initialize the LLM with tools
    llm = ChatOpenAI(
        model=AGENT_MODEL,
        temperature=TEMPERATURE,
        api_key=OPENAI_API_KEY
    )
    
    # Bind tools to the LLM
    tools = [search_by_title, search_by_text, explore_nodes, list_collections]
    llm_with_tools = llm.bind_tools(tools)
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    def should_continue(state: AgentState) -> str:
        """Determine whether to continue or end."""
        messages = state["messages"]
        last_message = messages[-1]
        iteration_count = state.get("iteration_count", 0)
        
        logger.debug(f"[DECISION] Iteration {iteration_count}, checking if should continue")
        
        # Check iteration limit
        if iteration_count >= MAX_ITERATIONS:
            logger.info(f"[DECISION] Reached max iterations ({MAX_ITERATIONS}), ending")
            return "end"
        
        # If no tool calls, we're done
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            logger.info("[DECISION] No tool calls, ending agent execution")
            return "end"
        
        num_tool_calls = len(last_message.tool_calls)
        logger.info(f"[DECISION] Continuing with {num_tool_calls} tool call(s)")
        return "continue"
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Call the LLM with the current state."""
        messages = state["messages"]
        iteration_count = state.get("iteration_count", 0) + 1
        
        logger.info(f"[AGENT] Iteration {iteration_count}: Calling LLM")
        logger.debug(f"[AGENT] Current message count: {len(messages)}")
        
        # Add system message if this is the first call
        if len(messages) == 1:  # Only user message
            logger.debug("[AGENT] Adding system prompt to messages")
            # Inject collection names into system prompt
            enhanced_prompt = SYSTEM_PROMPT + f"""

COLLECTION NAMES (MUST USE THESE EXACT NAMES):
- For search_by_title, use collection_name: "{title_collection}"
- For search_by_text, use collection_name: "{text_collection}"
- For explore_nodes, use collection_name: "{title_collection}" or "{text_collection}" (either works)

CRITICAL: You MUST use these exact collection names when calling the tools. Do not make up collection names.

======================================================================================
CITATION REQUIREMENT - READ THIS BEFORE RESPONDING:
======================================================================================
Your final answer MUST include citations in this EXACT format:

For inline citations, use: [Node XXXX: Title]
Example: "The two kinds of charges are positive and negative [Node 0005: Two kinds of charges]."

For reference style, use numbered citations then list sources:
Example: "There are two kinds of charges [1][2]."
Then at the end:
"Sources:
[1] Node 0005: Two kinds of charges
[2] Node 0006: Cross-link"

EVERY SINGLE FACT must have a citation. No exceptions. If you provide ANY information without 
citing the source node ID, your answer is INCORRECT and UNACCEPTABLE.

Track which nodes you get information from during retrieval and cite them in your final answer.
======================================================================================
"""
            messages = [SystemMessage(content=enhanced_prompt)] + messages
        
        try:
            response = llm_with_tools.invoke(messages)
            
            # Log tool calls if any
            if hasattr(response, "tool_calls") and response.tool_calls:
                logger.info(f"[AGENT] LLM requested {len(response.tool_calls)} tool call(s)")
                for i, tool_call in enumerate(response.tool_calls, 1):
                    logger.debug(f"[AGENT]   Tool {i}: {tool_call.get('name', 'unknown')}")
            else:
                logger.info("[AGENT] LLM provided final answer (no tool calls)")
            
            return {
                "messages": [response],
                "iteration_count": iteration_count
            }
        except Exception as e:
            logger.error(f"[AGENT] Error calling LLM: {e}", exc_info=True)
            raise
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    
    # Tools always go back to agent
    workflow.add_edge("tools", "agent")
    
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
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "title_collection": title_collection,
        "text_collection": text_collection,
        "final_answer": "",
        "iteration_count": 0
    }
    
    # Run the agent
    if verbose:
        print(f"\n{'='*80}")
        print(f"QUERY: {user_query}")
        print(f"{'='*80}\n")
    
    try:
        logger.info("Invoking agent")
        final_state = agent.invoke(
            initial_state,
            {"recursion_limit": RECURSION_LIMIT}
        )
        
        # Extract final answer
        messages = final_state["messages"]
        final_message = messages[-1]
        iteration_count = final_state.get("iteration_count", 0)
        
        if isinstance(final_message, AIMessage):
            final_answer = final_message.content
        else:
            final_answer = str(final_message)
        
        logger.info(f"Agent completed successfully in {iteration_count} iterations")
        logger.info(f"Answer length: {len(final_answer)} characters")
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"FINAL ANSWER:")
            print(f"{'='*80}")
            print(final_answer)
            print(f"\n{'='*80}")
            print(f"Iterations: {iteration_count}")
            print(f"{'='*80}\n")
        
        # Reset deep log flag
        _DEEP_LOG_ENABLED = False
        
        return {
            "answer": final_answer,
            "messages": messages,
            "iteration_count": iteration_count
        }
        
    except Exception as e:
        error_msg = f"Error during agent execution: {str(e)}"
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        
        # Reset deep log flag
        _DEEP_LOG_ENABLED = False
        
        if verbose:
            print(f"\nERROR: {error_msg}\n")
        return {
            "answer": error_msg,
            "messages": [],
            "iteration_count": 0
        }

