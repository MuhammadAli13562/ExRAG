"""
Agent package for LangGraph-based agentic retrieval system.

This package provides a modular architecture for the retrieval agent,
with dependency injection for better testability.

Usage:
    from retrieval.agent import create_agent, query_agent
    
    # Create and run agent
    agent = create_agent(title_collection, text_collection)
    result = agent.invoke(initial_state)
    
    # Or use the convenience function
    result = query_agent(user_query, title_collection, text_collection)
"""
import logging
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from .state import AgentState, NavigatorState, RetrievalLoopState
from .graphs.navigator import create_navigator_subgraph
from .graphs.retrieval_loop import create_retrieval_loop_subgraph
from .nodes.planner import create_planner
from .nodes.navigate import create_navigate_structure
from .nodes.retrieve import create_retrieve_seeds
from .nodes.loop_runner import create_run_retrieval_loop

from .constants import (
    SCOPE_PARAMS,
    MAX_SCOPE_LEVEL,
    NAVIGATOR_MAX_ITERATIONS,
    RETRIEVAL_SCOPE_PARAMS,
    MAX_RETRIEVAL_SCOPE,
    MAX_SYNTHESIS_RETRIES,
    RETRIEVAL_MAX_ITERATIONS,
)
from .prompts import (
    RETRIEVAL_REFLECTION_PROMPT,
    NAVIGATOR_GOAL_PROMPT,
    NAVIGATOR_REFLECTION_PROMPT,
    SYSTEM_PROMPT,
)
from .helpers import (
    detect_structural_intent,
    is_evidence_sufficient,
    should_early_exit,
    count_evidence_grades,
)
from .grading import grade_single_node, grade_neighbors_batch

from ..config import AGENT_MODEL, TEMPERATURE, OPENAI_API_KEY, MAX_ITERATIONS, RECURSION_LIMIT
from ..tools import RetrievalTools
from ..langfuse_tracing import get_tracer

logger = logging.getLogger(__name__)

__all__ = [
    # Main API
    'create_agent',
    'query_agent',
    
    # State types
    'AgentState',
    'NavigatorState',
    'RetrievalLoopState',
    
    # Sub-graph creators
    'create_navigator_subgraph',
    'create_retrieval_loop_subgraph',
    
    # Constants
    'SCOPE_PARAMS',
    'MAX_SCOPE_LEVEL',
    'NAVIGATOR_MAX_ITERATIONS',
    'RETRIEVAL_SCOPE_PARAMS',
    'MAX_RETRIEVAL_SCOPE',
    'MAX_SYNTHESIS_RETRIES',
    'RETRIEVAL_MAX_ITERATIONS',
    
    # Prompts
    'RETRIEVAL_REFLECTION_PROMPT',
    'NAVIGATOR_GOAL_PROMPT',
    'NAVIGATOR_REFLECTION_PROMPT',
    'SYSTEM_PROMPT',
    
    # Helpers
    'detect_structural_intent',
    'is_evidence_sufficient',
    'should_early_exit',
    'count_evidence_grades',
    
    # Grading
    'grade_single_node',
    'grade_neighbors_batch',
]


def create_agent(
    title_collection: str,
    text_collection: str,
    retrieval_tools: Optional[RetrievalTools] = None,
    llm: Optional[ChatOpenAI] = None,
    tracer=None,
):
    """
    Create a LangGraph agent for retrieval with dependency injection.
    
    Args:
        title_collection: Name of the title-indexed collection
        text_collection: Name of the text-indexed collection
        retrieval_tools: Optional RetrievalTools instance (creates default if not provided)
        llm: Optional ChatOpenAI instance (creates default if not provided)
        tracer: Optional tracer instance (creates default if not provided)
    
    Returns:
        Compiled LangGraph agent
    """
    # Initialize defaults if not provided
    if retrieval_tools is None:
        logger.info("Initializing default RetrievalTools")
        retrieval_tools = RetrievalTools()
    
    if llm is None:
        logger.info(f"Initializing default LLM: {AGENT_MODEL}")
        llm = ChatOpenAI(
            model=AGENT_MODEL,
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY
        )
    
    if tracer is None:
        tracer = get_tracer()
    
    # Create sub-graphs
    navigator_subgraph = create_navigator_subgraph(
        llm=llm,
        retrieval_tools_instance=retrieval_tools,
        title_collection=title_collection,
        text_collection=text_collection,
        tracer=tracer
    )
    
    retrieval_loop_subgraph = create_retrieval_loop_subgraph(
        llm=llm,
        retrieval_tools_instance=retrieval_tools,
        text_collection=text_collection,
        tracer=tracer
    )
    
    # Create node functions
    planner = create_planner(llm, tracer, AGENT_MODEL)
    # Pass retrieval_tools for deterministic navigation (Anthropic: "start simple")
    navigate_structure = create_navigate_structure(
        navigator_subgraph, 
        tracer, 
        retrieval_tools=retrieval_tools, 
        text_collection=text_collection
    )
    retrieve_seeds = create_retrieve_seeds(retrieval_tools, text_collection, tracer)
    run_retrieval_loop = create_run_retrieval_loop(retrieval_loop_subgraph, text_collection, tracer)
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner)
    workflow.add_node("navigate", navigate_structure)
    workflow.add_node("retrieve", retrieve_seeds)
    workflow.add_node("retrieval_loop", run_retrieval_loop)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Main flow: planner → navigate → retrieve → retrieval_loop → END
    workflow.add_edge("planner", "navigate")
    workflow.add_edge("navigate", "retrieve")
    workflow.add_edge("retrieve", "retrieval_loop")
    workflow.add_edge("retrieval_loop", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def query_agent(
    user_query: str,
    title_collection: str,
    text_collection: str,
    verbose: bool = True,
    deep_log: bool = False,
    retrieval_tools: Optional[RetrievalTools] = None,
    llm: Optional[ChatOpenAI] = None,
) -> Dict[str, Any]:
    """
    Query the retrieval agent.
    
    Args:
        user_query: The user's question/query
        title_collection: Name of the title-indexed collection
        text_collection: Name of the text-indexed collection
        verbose: Whether to print intermediate steps
        deep_log: Whether to log all tool inputs and outputs in detail
        retrieval_tools: Optional RetrievalTools instance (for testing)
        llm: Optional ChatOpenAI instance (for testing)
    
    Returns:
        Dictionary containing the final answer and conversation history
    """
    logger.info("="*80)
    logger.info(f"Starting agent query: '{user_query}'")
    logger.info(f"Title collection: {title_collection}")
    logger.info(f"Text collection: {text_collection}")
    if deep_log:
        logger.info("🔍 DEEP LOGGING ENABLED - All tool inputs/outputs will be logged")
    logger.info("="*80)
    
    # Create the agent with optional dependency injection
    logger.info("Creating agent graph")
    tracer = get_tracer()
    
    agent = create_agent(
        title_collection=title_collection,
        text_collection=text_collection,
        retrieval_tools=retrieval_tools,
        llm=llm,
        tracer=tracer,
    )
    
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
        effective_recursion_limit = max(RECURSION_LIMIT, (MAX_ITERATIONS + 5) * 10)

        logger.info(f"Invoking agent (recursion_limit={effective_recursion_limit}, max_attempts={MAX_ITERATIONS})")
        try:
            final_state = agent.invoke(
                initial_state,
                {
                    "recursion_limit": effective_recursion_limit,
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
            # Best-effort fallback: if recursion_limit is hit anyway, rerun with a higher cap
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
        
        if verbose:
            print(f"\nERROR: {error_msg}\n")
        return {
            "answer": error_msg,
            "messages": [],
            "evidence_count": 0
        }
