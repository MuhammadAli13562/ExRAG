"""
State TypedDicts for the retrieval agent and its sub-graphs.
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
import operator


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


class RetrievalLoopState(TypedDict):
    """State for the RetrievalLoop sub-graph (ReAct-style retrieval)."""
    # Input from parent
    user_query: str
    subqueries: List[str]
    text_collection: str
    initial_seeds: List[Dict[str, Any]]        # Seeds from retrieve_seeds
    
    # Processing state
    pending_seeds: List[Dict[str, Any]]        # Seeds waiting to be processed
    visited_node_ids: List[str]                # MEMORY: don't re-process nodes
    evidence_pool: List[Dict[str, Any]]        # Accumulated relevant nodes (ranked)
    
    # Loop control
    iteration: int                             # Current iteration count
    max_iterations: int                        # Max allowed per scope
    scope_level: int                           # 0=focused, 1=expanded, 2=broad
    status: str                                # processing|reflecting|checking|retrying_search|synthesizing|validating|retrying_synthesis|success|best_effort
    seeds_processed: int                       # Count of seeds processed
    total_seeds: int                           # Total seeds at current scope
    
    # Reflection state
    reflection: str                            # Agent's reflection on progress
    last_action: str                           # Description of last action taken
    high_grade_count: int                      # Count of high-grade evidence nodes
    medium_grade_count: int                    # Count of medium-grade evidence nodes
    
    # Synthesis state
    final_answer: str                          # Generated answer
    synthesis_attempts: int                    # Number of synthesis attempts
    validation: Dict[str, Any]                 # Citation validation results
    synthesis_feedback: str                    # Feedback for retry
    
    # Output
    result: Dict[str, Any]                     # Final output
