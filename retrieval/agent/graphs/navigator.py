"""
Navigator sub-graph for structural document navigation.
"""
import json
import time
import logging
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field as PydanticField

from ..state import NavigatorState
from ..constants import SCOPE_PARAMS, MAX_SCOPE_LEVEL, NAVIGATOR_MAX_ITERATIONS
from ..prompts import NAVIGATOR_GOAL_PROMPT, NAVIGATOR_REFLECTION_PROMPT
from ...tools import RetrievalTools

logger = logging.getLogger(__name__)


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
