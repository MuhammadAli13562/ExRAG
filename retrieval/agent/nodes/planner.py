"""
Planner node for the retrieval agent.
"""
import json
import time
import logging
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..state import AgentState
from ..helpers import detect_structural_intent

logger = logging.getLogger(__name__)


def create_planner(llm: ChatOpenAI, tracer, agent_model: str):
    """
    Create the planner node function.
    
    Args:
        llm: The LLM instance to use
        tracer: Langfuse tracer for observability
        agent_model: Model name for logging
    
    Returns:
        Planner node function
    """
    
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
    
    def _clamp_int(val: Any, lo: int, hi: int, default: int) -> int:
        try:
            n = int(val)
        except Exception:
            return default
        return max(lo, min(hi, n))
    
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
                metadata={"duration_ms": dt, "model": agent_model},
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
    
    return planner
